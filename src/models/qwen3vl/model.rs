use anyhow::{Result, anyhow};
use candle_core::{D, DType, IndexOp, Shape, Tensor};
use candle_nn::{
    Activation, Embedding, Init, LayerNorm, Linear, Module, RmsNorm, VarBuilder, embedding, linear,
    linear_no_bias, rms_norm,
};

use crate::{
    models::{
        common::{GateUpDownMLP, TwoLinearMLP, eager_attention_forward, get_layer_norm},
        qwen3vl::config::{Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig},
    },
    position_embed::rope::{
        Qwen2_5VisionRotaryEmbedding, Qwen3VLTextRotaryEmbedding, apply_rotary_pos_emb,
        apply_rotary_pos_emb_vision,
    },
    utils::tensor_utils::{
        bitor_tensor, get_vision_next_indices, linspace, mask_index_add, masked_scatter_dim0,
        nonzero_index, prepare_causal_attention_mask, prod_tensor_last_dim, split_tensor,
        zero_index,
    },
};

pub struct Qwen3VLVisionPatchEmbed {
    conv3d_weight: Tensor,
    conv3d_bias: Tensor,
}

impl Qwen3VLVisionPatchEmbed {
    pub fn new(cfg: &Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let patch_size = cfg.patch_size;
        let temporal_patch_size = cfg.temporal_patch_size;
        let in_channels = cfg.in_channels;
        let embed_dim = cfg.hidden_size;
        // conv3d weight key: visual.patch_embed.proj.weight, value: Tensor[dims 1024, 3, 2, 16, 16; bf16, cuda:0]
        // (1024, 3, 2, 16, 16) -> (1024, 1536) -> (1536, 1024)
        let conv3d_weight = vb
            .get_with_hints(
                (
                    embed_dim,
                    in_channels,
                    temporal_patch_size,
                    patch_size,
                    patch_size,
                ),
                "proj.weight",
                Init::Const(1.),
            )?
            .flatten(1, 4)?
            .t()?;
        // (1024) -> (1, 1024)
        let conv3d_bias = vb
            .get_with_hints((embed_dim,), "proj.bias", Init::Const(0.))?
            .unsqueeze(0)?;
        Ok(Self {
            conv3d_weight,
            conv3d_bias,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // hidden_states shape:  (grid_t*grid_h*grid_w, c*temporal_patch_size*patch_size*patch_size)
        // ((), 1536) matmul (1536, 1024) -> ((), 1024)
        let hidden_states = hidden_states.matmul(&self.conv3d_weight)?;
        let hidden_states = hidden_states.broadcast_add(&self.conv3d_bias)?;
        Ok(hidden_states)
    }
}

pub struct Qwen3VLVisionPatchMerger {
    hidden_size: usize,
    use_postshuffle_norm: bool,
    norm: LayerNorm,
    linear_fc1: Linear,
    act_fn: Activation,
    linear_fc2: Linear,
}

impl Qwen3VLVisionPatchMerger {
    pub fn new(
        config: &Qwen3VLVisionConfig,
        vb: VarBuilder,
        use_postshuffle_norm: bool,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size * config.spatial_merge_size.pow(2);
        let norm_size = if use_postshuffle_norm {
            hidden_size
        } else {
            config.hidden_size
        };
        let norm = get_layer_norm(vb.pp("norm"), 1e-6, norm_size)?;
        let linear_fc1 = linear(hidden_size, hidden_size, vb.pp("linear_fc1"))?;
        let act_fn = Activation::Gelu;
        let linear_fc2 = linear(hidden_size, config.out_hidden_size, vb.pp("linear_fc2"))?;
        Ok(Self {
            hidden_size,
            use_postshuffle_norm,
            norm,
            linear_fc1,
            act_fn,
            linear_fc2,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.use_postshuffle_norm {
            xs.reshape(((), self.hidden_size))?
        } else {
            xs.clone()
        };
        let xs = self.norm.forward(&xs)?.reshape(((), self.hidden_size))?;
        let xs = self
            .linear_fc2
            .forward(&self.act_fn.forward(&self.linear_fc1.forward(&xs)?)?)?;
        Ok(xs)
    }
}

pub struct Qwen3VLVisionAttention {
    num_heads: usize,
    qkv: Linear,
    proj: Linear,
    scaling: f64,
}

impl Qwen3VLVisionAttention {
    pub fn new(config: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_heads;
        let head_dim = hidden_size / num_heads;
        let qkv = linear(hidden_size, hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(hidden_size, hidden_size, vb.pp("proj"))?;
        let scaling = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            num_heads,
            qkv,
            proj,
            scaling,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cu_seqlens: &Tensor,
    ) -> Result<Tensor> {
        // xs: (seq_len, hidden_size)
        let seq_length = xs.dim(0)?;
        // (seq_len, hidden_size) -> (seq_len, hidden_size*3)
        // -> (seq_len, 3, num_heads, head_dim)
        // -> (3, seq_len, num_heads, head_dim)
        let qkv_states = xs
            .apply(&self.qkv)?
            .reshape((seq_length, 3, self.num_heads, ()))?
            .permute((1, 0, 2, 3))?;
        // (seq_len, num_heads, head_dim)
        let query_states = qkv_states.i(0)?.contiguous()?;
        let key_states = qkv_states.i(1)?.contiguous()?;
        let value_states = qkv_states.i(2)?.contiguous()?;
        let (query_states, key_states) =
            apply_rotary_pos_emb_vision(&query_states, &key_states, cos, sin)?;
        // (seq_len, num_heads, head_dim) -> (num_heads, seq_len, head_dim) -> (1, num_heads, seq_len, head_dim)
        let query_states = query_states.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let key_states = key_states.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let value_states = value_states.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let cu_last_id = cu_seqlens.dim(0)? - 1;
        let lengths = cu_seqlens.i(1..)?.sub(&cu_seqlens.i(..cu_last_id)?)?;
        let chunks: Vec<usize> = lengths
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as usize)
            .collect();
        let q_splits = split_tensor(&query_states, &chunks, 2)?;
        let k_splits = split_tensor(&key_states, &chunks, 2)?;
        let v_splits = split_tensor(&value_states, &chunks, 2)?;

        let mut attn_outputs = Vec::new();
        for (q, (k, v)) in q_splits.iter().zip(k_splits.iter().zip(v_splits.iter())) {
            let output = eager_attention_forward(q, k, v, None, None, self.scaling)?;
            attn_outputs.push(output);
        }
        let attn_output = Tensor::cat(&attn_outputs, 1)?;
        let attn_output = attn_output.reshape((seq_length, ()))?.contiguous()?;
        let attn_ouput = attn_output.apply(&self.proj)?;
        Ok(attn_ouput)
    }
}

pub struct Qwen3VLVisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: Qwen3VLVisionAttention,
    mlp: TwoLinearMLP,
}

impl Qwen3VLVisionBlock {
    pub fn new(config: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-6, config.hidden_size)?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-6, config.hidden_size)?;
        let attn = Qwen3VLVisionAttention::new(config.clone(), vb.pp("attn"))?;
        let mlp = TwoLinearMLP::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            true,
            "linear_fc1",
            "linear_fc2",
        )?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cu_seqlens: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs = self.attn.forward(&xs, cos, sin, cu_seqlens)?;
        let xs = (residual + xs)?;
        let residual = xs.clone();
        let xs = self.mlp.forward(&self.norm2.forward(&xs)?)?;
        let xs = (residual + xs)?;
        Ok(xs)
    }
}

pub struct Qwen3VLVisionModel {
    spatial_merge_size: usize,
    patch_embed: Qwen3VLVisionPatchEmbed,
    pos_embed: Embedding,
    num_grid_per_side: u32,
    rotary_pos_emb: Qwen2_5VisionRotaryEmbedding,
    blocks: Vec<Qwen3VLVisionBlock>,
    merger: Qwen3VLVisionPatchMerger,
    deepstack_visual_indexes: Vec<usize>,
    deepstack_merger_list: Vec<Qwen3VLVisionPatchMerger>,
    dtype: DType,
}

impl Qwen3VLVisionModel {
    pub fn new(config: Qwen3VLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let spatial_merge_size = config.spatial_merge_size;
        let patch_embed = Qwen3VLVisionPatchEmbed::new(&config, vb.pp("patch_embed"))?;
        let pos_embed = embedding(
            config.num_position_embeddings,
            config.hidden_size,
            vb.pp("pos_embed"),
        )?;
        let num_grid_per_side = (config.num_position_embeddings as f32).sqrt() as u32;
        let head_dim = config.hidden_size / config.num_heads;
        let rotary_pos_emb = Qwen2_5VisionRotaryEmbedding::new(head_dim / 2, None);
        let mut blocks = Vec::new();
        let vb_blocks = vb.pp("blocks");
        for i in 0..config.depth {
            let block = Qwen3VLVisionBlock::new(config.clone(), vb_blocks.pp(i))?;
            blocks.push(block);
        }
        let merger = Qwen3VLVisionPatchMerger::new(&config, vb.pp("merger"), false)?;
        let deepstack_visual_indexes = config.deepstack_visual_indexes.clone();
        let mut deepstack_merger_list = Vec::new();
        let vb_deepstack = vb.pp("deepstack_merger_list");
        for i in 0..deepstack_visual_indexes.len() {
            let merger_i = Qwen3VLVisionPatchMerger::new(&config, vb_deepstack.pp(i), true)?;
            deepstack_merger_list.push(merger_i);
        }
        Ok(Self {
            spatial_merge_size,
            patch_embed,
            pos_embed,
            num_grid_per_side,
            rotary_pos_emb,
            blocks,
            merger,
            deepstack_visual_indexes,
            deepstack_merger_list,
            dtype: vb.dtype(),
        })
    }

    pub fn fast_pos_embed_interpolate(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let mut idx_list = vec![vec![]; 4];
        let mut weight_list = vec![vec![]; 4];
        let mut split_idx = vec![];
        for i in 0..grid_thw.dim(0)? {
            let [_, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            split_idx.push((h * w) as usize);
            let num_grid_per_side_sub_one = (self.num_grid_per_side - 1) as f32;
            let h_idxs = linspace(
                0.0,
                num_grid_per_side_sub_one,
                h as usize,
                grid_thw.device(),
            )?;
            let w_idxs = linspace(
                0.0,
                num_grid_per_side_sub_one,
                w as usize,
                grid_thw.device(),
            )?;
            let h_idxs_floor = h_idxs.to_dtype(candle_core::DType::U32)?;
            let w_idxs_floor = w_idxs.to_dtype(candle_core::DType::U32)?;
            let h_idxs_ceil = h_idxs_floor
                .affine(1.0, 1.0)?
                .clamp(0u32, num_grid_per_side_sub_one as u32)?;
            let w_idxs_ceil = w_idxs_floor
                .affine(1.0, 1.0)?
                .clamp(0u32, num_grid_per_side_sub_one as u32)?;
            let dh = h_idxs
                .sub(&h_idxs_floor.to_dtype(h_idxs.dtype())?)?
                .unsqueeze(D::Minus1)?;
            let dw = w_idxs
                .sub(&w_idxs_floor.to_dtype(h_idxs.dtype())?)?
                .unsqueeze(0)?;
            let base_h = h_idxs_floor
                .affine(self.num_grid_per_side as f64, 0.0)?
                .unsqueeze(D::Minus1)?;
            let base_h_ceil = h_idxs_ceil
                .affine(self.num_grid_per_side as f64, 0.0)?
                .unsqueeze(D::Minus1)?;
            idx_list[0].extend_from_slice(
                &base_h
                    .broadcast_add(&w_idxs_floor.unsqueeze(0)?)?
                    .flatten_all()?
                    .to_vec1::<u32>()?,
            );
            idx_list[1].extend_from_slice(
                &base_h
                    .broadcast_add(&w_idxs_ceil.unsqueeze(0)?)?
                    .flatten_all()?
                    .to_vec1::<u32>()?,
            );
            idx_list[2].extend_from_slice(
                &base_h_ceil
                    .broadcast_add(&w_idxs_floor.unsqueeze(0)?)?
                    .flatten_all()?
                    .to_vec1::<u32>()?,
            );
            idx_list[3].extend_from_slice(
                &base_h_ceil
                    .broadcast_add(&w_idxs_ceil.unsqueeze(0)?)?
                    .flatten_all()?
                    .to_vec1::<u32>()?,
            );

            let one_sub_dh = Tensor::ones_like(&dh)?.sub(&dh)?;
            let one_sub_dw = Tensor::ones_like(&dw)?.sub(&dw)?;

            weight_list[0].extend_from_slice(
                &one_sub_dh
                    .broadcast_mul(&one_sub_dw)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
            weight_list[1].extend_from_slice(
                &one_sub_dh
                    .broadcast_mul(&dw)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
            weight_list[2].extend_from_slice(
                &dh.broadcast_mul(&one_sub_dw)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
            weight_list[3]
                .extend_from_slice(&dh.broadcast_mul(&dw)?.flatten_all()?.to_vec1::<f32>()?);
        }
        let idx_tensor = Tensor::new(idx_list, grid_thw.device())?;
        let weight_tensor = Tensor::new(weight_list, grid_thw.device())?.to_dtype(self.dtype)?;
        let pos_embeds = self
            .pos_embed
            .forward(&idx_tensor)?
            .broadcast_mul(&weight_tensor.unsqueeze(D::Minus1)?)?;
        let patch_pos_embeds = pos_embeds
            .i(0)?
            .add(&pos_embeds.i(1)?)?
            .add(&pos_embeds.i(2)?)?
            .add(&pos_embeds.i(3)?)?;
        let mut patch_pos_embeds_permute = vec![];
        let patch_pos_embeds = split_tensor(&patch_pos_embeds, &split_idx, 0)?;
        let merge_size = self.spatial_merge_size;
        for (i, pos_embed) in patch_pos_embeds.iter().enumerate() {
            let [t, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            // let pos_embed = &patch_pos_embeds[i];
            let pos_emebd_last_dim = pos_embed.dim(D::Minus1)?;
            let pos_embed = pos_embed.repeat((t as usize, 1))?;
            let shape = Shape::from(vec![
                t as usize,
                h as usize / merge_size,
                merge_size,
                w as usize / merge_size,
                merge_size,
                pos_emebd_last_dim,
            ]);
            let pos_embed = pos_embed
                .reshape(shape)?
                .permute((0, 1, 3, 2, 4, 5))?
                .flatten(0, 4)?;
            patch_pos_embeds_permute.push(pos_embed);
        }
        let patch_pos_embeds = Tensor::cat(&patch_pos_embeds_permute, 0)?;
        Ok(patch_pos_embeds)
    }

    pub fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let merge_size = self.spatial_merge_size;
        let max_hw = grid_thw.i((.., 1..))?.max_all()?.to_scalar::<u32>()?;
        let freq_table = self
            .rotary_pos_emb
            .forward(max_hw as usize, grid_thw.device())?;
        let mut pos_ids_vec = vec![];
        for i in 0..grid_thw.dim(0)? {
            let [t, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            let merged_h = h / merge_size as u32;
            let merged_w = w / merge_size as u32;
            let blocks_rows = Tensor::arange(0, merged_h, grid_thw.device())?;
            let blocks_cols = Tensor::arange(0, merged_w, grid_thw.device())?;
            let intra_row = Tensor::arange(0, merge_size as u32, grid_thw.device())?;
            let intra_col = Tensor::arange(0, merge_size as u32, grid_thw.device())?;

            let row_idx = blocks_rows
                .reshape(((), 1, 1, 1))?
                .contiguous()?
                .affine(merge_size as f64, 0.0)?
                .broadcast_add(&intra_row.reshape((1, 1, (), 1))?.contiguous()?)?;
            let col_idx = blocks_cols
                .reshape((1, (), 1, 1))?
                .contiguous()?
                .affine(merge_size as f64, 0.0)?
                .broadcast_add(&intra_col.reshape((1, 1, 1, ()))?.contiguous()?)?;
            let row_idx = row_idx
                .expand((merged_h as usize, merged_w as usize, merge_size, merge_size))?
                .flatten_all()?;
            let col_idx = col_idx
                .expand((merged_h as usize, merged_w as usize, merge_size, merge_size))?
                .flatten_all()?;
            let mut coords = Tensor::stack(&[row_idx, col_idx], D::Minus1)?.contiguous()?;
            if t > 1 {
                coords = coords.repeat((t as usize, 1))?;
            }
            pos_ids_vec.push(coords);
        }
        let pos_ids = Tensor::cat(&pos_ids_vec, 0)?;
        let pos_ids_h = pos_ids.i((.., 0))?.contiguous()?;
        // 第二列是w维度的索引
        let pos_ids_w = pos_ids.i((.., 1))?.contiguous()?;
        let rotary_pos_emb_h = freq_table.index_select(&pos_ids_h, 0)?;
        let rotary_pos_emb_w = freq_table.index_select(&pos_ids_w, 0)?;
        // 每个patch融合h索引和w索引两个的位置编码信息
        let rotary_pos_emb = Tensor::cat(&[rotary_pos_emb_h, rotary_pos_emb_w], 1)?.contiguous()?;
        Ok(rotary_pos_emb)
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        grid_thw: &Tensor,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let hidden_states = self.patch_embed.forward(hidden_states)?;
        let pos_embeds = self.fast_pos_embed_interpolate(grid_thw)?;
        let hidden_states = hidden_states.broadcast_add(&pos_embeds)?;
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let mut hidden_states = hidden_states.reshape((seq_len, ()))?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        let cu_seqlens = grid_thw.i((.., 1))?.mul(&grid_thw.i((.., 2))?)?;
        let grid_t = grid_thw.i((.., 0))?.to_vec1::<u32>()?;
        let mut cu_seqlens_repeat = Vec::new();
        for (index, t) in grid_t.iter().enumerate() {
            cu_seqlens_repeat.push(cu_seqlens.i(index)?.repeat(*t as usize)?);
        }
        let cu_seqlens_full = Tensor::cat(&cu_seqlens_repeat, 0)?.flatten_all()?;
        let cu_seqlens = cu_seqlens_full
            .to_dtype(DType::F64)?
            .cumsum(0)?
            .to_dtype(DType::U32)?
            .pad_with_zeros(D::Minus1, 1, 0)?;
        let mut deepstack_feature_lists = vec![];
        for (layer_num, block) in self.blocks.iter().enumerate() {
            hidden_states = block.forward(&hidden_states, &cu_seqlens, &cos, &sin)?;
            if self.deepstack_visual_indexes.contains(&layer_num) {
                if let Some(index) = self
                    .deepstack_visual_indexes
                    .iter()
                    .position(|&x| x == layer_num)
                {
                    let deepstack_feature =
                        self.deepstack_merger_list[index].forward(&hidden_states)?;
                    deepstack_feature_lists.push(deepstack_feature);
                } else {
                    println!("Value not found");
                }
            }
        }
        hidden_states = self.merger.forward(&hidden_states)?;
        Ok((hidden_states, deepstack_feature_lists))
    }
}

pub struct Qwen3VLTextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3VLTextAttention {
    pub fn new(config: Qwen3VLTextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let (q_proj, k_proj, v_proj, o_proj) = if config.attention_bias {
            let q_proj = linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
            let k_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
            let v_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
            let o_proj = linear(num_attention_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
            (q_proj, k_proj, v_proj, o_proj)
        } else {
            let q_proj =
                linear_no_bias(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
            let k_proj =
                linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
            let v_proj =
                linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
            let o_proj =
                linear_no_bias(num_attention_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
            (q_proj, k_proj, v_proj, o_proj)
        };
        let q_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        let key_states = self.k_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
        )?;
        let attn_output =
            attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

pub struct Qwen3VLTextDecoderLayer {
    self_attn: Qwen3VLTextAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen3VLTextDecoderLayer {
    pub fn new(config: Qwen3VLTextConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen3VLTextAttention::new(config.clone(), vb.pp("self_attn"))?;
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            false,
        )?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        let xs = residual.add(&xs)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct Qwen3VLTextModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3VLTextDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: Qwen3VLTextRotaryEmbedding,
    mrope_section: Vec<usize>,
}

impl Qwen3VLTextModel {
    pub fn new(config: Qwen3VLTextConfig, vb: VarBuilder) -> Result<Self> {
        let vocab_size = config.vocab_size;
        let embed_tokens = embedding(vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_l = vb.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = Qwen3VLTextDecoderLayer::new(config.clone(), vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let head_dim = config.head_dim;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(head_dim, config.rope_theta);
        let mrope_section = config.rope_scaling.mrope_section.clone();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section,
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        position_ids: Option<&Tensor>,
        visual_pos_masks: Option<&Tensor>,
        deepstack_visual_embeds: Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(
                seqlen_offset as u32,
                (seq_len + seqlen_offset) as u32,
                inputs_embeds.device(),
            )?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((3, b_size, seq_len))?,
        };
        let (cos, sin) = self.rotary_emb.forward(
            &position_ids,
            inputs_embeds.dtype(),
            self.mrope_section.clone(),
        )?;
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<&Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(&prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    0,
                    inputs_embeds.device(),
                )?)
            }
        };
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask)?;
            if let Some(deepstack_embeds) = deepstack_visual_embeds.as_ref()
                && layer_idx < deepstack_embeds.len()
            {
                xs = mask_index_add(
                    &xs.squeeze(0)?,
                    &visual_pos_masks.unwrap().squeeze(0)?,
                    &deepstack_embeds[layer_idx],
                )?
                .unsqueeze(0)?;
            }
        }
        let xs = xs.apply(&self.norm)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

pub struct Qwen3VLModel {
    config: Qwen3VLConfig,
    visual: Qwen3VLVisionModel,
    language_model: Qwen3VLTextModel,
    lm_head: Linear,
    rope_deltas: Option<Tensor>,
}

impl Qwen3VLModel {
    pub fn new(config: Qwen3VLConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let config = config.clone();
        let visual = Qwen3VLVisionModel::new(config.vision_config.clone(), vb_m.pp("visual"))?;
        let language_model =
            Qwen3VLTextModel::new(config.text_config.clone(), vb_m.pp("language_model"))?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(language_model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };
        Ok(Self {
            config,
            visual,
            language_model,
            lm_head,
            rope_deltas: None,
        })
    }

    fn get_vision_features(
        &self,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
        let (image_embeds, deepstack_image_embeds) =
            self.visual.forward(pixel_values, image_grid_thw)?;
        // torch.prod
        let split_sizes: Vec<usize> = prod_tensor_last_dim(image_grid_thw)?
            .to_vec1::<u32>()?
            .iter()
            .map(|&x| x as usize / self.visual.spatial_merge_size.pow(2))
            .collect();
        let image_embeds = split_tensor(&image_embeds, &split_sizes, 0)?;
        Ok((image_embeds, deepstack_image_embeds))
    }

    fn get_placeholder_mask(&self, input_ids: &Tensor, is_image: bool) -> Result<Tensor> {
        let special_token = if is_image {
            Tensor::new(vec![self.config.image_token_id as u32], input_ids.device())?
        } else {
            Tensor::new(vec![self.config.video_token_id as u32], input_ids.device())?
        };
        let special_mask = input_ids
            .broadcast_eq(&special_token)?
            .to_dtype(candle_core::DType::U32)?;
        Ok(special_mask)
    }

    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let video_grid_thw = match video_grid_thw {
            Some(thw) => {
                let grid_t = thw.i((.., 0))?.to_vec1::<u32>()?;
                let mut v_thw_vec = Vec::new();
                for (index, t) in grid_t.iter().enumerate() {
                    let mut thw_i = thw.i(index)?.to_vec1::<u32>()?;
                    // [12, 30, 50]
                    // [1, 30, 50]*t
                    thw_i[0] = 1;
                    v_thw_vec.push(
                        Tensor::new(thw_i, thw.device())?
                            .repeat(*t as usize)?
                            .reshape((*t as usize, ()))?,
                    );
                }
                Some(&Tensor::cat(&v_thw_vec, 0)?)
            }
            None => None,
        };

        let spatial_merge_size = self.config.vision_config.spatial_merge_size;
        let image_token_id = self.config.image_token_id;
        let video_token_id = self.config.video_token_id;
        let vision_start_token_id = self.config.vision_start_token_id;
        let mut mrope_position_deltas = vec![];
        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            let total_input_ids = input_ids.clone();
            let mask_ = mask
                .cloned()
                .unwrap_or(Tensor::ones_like(&total_input_ids)?)
                .to_device(input_ids.device())?;
            let mut position_ids = Tensor::ones(
                (3, input_ids.dim(0)?, input_ids.dim(1)?),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            let mut image_index = 0;
            let mut video_index = 0;

            for i in 0..total_input_ids.dim(0)? {
                let mut input_ids_i = total_input_ids.i(i)?;
                let mask_i = mask_.i(i)?;
                // 推理时, attention_mask如果是全1向量,取非0索引的操作没必要
                if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                    let nonzero_idx = nonzero_index(&mask_i)?;
                    input_ids_i = input_ids_i.gather(&nonzero_idx, 0)?;
                }
                let mut text_start = 0;
                let mut text_end = 0;
                let mut thw = vec![];
                let mut llm_pos_ids_list: Vec<Tensor> = Vec::new();
                // vision start的下一个索引
                let vision_indices =
                    get_vision_next_indices(&input_ids_i, vision_start_token_id as u32);

                match vision_indices {
                    Ok(indeices) => {
                        let vision_tokens = input_ids_i.gather(&indeices, 0)?.to_vec1::<u32>()?;
                        let vision_indices_vec = indeices.to_vec1::<u32>()?;
                        for (j, &token) in vision_tokens.iter().enumerate() {
                            if token == image_token_id as u32 {
                                thw = image_grid_thw.unwrap().i(image_index)?.to_vec1::<u32>()?;
                                image_index += 1;
                                text_end = vision_indices_vec[j];
                            }
                            if token == video_token_id as u32 {
                                thw = video_grid_thw.unwrap().i(video_index)?.to_vec1::<u32>()?;
                                text_end = vision_indices_vec[j];
                                video_index += 1;
                            }
                            let llm_grid_t = thw[0];
                            let llm_grid_h = thw[1] / spatial_merge_size as u32;
                            let llm_grid_w = thw[2] / spatial_merge_size as u32;
                            let text_len = text_end - text_start;
                            let start_idx = if !llm_pos_ids_list.is_empty() {
                                llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                                    .max_all()?
                                    .to_scalar::<u32>()?
                                    + 1
                            } else {
                                0
                            };
                            let pos_ids = Tensor::arange(
                                start_idx,
                                start_idx + text_len,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .broadcast_as((3usize, text_len as usize))?;
                            llm_pos_ids_list.push(pos_ids);

                            let t_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_t,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(D::Minus1)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize * llm_grid_w as usize,
                            ))?
                            .flatten_all()?;
                            let h_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_h,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(D::Minus1)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;
                            let w_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_w,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(0)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;

                            let thw_index = Tensor::stack(&[t_index, h_index, w_index], 0)?;
                            llm_pos_ids_list.push(thw_index);
                            text_start = text_end + llm_grid_t * llm_grid_h * llm_grid_w;
                        }
                    }
                    Err(e) => {
                        println!("get vision_indices err: {}", e);
                    }
                };
                if text_start < input_ids_i.dim(0)? as u32 {
                    let start_idx = if !llm_pos_ids_list.is_empty() {
                        llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                            .max_all()?
                            .to_scalar::<u32>()?
                            + 1
                    } else {
                        0
                    };
                    let text_len = input_ids_i.dim(0)? as u32 - text_start;
                    let pos_ids =
                        Tensor::arange(start_idx, start_idx + text_len, input_ids_i.device())?
                            .unsqueeze(0)?
                            .broadcast_as((3usize, text_len as usize))?;
                    llm_pos_ids_list.push(pos_ids);
                }
                let llm_position = Tensor::cat(&llm_pos_ids_list, 1)?.reshape((3, 1, ()))?;
                position_ids = position_ids
                    .slice_assign(&[(0..3), (i..i + 1), (0..input_ids.dim(1)?)], &llm_position)?;
                let position_deltas = llm_position.max_all()?.to_scalar::<u32>()? as i64 + 1
                    - input_ids_i.dim(0)? as i64;
                mrope_position_deltas.push(position_deltas);
            }
            let mut mrope_position_deltas = Tensor::new(mrope_position_deltas, input_ids.device())?;
            if mrope_position_deltas.rank() == 1 {
                mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
            }
            Ok((position_ids.contiguous()?, mrope_position_deltas))
        } else if let Some(mask) = mask {
            let mut position_ids = mask
                .to_dtype(candle_core::DType::F64)?
                .cumsum(D::Minus1)?
                .to_dtype(candle_core::DType::U32)?
                .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;
            for i in 0..position_ids.dim(0)? {
                let mut position_ids_i = position_ids.i(i)?;
                let mask_i = mask.i(i)?;
                // 如果有pad, 将填充位置置为1
                // 当bs>1, 可能存在不同序列长度，需要添加pad使seq_len长度一致
                if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                    let zero_indices = zero_index(&mask_i)?;
                    let replace_1 = Tensor::ones(
                        zero_indices.dim(0)?,
                        candle_core::DType::U32,
                        input_ids.device(),
                    )?;
                    position_ids_i = position_ids_i
                        .scatter(&zero_indices, &replace_1, 0)?
                        .unsqueeze(0)?;
                    position_ids = position_ids
                        .slice_assign(&[(i..i + 1), (0..position_ids.dim(1)?)], &position_ids_i)?;
                }
            }
            position_ids = position_ids
                .unsqueeze(0)?
                .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?
                .contiguous()?;
            let mut mrope_position_deltas = position_ids
                .max(0)?
                .max(D::Minus1)?
                .broadcast_sub(&Tensor::new(
                    vec![mask.dim(D::Minus1)? as u32 - 1],
                    input_ids.device(),
                )?)?
                .contiguous()?;
            if mrope_position_deltas.rank() == 1 {
                mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
            }
            Ok((position_ids, mrope_position_deltas))
        } else {
            let position_ids =
                Tensor::arange(0_u32, input_ids.dim(D::Minus1)? as u32, input_ids.device())?
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(D::Minus1)?))?
                    .contiguous()?;
            let mrope_position_deltas = Tensor::zeros(
                (input_ids.dim(0)?, 1),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            Ok((position_ids, mrope_position_deltas))
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.language_model.embed_tokens.forward(input_ids)?;
        let mut image_mask = None;
        let mut video_mask = None;
        let mut deepstack_image_embeds = None;
        let mut deepstack_video_embeds = None;
        if let Some(pixel_values) = pixel_values
            && let Some(image_grid_thw) = image_grid_thw
        {
            let (image_embeds, deepstack_img_embed) =
                self.get_vision_features(pixel_values, image_grid_thw)?;
            let image_embeds = Tensor::cat(&image_embeds, 0)?;
            let vision_mask = self.get_placeholder_mask(input_ids, true)?;
            let n_image_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;
            if n_image_tokens as usize != image_embeds.dim(0)? {
                return Err(anyhow!(format!(
                    "n_image_token num: {} not equal to image_embed len: {}",
                    n_image_tokens,
                    image_embeds.dim(0)?
                )));
            }
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
            image_mask = Some(vision_mask);
            deepstack_image_embeds = Some(deepstack_img_embed);
        }
        if let Some(pixel_values_video) = pixel_values_video
            && let Some(video_grid_thw) = video_grid_thw
        {
            let (video_embeds, deepstack_video_embed) =
                self.get_vision_features(pixel_values_video, video_grid_thw)?;
            let video_embeds = Tensor::cat(&video_embeds, 0)?;
            let vision_mask = self.get_placeholder_mask(input_ids, false)?;
            let n_video_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;
            if n_video_tokens as usize != video_embeds.dim(0)? {
                return Err(anyhow!(format!(
                    "n_image_token num: {} not equal to image_embed len: {}",
                    n_video_tokens,
                    video_embeds.dim(0)?
                )));
            }
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embeds, &vision_mask)?;
            video_mask = Some(vision_mask);
            deepstack_video_embeds = Some(deepstack_video_embed);
        }
        let mut visual_pos_mask = None;
        let mut deepstack_visual_embeds = None;
        if let Some(image_mask_) = image_mask {
            if let Some(video_mask_) = video_mask {
                let image_mask_ = image_mask_.squeeze(0)?;
                let video_mask_ = video_mask_.squeeze(0)?;
                let visual_mask = bitor_tensor(&image_mask_, &video_mask_)?;
                let visual_none_zero_index = nonzero_index(&visual_mask)?;
                let image_mask_joint = image_mask_.gather(&visual_none_zero_index, 0)?;
                let image_nonzero_joint = nonzero_index(&image_mask_joint)?;
                let video_mask_joint = video_mask_.gather(&visual_none_zero_index, 0)?;
                let video_nonzero_joint = nonzero_index(&video_mask_joint)?;
                let mut deepstack_embeds = vec![];
                let visual_len = visual_none_zero_index.dim(0)?;
                for (img_embed, vid_embed) in deepstack_image_embeds
                    .unwrap()
                    .iter()
                    .zip(deepstack_video_embeds.unwrap().iter())
                {
                    let embed_joint = Tensor::zeros(
                        (visual_len, img_embed.dim(D::Minus1)?),
                        img_embed.dtype(),
                        img_embed.device(),
                    )?;
                    let embed_joint = embed_joint.index_add(&image_nonzero_joint, img_embed, 0)?;
                    let embed_joint = embed_joint.index_add(&video_nonzero_joint, vid_embed, 0)?;
                    deepstack_embeds.push(embed_joint);
                }
                visual_pos_mask = Some(visual_mask.unsqueeze(0)?);
                deepstack_visual_embeds = Some(deepstack_embeds);
            } else {
                visual_pos_mask = Some(image_mask_);
                deepstack_visual_embeds = deepstack_image_embeds;
            }
        } else if let Some(video_mask_) = video_mask {
            visual_pos_mask = Some(video_mask_);
            deepstack_visual_embeds = deepstack_video_embeds;
        }

        let position_ids;
        let rope_deltas;
        if (cache_position.is_some() && cache_position.unwrap().i(0)?.to_scalar::<u32>()? == 0)
            || self.rope_deltas.is_none()
        {
            (position_ids, rope_deltas) =
                self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, None)?;
            self.rope_deltas = Some(rope_deltas);
        } else {
            let (bs, seq_len, _) = inputs_embeds.dims3()?;
            let delta = if let Some(cache_position) = cache_position {
                cache_position
                    .i(0)?
                    .to_dtype(self.rope_deltas.as_ref().unwrap().dtype())?
                    .broadcast_add(self.rope_deltas.as_ref().unwrap())?
                    .contiguous()?
                    .to_dtype(candle_core::DType::U32)?
            } else {
                Tensor::zeros(1, inputs_embeds.dtype(), inputs_embeds.device())?
            };
            position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
                .unsqueeze(0)?
                .broadcast_as((bs, seq_len))?
                .broadcast_add(&delta)?
                .unsqueeze(0)?
                .broadcast_as((3, bs, seq_len))?
                .contiguous()?;
        }
        let outputs = self.language_model.forward(
            &inputs_embeds,
            seqlen_offset,
            Some(&position_ids),
            visual_pos_mask.as_ref(),
            deepstack_visual_embeds,
        )?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}
