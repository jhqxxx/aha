use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Init, Linear, Module, RmsNorm, VarBuilder, linear, linear_no_bias, rms_norm};

use crate::{
    models::{
        common::{GateUpDownMLP, eager_attention_forward},
        qwen2_5vl::config::{Qwen2_5VLConfig, RopeScaling},
    },
    position_embed::rope::{
        Qwen2_5VLTextRotaryEmbedding, Qwen2_5VisionRotaryEmbedding, apply_rotary_pos_emb,
        apply_rotary_pos_emb_vision,
    },
    utils::tensor_utils::{
        get_equal_mask, get_vision_next_indices, masked_scatter_dim0, nonzero_index,
        prepare_causal_attention_mask, safe_arg_sort_last_dim, zero_index,
    },
};

pub struct Qwen2_5VisionPatchEmbed {
    conv3d_weight: Tensor,
}

impl Qwen2_5VisionPatchEmbed {
    pub fn new(cfg: &Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let patch_size = cfg.vision_config.patch_size;
        let temporal_patch_size = cfg.vision_config.temporal_patch_size;
        let in_channels = cfg.vision_config.in_chans;
        let embed_dim = cfg.vision_config.hidden_size;
        // conv3d weight key: visual.patch_embed.proj.weight, value: Tensor[dims 1280, 3, 2, 14, 14; bf16, cuda:0]
        // (1280, 3, 2, 14, 14) -> (1280, 1176) -> (1176, 1280)
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
        Ok(Self { conv3d_weight })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // hidden_states shape:  (grid_t*grid_h*grid_w, c*temporal_patch_size*patch_size*patch_size)
        // ((), 1176) matmul (1176, 1280) -> ((), 1280)
        let hidden_states = hidden_states.matmul(&self.conv3d_weight)?;
        Ok(hidden_states)
    }
}

pub struct Qwen2_5VLPatchMerger {
    hidden_size: usize,
    ln_q: RmsNorm,
    mlp_0: Linear,
    mlp_2: Linear,
}

impl Qwen2_5VLPatchMerger {
    pub fn new(cfg: &Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size =
            cfg.vision_config.hidden_size * (cfg.vision_config.spatial_merge_size.pow(2));
        let ln_q = rms_norm(
            cfg.vision_config.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("ln_q"),
        )?;
        let mlp_0 = linear(hidden_size, hidden_size, vb.pp("mlp.0"))?;
        let mlp_2 = linear(
            hidden_size,
            cfg.vision_config.out_hidden_size,
            vb.pp("mlp.2"),
        )?;
        Ok(Self {
            hidden_size,
            ln_q,
            mlp_0,
            mlp_2,
        })
    }
}
impl Module for Qwen2_5VLPatchMerger {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs = xs.apply(&self.ln_q)?.reshape(((), self.hidden_size))?;
        let xs = xs.apply(&self.mlp_0)?.gelu()?.apply(&self.mlp_2)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLVisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    // scale: Tensor,
    scale: f64,
}

impl Qwen2_5VLVisionAttention {
    fn new(cfg: &Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.vision_config.hidden_size;
        let num_heads = cfg.vision_config.num_heads;
        let head_dim = hidden_size / num_heads;
        let qkv = linear(hidden_size, hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(hidden_size, hidden_size, vb.pp("proj"))?;
        // let scale = Tensor::new(vec![1f32 / (head_dim as f32).sqrt()], vb.device())?
        //     .to_dtype(vb.dtype())?;
        let scale = 1f64 / (head_dim as f64).sqrt();
        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        // xs: (seq_len, hidden_size)
        let seq_length = xs.dim(0)?;
        // (seq_len, hidden_size) -> (seq_len, hidden_size*3)
        // -> (seq_len, 3, num_heads, head_dim) -> (3, seq_len, num_heads, head_dim)
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
        //(seq_len, num_heads, head_dim) -> (num_heads, seq_len, head_dim) -> (1, num_heads, seq_len, head_dim)
        let query_states = query_states.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let key_states = key_states.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let value_states = value_states.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            None,
            Some(attention_mask),
            self.scale,
        )?;
        //(1, seq_len, n_head, dim) -> (seq_len, n_head, dim)
        let attn_output = attn_output
            .squeeze(0)?
            .reshape((seq_length, ()))?
            .contiguous()?;

        let attn_ouput = attn_output.apply(&self.proj)?;
        Ok(attn_ouput)
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLVisionBlock {
    attn: Qwen2_5VLVisionAttention,
    mlp: GateUpDownMLP,
    norm1: RmsNorm,
    norm2: RmsNorm,
}

impl Qwen2_5VLVisionBlock {
    fn new(cfg: &Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let attn = Qwen2_5VLVisionAttention::new(cfg, vb.pp("attn"))?;
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            cfg.vision_config.hidden_size,
            cfg.vision_config.intermediate_size,
            cfg.vision_config.hidden_act,
            true,
        )?;
        let norm1 = rms_norm(
            cfg.vision_config.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("norm1"),
        )?;
        let norm2 = rms_norm(
            cfg.vision_config.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("norm2"),
        )?;

        Ok(Self {
            attn,
            mlp,
            norm1,
            norm2,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.norm1.forward(xs)?;
        let xs = self.attn.forward(&xs, cos, sin, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.norm2)?.apply(&self.mlp)?;
        let xs = (residual + xs)?;
        Ok(xs)
    }
}

pub struct Qwen2_5VLVisionModel {
    spatial_merge_size: usize,
    patch_size: usize,
    fullatt_block_indexes: Vec<usize>,
    window_size: usize,
    spatial_merge_unit: usize,
    patch_embed: Qwen2_5VisionPatchEmbed,
    rotary_pos_emb: Qwen2_5VisionRotaryEmbedding,
    blocks: Vec<Qwen2_5VLVisionBlock>,
    merger: Qwen2_5VLPatchMerger,
    dtype: DType,
}

impl Qwen2_5VLVisionModel {
    pub fn new(cfg: &Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let spatial_merge_size = cfg.vision_config.spatial_merge_size;
        let patch_size = cfg.vision_config.patch_size;
        let fullatt_block_indexes = cfg.vision_config.fullatt_block_indexes.clone();
        let window_size = cfg.vision_config.window_size;
        let spatial_merge_unit = spatial_merge_size * spatial_merge_size;
        let head_dim = cfg.vision_config.hidden_size / cfg.vision_config.num_heads;
        let patch_embed = Qwen2_5VisionPatchEmbed::new(cfg, vb.pp("patch_embed"))?;
        let rotary_pos_emb = Qwen2_5VisionRotaryEmbedding::new(head_dim / 2, None);
        let mut blocks = Vec::new();
        let vb_blocks = vb.pp("blocks");
        for i in 0..cfg.vision_config.depth {
            let block = Qwen2_5VLVisionBlock::new(cfg, vb_blocks.pp(i))?;
            blocks.push(block);
        }
        let merger = Qwen2_5VLPatchMerger::new(cfg, vb.pp("merger"))?;
        let dtype = vb.dtype();
        Ok(Self {
            spatial_merge_size,
            patch_size,
            fullatt_block_indexes,
            window_size,
            spatial_merge_unit,
            patch_embed,
            rotary_pos_emb,
            blocks,
            merger,
            dtype,
        })
    }

    pub fn rot_pos_emb(&self, grid_thw: &Tensor) -> Result<Tensor> {
        let mut pos_ids = Vec::new();
        for i in 0..grid_thw.dim(0)? {
            let [t, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            // hpos_ids shape (h, w)
            let hpos_ids = Tensor::arange(0, h, grid_thw.device())?
                .unsqueeze(1)?
                .expand((h as usize, w as usize))?;
            let hpos_ids = hpos_ids.reshape((
                h as usize / self.spatial_merge_size,
                self.spatial_merge_size,
                w as usize / self.spatial_merge_size,
                self.spatial_merge_size,
            ))?;
            let hpos_ids = hpos_ids.permute((0, 2, 1, 3))?.flatten_all()?;
            let wpos_ids = Tensor::arange(0, w, grid_thw.device())?
                .unsqueeze(0)?
                .expand((h as usize, w as usize))?;
            let wpos_ids = wpos_ids.reshape((
                h as usize / self.spatial_merge_size,
                self.spatial_merge_size,
                w as usize / self.spatial_merge_size,
                self.spatial_merge_size,
            ))?;
            let wpos_ids = wpos_ids.permute((0, 2, 1, 3))?.flatten_all()?;
            // thw_pos_ids shape (h*w, 2)
            let thw_pos_ids =
                Tensor::stack(&[&hpos_ids, &wpos_ids], D::Minus1)?.repeat((t as usize, 1))?;
            pos_ids.push(thw_pos_ids);
        }
        let pos_ids = Tensor::cat(&pos_ids, 0)?.contiguous()?;
        let max_grid_size = grid_thw.i((.., 1..))?.max_all()?.to_scalar::<u32>()?;
        let rotary_pos_emb_full = self
            .rotary_pos_emb
            .forward(max_grid_size as usize, grid_thw.device())?;

        // contiguous()一定要加！！！很重要！！！！，不然index_select出来的是错的
        // 找错找了半天，都是泪啊，做维度索引操作后contiguous顺手写上总没错
        // 第一列是h维度的索引
        let pos_ids_h = pos_ids.i((.., 0))?.contiguous()?;
        // 第二列是w维度的索引
        let pos_ids_w = pos_ids.i((.., 1))?.contiguous()?;
        let rotary_pos_emb_h = rotary_pos_emb_full.index_select(&pos_ids_h, 0)?;
        let rotary_pos_emb_w = rotary_pos_emb_full.index_select(&pos_ids_w, 0)?;
        // 每个patch融合h索引和w索引两个的位置编码信息
        let rotary_pos_emb = Tensor::cat(&[rotary_pos_emb_h, rotary_pos_emb_w], 1)?.contiguous()?;
        Ok(rotary_pos_emb)
    }

    pub fn get_window_index(&self, grid_thw: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut window_index = Vec::new();
        let mut cu_window_seqlens = vec![0];
        let mut window_index_id = 0_i64;

        let vit_merger_window_size =
            (self.window_size / self.spatial_merge_size / self.patch_size) as u32;
        for i in 0..grid_thw.dim(0)? {
            let [grid_t, grid_h, grid_w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            let llm_grid_h = grid_h / self.spatial_merge_size as u32;
            let llm_grid_w = grid_w / self.spatial_merge_size as u32;
            // 因为后续需要使用-100来做填充，所以需要int类型
            // candle好像不支持i32， DType里面都没有定义i32, 所以这里使用i64
            let mut index = Tensor::arange(
                window_index_id,
                window_index_id + (grid_t * llm_grid_h * llm_grid_w) as i64,
                grid_thw.device(),
            )?
            .reshape((grid_t as usize, llm_grid_h as usize, llm_grid_w as usize))?
            .contiguous()?;
            // python transformers 中实现如下
            // let pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size);
            // 后面加上 % vit_merger_window_size，保证llm_grid_h能整除vit_merger_window_size时不需要pad
            // 按理说能整除应该是不需要pad的，transformers中这样实现不知道是不是有什么其他原因
            let pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size)
                % vit_merger_window_size;
            let pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size)
                % vit_merger_window_size;
            let num_window_h = (llm_grid_h + pad_h) / vit_merger_window_size;
            let num_window_w = (llm_grid_w + pad_w) / vit_merger_window_size;
            if pad_h > 0 {
                let pad_h_t = Tensor::new(vec![-100_i64], grid_thw.device())?
                    .broadcast_as((grid_t as usize, pad_h as usize, llm_grid_w as usize))?
                    .contiguous()?;
                index = Tensor::cat(&[&index, &pad_h_t], 1)?;
            }
            if pad_w > 0 {
                let pad_w_t = Tensor::new(vec![-100_i64], grid_thw.device())?
                    .broadcast_as((
                        grid_t as usize,
                        (llm_grid_h + pad_h) as usize,
                        pad_w as usize,
                    ))?
                    .contiguous()?;
                index = Tensor::cat(&[&index, &pad_w_t], 2)?;
            }
            let index_padded = index
                .reshape((
                    grid_t as usize,
                    num_window_h as usize,
                    vit_merger_window_size as usize,
                    num_window_w as usize,
                    vit_merger_window_size as usize,
                ))?
                .permute((0, 1, 3, 2, 4))?;
            let index_padded = index_padded
                .reshape((
                    grid_t as usize,
                    (num_window_h * num_window_w) as usize,
                    vit_merger_window_size as usize,
                    vit_merger_window_size as usize,
                ))?
                .contiguous()?;
            let is_pad = Tensor::new(vec![-100_i64], grid_thw.device())?;
            let seqlens = index_padded
                .broadcast_ne(&is_pad)?
                .sum((2, 3))?
                .flatten_all()?;
            let index_padded = index_padded.flatten_all()?;
            let not_pad = index_padded.broadcast_ne(&is_pad)?.to_vec1::<u8>()?;
            let indices: Vec<u32> = not_pad
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val != 0 { Some(idx as u32) } else { None })
                .collect();
            let indices_tensor = Tensor::from_slice(&indices, indices.len(), grid_thw.device())?;
            let index_new = index_padded.gather(&indices_tensor, 0)?;
            window_index.push(index_new);

            let seq_len_last = cu_window_seqlens[cu_window_seqlens.len() - 1];
            // cumsum方法i64类型执行会报错，先转成F64计算后再转回i64
            let cu_seqlens_tmp = seqlens
                .to_dtype(candle_core::DType::F64)?
                .cumsum(0)?
                .to_dtype(candle_core::DType::I64)?
                .broadcast_mul(&Tensor::new(
                    vec![self.spatial_merge_unit as i64],
                    grid_thw.device(),
                )?)?
                .broadcast_add(&Tensor::new(vec![seq_len_last], grid_thw.device())?)?;
            cu_window_seqlens.extend_from_slice(&cu_seqlens_tmp.to_vec1::<i64>()?);
            window_index_id += (grid_t * llm_grid_h * llm_grid_w) as i64;
        }
        let window_index_tensor = Tensor::cat(&window_index, 0)?;
        let cu_window_seqlens_tensor = Tensor::from_slice(
            &cu_window_seqlens,
            cu_window_seqlens.len(),
            grid_thw.device(),
        )?
        .to_dtype(candle_core::DType::U32)?;
        Ok((window_index_tensor, cu_window_seqlens_tensor))
    }

    pub fn get_attention_mask(
        &self,
        cu_seqlens: &Tensor,
        seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mut attention_mask =
            Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as((1, seq_len, seq_len))?;
        for i in 1..cu_seqlens.dim(0)? {
            let start = cu_seqlens.i(i - 1)?.to_scalar::<u32>()? as usize;
            let end = cu_seqlens.i(i)?.to_scalar::<u32>()? as usize;
            let block_size = end - start;
            let zeros =
                Tensor::zeros((1, block_size, block_size), candle_core::DType::F32, device)?;
            attention_mask =
                attention_mask.slice_assign(&[(0..1), (start..end), (start..end)], &zeros)?;
        }
        let attention_mask = attention_mask.to_dtype(dtype)?.contiguous()?;
        Ok(attention_mask)
    }

    pub fn forward(&self, hidden_states: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        // hidden_states: (seq_len, hidden_size)
        // grid_thw: (num_images_or_videos, 3), temporal, height, width
        let hidden_states = hidden_states.to_dtype(self.dtype)?;
        let hidden_states = self.patch_embed.forward(&hidden_states)?;
        let rotary_pos_emb = self.rot_pos_emb(grid_thw)?;
        let (window_index, cu_window_seqlens) = self.get_window_index(grid_thw)?;
        let seq_len = hidden_states.dim(0)?;
        let hidden_states = hidden_states
            .reshape((
                seq_len / self.spatial_merge_unit,
                self.spatial_merge_unit,
                (),
            ))?
            .contiguous()?;
        let hidden_states = hidden_states.index_select(&window_index, 0)?;
        let mut hidden_states = hidden_states.reshape((seq_len, ()))?;
        let rotary_pos_emb = rotary_pos_emb.reshape((
            seq_len / self.spatial_merge_unit,
            self.spatial_merge_unit,
            (),
        ))?;
        let rotary_pos_emb = rotary_pos_emb.index_select(&window_index, 0)?;
        let rotary_pos_emb = rotary_pos_emb.reshape((seq_len, ()))?;
        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], D::Minus1)?;
        let cos = emb.cos()?.to_dtype(hidden_states.dtype())?;
        let sin = emb.sin()?.to_dtype(hidden_states.dtype())?;
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
            .to_dtype(DType::U32)?;
        let pad_zero = Tensor::from_vec(vec![0_u32], 1, hidden_states.device())?;
        let cu_seqlens = Tensor::cat(&[&pad_zero, &cu_seqlens], D::Minus1)?;
        let attention_mask_window = self.get_attention_mask(
            &cu_window_seqlens,
            seq_len,
            hidden_states.device(),
            hidden_states.dtype(),
        )?;
        let attention_mask_full = self.get_attention_mask(
            &cu_seqlens,
            seq_len,
            hidden_states.device(),
            hidden_states.dtype(),
        )?;
        let mut attention_mask;
        for (layer_num, block) in self.blocks.iter().enumerate() {
            if self.fullatt_block_indexes.contains(&layer_num) {
                attention_mask = attention_mask_full.clone();
            } else {
                attention_mask = attention_mask_window.clone();
            }
            hidden_states = block.forward(&hidden_states, &cos, &sin, &attention_mask)?;
        }
        let hidden_states = self.merger.forward(&hidden_states)?;
        let reverse_indices = safe_arg_sort_last_dim(&window_index, true)?;
        let hidden_states = hidden_states.index_select(&reverse_indices, 0)?;
        Ok(hidden_states)
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLTextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen2_5VLTextAttention {
    fn new(cfg: &Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_size / num_heads;
        let q_proj = linear(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(hidden_size, hidden_size, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
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
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.hidden_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct Qwen2_5VLTextDecoderLayer {
    self_attn: Qwen2_5VLTextAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen2_5VLTextDecoderLayer {
    fn new(cfg: &Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Qwen2_5VLTextAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.hidden_act,
            false,
        )?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        let xs = (residual + xs)?;
        Ok(xs)
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2_5VLTextModel {
    pub embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen2_5VLTextDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: Qwen2_5VLTextRotaryEmbedding,
    dtype: DType,
    rope_scaling: RopeScaling,
}

impl Qwen2_5VLTextModel {
    pub fn new(cfg: &Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary_emb = Qwen2_5VLTextRotaryEmbedding::new(head_dim, cfg.rope_theta);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = Qwen2_5VLTextDecoderLayer::new(cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rope_scaling = cfg.rope_scaling.clone();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            dtype: vb.dtype(),
            rope_scaling,
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        position_ids: Option<&Tensor>,
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
            self.dtype,
            self.rope_scaling.mrope_section.clone(),
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
                    xs.device(),
                )?)
            }
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask)?;
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

pub struct Qwen2_5VLModel {
    visual: Qwen2_5VLVisionModel,
    model: Qwen2_5VLTextModel,
    pub cfg: Qwen2_5VLConfig,
    lm_head: Linear,
    rope_deltas: Option<Tensor>,
}

impl Qwen2_5VLModel {
    pub fn new(cfg: Qwen2_5VLConfig, vb: VarBuilder) -> Result<Self> {
        let visual = Qwen2_5VLVisionModel::new(&cfg, vb.pp("visual"))?;
        let model = Qwen2_5VLTextModel::new(&cfg, vb.pp("model"))?;
        let vocab_size = cfg.vocab_size;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            visual,
            model,
            cfg,
            lm_head,
            rope_deltas: None,
        })
    }

    pub fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        mask: Option<&Tensor>,
        second_per_grid_ts: Option<Vec<f32>>,
    ) -> Result<(Tensor, Tensor)> {
        let spatial_merge_size = self.cfg.vision_config.spatial_merge_size;
        let mut mrope_position_deltas: Vec<i64> = Vec::new();
        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            let total_input_ids = input_ids.clone();
            let mask_ = mask
                .cloned()
                .unwrap_or(Tensor::ones_like(&total_input_ids)?);
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
                let mut second_per_grid_t = 0_f32;
                let mut llm_pos_ids_list: Vec<Tensor> = Vec::new();
                // vision start的下一个索引
                let vision_indices =
                    get_vision_next_indices(&input_ids_i, self.cfg.vision_start_token_id as u32);
                match vision_indices {
                    Ok(indeices) => {
                        let vision_tokens = input_ids_i.gather(&indeices, 0)?.to_vec1::<u32>()?;
                        let vision_indices_vec = indeices.to_vec1::<u32>()?;
                        for (j, &token) in vision_tokens.iter().enumerate() {
                            if token == self.cfg.image_token_id as u32 {
                                thw = image_grid_thw.unwrap().i(image_index)?.to_vec1::<u32>()?;
                                image_index += 1;
                                text_end = vision_indices_vec[j];
                                second_per_grid_t = 0.0;
                            }
                            if token == self.cfg.video_token_id as u32 {
                                thw = video_grid_thw.unwrap().i(video_index)?.to_vec1::<u32>()?;
                                text_end = vision_indices_vec[j];
                                second_per_grid_t = match second_per_grid_ts {
                                    None => 1.0,
                                    Some(ref vec) => vec[video_index],
                                };
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
                            let range_tensor = Tensor::arange(0, llm_grid_t, input_ids_i.device())?
                                .unsqueeze(D::Minus1)?;
                            let expanded_range = range_tensor.broadcast_as((
                                llm_grid_t as usize,
                                (llm_grid_h * llm_grid_w) as usize,
                            ))?;
                            let time_tensor = expanded_range
                                .broadcast_mul(&Tensor::new(
                                    vec![
                                        (second_per_grid_t
                                            * self.cfg.vision_config.tokens_per_second as f32)
                                            as u32,
                                    ],
                                    input_ids_i.device(),
                                )?)?
                                .broadcast_add(&Tensor::new(
                                    vec![start_idx + text_len],
                                    input_ids_i.device(),
                                )?)?;
                            let t_index = time_tensor.flatten_all()?;
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
        mask: &Tensor,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
        second_per_grid_ts: Option<Vec<f32>>,
    ) -> Result<Tensor> {
        // input_ids shape: (bs, seq_len)
        let mut inputs_embeds = self.model.embed_tokens.forward(input_ids)?;
        // inputs_embeds shape: (bs, seq_len, hidden_dim)
        if let Some(pixel_values) = pixel_values
            && let Some(image_grid_thw) = image_grid_thw
        {
            // image_embed shape: (seq_len, hidden_dim)
            let image_embed = self.visual.forward(pixel_values, image_grid_thw)?;
            let vision_mask = get_equal_mask(input_ids, self.cfg.image_token_id as u32)?;

            let n_image_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;
            if n_image_tokens as usize != image_embed.dim(0)? {
                return Err(anyhow!(format!(
                    "n_image_token num: {} not equal to image_embed len: {}",
                    n_image_tokens,
                    image_embed.dim(0)?
                )));
            }
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embed, &vision_mask)?;
        }
        if let Some(pixel_values_video) = pixel_values_video
            && let Some(video_grid_thw) = video_grid_thw
        {
            let video_embed = self.visual.forward(pixel_values_video, video_grid_thw)?;

            let vision_mask = get_equal_mask(input_ids, self.cfg.video_token_id as u32)?;
            let n_video_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;
            if n_video_tokens as usize != video_embed.dim(0)? {
                return Err(anyhow!(format!(
                    "n_image_token num: {} not equal to image_embed len: {}",
                    n_video_tokens,
                    video_embed.dim(0)?
                )));
            }
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embed, &vision_mask)?;
        }
        let position_ids;
        let rope_deltas;
        if (cache_position.is_some() && cache_position.unwrap().i(0)?.to_scalar::<u32>()? == 0)
            || self.rope_deltas.is_none()
        {
            (position_ids, rope_deltas) = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                Some(mask),
                second_per_grid_ts,
            )?;
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
        let outputs = self
            .model
            .forward(&inputs_embeds, seqlen_offset, Some(&position_ids))?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
