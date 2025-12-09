use anyhow::{Result, anyhow};
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{
    Conv2d, Embedding, Init, Linear, Module, RmsNorm, VarBuilder, embedding, linear,
    linear_no_bias, rms_norm,
};

use crate::{
    models::{
        common::{GateUpDownMLP, NaiveAttnTwoLinearMLPBlock, eager_attention_forward, get_conv2d},
        hunyuan_ocr::config::{HunYuanVLConfig, HunYuanVLVisionConfig},
    },
    position_embed::rope::{RoPE, apply_rotary_pos_emb, get_xd_cos_sin},
    utils::tensor_utils::{
        interpolate_bilinear, masked_scatter_dim0, prepare_causal_attention_mask, split_tensor,
    },
};

pub struct HunYuanVisionPatchEmbed {
    patch_embedding: Conv2d,
    // position_embedding: Embedding,
    num_channels: usize,
    patch_size: usize,
    // num_positions: usize,
    // position_edge: usize,
    embed_dim: usize,
    patch_pos_embed: Tensor,
}

impl HunYuanVisionPatchEmbed {
    pub fn new(vb: VarBuilder, config: &HunYuanVLVisionConfig) -> Result<Self> {
        let patch_embedding = get_conv2d(
            vb.pp("patch_embedding"),
            config.num_channels,
            config.hidden_size,
            config.patch_size,
            0,
            config.patch_size,
            1,
            1,
            true,
        )?;
        let num_channels = config.num_channels;
        let patch_size = config.patch_size;
        let position_edge = config.max_image_size / patch_size;
        let num_positions = (position_edge).pow(2) + 1;
        let embed_dim = config.hidden_size;
        let position_embedding = embedding(num_positions, embed_dim, vb.pp("position_embedding"))?;
        let patch_pos_embed = position_embedding
            .embeddings()
            .i(1..)?
            .reshape((1, position_edge, position_edge, embed_dim))?
            .permute((0, 3, 1, 2))?;
        Ok(Self {
            patch_embedding,
            // position_embedding,
            num_channels,
            patch_size,
            // num_positions,
            // position_edge,
            embed_dim,
            patch_pos_embed,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let (num_patches, _) = pixel_values.dims2()?;
        let pixel_values = pixel_values.reshape((
            num_patches,
            self.num_channels,
            self.patch_size,
            self.patch_size,
        ))?;
        let patch_embeds = self.patch_embedding.forward(&pixel_values)?;
        let patch_embeds = patch_embeds
            .squeeze(D::Minus1)?
            .squeeze(D::Minus1)?
            .unsqueeze(0)?;
        let mut patch_pos_embed_list = vec![];
        let img_num = grid_thw.dim(0)?;
        for i in 0..img_num {
            let grid_i = grid_thw.i(i)?;
            let grid_h = grid_i.i(1)?.to_scalar::<u32>()? as usize;
            let grid_w = grid_i.i(2)?.to_scalar::<u32>()? as usize;
            let patch_pos_embed_ =
                interpolate_bilinear(&self.patch_pos_embed, (grid_h, grid_w), Some(false))?;
            let patch_pos_embed_ = patch_pos_embed_
                .reshape((self.embed_dim, ()))?
                .transpose(0, 1)?
                .unsqueeze(0)?;
            patch_pos_embed_list.push(patch_pos_embed_);
        }
        let patch_pos_embed = Tensor::cat(&patch_pos_embed_list, 1)?;
        let embedding = patch_embeds.add(&patch_pos_embed)?;
        Ok(embedding)
    }
}

pub struct HunYuanVisionPatchMerger {
    proj_0: Conv2d,
    proj_2: Conv2d,
    mlp: Linear,
    image_newline: Tensor,
    image_begin: Tensor,
    image_end: Tensor,
    // image_sep: Tensor,
    before_rms: RmsNorm,
    after_rms: RmsNorm,
}

impl HunYuanVisionPatchMerger {
    pub fn new(vb: VarBuilder, config: &HunYuanVLVisionConfig) -> Result<Self> {
        let proj_0 = get_conv2d(
            vb.pp("proj.0"),
            config.hidden_size,
            config.hidden_size * 2,
            config.spatial_merge_size,
            0,
            config.spatial_merge_size,
            1,
            1,
            true,
        )?;
        let proj_2 = get_conv2d(
            vb.pp("proj.2"),
            config.hidden_size * 2,
            config.hidden_size * 4,
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let mlp = linear(config.hidden_size * 4, config.out_hidden_size, vb.pp("mlp"))?;
        let image_newline =
            vb.get_with_hints(config.hidden_size * 4, "image_newline", Init::Const(0.))?;
        let image_begin =
            vb.get_with_hints(config.out_hidden_size, "image_begin", Init::Const(0.))?;
        let image_end = vb.get_with_hints(config.out_hidden_size, "image_end", Init::Const(0.))?;
        // let image_sep = vb.get_with_hints(config.out_hidden_size, "image_sep", Init::Const(0.))?;
        let before_rms = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("before_rms"))?;
        let after_rms = rms_norm(
            config.out_hidden_size,
            config.rms_norm_eps,
            vb.pp("after_rms"),
        )?;
        Ok(Self {
            proj_0,
            proj_2,
            mlp,
            image_newline,
            image_begin,
            image_end,
            // image_sep,
            before_rms,
            after_rms,
        })
    }
    pub fn forward(&self, xs: &Tensor, size: (usize, usize)) -> Result<Tensor> {
        let xs = self.before_rms.forward(xs)?;
        let (h, w) = size;
        let xs = xs.permute((0, 2, 1))?.reshape((xs.dim(0)?, (), h, w))?;
        let xs = self.proj_0.forward(&xs)?.gelu()?;
        let xs = self.proj_2.forward(&xs)?;
        let (b, c, h, _) = xs.dims4()?;
        let image_newline = self
            .image_newline
            .reshape((1, c, 1, 1))?
            .broadcast_as((b, c, h, 1))?
            .to_dtype(xs.dtype())?;
        let xs = Tensor::cat(&[xs, image_newline], D::Minus1)?;
        let xs = xs.reshape((b, c, ()))?.permute((0, 2, 1))?;
        let xs = self.mlp.forward(&xs)?;
        let begin = self
            .image_begin
            .reshape((1, 1, ()))?
            .broadcast_as((b, 1, xs.dim(D::Minus1)?))?
            .to_dtype(xs.dtype())?;
        let end = self
            .image_end
            .reshape((1, 1, ()))?
            .broadcast_as((b, 1, xs.dim(D::Minus1)?))?
            .to_dtype(xs.dtype())?;
        let xs = Tensor::cat(&[begin, xs, end], 1)?;
        let xs = self.after_rms.forward(&xs)?;
        Ok(xs)
    }
}

pub struct HunYuanVisionTransformer {
    embeddings: HunYuanVisionPatchEmbed,
    layers: Vec<NaiveAttnTwoLinearMLPBlock>,
    perceive: HunYuanVisionPatchMerger,
}

impl HunYuanVisionTransformer {
    pub fn new(vb: VarBuilder, config: &HunYuanVLVisionConfig) -> Result<Self> {
        let embeddings = HunYuanVisionPatchEmbed::new(vb.pp("embeddings"), config)?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            let layer_i = NaiveAttnTwoLinearMLPBlock::new(
                vb_layers.pp(i),
                config.hidden_size,
                config.num_attention_heads,
                None,
                None,
                true,
                "self_attn",
                None,
                config.intermediate_size,
                config.hidden_act,
                true,
                "mlp",
                "dense_h_to_4h",
                "dense_4h_to_h",
                config.rms_norm_eps,
                "input_layernorm",
                "post_attention_layernorm",
            )?;
            layers.push(layer_i);
        }
        let perceive = HunYuanVisionPatchMerger::new(vb.pp("perceive"), config)?;
        Ok(Self {
            embeddings,
            layers,
            perceive,
        })
    }

    pub fn forward(&self, xs: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(xs, grid_thw)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, None, None, None, false)?;
        }
        let mut cu_seqlens = vec![];
        for i in 0..grid_thw.dim(0)? {
            let [_, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            cu_seqlens.push((h * w) as usize);
        }
        let split_items = split_tensor(&hidden_states, &cu_seqlens, 1)?;
        let mut processed_item = vec![];
        for i in 0..grid_thw.dim(0)? {
            let [_, h, w] = grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            let processed = self
                .perceive
                .forward(&split_items[i], (h as usize, w as usize))?;
            processed_item.push(processed);
        }
        let xs = Tensor::cat(&processed_item, 1)?;
        Ok(xs)
    }
}

pub struct HunYuanVLAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    query_layernorm: RmsNorm,
    key_layernorm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl HunYuanVLAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        head_dim: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        attention_bias: bool,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let (q_proj, k_proj, v_proj, o_proj) = if attention_bias {
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
        let query_layernorm = rms_norm(head_dim, rms_norm_eps, vb.pp("query_layernorm"))?;
        let key_layernorm = rms_norm(head_dim, rms_norm_eps, vb.pp("key_layernorm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            query_layernorm,
            key_layernorm,
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
        let query_states = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;

        let key_states = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        let query_states = self.query_layernorm.forward(&query_states)?;
        let key_states = self.key_layernorm.forward(&key_states)?;
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

pub struct HunYuanVLDecoderLayer {
    self_attn: HunYuanVLAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl HunYuanVLDecoderLayer {
    pub fn new(config: &HunYuanVLConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = HunYuanVLAttention::new(
            vb.pp("self_attn"),
            config.hidden_size,
            config.head_dim,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.attention_bias,
            config.rms_norm_eps,
        )?;
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

pub struct HunYuanVLTextModel {
    embed_tokens: Embedding,
    layers: Vec<HunYuanVLDecoderLayer>,
    norm: RmsNorm,
    rope: RoPE,
    xdrope_section: Vec<usize>,
}

impl HunYuanVLTextModel {
    pub fn new(vb: VarBuilder, config: &HunYuanVLConfig) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            let layer = HunYuanVLDecoderLayer::new(config, vb_layers.pp(i))?;
            layers.push(layer);
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let base = config.rope_theta
            * config
                .rope_scaling
                .alpha
                .powf(config.head_dim as f64 / (config.head_dim - 2) as f64);
        let rope = RoPE::new(config.head_dim, base as f32, vb.device())?;
        let xdrope_section = config.rope_scaling.xdrope_section.clone();
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rope,
            xdrope_section,
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        position_ids: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;

        // let position_ids = match position_ids {
        //     Some(ids) => ids.clone(),
        //     None => Tensor::arange(
        //         seqlen_offset as u32,
        //         (seq_len + seqlen_offset) as u32,
        //         inputs_embeds.device(),
        //     )?
        //     .unsqueeze(0)?,
        // };
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

        let (cos, sin) = self
            .rope
            .forward(seqlen_offset, seq_len, inputs_embeds.device())?;
        let mut xs = inputs_embeds.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if i == 0
                && let Some(position_ids) = position_ids
            {
                let (cos, sin) =
                    get_xd_cos_sin(&cos, &sin, position_ids, self.xdrope_section.clone())?;
                xs = layer.forward(&xs, &cos, &sin, attention_mask)?;
            } else {
                xs = layer.forward(&xs, &cos, &sin, attention_mask)?;
            }
        }
        let xs = self.norm.forward(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

pub struct HunyuanVLModel {
    // config: HunYuanVLConfig,
    vit: HunYuanVisionTransformer,
    model: HunYuanVLTextModel,
    lm_head: Linear,
}

impl HunyuanVLModel {
    pub fn new(vb: VarBuilder, config: HunYuanVLConfig) -> Result<Self> {
        let vit = HunYuanVisionTransformer::new(vb.pp("vit"), &config.vision_config)?;
        let model = HunYuanVLTextModel::new(vb.pp("model"), &config)?;
        let lm_head = Linear::new(model.embed_tokens.embeddings().clone(), None);
        Ok(Self {
            // config,
            vit,
            model,
            lm_head,
        })
    }
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        image_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.model.embed_tokens.forward(input_ids)?;
        if let Some(pixel_values) = pixel_values
            && let Some(grid_thw) = image_grid_thw
            && let Some(image_mask) = image_mask
        {
            let image_embeds = self.vit.forward(pixel_values, grid_thw)?.squeeze(0)?;
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, image_mask)?;
        }
        let outputs = self
            .model
            .forward(&inputs_embeds, position_ids, seqlen_offset)?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
