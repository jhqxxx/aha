use crate::{
    models::{
        common::{GateUpDownMLP, QKNormAttention, conv1d_depthwise, get_conv1d},
        lfm2::config::Lfm2Config,
    },
    position_embed::rope::RoPE,
    utils::tensor_utils::prepare_causal_attention_mask,
};
use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::{
    Conv1d, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_b, rms_norm,
};

pub struct Lfm2ShortConv {
    l_cache: usize,
    conv: Conv1d,
    in_proj: Linear,
    out_proj: Linear,
    cache: Option<Tensor>,
}

impl Lfm2ShortConv {
    pub fn new(vb: VarBuilder, config: &Lfm2Config) -> Result<Self> {
        let l_cache = config.conv_l_cache;
        let bias = config.conv_bias;
        let conv = get_conv1d(
            vb.pp("conv"),
            config.hidden_size,
            config.hidden_size,
            l_cache,
            l_cache - 1,
            1,
            1,
            config.hidden_size,
            bias,
        )?;
        let in_proj = linear_b(
            config.hidden_size,
            config.hidden_size * 3,
            bias,
            vb.pp("in_proj"),
        )?;
        let out_proj = linear_b(
            config.hidden_size,
            config.hidden_size,
            bias,
            vb.pp("out_proj"),
        )?;
        Ok(Self {
            l_cache,
            conv,
            in_proj,
            out_proj,
            cache: None,
        })
    }

    pub fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        let seq_len = xs.dim(1)?;
        let bc_x = self.in_proj.forward(xs)?.transpose(D::Minus1, D::Minus2)?;
        let chunk = bc_x.chunk(3, D::Minus2)?;
        let bx = chunk[0].mul(&chunk[2])?;
        let c: &Tensor = &chunk[1];
        let conv_out = if self.cache.is_none() && seq_len > 1 {
            let pad_num = self.l_cache as isize - seq_len as isize;
            let conv_state = if pad_num > 0 {
                bx.pad_with_zeros(D::Minus1, pad_num as usize, 0)?
            } else {
                bx.narrow(D::Minus1, pad_num.unsigned_abs(), self.l_cache)?
            };
            self.cache = Some(conv_state);
            let bx = bx.pad_with_zeros(D::Minus1, self.l_cache - 1, self.l_cache - 1)?;
            let bx = conv1d_depthwise(&bx, self.conv.weight(), self.conv.bias())?;
            bx.narrow(D::Minus1, 0, seq_len)?
        } else {
            let conv_state = self.cache.as_ref().unwrap();
            let conv_state = Tensor::cat(&[conv_state, &bx], D::Minus1)?;
            let conv_state = conv_state.narrow(D::Minus1, 1, self.l_cache)?;
            let conv_out = conv1d_depthwise(&conv_state, self.conv.weight(), self.conv.bias())?;
            self.cache = Some(conv_state);
            conv_out
        };
        let y = c.mul(&conv_out)?;
        let y = y.transpose(D::Minus1, D::Minus2)?.contiguous()?;
        let y = self.out_proj.forward(&y)?;
        Ok(y)
    }

    pub fn clear_cache(&mut self) {
        self.cache = None;
    }
}

enum LayerKind {
    SelfAttn(QKNormAttention),
    Conv(Lfm2ShortConv),
}

impl LayerKind {
    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        match self {
            LayerKind::SelfAttn(attn) => attn.forward(xs, cos, sin, attention_mask),
            LayerKind::Conv(conv) => conv.forward(xs),
        }
    }
}

pub struct Lfm2DecoderLayer {
    layer: LayerKind,
    feed_forward: GateUpDownMLP,
    operator_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl Lfm2DecoderLayer {
    pub fn new(vb: VarBuilder, config: &Lfm2Config, layer_type: &str) -> Result<Self> {
        let layer = if layer_type.eq("full_attention") {
            let attn = QKNormAttention::new(
                vb.pp("self_attn"),
                config.hidden_size,
                config.num_attention_heads,
                None,
                Some(config.num_key_value_heads),
                false,
                config.block_norm_eps,
                Some("q_proj"),
                Some("k_proj"),
                Some("v_proj"),
                Some("out_proj"),
                Some("q_layernorm"),
                Some("k_layernorm"),
            )?;
            LayerKind::SelfAttn(attn)
        } else {
            let conv = Lfm2ShortConv::new(vb.pp("conv"), config)?;
            LayerKind::Conv(conv)
        };
        let intermediate_size = if config.block_auto_adjust_ff_dim {
            let inter_size = 2 * config.block_ff_dim / 3;
            let inter_size = (config.block_ffn_dim_multiplier * inter_size as f64) as usize;
            config.block_multiple_of
                * ((inter_size + config.block_multiple_of - 1) / config.block_multiple_of)
        } else {
            config.block_ff_dim
        };
        let feed_forward = GateUpDownMLP::new(
            vb.pp("feed_forward"),
            config.hidden_size,
            intermediate_size,
            candle_nn::Activation::Silu,
            false,
            Some("w1"),
            Some("w3"),
            Some("w2"),
        )?;

        let operator_norm = rms_norm(config.hidden_size, config.norm_eps, vb.pp("operator_norm"))?;
        let ffn_norm = rms_norm(config.hidden_size, config.norm_eps, vb.pp("ffn_norm"))?;
        Ok(Self {
            layer,
            feed_forward,
            operator_norm,
            ffn_norm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let res = xs.clone();
        let xs = self.operator_norm.forward(xs)?;
        let xs = self.layer.forward(&xs, cos, sin, attention_mask)?;
        let res = xs.add(&res)?;
        let xs = self.ffn_norm.forward(&res)?;
        let xs = self.feed_forward.forward(&xs)?;
        let xs = xs.add(&res)?;
        Ok(xs)
    }

    pub fn clear_cache(&mut self) {
        match &mut self.layer {
            LayerKind::SelfAttn(attn) => attn.clear_kv_cache(),
            LayerKind::Conv(conv) => conv.clear_cache(),
        }
    }
}

pub struct Lfm2Decoder {
    embed_tokens: Embedding,
    layers: Vec<Lfm2DecoderLayer>,
    // rotary_emb: RoPE,
    pos_emb: RoPE,
    embedding_norm: RmsNorm,
}

impl Lfm2Decoder {
    pub fn new(vb: VarBuilder, config: &Lfm2Config) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        // let layer_types = config.layer_types.as_ref().unwrap();
        let layer_types = config.get_layer_types()?;
        for i in 0..config.num_hidden_layers {
            let layer_type = layer_types.get(i).unwrap();
            let layer = Lfm2DecoderLayer::new(vb_layers.pp(i), config, layer_type)?;
            layers.push(layer);
        }
        let dim = config.hidden_size / config.num_attention_heads;
        let pos_emb = RoPE::new(dim, config.rope_theta, vb.device())?;
        let embedding_norm =
            rms_norm(config.hidden_size, config.norm_eps, vb.pp("embedding_norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            pos_emb,
            embedding_norm,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        inputs_embeds: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let inputs_embeds = if let Some(embed) = inputs_embeds {
            embed.clone()
        } else {
            self.embed_tokens.forward(input_ids)?
        };

        let (bs, seq_len, _) = inputs_embeds.dims3()?;
        let attention_mask = if seq_len > 1 {
            Some(prepare_causal_attention_mask(
                bs,
                seq_len,
                seqlen_offset,
                inputs_embeds.device(),
            )?)
        } else {
            None
        };
        let (cos, sin) = self
            .pos_emb
            .forward(seqlen_offset, seq_len, inputs_embeds.device())?;
        let mut xs = inputs_embeds;
        for layer in &mut self.layers {
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
        }
        let xs = self.embedding_norm.forward(&xs)?;
        Ok(xs)
    }

    pub fn clear_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_cache()
        }
    }
}

pub struct Lfm2Model {
    model: Lfm2Decoder,
    lm_head: Linear,
}

impl Lfm2Model {
    pub fn new(vb: VarBuilder, config: &Lfm2Config) -> Result<Self> {
        let model = Lfm2Decoder::new(vb.pp("model"), config)?;
        let lm_head = if let Some(flag) = config.tie_embedding
            && flag
        {
            Linear::new(model.embed_tokens.embeddings().clone(), None)
        } else {
            let linear = linear_b(
                config.hidden_size,
                config.vocab_size,
                false,
                vb.pp("lm_head"),
            );
            match linear {
                Ok(linear) => linear,
                Err(_) => Linear::new(model.embed_tokens.embeddings().clone(), None),
            }
        };
        Ok(Self { model, lm_head })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let xs = self.model.forward(input_ids, None, seqlen_offset)?;
        let seq_len = xs.dim(1)?;
        let last_xs = xs.narrow(1, seq_len - 1, 1)?;
        let xs = self.lm_head.forward(&last_xs)?;
        Ok(xs)
    }

    pub fn clear_cache(&mut self) {
        self.model.clear_cache();
    }
}
