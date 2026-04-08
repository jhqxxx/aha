use anyhow::{Ok, Result, anyhow};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Embedding, Module, RmsNorm, VarBuilder, embedding, rms_norm};

use crate::{
    models::{
        common::modules::{GateUpDownMLP, NaiveAttention},
        voxcpm::config::VoxMiniCPM4Config,
    },
    position_embed::rope::compute_default_rope_parameters,
    utils::tensor_utils::prepare_causal_attention_mask,
};

pub struct MiniCPMLongRoPE {
    short_factor: Vec<f32>,
    long_factor: Vec<f32>,
    original_max_position_embeddings: usize,
    max_seq_len_cached: usize,
    scaling_factor: f64,
    inv_freq: Tensor,
    cos_cached: Tensor,
    sin_cached: Tensor,
    device: Device,
    dtype: DType,
}
impl MiniCPMLongRoPE {
    pub fn new(cfg: &VoxMiniCPM4Config, device: &Device, dtype: DType) -> Result<Self> {
        let head_dim = cfg
            .kv_channels
            .unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rope_theta = cfg.rope_theta;
        let short_factor = cfg.rope_scaling.short_factor.clone();
        let long_factor = cfg.rope_scaling.short_factor.clone();
        let original_max_position_embeddings = cfg.rope_scaling.original_max_position_embeddings;
        let max_position_embeddings = cfg.max_position_embeddings;
        let scale = max_position_embeddings as f64 / original_max_position_embeddings as f64;
        let scaling_factor =
            (1.0 + scale.ln() / (original_max_position_embeddings as f64).ln()).sqrt();
        let inv_freq = compute_default_rope_parameters(head_dim, rope_theta);
        let inv_freq = Tensor::from_slice(&inv_freq, (1, inv_freq.len()), device)?;
        let max_seq_len_cached = max_position_embeddings;
        let t = Tensor::arange(0.0_f32, max_position_embeddings as f32, device)?
            .reshape((max_position_embeddings, 1))?;
        // short_factor.len() = 32
        // head_dim = 1024 / 16 = 64, inv_freq.len() = 32
        let ext_factors = Tensor::from_slice(&short_factor, (1, short_factor.len()), device)?;
        let ext_factors = Tensor::ones_like(&ext_factors)?.div(&ext_factors)?;
        // (seq_len, 1) matmul (1, 32) -> (seq_len, 32) * (1, 32)-> (seq_len, 32)
        let freqs = t.matmul(&ext_factors)?.broadcast_mul(&inv_freq)?;

        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos_cached = emb.cos()?.affine(scaling_factor, 0.0)?.to_dtype(dtype)?;
        let sin_cached = emb.sin()?.affine(scaling_factor, 0.0)?.to_dtype(dtype)?;
        Ok(Self {
            short_factor,
            long_factor,
            original_max_position_embeddings,
            max_seq_len_cached,
            scaling_factor,
            inv_freq,
            cos_cached,
            sin_cached,
            device: device.clone(),
            dtype,
        })
    }
    pub fn update_cos_sin_cache(&mut self, seqlen: usize) -> Result<()> {
        self.max_seq_len_cached = seqlen;
        let t = Tensor::arange(0.0_f32, seqlen as f32, &self.device)?.reshape((seqlen, 1))?;
        let mut ext_factors = Tensor::from_slice(
            &self.short_factor,
            (1, self.short_factor.len()),
            &self.device,
        )?;
        if seqlen > self.original_max_position_embeddings {
            ext_factors =
                Tensor::from_slice(&self.long_factor, (1, self.long_factor.len()), &self.device)?;
        }
        let ext_factors = Tensor::ones_like(&ext_factors)?.div(&ext_factors)?;
        let freqs = t.matmul(&ext_factors)?.broadcast_mul(&self.inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        let cos_cached = emb
            .cos()?
            .affine(self.scaling_factor, 0.0)?
            .to_dtype(self.dtype)?;
        let sin_cached = emb
            .sin()?
            .affine(self.scaling_factor, 0.0)?
            .to_dtype(self.dtype)?;
        self.cos_cached = cos_cached;
        self.sin_cached = sin_cached;
        Ok(())
    }
    pub fn forward(&mut self, pos_offset: usize, seqlen: usize) -> Result<(Tensor, Tensor)> {
        if pos_offset + seqlen > self.max_seq_len_cached {
            self.update_cos_sin_cache(pos_offset + seqlen)?;
        }
        let cos = self.cos_cached.narrow(0, pos_offset, seqlen)?;
        let sin = self.sin_cached.narrow(0, pos_offset, seqlen)?;

        Ok((cos, sin))
    }
}

pub struct MiniCPMDecoderLayer {
    self_attn: NaiveAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    scale_depth: f32,
    num_hidden_layers: usize,
    use_mup: bool,
}

impl MiniCPMDecoderLayer {
    pub fn new(vb: VarBuilder, cfg: &VoxMiniCPM4Config) -> Result<Self> {
        let self_attn = NaiveAttention::new(
            vb.pp("self_attn"),
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            Some(
                cfg.kv_channels
                    .unwrap_or(cfg.hidden_size / cfg.num_attention_heads),
            ),
            false,
            None,
            None,
            None,
            None,
        )?;
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            cfg.hidden_size,
            cfg.intermediate_size,
            candle_nn::Activation::Silu,
            false,
            None,
            None,
            None,
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
            scale_depth: cfg.scale_depth,
            num_hidden_layers: cfg.num_hidden_layers,
            use_mup: cfg.use_mup,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, cos, sin, attention_mask, true)?;
        let xs = if self.use_mup {
            (residual
                + xs.affine(
                    self.scale_depth as f64 / (self.num_hidden_layers as f64).sqrt(),
                    0.0,
                ))?
        } else {
            (residual + xs)?
        };
        let residual = xs.clone();
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = if self.use_mup {
            (residual
                + xs.affine(
                    self.scale_depth as f64 / (self.num_hidden_layers as f64).sqrt(),
                    0.0,
                ))?
        } else {
            (residual + xs)?
        };
        Ok(xs)
    }

    pub fn forward_with_cache(
        &mut self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_with_cache(&xs, cos, sin, attention_mask, true)?;
        let xs = if self.use_mup {
            (residual
                + xs.affine(
                    self.scale_depth as f64 / (self.num_hidden_layers as f64).sqrt(),
                    0.0,
                )?)?
        } else {
            (residual + xs)?
        };
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        let xs = if self.use_mup {
            (residual
                + xs.affine(
                    self.scale_depth as f64 / (self.num_hidden_layers as f64).sqrt(),
                    0.0,
                )?)?
        } else {
            (residual + xs)?
        };
        Ok(xs)
    }
    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct MiniCPMModel {
    // cfg: VoxMiniCPM4Config,
    pub embed_tokens: Option<Embedding>,
    layers: Vec<MiniCPMDecoderLayer>,
    norm: RmsNorm,
    rope_emb: Option<MiniCPMLongRoPE>,
}

impl MiniCPMModel {
    pub fn new(vb: VarBuilder, cfg: VoxMiniCPM4Config) -> Result<Self> {
        // let vb = vb.pp("model");
        let embed_tokens = if cfg.vocab_size > 0 {
            Some(embedding(
                cfg.vocab_size,
                cfg.hidden_size,
                vb.pp("embed_tokens"),
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let layer = MiniCPMDecoderLayer::new(vb_layers.pp(i), &cfg)?;
            layers.push(layer);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        let rope_emb = if let Some(flag) = cfg.no_rope
            && flag
        {
            None
        } else {
            Some(MiniCPMLongRoPE::new(&cfg, vb.device(), vb.dtype())?)
        };

        Ok(Self {
            // cfg,
            embed_tokens,
            layers,
            norm,
            rope_emb,
        })
    }

    pub fn forward(
        &mut self,
        input_embeds: &Tensor,
        position_id: usize,
        is_causal: bool,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = input_embeds.dims3()?;
        let attention_mask: Option<Tensor> = {
            if !is_causal || seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    bs,
                    seq_len,
                    0,
                    input_embeds.device(),
                )?)
            }
        };
        let (cos, sin) = if let Some(rope_emb) = &mut self.rope_emb {
            let (cos, sin) = rope_emb.forward(position_id, seq_len)?;
            (Some(cos), Some(sin))
        } else {
            (None, None)
        };
        let mut hidden_states = input_embeds.clone();
        for decode_layer in &self.layers {
            hidden_states = decode_layer.forward(
                &hidden_states,
                cos.as_ref(),
                sin.as_ref(),
                attention_mask.as_ref(),
            )?;
        }
        hidden_states = self.norm.forward(&hidden_states)?;
        Ok(hidden_states)
    }

    pub fn forward_with_cache(
        &mut self,
        input_embeds: &Tensor,
        position_id: usize,
    ) -> Result<Tensor> {
        let input_embeds = match input_embeds.rank() {
            2 => input_embeds.unsqueeze(1)?,
            3 => input_embeds.clone(),
            _ => return Err(anyhow!("MiniCPMModelinput_embeds illigal")),
        };
        let (bs, seq_len, _) = input_embeds.dims3()?;
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    bs,
                    seq_len,
                    0,
                    input_embeds.device(),
                )?)
            }
        };
        // let (cos, sin) = self.rope_emb.forward(position_id, seq_len)?;
        let (cos, sin) = if let Some(rope_emb) = &mut self.rope_emb {
            let (cos, sin) = rope_emb.forward(position_id, seq_len)?;
            (Some(cos), Some(sin))
        } else {
            (None, None)
        };
        let mut hidden_states = input_embeds.clone();
        for decode_layer in &mut self.layers {
            hidden_states = decode_layer.forward_with_cache(
                &hidden_states,
                cos.as_ref(),
                sin.as_ref(),
                attention_mask.as_ref(),
            )?;
        }
        hidden_states = self.norm.forward(&hidden_states)?;

        Ok(hidden_states)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
