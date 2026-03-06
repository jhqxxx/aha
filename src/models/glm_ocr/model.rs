//! GLM-OCR Model Implementation

use anyhow::Result;
use candle_core::{D, DType, IndexOp, Tensor};
use candle_nn::{
    Activation, Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, Module, RmsNorm, VarBuilder,
    conv2d, embedding, layer_norm, linear, linear_no_bias, rms_norm,
};

use crate::{
    models::{
        common::GateUpDownMLP,
        glm_ocr::config::{
            GlmOcrConfig, GlmOcrProjectorConfig, GlmOcrTextConfig, GlmOcrVisionConfig,
        },
    },
    position_embed::rope::{apply_rotary_pos_emb_vision, glm_ocr_apply_rotary_pos_emb},
    utils::{
        tensor_utils::{prepare_causal_attention_mask, repeat_kv},
    },
};

pub struct GlmOcrRMSNorm(RmsNorm);

impl GlmOcrRMSNorm {
    pub fn new(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        let rms = rms_norm(hidden_size, eps, vb)?;
        Ok(Self(rms))
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(xs)?)
    }
    pub fn extra_repr(&self) -> String {
        "GlmOcrRMSNorm".to_string()
    }
}

pub struct GlmOcrVisionMlp(GateUpDownMLP);

impl GlmOcrVisionMlp {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let mlp = GateUpDownMLP::new(
            vb,
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            config.attention_bias,
            Some("gate_proj"),
            Some("up_proj"),
            Some("down_proj"),
        )?;
        Ok(Self(mlp))
    }

    pub fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        Ok(self.0.forward(hidden_state)?)
    }
}

fn eager_attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
    dropout: f64,
) -> Result<(Tensor, Tensor)> {
    let key_states = match num_key_value_groups {
        Some(g) => repeat_kv(key_states.clone(), g)?.contiguous()?,
        None => key_states.clone(),
    };
    let value_states = match num_key_value_groups {
        Some(g) => repeat_kv(value_states.clone(), g)?.contiguous()?,
        None => value_states.clone(),
    };
    let query_states = query_states.contiguous()?;
    let key_states = key_states.contiguous()?;
    let value_states = value_states.contiguous()?;

    let output = {
        #[cfg(feature = "flash-attn")]
        {
            // Flash attention: causal iff attention_mask is present.
            // Explicit contiguous() ensures proper memory layout for flash_attn kernel
            let q = query_states.transpose(1, 2)?.contiguous()?;
            let k = key_states.transpose(1, 2)?.contiguous()?;
            let v = value_states.transpose(1, 2)?.contiguous()?;
            candle_flash_attn::flash_attn(&q, &k, &v, scaling as f32, attention_mask.is_some())?
                // flash_attn returns [batch, q_len, heads, head_dim] — already in final layout
        }
        #[cfg(not(feature = "flash-attn"))]
        {
            // Chunked Q-attention: process Q in blocks so the attention matrix
            // [batch, heads, CHUNK, k_len] stays bounded regardless of q_len.
            // Peak memory per chunk: CHUNK × k_len × heads × 4 bytes (f32 softmax).
            // Mathematically equivalent to full attention.
            // CHUNK_SIZE=512 is empirically optimal for most hardware (CPU/GPU balance)
            const CHUNK_SIZE: usize = 512;
            let q_len = query_states.dim(2)?;
            let k_t = key_states.transpose(D::Minus2, D::Minus1)?.contiguous()?;

            let raw = if q_len > CHUNK_SIZE {
                let mut chunks: Vec<Tensor> = Vec::with_capacity((q_len + CHUNK_SIZE - 1) / CHUNK_SIZE);
                let mut start = 0;
                while start < q_len {
                    let len = CHUNK_SIZE.min(q_len - start);
                    let q_chunk = query_states.narrow(2, start, len)?;
                    let attn = (q_chunk.matmul(&k_t)? * scaling)?;
                    let attn = match attention_mask {
                        None => attn,
                        Some(mask) => {
                            attn.broadcast_add(&mask.narrow(2, start, len)?.to_dtype(attn.dtype())?)?
                        }
                    };
                    // Softmax computation: Optimize dtype conversions for CPU (which uses F32)
                    let attn = if query_states.dtype() == DType::F32 {
                        candle_nn::ops::softmax_last_dim(&attn)?
                    } else {
                        candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?
                            .to_dtype(query_states.dtype())?
                    };
                    // Apply dropout uniformly across chunked and non-chunked paths for consistency
                    let attn = candle_nn::ops::dropout(&attn, dropout as f32)?;
                    chunks.push(attn.matmul(&value_states)?);
                    start += len;
                }
                Tensor::cat(&chunks, 2)? // [batch, heads, q_len, head_dim]
            } else {
                let attn = (query_states.matmul(&k_t)? * scaling)?;
                let attn = match attention_mask {
                    None => attn,
                    Some(mask) => attn.broadcast_add(&mask.to_dtype(attn.dtype())?)?,
                };
                // Softmax computation: Same optimization as chunked path
                let attn = if query_states.dtype() == DType::F32 {
                    candle_nn::ops::softmax_last_dim(&attn)?
                } else {
                    candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?
                        .to_dtype(query_states.dtype())?
                };
                // Apply dropout uniformly (now consistent across both paths)
                let attn = candle_nn::ops::dropout(&attn, dropout as f32)?;
                attn.matmul(&value_states)?
            };
            // [batch, heads, q_len, head_dim] -> [batch, q_len, heads, head_dim]
            raw.transpose(1, 2)?.contiguous()?
        }
    };

    // output layout: [batch, q_len, heads, head_dim]
    let placeholder = Tensor::zeros((0,), query_states.dtype(), query_states.device())?;
    Ok((output, placeholder))
}

pub struct GlmOcrTextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl GlmOcrTextAttention {
    pub fn new(
        vb: VarBuilder,
        config: &GlmOcrTextConfig,
        _layer_idx: Option<usize>,
    ) -> Result<Self> {
        let head_dim = config.head_dim.unwrap_or_else(|| {
            // Integer division, panics if num_attention_heads is 0 (like Python)
            config.hidden_size / config.num_attention_heads
        });
        let num_kv_groups = config.num_attention_heads / config.num_key_value_heads;

        let scaling = 1.0 / (head_dim as f64).sqrt();

        let q_proj = linear_no_bias(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_no_bias(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_no_bias(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_no_bias(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (bs, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((bs, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((bs, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (cos, sin) = position_embeddings;
        let (query_states, key_states) =
            glm_ocr_apply_rotary_pos_emb(&query_states, &key_states, cos, sin)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        // Python: dropout=0.0 if not self.training else self.attention_dropout
        // Rust is inference-only, so always 0.0
        let (attn_output, attn_weights) = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
            0.0,
        )?;

        let attn_output = attn_output.reshape((bs, q_len, ()))?;
        Ok((self.o_proj.forward(&attn_output)?, attn_weights))
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

pub struct GlmOcrVisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl GlmOcrVisionRotaryEmbedding {
    pub fn new(dim: usize, theta: f32, device: &candle_core::Device, dtype: DType) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq =
            Tensor::from_vec(inv_freq, (dim / 2,), device)?.to_dtype(dtype)?;
        Ok(Self { inv_freq })
    }

    pub fn forward(&self, seqlen: usize) -> Result<Tensor> {
        // Python: freqs = torch.outer(seq, self.inv_freq) -> (seqlen, dim/4)
        let seq = Tensor::arange(0f32, seqlen as f32, self.inv_freq.device())?;
        let seq = seq.to_dtype(self.inv_freq.dtype())?;
        let freqs = seq.unsqueeze(1)?.matmul(&self.inv_freq.unsqueeze(0)?)?;
        Ok(freqs)
    }

    pub fn rot_pos_emb(
        &self,
        grid_thw: &[(usize, usize, usize)],
        spatial_merge_size: usize,
    ) -> Result<(Tensor, Tensor)> {
        let sms = spatial_merge_size;
        let mut all_hpos: Vec<u32> = Vec::new();
        let mut all_wpos: Vec<u32> = Vec::new();
        let mut max_grid_size: usize = 0;

        for &(t, h, w) in grid_thw {
            max_grid_size = max_grid_size.max(h).max(w);

            for _ in 0..t {
                for hi in 0..h {
                    for wi in 0..w {
                        // Apply spatial merge rearrangement
                        let _hb = hi / sms;
                        let _si = hi % sms;
                        let _wb = wi / sms;
                        let _sj = wi % sms;
                        
                        // After permute(0,2,1,3): position = (hb, wb, si, sj)
                        // Flatten: idx = hb * w_blocks * sms * sms + wb * sms * sms + si * sms + sj
                        // But we just need the h and w positions for rotary embedding
                        all_hpos.push(hi as u32);
                        all_wpos.push(wi as u32);
                    }
                }
            }
        }

        let total_seq = all_hpos.len();
        let freqs_full = self.forward(max_grid_size)?; // (max_grid_size, dim/4)

        let h_indices = Tensor::from_vec(all_hpos, (total_seq,), self.inv_freq.device())?;
        let w_indices = Tensor::from_vec(all_wpos, (total_seq,), self.inv_freq.device())?;
        let h_freqs = freqs_full.index_select(&h_indices, 0)?; // (total_seq, dim/4)
        let w_freqs = freqs_full.index_select(&w_indices, 0)?; // (total_seq, dim/4)

        // Concatenate h and w freqs: (total_seq, dim/2)
        let rotary_pos_emb = Tensor::cat(&[&h_freqs, &w_freqs], 1)?;

        let emb = Tensor::cat(&[&rotary_pos_emb, &rotary_pos_emb], 1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;

        Ok((cos, sin))
    }
}

pub struct GlmOcrTextMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GlmOcrTextMLP {
    pub fn new(vb: VarBuilder, config: &GlmOcrTextConfig) -> Result<Self> {
        let gate_up_proj = linear_no_bias(
            config.hidden_size,
            2 * config.intermediate_size,
            vb.pp("gate_up_proj"),
        )?;
        let down_proj = linear_no_bias(
            config.intermediate_size,
            config.hidden_size,
            vb.pp("down_proj"),
        )?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            act_fn: config.hidden_act,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let up_states = self.gate_up_proj.forward(xs)?;
        let dim = up_states.dims().len() - 1;
        let chunks = up_states.chunk(2, dim)?;
        let gate = &chunks[0];
        let up = &chunks[1];
        let up_states = up.broadcast_mul(&self.act_fn.forward(gate)?)?;
        Ok(self.down_proj.forward(&up_states)?)
    }
}

pub struct GlmOcrTextDecoderLayer {
    self_attn: GlmOcrTextAttention,
    mlp: GlmOcrTextMLP,
    input_layernorm: GlmOcrRMSNorm,
    post_attention_layernorm: GlmOcrRMSNorm,
    post_self_attn_layernorm: GlmOcrRMSNorm,
    post_mlp_layernorm: GlmOcrRMSNorm,
}

impl GlmOcrTextDecoderLayer {
    pub fn new(vb: VarBuilder, config: &GlmOcrTextConfig, layer_idx: usize) -> Result<Self> {
        let self_attn = GlmOcrTextAttention::new(vb.pp("self_attn"), config, Some(layer_idx))?;
        let mlp = GlmOcrTextMLP::new(vb.pp("mlp"), config)?;
        let input_layernorm = GlmOcrRMSNorm::new(
            vb.pp("input_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_attention_layernorm = GlmOcrRMSNorm::new(
            vb.pp("post_attention_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_self_attn_layernorm = GlmOcrRMSNorm::new(
            vb.pp("post_self_attn_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;
        let post_mlp_layernorm = GlmOcrRMSNorm::new(
            vb.pp("post_mlp_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            post_self_attn_layernorm,
            post_mlp_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        position_embeddings: (&Tensor, &Tensor),
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let (xs, _attn_weights) =
            self.self_attn
                .forward(&xs, position_embeddings, attention_mask)?;
        let xs = self.post_self_attn_layernorm.forward(&xs)?;
        let xs = residual.add(&xs)?;

        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = self.post_mlp_layernorm.forward(&xs)?;
        Ok(xs.add(&residual)?)
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct GlmOcrVisionAttention {
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    qkv: Linear,
    proj: Linear,
    q_norm: GlmOcrRMSNorm,
    k_norm: GlmOcrRMSNorm,
}

impl GlmOcrVisionAttention {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let qkv = linear(config.hidden_size, config.hidden_size * 3, vb.pp("qkv"))?;
        let proj = linear(config.hidden_size, config.hidden_size, vb.pp("proj"))?;
        let q_norm = GlmOcrRMSNorm::new(vb.pp("q_norm"), head_dim, config.rms_norm_eps)?;
        let k_norm = GlmOcrRMSNorm::new(vb.pp("k_norm"), head_dim, config.rms_norm_eps)?;

        Ok(Self {
            num_heads: config.num_heads,
            head_dim,
            scaling,
            qkv,
            proj,
            q_norm,
            k_norm,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        position_embeddings: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;
        let qkv = self.qkv.forward(xs)?;
        let qkv = qkv
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (cos, sin) = if let Some((cos, sin)) = position_embeddings {
            (cos, sin)
        } else {
            return Err(anyhow::anyhow!(
                "Position embeddings required for vision attention"
            ));
        };

        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        let q = q.transpose(0, 1)?.unsqueeze(0)?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?;
        let v = v.transpose(0, 1)?.unsqueeze(0)?;

        // Vision attention always uses num_key_value_groups = 1
        let (attn_output, _attn_weights) =
            eager_attention_forward(&q, &k, &v, Some(1), None, self.scaling, 0.0)?;
        let attn_output = attn_output.reshape((seq_len, ()))?;
        Ok(self.proj.forward(&attn_output)?)
    }

    pub fn forward_with_params(
        &self,
        xs: &Tensor,
        _cu_seqlens: &Tensor,
        _rotary_pos_emb: Option<&Tensor>,
        position_embeddings: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let (seq_len, _) = xs.dims2()?;
        let qkv = self.qkv.forward(xs)?;
        let qkv = qkv
            .reshape((seq_len, 3, self.num_heads, self.head_dim))?
            .permute((1, 0, 2, 3))?;

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (cos, sin) = if let Some((cos, sin)) = position_embeddings {
            (cos, sin)
        } else {
            return Err(anyhow::anyhow!(
                "Position embeddings required for vision attention"
            ));
        };

        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        let q = q.transpose(0, 1)?.unsqueeze(0)?;
        let k = k.transpose(0, 1)?.unsqueeze(0)?;
        let v = v.transpose(0, 1)?.unsqueeze(0)?;

        // Vision attention always uses num_key_value_groups = 1
        let (attn_output, _attn_weights) =
            eager_attention_forward(&q, &k, &v, Some(1), None, self.scaling, 0.0)?;
        let attn_output = attn_output.reshape((seq_len, ()))?;
        Ok(self.proj.forward(&attn_output)?)
    }
}

pub struct GlmOcrVisionBlock {
    norm1: GlmOcrRMSNorm,
    norm2: GlmOcrRMSNorm,
    attn: GlmOcrVisionAttention,
    mlp: GlmOcrVisionMlp,
}

impl GlmOcrVisionBlock {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let norm1 = GlmOcrRMSNorm::new(vb.pp("norm1"), config.hidden_size, config.rms_norm_eps)?;
        let attn = GlmOcrVisionAttention::new(vb.pp("attn"), config)?;
        let norm2 = GlmOcrRMSNorm::new(vb.pp("norm2"), config.hidden_size, config.rms_norm_eps)?;
        let mlp = GlmOcrVisionMlp::new(vb.pp("mlp"), config)?;

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
        rotary_pos_emb: Option<&Tensor>,
        position_embeddings: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs =
            self.attn
                .forward_with_params(&xs, cu_seqlens, rotary_pos_emb, position_embeddings)?;
        let xs = residual.add(&xs)?;

        let residual = xs.clone();
        let xs = self.norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        Ok(xs.add(&residual)?)
    }
}

pub struct GlmOcrVisionPatchMerger {
    proj: Linear,
    post_projection_norm: LayerNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GlmOcrVisionPatchMerger {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let proj = linear_no_bias(
            config.out_hidden_size,
            config.out_hidden_size,
            vb.pp("proj"),
        )?;

        let post_projection_norm = layer_norm(
            config.out_hidden_size,
            config.rms_norm_eps,
            vb.pp("post_projection_norm"),
        )?;

        let context_dim = config.out_hidden_size * config.in_channels;
        let gate_proj = linear_no_bias(
            config.out_hidden_size,
            context_dim,
            vb.pp("gate_proj"),
        )?;
        let up_proj = linear_no_bias(
            config.out_hidden_size,
            context_dim,
            vb.pp("up_proj"),
        )?;
        let down_proj = linear_no_bias(
            context_dim,
            config.out_hidden_size,
            vb.pp("down_proj"),
        )?;

        Ok(Self {
            proj,
            post_projection_norm,
            gate_proj,
            up_proj,
            down_proj,
            act_fn: config.hidden_act,
        })
    }

    pub fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let mut hidden_state = self.proj.forward(hidden_state)?;
        hidden_state = self.post_projection_norm.forward(&hidden_state)?;
        hidden_state = hidden_state.gelu()?;

        let gate = self.gate_proj.forward(&hidden_state)?;
        let gate = self.act_fn.forward(&gate)?;
        let up = self.up_proj.forward(&hidden_state)?;
        let result = gate.broadcast_mul(&up)?;

        Ok(self.down_proj.forward(&result)?)
    }
}

pub struct GlmOcrVisionPatchEmbed {
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    #[allow(dead_code)] embed_dim: usize,
    proj: Linear,
}

impl GlmOcrVisionPatchEmbed {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let patch_dim =
            config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;

        let weight = vb
            .get(
                (
                    config.hidden_size,
                    config.in_channels,
                    config.temporal_patch_size,
                    config.patch_size,
                    config.patch_size,
                ),
                "proj.weight",
            )?
            .reshape((config.hidden_size, patch_dim))?;

        let bias = vb.get(config.hidden_size, "proj.bias").ok();

        let proj = candle_nn::Linear::new(weight, bias);

        Ok(Self {
            patch_size: config.patch_size,
            temporal_patch_size: config.temporal_patch_size,
            in_channels: config.in_channels,
            embed_dim: config.hidden_size,
            proj,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let rank = pixel_values.rank();
        
        if rank == 2 {
            let hidden_states = self.proj.forward(pixel_values)?;
            Ok(hidden_states)
        } else {
            let (batch, _c, h, w) = pixel_values.dims4()?;

            let patches_h = h / self.patch_size;
            let patches_w = w / self.patch_size;
            let num_patches = patches_h * patches_w;

            let pv = pixel_values.reshape((
                batch,
                patches_h,
                self.patch_size,
                patches_w,
                self.patch_size,
                self.in_channels,
            ))?;

            let pv = pv.permute((0, 1, 3, 5, 2, 4))?;

            let pv = pv.reshape((
                batch * num_patches,
                self.in_channels * self.patch_size * self.patch_size,
            ))?;

            let pv = pv.unsqueeze(1)?;
            let ones_shape: Vec<usize> = vec![1, self.temporal_patch_size];
            let pv = pv.broadcast_mul(&Tensor::ones(ones_shape, pv.dtype(), pv.device())?)?;
            let pv = pv.reshape((
                batch * num_patches,
                self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size,
            ))?;

            let hidden_states = self.proj.forward(&pv)?;
            Ok(hidden_states)
        }
    }
}

pub struct GlmOcrVisionModel {
    patch_embed: GlmOcrVisionPatchEmbed,
    rotary_pos_emb: GlmOcrVisionRotaryEmbedding,
    blocks: Vec<GlmOcrVisionBlock>,
    merger: GlmOcrVisionPatchMerger,
    downsample: Conv2d,
    post_layernorm: GlmOcrRMSNorm,
    config: GlmOcrVisionConfig,
}

impl GlmOcrVisionModel {
    pub fn new(vb: VarBuilder, config: &GlmOcrVisionConfig) -> Result<Self> {
        let patch_embed = GlmOcrVisionPatchEmbed::new(vb.pp("patch_embed"), config)?;

        let head_dim = config.hidden_size / config.num_heads;
        let rotary_pos_emb =
            GlmOcrVisionRotaryEmbedding::new(head_dim / 2, config.rope_theta, vb.device(), vb.dtype())?;

        let mut blocks = Vec::new();
        let depth = config.depth;
        for i in 0..depth {
            let block = GlmOcrVisionBlock::new(vb.pp("blocks").pp(i), config)?;
            blocks.push(block);
        }

        let merger = GlmOcrVisionPatchMerger::new(vb.pp("merger"), config)?;

        let downsample = conv2d(
            config.hidden_size,
            config.out_hidden_size,
            config.spatial_merge_size,
            Conv2dConfig {
                stride: config.spatial_merge_size,
                ..Default::default()
            },
            vb.pp("downsample"),
        )?;

        let post_layernorm = GlmOcrRMSNorm::new(
            vb.pp("post_layernorm"),
            config.hidden_size,
            config.rms_norm_eps,
        )?;

        Ok(Self {
            patch_embed,
            rotary_pos_emb,
            blocks,
            merger,
            downsample,
            post_layernorm,
            config: config.clone(),
        })
    }

    pub fn forward(&self, pixel_values: &Tensor, grid_thw: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.patch_embed.forward(pixel_values)?;

        // Parse grid_thw - may be shape (3,) or (N, 3)
        let grid_thw_parsed = if grid_thw.dims().len() == 1 {
            let t = grid_thw.i(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
            let h = grid_thw.i(1)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
            let w = grid_thw.i(2)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
            vec![(t, h, w)]
        } else {
            let grid_thw = grid_thw.to_dtype(DType::F32)?;
            let n = grid_thw.dim(0)?;
            let mut result = Vec::new();
            for i in 0..n {
                let row = grid_thw.i(i)?;
                let t = row.i(0)?.to_scalar::<f32>()? as usize;
                let h = row.i(1)?.to_scalar::<f32>()? as usize;
                let w = row.i(2)?.to_scalar::<f32>()? as usize;
                result.push((t, h, w));
            }
            result
        };

        let (cos, sin) = self
            .rotary_pos_emb
            .rot_pos_emb(&grid_thw_parsed, self.config.spatial_merge_size)?;

        let rotary_pos_emb = Tensor::cat(&[&cos, &sin], D::Minus1)?;
        let position_embeddings = (&cos, &sin);

        let mut cu_seqlens_values: Vec<i32> = vec![0];
        let mut cumsum: i32 = 0;
        for (t, h, w) in &grid_thw_parsed {
            let spatial_patches = (h * w) as i32;
            for _ in 0..*t {
                cumsum += spatial_patches;
                cu_seqlens_values.push(cumsum);
            }
        }
        let cu_seqlens = Tensor::from_slice(
            &cu_seqlens_values,
            &[cu_seqlens_values.len()],
            hidden_states.device(),
        )?;

        for block in self.blocks.iter() {
            hidden_states = block.forward(
                &hidden_states,
                &cu_seqlens,
                Some(&rotary_pos_emb),
                Some(position_embeddings),
            )?;
        }

        let hidden_states = self.post_layernorm.forward(&hidden_states)?;

        let sms = self.config.spatial_merge_size;
        let hidden_dim = hidden_states.dim(hidden_states.dims().len() - 1)?;
        
        let total_patches = hidden_states.dim(0)?;  // 2816
        let merged_patches = total_patches / (sms * sms);  // 704
        let hidden_states = hidden_states.reshape((merged_patches, sms, sms, hidden_dim))?;
        let hidden_states = hidden_states.permute((0, 3, 1, 2))?;  // [704, 1024, 2, 2]
        let hidden_states = self.downsample.forward(&hidden_states)?;  // [704, 1536, 1, 1]
        let hidden_states = hidden_states.reshape((merged_patches, self.config.out_hidden_size))?;  // [704, 1536]

        let merged = self.merger.forward(&hidden_states)?;

        let merged = merged.unsqueeze(0)?;
        Ok(merged)
    }
}

pub struct GlmOcrProjector {
    #[allow(dead_code)] query_embed: Option<Tensor>,
    proj: Linear,
    norm: LayerNorm,
    #[allow(dead_code)] num_queries: usize,
}

impl GlmOcrProjector {
    pub fn new(
        vb: VarBuilder,
        vision_config: &GlmOcrVisionConfig,
        config: &GlmOcrProjectorConfig,
    ) -> Result<Self> {
        let query_embed = vb
            .get(
                (1, config.num_queries, vision_config.out_hidden_size),
                "query_embed",
            )
            .ok();

        let proj = linear_no_bias(
            vision_config.out_hidden_size,
            config.hidden_size,
            vb.pp("proj"),
        )?;
        let norm = layer_norm(config.hidden_size, 1e-5, vb.pp("norm"))?;

        Ok(Self {
            query_embed,
            proj,
            norm,
            num_queries: config.num_queries,
        })
    }

    pub fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        let projected = self.proj.forward(image_features)?;
        Ok(self.norm.forward(&projected)?)
    }
}

pub struct GlmOcrTextRotaryEmbedding {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
}

impl GlmOcrTextRotaryEmbedding {
    pub fn new(
        config: &GlmOcrTextConfig,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Self> {
        let rope_theta = config.rope_theta;
        let head_dim = config.head_dim.unwrap_or_else(|| {
            // Integer division, panics if num_attention_heads is 0 (like Python)
            config.hidden_size / config.num_attention_heads
        });
        let partial_rotary_factor = if config.partial_rotary_factor == 0.0 {
            1.0
        } else {
            config.partial_rotary_factor
        };
        let dim = (head_dim as f32 * partial_rotary_factor) as usize;

        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (rope_theta as f64).powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq =
            Tensor::from_slice(&inv_freq, (1, inv_freq.len()), device)?.to_dtype(dtype)?;

        let mrope_section = if config.mrope_section.is_empty() {
            vec![8, 12, 12]
        } else {
            config.mrope_section.clone()
        };

        Ok(Self {
            inv_freq,
            mrope_section,
        })
    }

    fn apply_mrope(&self, freqs: &Tensor) -> Result<Tensor> {
        // freqs: (3, bs, seq_len, head_dim/2)
        // Split by mrope_section and select from different axes
        let section = &self.mrope_section;
        let mut chunks = Vec::new();
        let mut offset = 0;
        for &s in section.iter() {
            let chunk = freqs.narrow(D::Minus1, offset, s)?;
            chunks.push(chunk);
            offset += s;
        }
        // Select chunk[i % 3] from axis 0
        let mut result_parts = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let selected = chunk.i(i % 3)?; // (bs, seq_len, section_size)
            result_parts.push(selected);
        }
        Ok(Tensor::cat(&result_parts, D::Minus1)?)
    }
    
     pub fn forward_with_position_ids(&self, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, bs, _seq_len) = position_ids.dims3()?;
        let inv_freq_len = self.inv_freq.dim(1)?; 

        // inv_freq: (1, inv_freq_len) -> broadcast to (3, bs, inv_freq_len, 1)
        let inv_freq = self.inv_freq.unsqueeze(0)?.unsqueeze(D::Minus1)?; // (1, 1, hd/2, 1)
        let inv_freq = inv_freq.broadcast_as((3, bs, inv_freq_len, 1))?;
        let inv_freq = inv_freq.to_dtype(DType::F32)?.contiguous()?;

        // position_ids: (3, bs, seq_len) -> (3, bs, 1, seq_len)
        let pos_expanded = position_ids
            .unsqueeze(D::Minus2)?
            .to_dtype(DType::F32)?
            .contiguous()?;

        // freqs = inv_freq @ pos_expanded -> (3, bs, hd/2, seq_len) -> T -> (3, bs, seq_len, hd/2)
        let freqs = inv_freq.matmul(&pos_expanded)?.transpose(2, 3)?;

        // Apply M-RoPE section selection
        let freqs = self.apply_mrope(&freqs)?; // (bs, seq_len, hd/2)

        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        Ok((
            emb.cos()?.to_dtype(self.inv_freq.dtype())?,
            emb.sin()?.to_dtype(self.inv_freq.dtype())?,
        ))
    }

    pub fn forward(
        &self,
        seq_len: usize,
        seqlen_offset: usize,
        device: &candle_core::Device,
    ) -> Result<(Tensor, Tensor)> {
        // For text-only (no image), all 3 axes have the same position IDs
        let positions = Tensor::arange(
            seqlen_offset as f32,
            (seqlen_offset + seq_len) as f32,
            device,
        )?
        .to_dtype(self.inv_freq.dtype())?;

        // position_ids: (3, 1, seq_len)
        let positions = positions.unsqueeze(0)?; // (1, seq_len)
        let positions_3d = positions.unsqueeze(0)?.expand((3, 1, seq_len))?; // (3, 1, seq_len)

        // inv_freq: (1, head_dim/2) -> (1, 1, head_dim/2, 1) -> (3, 1, head_dim/2, 1)
        let inv_freq = self.inv_freq.unsqueeze(0)?.unsqueeze(D::Minus1)?; // (1, 1, hd/2, 1)
        let inv_freq = inv_freq.broadcast_as((3, 1, self.inv_freq.dim(1)?, 1))?; // (3, 1, hd/2, 1)
        let inv_freq = inv_freq.to_dtype(DType::F32)?.contiguous()?;

        // position_ids: (3, 1, 1, seq_len)
        let positions_expanded = positions_3d
            .unsqueeze(D::Minus2)?
            .to_dtype(DType::F32)?
            .contiguous()?;

        // freqs = inv_freq @ positions -> (3, 1, hd/2, seq_len) -> transpose -> (3, 1, seq_len, hd/2)
        let freqs = inv_freq.matmul(&positions_expanded)?.transpose(2, 3)?;

        // Apply M-RoPE
        let freqs = self.apply_mrope(&freqs)?; // (1, seq_len, hd/2)

        // Double: emb = cat(freqs, freqs) -> (1, seq_len, head_dim)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;

        Ok((
            cos.to_dtype(self.inv_freq.dtype())?,
            sin.to_dtype(self.inv_freq.dtype())?,
        ))
    }
}

pub struct GlmOcrTextModel {
    embed_tokens: Embedding,
    layers: Vec<GlmOcrTextDecoderLayer>,
    norm: GlmOcrRMSNorm,
    lm_head: Linear,
    rotary_emb: GlmOcrTextRotaryEmbedding,
    config: GlmOcrTextConfig,
    spatial_merge_size: usize,
    /// max_mrope_position + 1 after prefill (stored for decode-pass position computation)
    next_mrope_pos: usize,
    /// Number of tokens in the prefill pass
    prefill_seq_len: usize,
}

impl GlmOcrTextModel {
    pub fn new(vb: VarBuilder, config: GlmOcrTextConfig, spatial_merge_size: usize) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let layer = GlmOcrTextDecoderLayer::new(vb.pp("layers").pp(i), &config, i)?;
            layers.push(layer);
        }

        let norm = GlmOcrRMSNorm::new(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        let root_vb = vb.root();
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, root_vb.pp("lm_head"))?;

        let rotary_emb = GlmOcrTextRotaryEmbedding::new(&config, vb.device(), vb.dtype())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary_emb,
            config,
            spatial_merge_size,
            next_mrope_pos: 0,
            prefill_seq_len: 0,
        })
    }

    fn compute_mrope_position_ids(
        &mut self,
        image_mask: &Tensor,
        grid_thw: &Tensor,
        seq_len: usize,
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        // Parse original grid dimensions (before spatial merge)
        let t_dim = grid_thw.i(0)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
        let h_dim = grid_thw.i(1)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;
        let w_dim = grid_thw.i(2)?.to_dtype(DType::F32)?.to_scalar::<f32>()? as usize;

        // Merged grid dimensions (what the LLM sees as image tokens)
        let llm_grid_t = t_dim;
        let llm_grid_h = h_dim / self.spatial_merge_size;
        let llm_grid_w = w_dim / self.spatial_merge_size;
        let num_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w;

        // Image mask as bool vec (shape (1, seq_len) -> (seq_len,))
        let mask_vec = image_mask.squeeze(0)?.to_dtype(DType::U8)?.to_vec1::<u8>()?;

        let mut t_ids: Vec<i64> = Vec::with_capacity(seq_len);
        let mut h_ids: Vec<i64> = Vec::with_capacity(seq_len);
        let mut w_ids: Vec<i64> = Vec::with_capacity(seq_len);

        let mut st_idx: i64 = 0; // start index for the next group
        let mut i = 0usize;

        while i < seq_len {
            let is_img = mask_vec[i] == 1;
            let start = i;
            while i < seq_len && (mask_vec[i] == 1) == is_img {
                i += 1;
            }
            let run_len = i - start;

            if is_img {
                // Assign 3D (t, h, w) positions for merged image grid
                assert_eq!(
                    run_len, num_image_tokens,
                    "image token count mismatch: mask={}, grid={}",
                    run_len, num_image_tokens
                );
                for ti in 0..llm_grid_t {
                    for hi in 0..llm_grid_h {
                        for wi in 0..llm_grid_w {
                            t_ids.push(ti as i64 + st_idx);
                            h_ids.push(hi as i64 + st_idx);
                            w_ids.push(wi as i64 + st_idx);
                        }
                    }
                }
                // Next group starts after max(t, h, w) + 1
                let max_offset = (llm_grid_t as i64 - 1)
                    .max(llm_grid_h as i64 - 1)
                    .max(llm_grid_w as i64 - 1);
                st_idx += max_offset + 1;
            } else {
                // Sequential positions for text tokens
                for j in 0..run_len {
                    let pos = st_idx + j as i64;
                    t_ids.push(pos);
                    h_ids.push(pos);
                    w_ids.push(pos);
                }
                st_idx += run_len as i64;
            }
        }

        // st_idx is now max_mrope_pos + 1; store for decode passes
        self.next_mrope_pos = st_idx as usize;
        self.prefill_seq_len = seq_len;

        let t_t = Tensor::from_vec(t_ids, (1, seq_len), device)?;
        let h_t = Tensor::from_vec(h_ids, (1, seq_len), device)?;
        let w_t = Tensor::from_vec(w_ids, (1, seq_len), device)?;
        Ok(Tensor::stack(&[&t_t, &h_t, &w_t], 0)?) // (3, 1, seq_len)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        image_features: Option<&Tensor>,
        image_mask: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len) = input_ids.dims2()?;
        let mut inputs_embeds = self.embed_tokens.forward(input_ids)?;

        // Merge image features into embeddings at image token positions
        if let (Some(img_feats), Some(img_mask)) = (image_features, image_mask) {
            // img_feats: (1, num_features, hidden_size)
            // img_mask: (1, seq_len) with 1s at image_token positions
            let img_mask_bool = img_mask.squeeze(0)?.to_dtype(DType::U8)?.to_vec1::<u8>()?;
            let _hidden_size = inputs_embeds.dim(2)?;

            // Collect image token indices
            let image_indices: Vec<usize> = img_mask_bool
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v == 1)
                .map(|(i, _)| i)
                .collect();

            let num_features = img_feats.dim(1)?;
            let num_to_replace = image_indices.len().min(num_features);

            // Replace embeddings at image positions with image features
            // Build the merged embeddings by copying
            let embeds_flat = inputs_embeds.squeeze(0)?; // (seq_len, hidden_size)
            let mut embeds_vec: Vec<Tensor> = Vec::new();

            let mut feat_idx = 0;
            let mut pos = 0;
            for &img_pos in image_indices.iter().take(num_to_replace) {
                if img_pos > pos {
                    embeds_vec.push(embeds_flat.narrow(0, pos, img_pos - pos)?);
                }
                embeds_vec.push(img_feats.i((0, feat_idx, ..))?.unsqueeze(0)?);
                feat_idx += 1;
                pos = img_pos + 1;
            }
            if pos < seq_len {
                embeds_vec.push(embeds_flat.narrow(0, pos, seq_len - pos)?);
            }

            let refs: Vec<&Tensor> = embeds_vec.iter().collect();
            inputs_embeds = Tensor::cat(&refs, 0)?.unsqueeze(0)?;
        }

        let attention_mask = if seq_len > 1 {
            Some(prepare_causal_attention_mask(
                bs,
                seq_len,
                seqlen_offset,
                input_ids.device(),
            )?)
        } else {
            None
        };

        let (cos, sin) = if seqlen_offset == 0 {
            if let (Some(mask), Some(thw)) = (image_mask, image_grid_thw) {
                // Prefill with image: compute 3D M-RoPE position IDs
                let pos_ids = self.compute_mrope_position_ids(mask, thw, seq_len, input_ids.device())?;
                self.prefill_seq_len = seq_len;
                self.rotary_emb.forward_with_position_ids(&pos_ids)?
            } else {
                // Pure text prefill: all three axes have sequential positions
                self.next_mrope_pos = seq_len;
                self.prefill_seq_len = seq_len;
                self.rotary_emb.forward(seq_len, 0, input_ids.device())?
            }
        } else {
            // Decode pass: single token, mrope position = next_mrope_pos + decode_step
            let decode_pos = self.next_mrope_pos + (seqlen_offset - self.prefill_seq_len);
            self.rotary_emb.forward(1, decode_pos, input_ids.device())?
        };

        let mut hidden_states = inputs_embeds;
        for layer in self.layers.iter_mut() {
            hidden_states = layer.forward(&hidden_states, (&cos, &sin), attention_mask.as_ref())?;
        }

        hidden_states = self.norm.forward(&hidden_states)?;
        let logits = self.lm_head.forward(&hidden_states)?;

        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

pub struct GlmOcrModel {
    vision_encoder: GlmOcrVisionModel,
    language_model: GlmOcrTextModel,
}

impl GlmOcrModel {
    pub fn new(vb: VarBuilder, config: GlmOcrConfig) -> Result<Self> {
        let vision_encoder =
            GlmOcrVisionModel::new(vb.pp("model").pp("visual"), &config.vision_config)?;
        let language_model = GlmOcrTextModel::new(
            vb.pp("model").pp("language_model"),
            config.text_config,
            config.vision_config.spatial_merge_size,
        )?;

        Ok(Self {
            vision_encoder,
            language_model,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        image_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let image_features = if let Some(pixels) = pixel_values {
            let grid_thw = if let Some(grid) = image_grid_thw {
                grid.clone()
            } else {
                Tensor::new(
                    &[
                        1u32,
                        (pixels.dim(0)? / 44) as u32,  // Approximate
                        (pixels.dim(1)? / 44) as u32,
                    ],
                    input_ids.device(),
                )?
            };

            let vision_output = self.vision_encoder.forward(pixels, &grid_thw)?;

            Some(vision_output)
        } else {
            None
        };

        let result = self.language_model.forward(
            input_ids,
            image_features.as_ref(),
            image_mask,
            image_grid_thw,
            seqlen_offset,
        );

        result
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}
