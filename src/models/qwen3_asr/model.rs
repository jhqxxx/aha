use anyhow::Result;
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{Conv1d, LayerNorm, Linear, Module, VarBuilder, linear, ops::softmax_last_dim};

use crate::{
    models::{
        common::{
            NaiveAttention, TwoLinearMLP, eager_attention_forward, get_conv1d, get_layer_norm,
        },
        fun_asr_nano::config::FunASRNanoConfig,
        qwen3::{config::Qwen3Config, model::Qwen3Model},
    },
    position_embed::sinusoidal_pe::SinusoidalPositionEncoderCat,
    utils::tensor_utils::{get_equal_mask, attn_masked_fill, masked_scatter_dim0},
};

pub struct MultiHeadedAttentionSANM {
    head_dim: usize,
    n_head: usize,
    linear_out: Linear,
    linear_q_k_v: Linear,
    fsmn_block: Conv1d,
    left_padding: usize,
    right_padding: usize,
    scaling: f64,
}

impl MultiHeadedAttentionSANM {
    pub fn new(
        vb: VarBuilder,
        n_head: usize,
        in_dim: usize,
        hidden_dim: usize,
        kernel_size: usize,
        sanm_shfit: usize,
    ) -> Result<Self> {
        let head_dim = hidden_dim / n_head;
        let linear_out = linear(hidden_dim, hidden_dim, vb.pp("linear_out"))?;
        let linear_q_k_v = linear(in_dim, hidden_dim * 3, vb.pp("linear_q_k_v"))?;
        let fsmn_block = get_conv1d(
            vb.pp("fsmn_block"),
            hidden_dim,
            hidden_dim,
            kernel_size,
            0,
            1,
            1,
            hidden_dim,
            false,
        )?;
        let mut left_padding = (kernel_size - 1) / 2;
        if sanm_shfit > 0 {
            left_padding += sanm_shfit;
        }
        let right_padding = kernel_size - 1 - left_padding;
        let scaling = (head_dim as f64).powf(-0.5);
        Ok(Self {
            head_dim,
            n_head,
            linear_out,
            linear_q_k_v,
            fsmn_block,
            left_padding,
            right_padding,
            scaling,
        })
    }

    pub fn forward_fsmn(
        &self,
        inputs: &Tensor,
        mask: Option<&Tensor>,
        mask_shfit_chunk: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut inputs = inputs.clone();
        let mask = if let Some(mask) = mask {
            let mut mask = mask.unsqueeze(D::Minus1)?.unsqueeze(0)?;
            if let Some(mask_shfit_chunk) = mask_shfit_chunk {
                mask = mask.broadcast_mul(mask_shfit_chunk)?;
            }
            inputs = inputs.broadcast_mul(&mask)?;
            Some(mask)
        } else {
            None
        };
        let xs = inputs.transpose(1, 2)?;
        let xs = xs.pad_with_zeros(D::Minus1, self.left_padding, self.right_padding)?;
        let xs = self.fsmn_block.forward(&xs)?;
        let xs = xs.transpose(1, 2)?;
        let mut xs = xs.add(&inputs)?;
        if let Some(mask) = mask {
            xs = xs.broadcast_mul(&mask)?;
        }
        Ok(xs)
    }
    pub fn forward_qkv(&self, xs: &Tensor) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let (b, t, _) = xs.dims3()?;
        let q_k_v = self
            .linear_q_k_v
            .forward(xs)?
            .reshape((b, t, 3, self.n_head, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let q_h = q_k_v.i(0)?.contiguous()?;
        let k_h = q_k_v.i(1)?.contiguous()?;
        let v_h = q_k_v.i(2)?.contiguous()?;
        let v = v_h.transpose(1, 2)?.reshape((b, t, ()))?;
        Ok((q_h, k_h, v_h, v))
    }

    pub fn forward_attention(
        &self,
        values: &Tensor,
        scores: &Tensor,
        mask: Option<&Tensor>,
        mask_att_chunk_encoder: Option<&Tensor>,
    ) -> Result<Tensor> {
        let bs = scores.dim(0)?;
        let attn = if let Some(mask) = mask {
            let mask = if let Some(mask_att_chunk_encoder) = mask_att_chunk_encoder {
                mask.mul(mask_att_chunk_encoder)?
            } else {
                mask.clone()
            };
            // mask: rank = 2
            let mask = get_equal_mask(&mask, 0)?;
            let scores = attn_masked_fill(scores, &mask, f32::NEG_INFINITY)?;
            let attn = softmax_last_dim(&scores)?;
            attn_masked_fill(&attn, &mask, 0.0)?
        } else {
            softmax_last_dim(scores)?
        };
        let xs = attn.matmul(values)?;
        let xs =
            xs.transpose(1, 2)?
                .contiguous()?
                .reshape((bs, (), self.n_head * self.head_dim))?;
        let xs = self.linear_out.forward(&xs)?;
        Ok(xs)
    }

    pub fn forward_simple(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let q_k_v = self.linear_q_k_v.forward(xs)?;
        let dim = self.head_dim * self.n_head;
        let q_h = q_k_v
            .narrow(D::Minus1, 0, dim)?
            .reshape((b, t, self.n_head, ()))?
            .permute((0, 2, 1, 3))?;
        let k_h = q_k_v
            .narrow(D::Minus1, dim, dim)?
            .reshape((b, t, self.n_head, ()))?
            .permute((0, 2, 1, 3))?;
        let v = q_k_v.narrow(D::Minus1, dim * 2, dim)?;
        let v_h = v.reshape((b, t, self.n_head, ()))?.permute((0, 2, 1, 3))?;
        let fsmn_memory = v.transpose(1, 2)?;
        let fsmn_memory = fsmn_memory
            .pad_with_zeros(D::Minus1, self.left_padding, self.right_padding)?
            .contiguous()?;
        let fsmn_memory = self.fsmn_block.forward(&fsmn_memory)?;
        // let fsmn_memory = conv1d_group_parallel(&fsmn_memory, &self.fsmn_block)?;

        let fsmn_memory = fsmn_memory.transpose(1, 2)?;
        let fsmn_memory = fsmn_memory.add(&v)?;
        let att_outs = eager_attention_forward(&q_h, &k_h, &v_h, None, None, self.scaling)?;
        let att_outs = att_outs.reshape((b, t, ()))?;
        let att_outs = self.linear_out.forward(&att_outs)?;
        let att_outs = att_outs.add(&fsmn_memory)?;
        Ok(att_outs)
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        mask_shfit_chunk: Option<&Tensor>,
        mask_att_chunk_encoder: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (q_h, k_h, v_h, v) = self.forward_qkv(xs)?;
        let fsmn_memory = self.forward_fsmn(&v, mask, mask_shfit_chunk)?;
        let q_h = q_h.affine(self.scaling, 0.0)?;
        let scores = q_h.matmul(&k_h.transpose(D::Minus2, D::Minus1)?)?;
        let attn_outs = self.forward_attention(&v_h, &scores, mask, mask_att_chunk_encoder)?;
        let att_outs = attn_outs.add(&fsmn_memory)?;
        Ok(att_outs)
    }
}

pub struct EncoderLayerSANM {
    self_attn: MultiHeadedAttentionSANM,
    feed_forward: TwoLinearMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
    concat_linear: Option<Linear>,
    normalize_before: bool,
    in_dim: usize,
    hidden_dim: usize,
}

impl EncoderLayerSANM {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        hidden_dim: usize,
        n_head: usize,
        kernel_size: usize,
        sanm_shfit: usize,
        hidden_units: usize,
        normalize_before: bool,
        concat_after: bool,
    ) -> Result<Self> {
        let self_attn = MultiHeadedAttentionSANM::new(
            vb.pp("self_attn"),
            n_head,
            in_dim,
            hidden_dim,
            kernel_size,
            sanm_shfit,
        )?;
        let feed_forward = TwoLinearMLP::new(
            vb.pp("feed_forward"),
            hidden_dim,
            hidden_units,
            hidden_dim,
            candle_nn::Activation::Relu,
            true,
            "w_1",
            "w_2",
        )?;
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-5, in_dim)?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-5, hidden_dim)?;
        let concat_linear = if concat_after {
            let lin = linear(hidden_dim * 2, hidden_dim, vb.pp("concat_linear"))?;
            Some(lin)
        } else {
            None
        };
        Ok(Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
            concat_linear,
            normalize_before,
            in_dim,
            hidden_dim,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        mask_shfit_chunk: Option<&Tensor>,
        mask_att_chunk_encoder: Option<&Tensor>,
    ) -> Result<Tensor> {
        let stoch_layer_coeff = 1.0f64;
        let residual = xs.clone();
        let mut xs = if self.normalize_before {
            self.norm1.forward(xs)?
        } else {
            xs.clone()
        };
        if self.concat_linear.is_some() {
            let attn =
                self.self_attn
                    .forward(&xs, mask, mask_shfit_chunk, mask_att_chunk_encoder)?;
            let x_concat = Tensor::cat(&[&xs, &attn], D::Minus1)?;
            if self.in_dim == self.hidden_dim {
                let x_concat = self
                    .concat_linear
                    .as_ref()
                    .unwrap()
                    .forward(&x_concat)?
                    .affine(stoch_layer_coeff, 0.0)?;
                xs = residual.add(&x_concat)?;
            } else {
                xs = self
                    .concat_linear
                    .as_ref()
                    .unwrap()
                    .forward(&x_concat)?
                    .affine(stoch_layer_coeff, 0.0)?;
            }
        } else if self.in_dim == self.hidden_dim {
            let attn = self
                .self_attn
                .forward(&xs, mask, mask_shfit_chunk, mask_att_chunk_encoder)?
                .affine(stoch_layer_coeff, 0.0)?;
            xs = residual.add(&attn)?;
        } else {
            xs = self
                .self_attn
                .forward(&xs, mask, mask_shfit_chunk, mask_att_chunk_encoder)?
                .affine(stoch_layer_coeff, 0.0)?;
        }

        if !self.normalize_before {
            xs = self.norm1.forward(&xs)?;
        }
        let residual = xs.clone();
        if self.normalize_before {
            xs = self.norm2.forward(&xs)?;
        }
        xs = self
            .feed_forward
            .forward(&xs)?
            .affine(stoch_layer_coeff, 0.0)?;
        xs = residual.add(&xs)?;
        if !self.normalize_before {
            xs = self.norm2.forward(&xs)?;
        }
        Ok(xs)
    }

    pub fn forward_simple(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = self.norm1.forward(xs)?;
        if self.in_dim == self.hidden_dim {
            let attn = self.self_attn.forward_simple(&xs)?;
            xs = residual.add(&attn)?;
        } else {
            xs = self.self_attn.forward_simple(&xs)?;
        }

        let residual = xs.clone();
        let xs = self.norm2.forward(&xs)?;

        let xs = self.feed_forward.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
}

pub struct SenseVoiceEncoderSmall {
    embed: SinusoidalPositionEncoderCat,
    encoders0: EncoderLayerSANM,
    encoders: Vec<EncoderLayerSANM>,
    tp_encoders: Vec<EncoderLayerSANM>,
    after_norm: LayerNorm,
    tp_norm: LayerNorm,
    scaling: f64,
}

impl SenseVoiceEncoderSmall {
    pub fn new(
        vb: VarBuilder,
        input_size: usize,
        output_size: usize,
        attention_heads: usize,
        linear_units: usize,
        num_blocks: usize,
        tp_blocks: usize,
        normalize_before: bool,
        kernel_size: usize,
        sanm_shfit: usize,
    ) -> Result<Self> {
        let embed = SinusoidalPositionEncoderCat::new(Some(input_size), true, vb.device())?;

        let encoders0 = EncoderLayerSANM::new(
            vb.pp("encoders0.0"),
            input_size,
            output_size,
            attention_heads,
            kernel_size,
            sanm_shfit,
            linear_units,
            normalize_before,
            false,
        )?;
        let mut encoders = vec![];
        let vb_encoders = vb.pp("encoders");
        for i in 0..(num_blocks - 1) {
            let encoder_i = EncoderLayerSANM::new(
                vb_encoders.pp(i),
                output_size,
                output_size,
                attention_heads,
                kernel_size,
                sanm_shfit,
                linear_units,
                normalize_before,
                false,
            )?;
            encoders.push(encoder_i);
        }
        let vb_tp_encoders = vb.pp("tp_encoders");
        let mut tp_encoders = vec![];
        for i in 0..tp_blocks {
            let tp_blocks_i = EncoderLayerSANM::new(
                vb_tp_encoders.pp(i),
                output_size,
                output_size,
                attention_heads,
                kernel_size,
                sanm_shfit,
                linear_units,
                normalize_before,
                false,
            )?;
            tp_encoders.push(tp_blocks_i);
        }
        let after_norm = get_layer_norm(vb.pp("after_norm"), 1e-5, output_size)?;
        let tp_norm = get_layer_norm(vb.pp("tp_norm"), 1e-5, output_size)?;
        let scaling = (output_size as f64).powf(0.5);
        Ok(Self {
            embed,
            encoders0,
            encoders,
            tp_encoders,
            after_norm,
            tp_norm,
            scaling,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.affine(self.scaling, 0.0)?;
        let xs = self.embed.forward(&xs, 0)?;
        let mut xs = self.encoders0.forward_simple(&xs)?;
        for encoder_layer in &self.encoders {
            xs = encoder_layer.forward_simple(&xs)?;
        }
        xs = self.after_norm.forward(&xs)?;
        for tp_layer in &self.tp_encoders {
            xs = tp_layer.forward_simple(&xs)?;
        }
        xs = self.tp_norm.forward(&xs)?;
        Ok(xs)
    }
}

pub struct AdaptorEncoderLayer {
    self_attn: NaiveAttention,
    feed_forward: TwoLinearMLP,
    norm1: LayerNorm,
    norm2: LayerNorm,
    concat_linear: Option<Linear>,
    normalize_before: bool,
}

impl AdaptorEncoderLayer {
    pub fn new(
        vb: VarBuilder,
        llm_dim: usize,
        n_head: usize,
        normalize_before: bool,
        concat_after: bool,
    ) -> Result<Self> {
        let self_attn = NaiveAttention::new(
            vb.pp("self_attn"),
            llm_dim,
            n_head,
            n_head,
            None,
            true,
            Some("linear_q"),
            Some("linear_k"),
            Some("linear_v"),
            Some("linear_out"),
        )?;
        let feed_forward = TwoLinearMLP::new(
            vb.pp("feed_forward"),
            llm_dim,
            llm_dim / 4,
            llm_dim,
            candle_nn::Activation::Relu,
            true,
            "w_1",
            "w_2",
        )?;
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-5, llm_dim)?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-5, llm_dim)?;
        let concat_linear = if concat_after {
            let lin = linear(llm_dim * 2, llm_dim, vb.pp("concat_linear"))?;
            Some(lin)
        } else {
            None
        };
        Ok(Self {
            self_attn,
            feed_forward,
            norm1,
            norm2,
            concat_linear,
            normalize_before,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let stoch_layer_coeff = 1.0f64;
        let residual = xs.clone();
        let mut xs = if self.normalize_before {
            self.norm1.forward(xs)?
        } else {
            xs.clone()
        };
        if self.concat_linear.is_some() {
            let attn = self.self_attn.forward(&xs, None, None, mask, false)?;
            let x_concat = Tensor::cat(&[&xs, &attn], D::Minus1)?;
            let x_concat = self
                .concat_linear
                .as_ref()
                .unwrap()
                .forward(&x_concat)?
                .affine(stoch_layer_coeff, 0.0)?;
            xs = residual.add(&x_concat)?;
        } else {
            let attn = self
                .self_attn
                .forward(&xs, None, None, mask, false)?
                .affine(stoch_layer_coeff, 0.0)?;
            xs = residual.add(&attn)?;
        }
        if !self.normalize_before {
            xs = self.norm1.forward(&xs)?;
        }
        let residual = xs.clone();
        if self.normalize_before {
            xs = self.norm2.forward(&xs)?;
        }
        xs = self
            .feed_forward
            .forward(&xs)?
            .affine(stoch_layer_coeff, 0.0)?;
        xs = residual.add(&xs)?;
        if !self.normalize_before {
            xs = self.norm2.forward(&xs)?;
        }
        Ok(xs)
    }
}

pub struct AudioAdaptor {
    k: usize,
    linear1: Linear,
    linear2: Linear,
    blocks: Vec<AdaptorEncoderLayer>,
}

impl AudioAdaptor {
    pub fn new(
        vb: VarBuilder,
        downsample_rate: usize,
        encoder_dim: usize,
        llm_dim: usize,
        ffn_dim: usize,
        n_layer: usize,
        attention_heads: usize,
    ) -> Result<Self> {
        let linear1 = linear(encoder_dim * downsample_rate, ffn_dim, vb.pp("linear1"))?;
        let linear2 = linear(ffn_dim, llm_dim, vb.pp("linear2"))?;
        let mut blocks = vec![];
        let vb_blocks = vb.pp("blocks");
        for i in 0..n_layer {
            let layer =
                AdaptorEncoderLayer::new(vb_blocks.pp(i), llm_dim, attention_heads, true, false)?;
            blocks.push(layer);
        }
        Ok(Self {
            k: downsample_rate,
            linear1,
            linear2,
            blocks,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, dim) = xs.dims3()?;
        let chunk_num = (seq_len - 1) / self.k + 1;
        let pad_num = chunk_num * self.k - seq_len;
        let xs = xs.pad_with_zeros(1, 0, pad_num)?;
        let xs = xs.contiguous()?.reshape((bs, chunk_num, dim * self.k))?;
        let xs = self.linear1.forward(&xs)?.relu()?;
        let mut xs = self.linear2.forward(&xs)?;
        for block in &self.blocks {
            xs = block.forward(&xs, None)?;
        }
        Ok(xs)
    }
}

pub struct FunAsrNanoModel {
    audio_encoder: SenseVoiceEncoderSmall,
    audio_adaptor: AudioAdaptor,
    llm: Qwen3Model,
}
impl FunAsrNanoModel {
    pub fn new(vb: VarBuilder, config: &FunASRNanoConfig, llm_cfg: &Qwen3Config) -> Result<Self> {
        let input_size = config.frontend_conf.lfr_m * config.frontend_conf.n_mels;
        let audio_encoder = SenseVoiceEncoderSmall::new(
            vb.pp("audio_encoder"),
            input_size,
            config.audio_encoder_conf.output_size,
            config.audio_encoder_conf.attention_heads,
            config.audio_encoder_conf.linear_units,
            config.audio_encoder_conf.num_blocks,
            config.audio_encoder_conf.tp_blocks,
            config.audio_encoder_conf.normalize_before,
            config.audio_encoder_conf.kernel_size,
            config.audio_encoder_conf.sanm_shfit,
        )?;
        let audio_adaptor = AudioAdaptor::new(
            vb.pp("audio_adaptor"),
            config.audio_adaptor_conf.downsample_rate,
            config.audio_adaptor_conf.encoder_dim,
            config.audio_adaptor_conf.llm_dim,
            config.audio_adaptor_conf.ffn_dim,
            config.audio_adaptor_conf.n_layer,
            8,
        )?;
        let llm = Qwen3Model::new(llm_cfg, vb.pp("llm"))?;
        Ok(Self {
            audio_encoder,
            audio_adaptor,
            llm,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        speech: Option<&Tensor>,
        fbank_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.llm.embedding_token_id(input_ids)?;
        if let Some(speech) = speech
            && let Some(fbank_mask) = fbank_mask
        {
            let speech = self.audio_encoder.forward(speech)?;
            let encoder_out = self.audio_adaptor.forward(&speech)?;
            let speech_token_len = fbank_mask.sum_all()?.to_scalar::<u32>()?;
            let audio_embed = encoder_out
                .squeeze(0)?
                .narrow(0, 0, speech_token_len as usize)?;
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &audio_embed, fbank_mask)?;
        }
        let logits = self
            .llm
            .forward(None, Some(&inputs_embeds), seqlen_offset)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.llm.clear_kv_cache();
    }
}
