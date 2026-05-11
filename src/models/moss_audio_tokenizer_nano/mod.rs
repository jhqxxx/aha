pub mod config;
use anyhow::{Result, anyhow};
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, linear_no_bias};

use crate::{
    models::{
        common::modules::{
            TwoLinearMLP, WNConv1d, eager_attention_forward, get_layer_norm, l2_normalize,
        },
        moss_audio_tokenizer_nano::config::{
            MossAudioTokenizerConfig, MossAudioTokenizerModuleConfig,
            MossAudioTokenizerQuantizerKwargs,
        },
    },
    position_embed::rope::{RoPE, apply_rotary_pos_emb_roformer},
};

pub struct MossAudioTokenizerPatchedPretransform {
    patch_size: usize,
    is_downsample: bool,
}

impl MossAudioTokenizerPatchedPretransform {
    pub fn new(patch_size: usize, is_downsample: bool) -> Self {
        Self {
            patch_size,
            is_downsample,
        }
    }

    pub fn encode(&self, x: &Tensor, input_lengths: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, d, _) = x.dims3()?;
        let x = x
            .reshape((b, d, (), self.patch_size))?
            .permute((0, 1, 3, 2))?
            .reshape((b, d * self.patch_size, ()))?;
        let out_lengths = input_lengths
            .affine(1.0 / self.patch_size as f64, 0.0)?
            .floor()?;
        Ok((x, out_lengths))
    }

    pub fn decode(&self, x: &Tensor, input_lengths: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, dh, l) = x.dims3()?;
        let d = dh / self.patch_size;
        let x = x
            .reshape((b, d, self.patch_size, l))?
            .permute((0, 1, 3, 2))?
            .reshape((b, d, l * self.patch_size))?;
        // let out_lengths = (input_lengths * self.patch_size as f64)?;
        let out_lengths = input_lengths.affine(self.patch_size as f64, 0.0)?;
        Ok((x, out_lengths))
    }

    pub fn forward(&self, x: &Tensor, input_lengths: &Tensor) -> Result<(Tensor, Tensor)> {
        if self.is_downsample {
            self.encode(x, input_lengths)
        } else {
            self.decode(x, input_lengths)
        }
    }
}

pub struct MossAudioTokenizerMultiheadAttention {
    num_heads: usize,
    scale: f64,
    in_proj: Linear,
    out_proj: Linear,
}

impl MossAudioTokenizerMultiheadAttention {
    pub fn new(vb: VarBuilder, embed_dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let scale = 1f64 / f64::sqrt(head_dim as f64);
        let in_proj = linear_no_bias(embed_dim, 3 * embed_dim, vb.pp("in_proj"))?;
        let out_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("out_proj"))?;
        Ok(Self {
            num_heads,
            scale,
            in_proj,
            out_proj,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        input_lengths: &Tensor,
    ) -> Result<Tensor> {
        let (bs, max_seqlen, _) = xs.dims3()?;
        let projected = self
            .in_proj
            .forward(xs)?
            .reshape((bs, max_seqlen, 3, self.num_heads, ()))?
            .permute((2, 0, 3, 1, 4))?;
        let [q, k, v] = projected
            .chunk(3, 0)?
            .try_into()
            .map_err(|_| anyhow!("Chunk size mismatch"))?;
        let q = q.squeeze(0)?.contiguous()?;
        let k = k.squeeze(0)?.contiguous()?;
        let v = v.squeeze(0)?.contiguous()?;
        let (q, k) = apply_rotary_pos_emb_roformer(&q, &k, cos, sin)?;
        // let (q, k) = self.apply_rope(&q, &k, cos, sin)?;
        let attn = eager_attention_forward(&q, &k, &v, None, Some(mask), self.scale)?;
        // (b, seq_len, n_head, dim) -> (b, n_head, seq_len, dim)
        let attn = attn.transpose(1, 2)?;
        let valid_q = Tensor::arange(0f32, max_seqlen as f32, xs.device())?
            .reshape((1, 1, max_seqlen, 1))?
            .broadcast_lt(
                &input_lengths
                    .reshape((bs, 1, 1, 1))?
                    .repeat((1, 1, max_seqlen, 1))?,
            )?
            .broadcast_as(attn.shape())?;
        let on_false = attn.zeros_like()?;
        let attn = valid_q.where_cond(&attn, &on_false)?;
        // (b, n_head, seq_len, dim) -> (b, seq_len, n_head, dim)
        let attn = attn.transpose(1, 2)?;
        let attn = attn.reshape((bs, max_seqlen, ()))?;
        let out = self.out_proj.forward(&attn)?;
        Ok(out)
    }
}

pub struct MossAudioTokenizerTransformerLayer {
    self_attn: MossAudioTokenizerMultiheadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    ffn: TwoLinearMLP,
    layer_scale_1: Tensor,
    layer_scale_2: Tensor,
}

impl MossAudioTokenizerTransformerLayer {
    pub fn new(vb: VarBuilder, config: &MossAudioTokenizerModuleConfig) -> Result<Self> {
        let self_attn = MossAudioTokenizerMultiheadAttention::new(
            vb.pp("self_attn"),
            config.d_model.unwrap(),
            config.num_heads.unwrap(),
        )?;
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-5, config.d_model.unwrap(), true)?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-5, config.d_model.unwrap(), true)?;
        let ffn = TwoLinearMLP::new(
            vb.pp("ffn"),
            config.d_model.unwrap(),
            config.dim_feedforward.unwrap(),
            config.d_model.unwrap(),
            candle_nn::Activation::Gelu,
            false,
            "0",
            "2",
        )?;
        let layer_scale_1 = vb
            .get(config.d_model.unwrap(), "layer_scale_1.scale")?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let layer_scale_2 = vb
            .get(config.d_model.unwrap(), "layer_scale_2.scale")?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        Ok(Self {
            self_attn,
            norm1,
            norm2,
            ffn,
            layer_scale_1,
            layer_scale_2,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
        input_lengths: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, mask, input_lengths)?;
        let xs = self.layer_scale_1.broadcast_mul(&xs)?;
        let residual = residual.add(&xs)?;
        let xs = self.norm2.forward(&residual)?;
        let xs = self.ffn.forward(&xs)?;
        let xs = self.layer_scale_2.broadcast_mul(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
}

pub struct MossAudioTokenizerTransformer {
    rope: RoPE, // use roformer
    context: usize,
    layers: Vec<MossAudioTokenizerTransformerLayer>,
}

impl MossAudioTokenizerTransformer {
    pub fn new(
        vb: VarBuilder,
        config: &MossAudioTokenizerModuleConfig,
        context: usize,
    ) -> Result<Self> {
        let dim = config.d_model.unwrap() / config.num_heads.unwrap();
        let rope = RoPE::new(dim, 10000.0, vb.device())?;
        let vb_layers = vb.pp("layers");
        let mut layers = vec![];
        for i in 0..config.num_layers.unwrap() {
            let layer = MossAudioTokenizerTransformerLayer::new(vb_layers.pp(i), config)?;
            layers.push(layer);
        }
        Ok(Self {
            rope,
            context,
            layers,
        })
    }

    pub fn forward(&self, input_embeds: &Tensor, input_lengths: &Tensor) -> Result<Tensor> {
        let t = input_embeds.dim(1)?;
        let (cos, sin) = self.rope.forward(0, t, input_embeds.device())?;
        let mut xs = input_embeds.clone();
        let mask = self.build_attn_bias(input_lengths, t)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, &cos, &sin, &mask, input_lengths)?;
        }
        Ok(xs)
    }

    fn build_attn_bias(&self, input_lengths: &Tensor, max_seqlen: usize) -> Result<Tensor> {
        let positions = Tensor::arange(0f32, max_seqlen as f32, input_lengths.device())?;
        let input_lengths = input_lengths.reshape(((), 1, 1))?;
        let valid_k = positions
            .reshape((1, 1, max_seqlen))?
            .broadcast_lt(&input_lengths)?;
        let delta = positions
            .reshape((1, max_seqlen, 1))?
            .broadcast_sub(&positions.reshape((1, 1, max_seqlen))?)?;
        let delta1 = delta.ge(&delta.zeros_like()?)?;
        let delta2 = delta
            .lt(&Tensor::new(self.context as f32, delta.device())?.broadcast_as(delta.shape())?)?;
        let mask = delta1.minimum(&delta2)?;
        let mask = mask.broadcast_minimum(&valid_k)?.unsqueeze(1)?;
        let on_true = mask.zeros_like()?.to_dtype(candle_core::DType::F32)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, mask.device())?.broadcast_as(mask.shape())?;
        let mask = mask.where_cond(&on_true, &on_false)?;
        Ok(mask)
    }
}

pub struct MossAudioTokenizerProjectedTransformer {
    input_proj: Linear,
    transformer: MossAudioTokenizerTransformer,
    output_proj: Linear,
}

impl MossAudioTokenizerProjectedTransformer {
    pub fn new(
        vb: VarBuilder,
        config: &MossAudioTokenizerModuleConfig,
        context: usize,
    ) -> Result<Self> {
        let input_proj = linear_no_bias(
            config.input_dimension.unwrap(),
            config.d_model.unwrap(),
            vb.pp("input_proj"),
        )?;
        let transformer =
            MossAudioTokenizerTransformer::new(vb.pp("transformer"), config, context)?;
        let output_proj = linear_no_bias(
            config.d_model.unwrap(),
            config.output_dimension.unwrap(),
            vb.pp("output_proj"),
        )?;
        Ok(Self {
            input_proj,
            transformer,
            output_proj,
        })
    }

    pub fn forward(
        &self,
        input_embeds: &Tensor,
        input_lengths: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let xs = self.input_proj.forward(&input_embeds.transpose(1, 2)?)?;
        let xs = self.transformer.forward(&xs, input_lengths)?;
        let xs = self.output_proj.forward(&xs)?.transpose(1, 2)?;
        Ok((xs, input_lengths.clone()))
    }
}

pub enum MossAudioTokenizerModule {
    PatchedPretransform(MossAudioTokenizerPatchedPretransform),
    ProjectedTransformer(MossAudioTokenizerProjectedTransformer),
}

impl MossAudioTokenizerModule {
    pub fn forward(
        &self,
        input_embeds: &Tensor,
        input_lengths: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        match self {
            MossAudioTokenizerModule::PatchedPretransform(patch) => {
                patch.forward(input_embeds, input_lengths)
            }
            MossAudioTokenizerModule::ProjectedTransformer(transformer) => {
                transformer.forward(input_embeds, input_lengths)
            }
        }
    }
}

pub struct MossAudioTokenizerLFQ {
    in_proj: Option<WNConv1d>,
    out_proj: Option<WNConv1d>,
    codebook: Embedding,
    codebook_l2_norm: Tensor,
}

impl MossAudioTokenizerLFQ {
    pub fn new(vb: VarBuilder, config: &MossAudioTokenizerQuantizerKwargs) -> Result<Self> {
        let in_proj = if config.rvq_dim != config.codebook_dim {
            Some(WNConv1d::new(
                vb.pp("in_proj"),
                config.rvq_dim,
                config.codebook_dim,
                1,
                1,
                0,
                1,
                1,
                true,
                Some("parametrizations.weight.original0"),
                Some("parametrizations.weight.original1"),
            )?)
        } else {
            None
        };

        let out_proj = if config.rvq_dim != config.codebook_dim {
            Some(WNConv1d::new(
                vb.pp("out_proj"),
                config.codebook_dim,
                config.rvq_dim,
                1,
                1,
                0,
                1,
                1,
                true,
                Some("parametrizations.weight.original0"),
                Some("parametrizations.weight.original1"),
            )?)
        } else {
            None
        };

        let codebook = embedding(config.codebook_size, config.codebook_dim, vb.pp("codebook"))?;
        let codebook_l2_norm = l2_normalize(codebook.embeddings(), 1)?;
        Ok(Self {
            in_proj,
            out_proj,
            codebook,
            codebook_l2_norm,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let z_e = if let Some(in_proj) = &self.in_proj {
            in_proj.forward(xs)?
        } else {
            xs.clone()
        };
        let (bs, len, _) = z_e.dims3()?;
        let encodings = z_e.transpose(1, 2)?.reshape(((), len))?;
        let encodings = l2_normalize(&encodings, 1)?;
        let dist1 = encodings.powf(2.0)?.sum_keepdim(1)?;
        let dist2 = encodings
            .affine(2.0, 0.0)?
            .matmul(&self.codebook_l2_norm.t()?)?;
        let dist3 = self.codebook_l2_norm.powf(2.0)?.sum_keepdim(1)?.t()?;
        let dist = dist1.broadcast_sub(&dist2)?.broadcast_add(&dist3)?;
        let indices = dist
            .affine(-1.0, 0.0)?
            .argmax(1)?
            .reshape((bs, ()))?
            .to_dtype(candle_core::DType::U32)?;
        let z_q = self.codebook.forward(&indices)?.transpose(1, 2)?;
        let mut z_q = z_e.add(&z_q.sub(&z_e)?)?;
        if let Some(out_proj) = &self.out_proj {
            z_q = out_proj.forward(&z_q)?;
        }
        Ok((z_q, indices))
    }

    pub fn decode_code(&self, codec: &Tensor) -> Result<Tensor> {
        let mut z_q = self.codebook.forward(codec)?.transpose(1, 2)?;
        if let Some(out_proj) = &self.out_proj {
            z_q = out_proj.forward(&z_q)?;
        }
        Ok(z_q)
    }
}

pub struct MossAudioTokenizerResidualLFQ {
    input_proj: Option<WNConv1d>,
    output_proj: Option<WNConv1d>,
    quantizers: Vec<MossAudioTokenizerLFQ>,
    rvq_dim: usize,
}

impl MossAudioTokenizerResidualLFQ {
    pub fn new(vb: VarBuilder, config: &MossAudioTokenizerQuantizerKwargs) -> Result<Self> {
        let input_proj = if config.input_dim != config.rvq_dim {
            Some(WNConv1d::new(
                vb.pp("input_proj"),
                config.input_dim,
                config.rvq_dim,
                1,
                1,
                0,
                1,
                1,
                true,
                Some("parametrizations.weight.original0"),
                Some("parametrizations.weight.original1"),
            )?)
        } else {
            None
        };

        let output_proj = if config.rvq_dim != config.output_dim {
            Some(WNConv1d::new(
                vb.pp("output_proj"),
                config.rvq_dim,
                config.output_dim,
                1,
                1,
                0,
                1,
                1,
                true,
                Some("parametrizations.weight.original0"),
                Some("parametrizations.weight.original1"),
            )?)
        } else {
            None
        };

        let vb_quantizers = vb.pp("quantizers");
        let mut quantizers = vec![];
        for i in 0..config.num_quantizers {
            let layer = MossAudioTokenizerLFQ::new(vb_quantizers.pp(i), config)?;
            quantizers.push(layer);
        }
        Ok(Self {
            input_proj,
            output_proj,
            quantizers,
            rvq_dim: config.rvq_dim,
        })
    }

    pub fn forward(&self, input_values: &Tensor, length: &Tensor) -> Result<Tensor> {
        let z = if let Some(proj) = &self.input_proj {
            proj.forward(input_values)?
        } else {
            input_values.clone()
        };
        let max_time = z.dim(2)?;
        let mask = Tensor::arange(0f32, max_time as f32, z.device())?
            .unsqueeze(0)?
            .broadcast_lt(&length.unsqueeze(1)?)?
            .unsqueeze(1)?;
        // let mut quantized_out = z.zeros_like()?;
        let mut residual = z.clone();
        let on_false = residual.zeros_like()?;
        let mask_reshape = mask.broadcast_as(residual.shape())?;
        let mut all_indices = vec![];
        for quantizer in self.quantizers.iter() {
            let masked_residual = mask_reshape.where_cond(&residual, &on_false)?;
            let (z_q_i, indices_i) = quantizer.forward(&masked_residual)?;
            all_indices.push(indices_i);
            let z_q_i_mask = mask_reshape.where_cond(&z_q_i, &on_false)?;
            residual = residual.sub(&z_q_i_mask)?;
        }
        let all_indices = Tensor::stack(&all_indices, 0)?;
        Ok(all_indices)
    }

    pub fn decode_codes(&self, codes: &Tensor) -> Result<Tensor> {
        let (_, bs, t) = codes.dims3()?;
        let mut emb = Tensor::zeros(
            (bs, self.rvq_dim, t),
            candle_core::DType::F32,
            codes.device(),
        )?;
        for (i, quantizer) in self.quantizers.iter().enumerate() {
            let code_i = quantizer.decode_code(&codes.i(i)?)?;
            emb = emb.add(&code_i)?;
        }
        if let Some(output_proj) = &self.output_proj {
            emb = output_proj.forward(&emb)?;
        }
        Ok(emb)
    }
}

pub struct MossAudioTokenizer {
    pub sampling_rate: usize,
    pub downsample_rate: usize,
    pub number_channels: usize,
    pub enable_channel_interleave: bool,
    encoder: Vec<MossAudioTokenizerModule>,
    quantizer: MossAudioTokenizerResidualLFQ,
    decoder: Vec<MossAudioTokenizerModule>,
}

impl MossAudioTokenizer {
    pub fn new(vb: VarBuilder, config: &MossAudioTokenizerConfig) -> Result<Self> {
        let channel_interleave_factor =
            if config.enable_channel_interleave && config.number_channels > 1 {
                config.number_channels
            } else {
                1
            };
        let current_frame_rate = config.sampling_rate * channel_interleave_factor;
        let vb_encoder = vb.pp("encoder");
        let mut encoder = vec![];
        for (layer_id, cfg) in config.encoder_kwargs.iter().enumerate() {
            if cfg.module_type == "PatchedPretransform"
                && let Some(patch_size) = cfg.patch_size
            {
                let layer = MossAudioTokenizerPatchedPretransform::new(patch_size, true);
                encoder.push(MossAudioTokenizerModule::PatchedPretransform(layer));
            } else if cfg.module_type == "Transformer" {
                let context_duration = cfg
                    .context_duration
                    .unwrap_or(config.causal_transformer_context_duration);
                let context = (current_frame_rate as f64 * context_duration).round() as usize;
                let layer = MossAudioTokenizerProjectedTransformer::new(
                    vb_encoder.pp(layer_id),
                    cfg,
                    context,
                )?;
                encoder.push(MossAudioTokenizerModule::ProjectedTransformer(layer));
            } else {
                return Err(anyhow!(
                    "Moss Module only sopport PatchedPretransform and Transformer, but get: {}",
                    cfg.module_type
                ));
            }
        }
        let quantizer =
            MossAudioTokenizerResidualLFQ::new(vb.pp("quantizer"), &config.quantizer_kwargs)?;
        let vb_decoder = vb.pp("decoder");
        let mut decoder = vec![];
        for (layer_id, cfg) in config.decoder_kwargs.iter().enumerate() {
            if cfg.module_type == "PatchedPretransform"
                && let Some(patch_size) = cfg.patch_size
            {
                let layer = MossAudioTokenizerPatchedPretransform::new(patch_size, false);
                decoder.push(MossAudioTokenizerModule::PatchedPretransform(layer));
            } else if cfg.module_type == "Transformer" {
                let context_duration = cfg
                    .context_duration
                    .unwrap_or(config.causal_transformer_context_duration);
                let context = (current_frame_rate as f64 * context_duration).round() as usize;
                let layer = MossAudioTokenizerProjectedTransformer::new(
                    vb_decoder.pp(layer_id),
                    cfg,
                    context,
                )?;
                decoder.push(MossAudioTokenizerModule::ProjectedTransformer(layer));
            } else {
                return Err(anyhow!(
                    "Moss Module only sopport PatchedPretransform and Transformer, but get: {}",
                    cfg.module_type
                ));
            }
        }

        Ok(Self {
            sampling_rate: config.sampling_rate,
            downsample_rate: config.downsample_rate,
            number_channels: config.number_channels,
            enable_channel_interleave: config.enable_channel_interleave,
            encoder,
            quantizer,
            decoder,
        })
    }

    fn flatten_channels_for_codec(
        &self,
        input_values: &Tensor,
        length: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (bs, _, audio_len) = input_values.dims3()?;
        let input_values = if audio_len % self.downsample_rate != 0 {
            let pad_length = self.downsample_rate - (audio_len % self.downsample_rate);
            input_values.pad_with_zeros(D::Minus1, 0, pad_length)?
        } else {
            input_values.clone()
        };
        if self.number_channels > 1 && self.enable_channel_interleave {
            let input_values = input_values
                .transpose(1, 2)?
                .contiguous()?
                .reshape((bs, 1, ()))?;
            let length = (length * self.number_channels as f64)?;
            Ok((input_values, length))
        } else {
            Ok((input_values, length.clone()))
        }
    }

    pub fn batch_encode(&self, input_values: &Tensor, length: &Tensor) -> Result<Vec<Tensor>> {
        let (mut encoder_hidden_states, mut encoder_hidden_lengths) =
            self.flatten_channels_for_codec(input_values, length)?;
        for layer in &self.encoder {
            (encoder_hidden_states, encoder_hidden_lengths) =
                layer.forward(&encoder_hidden_states, &encoder_hidden_lengths)?;
        }
        let audio_codes = self
            .quantizer
            .forward(&encoder_hidden_states, &encoder_hidden_lengths)?;
        // (dim, bs, len) -> (bs, len, dim)
        let audio_codes = audio_codes.permute((1, 2, 0))?;
        let mut audio_codes_vec = vec![];
        for index in 0..encoder_hidden_lengths.dim(0)? {
            let codes_i = audio_codes.i(index)?;
            let length = encoder_hidden_lengths.i(index)?.to_scalar::<f32>()? as usize;
            let codes_i = codes_i.narrow(0, 0, length)?;
            audio_codes_vec.push(codes_i);
        }
        Ok(audio_codes_vec)
    }

    pub fn encode_one(&self, wav: &Tensor) -> Result<Tensor> {
        // in: (channel, audio_len)
        let (c, len) = wav.dims2()?;
        if c != self.number_channels {
            return Err(anyhow!(
                "MossAudioTokenizer encode_one need number_channels: {} but the wav channel: {}",
                self.number_channels,
                c,
            ));
        }
        let input_values = wav.unsqueeze(0)?;
        let length = Tensor::new(vec![len as f32], wav.device())?;
        let audio_vec = self.batch_encode(&input_values, &length)?;
        Ok(audio_vec[0].clone())
    }

    pub fn encode_list(&self, wavs: &[Tensor]) -> Result<Vec<Tensor>> {
        if wavs.is_empty() {
            return Err(anyhow!(
                "MossAudioTokenizer encode_list need wavs len > 0, but the wavs is empty"
            ));
        }
        let mut length = vec![];
        for wav in wavs.iter() {
            let (c, len) = wav.dims2()?;
            if c != self.number_channels {
                return Err(anyhow!(
                    "MossAudioTokenizer encode_list need number_channels: {} but the wav channel: {}",
                    self.number_channels,
                    c,
                ));
            }
            length.push(len as u32);
        }
        let max_length = *length.iter().max().unwrap_or(&0) as usize;
        let mut input_values = vec![];
        for wav in wavs.iter() {
            let audio_len = wav.dim(1)?;
            let wav_ = if audio_len < max_length {
                wav.pad_with_zeros(D::Minus1, 0, max_length - audio_len)?
            } else {
                wav.clone()
            };
            input_values.push(wav_);
        }
        let input_values = Tensor::stack(&input_values, 0)?;
        let length_tensor = Tensor::new(length.clone(), input_values.device())?
            .to_dtype(candle_core::DType::F32)?;
        self.batch_encode(&input_values, &length_tensor)
    }

    pub fn decode_audio_token_ids_to_waveform(&self, audio_token_ids: &Tensor) -> Result<Tensor> {
        let decode_codes = audio_token_ids.t()?.unsqueeze(1)?; //(len, nq) -> (nq, len) -> (nq, 1, len)
        let len = decode_codes.dim(2)?;
        let mut audio = self.quantizer.decode_codes(&decode_codes)?;
        let mut audio_length = Tensor::new(&[len as f32], audio.device())?;
        for decoder_module in self.decoder.iter() {
            (audio, audio_length) = decoder_module.forward(&audio, &audio_length)?;
        }
        if self.number_channels == 1 || !self.enable_channel_interleave {
            Ok(audio)
        } else {
            let bs = audio.dim(0)?;
            Ok(audio
                .squeeze(1)?
                .contiguous()?
                .reshape((bs, (), self.number_channels))?
                .transpose(1, 2)?
                .contiguous()?)
        }
    }
}
