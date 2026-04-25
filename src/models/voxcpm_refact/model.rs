use crate::{
    models::voxcpm::{
        audio_vae::AudioVAE,
        config::VoxCPMConfig,
        minicpm4::MiniCPMModel,
        model::{ScalarQuantizationLayer, UnifiedCFM, VoxCPMLocDiT, VoxCPMLocEnc},
    },
    utils::tensor_utils::masked_scatter_dim0,
};
use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias};
use rocket::async_stream::stream;
use rocket::futures::Stream;

pub struct VoxCPMModelRefact {
    config: VoxCPMConfig,
    patch_size: usize,
    latent_dim: usize,
    decode_patch_len: usize,
    // audio_start_token: u32,
    // // audio_end_token: u32,
    // ref_audio_start_token: u32,
    // ref_audio_end_token: u32,
    // chunk_size: usize,
    // sample_rate: usize,
    base_lm: MiniCPMModel,
    residual_lm: MiniCPMModel,
    feat_encoder: VoxCPMLocEnc,
    feat_decoder: UnifiedCFM,
    fsq_layer: ScalarQuantizationLayer,
    enc_to_lm_proj: Linear,
    lm_to_dit_proj: Linear,
    res_to_dit_proj: Linear,
    fusion_concat_proj: Option<Linear>,
    stop_proj: Linear,
    stop_head: Linear,
    device: Device,
    dtype: DType,
}

impl VoxCPMModelRefact {
    pub fn new(
        vb: VarBuilder,
        config: VoxCPMConfig,
        latent_dim: usize,
        decode_chunk_size: usize,
    ) -> Result<Self> {
        let base_lm = MiniCPMModel::new(vb.pp("base_lm"), config.lm_config.clone())?;
        let mut residual_lm_config = config.lm_config.clone();
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers;
        residual_lm_config.vocab_size = 0;
        residual_lm_config.no_rope = config.residual_lm_no_rope;
        let residual_lm = MiniCPMModel::new(vb.pp("residual_lm"), residual_lm_config)?;
        let mut encoder_config = config.lm_config.clone();
        encoder_config.hidden_size = config.encoder_config.hidden_dim;
        encoder_config.intermediate_size = config.encoder_config.ffn_dim;
        encoder_config.num_attention_heads = config.encoder_config.num_heads;
        encoder_config.num_hidden_layers = config.encoder_config.num_layers;
        encoder_config.kv_channels = config.encoder_config.kv_channels;
        encoder_config.vocab_size = 0;
        let feat_encoder =
            VoxCPMLocEnc::new(vb.pp("feat_encoder"), encoder_config, config.feat_dim)?;

        let mut decoder_config = config.lm_config.clone();
        decoder_config.hidden_size = config.dit_config.hidden_dim;
        decoder_config.intermediate_size = config.dit_config.ffn_dim;
        decoder_config.num_attention_heads = config.dit_config.num_heads;
        decoder_config.num_hidden_layers = config.dit_config.num_layers;
        decoder_config.kv_channels = config.dit_config.kv_channels;
        decoder_config.vocab_size = 0;
        let estimator = VoxCPMLocDiT::new(
            vb.pp("feat_decoder.estimator"),
            decoder_config,
            config.feat_dim,
        )?;
        let feat_decoder = UnifiedCFM::new(
            config.feat_dim,
            config.dit_config.cfm_config.clone(),
            estimator,
            false,
            // config.architecture.clone(),
        )?;
        let fsq_layer = ScalarQuantizationLayer::new(
            vb.pp("fsq_layer"),
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
        )?;
        let enc_to_lm_proj = linear(
            config.encoder_config.hidden_dim,
            config.lm_config.hidden_size,
            vb.pp("enc_to_lm_proj"),
        )?;
        let lm_to_dit_proj = linear(
            config.lm_config.hidden_size,
            config.dit_config.hidden_dim,
            vb.pp("lm_to_dit_proj"),
        )?;
        let res_to_dit_proj = linear(
            config.lm_config.hidden_size,
            config.dit_config.hidden_dim,
            vb.pp("res_to_dit_proj"),
        )?;

        let fusion_concat_proj = if config.architecture.to_lowercase().eq("voxcpm2") {
            Some(linear(
                config.lm_config.hidden_size * 2,
                config.lm_config.hidden_size,
                vb.pp("fusion_concat_proj"),
            )?)
        } else {
            None
        };

        let stop_proj = linear(
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            vb.pp("stop_proj"),
        )?;
        let stop_head = linear_no_bias(config.lm_config.hidden_size, 2, vb.pp("stop_head"))?;

        let patch_size = config.patch_size;
        let decode_patch_len = patch_size * decode_chunk_size;
        Ok(Self {
            config,
            patch_size,
            latent_dim,
            decode_patch_len,
            // audio_start_token: 101,
            // // audio_end_token: 102,
            // ref_audio_start_token: 103,
            // ref_audio_end_token: 104,
            // chunk_size: audio_vae.chunk_size,
            // sample_rate: audio_vae.sample_rate,
            // tokenizer,
            // audio_vae,
            base_lm,
            residual_lm,
            feat_encoder,
            feat_decoder,
            fsq_layer,
            enc_to_lm_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            fusion_concat_proj,
            stop_proj,
            stop_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn inference(
        &mut self,
        text: &Tensor,
        audio_feat: Option<&Tensor>,
        audio_mask: Option<&Tensor>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        audio_vae: &AudioVAE,
    ) -> Result<Tensor> {
        let (b, t) = text.dims2()?;
        let scale_emb = if self.config.lm_config.use_mup {
            self.config.lm_config.scale_emb
        } else {
            1.0
        };
        let text_embed = self
            .base_lm
            .embed_tokens
            .as_ref()
            .unwrap()
            .forward(text)?
            .affine(scale_emb as f64, 0.0)?;
        let (combined_embed, mut prefix_feat_cond, feat_embed) = if let Some(audio_feat) =
            audio_feat
            && let Some(audio_mask) = audio_mask
        {
            let audio_feat = audio_feat.to_dtype(self.dtype)?;
            let audio_t = audio_feat.dim(1)?;
            let feat_embed = self.feat_encoder.forward(&audio_feat)?; // [b, audio_t, h_feat]
            let feat_embed = self.enc_to_lm_proj.forward(&feat_embed)?.squeeze(0)?;
            let embeds = masked_scatter_dim0(&text_embed, &feat_embed, audio_mask)?;
            let prefix_feat_cond = audio_feat.i((.., audio_t - 1, ..))?;
            (embeds, prefix_feat_cond, Some(feat_embed))
        } else {
            let prefix_feat_cond = Tensor::zeros(
                (b, self.patch_size, self.latent_dim),
                self.dtype,
                &self.device,
            )?;
            (text_embed, prefix_feat_cond, None)
        };
        let mut pred_feat_seq = Vec::new();
        // if feat_mask.i((1, t-1))?.to_scalar::<f32>()? == 0.0 {
        //     // TODO for stream
        // }
        let mut position_id = 0;
        let mut seq_len = t;
        let enc_outputs = self
            .base_lm
            .forward_with_cache(&combined_embed, position_id)?;

        let (mut lm_hidden, input_embeds) = if let Some(_) = audio_feat
            && let Some(audio_mask) = audio_mask
            && let Some(feat_embed) = feat_embed
        {
            let fsq_emb = self.fsq_layer.forward(&enc_outputs)?;
            let audio_mask_broadcast = audio_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(fsq_emb.shape())?;
            let enc_outputs = audio_mask_broadcast.where_cond(&fsq_emb, &enc_outputs)?;
            let lm_hidden = enc_outputs.i((.., t - 1, ..))?;
            let input_embeds = if let Some(fusion) = &self.fusion_concat_proj {
                let feat = enc_outputs.zeros_like()?;
                let feat = masked_scatter_dim0(&feat, &feat_embed, audio_mask)?;
                let concat = Tensor::cat(&[&enc_outputs, &feat], D::Minus1)?;
                fusion.forward(&concat)?
            } else {
                let feat = enc_outputs.zeros_like()?;
                let feat = masked_scatter_dim0(&feat, &feat_embed, audio_mask)?;
                enc_outputs.add(&feat)?
            };
            (lm_hidden, input_embeds)
        } else {
            let lm_hidden = enc_outputs.i((.., t - 1, ..))?;
            let input_embeds = if let Some(fusion) = &self.fusion_concat_proj {
                let feat = enc_outputs.zeros_like()?;
                let concat = Tensor::cat(&[&enc_outputs, &feat], D::Minus1)?;
                fusion.forward(&concat)?
            } else {
                enc_outputs
            };
            (lm_hidden, input_embeds)
        };
        let residual_enc_outputs = self
            .residual_lm
            .forward_with_cache(&input_embeds, position_id)?;
        let mut residual_hidden = residual_enc_outputs.i((.., t - 1, ..))?;
        for i in 0..max_len {
            let dit_hidden_1 = self.lm_to_dit_proj.forward(&lm_hidden)?; // [b, h_dit]
            let dit_hidden_2 = self.res_to_dit_proj.forward(&residual_hidden)?; // [b, h_dit]
            // let dit_hidden = dit_hidden_1.add(&dit_hidden_2)?;
            let dit_hidden = if self.fusion_concat_proj.is_some() {
                Tensor::cat(&[&dit_hidden_1, &dit_hidden_2], D::Minus1)?
            } else {
                dit_hidden_1.add(&dit_hidden_2)?
            };
            let cond = prefix_feat_cond.transpose(1, 2)?.contiguous()?;
            let pred_feat = self
                .feat_decoder
                .forward(
                    &dit_hidden,
                    inference_timesteps,
                    self.patch_size,
                    &cond,
                    1.0,
                    cfg_value,
                    1.0,
                    true,
                )?
                .transpose(1, 2)?; // [b, p, d]
            let curr_embed = self.feat_encoder.forward(&pred_feat.unsqueeze(1)?)?; // [b, 1, c]
            let curr_embed = self.enc_to_lm_proj.forward(&curr_embed)?;
            pred_feat_seq.push(pred_feat.unsqueeze(1)?);

            prefix_feat_cond = pred_feat;
            let stop_flag = self.stop_proj.forward(&lm_hidden)?.silu()?;
            let stop_flag = self
                .stop_head
                .forward(&stop_flag)?
                .argmax(D::Minus1)?
                .i(0)?
                .to_scalar::<u32>()?;
            if i > min_len && stop_flag == 1 {
                break;
            }
            position_id += seq_len;
            seq_len = 1;
            lm_hidden = self
                .base_lm
                .forward_with_cache(&curr_embed.i((.., 0, ..))?, position_id)?
                .squeeze(1)?;
            lm_hidden = self.fsq_layer.forward(&lm_hidden)?;
            let curr_residual_input = if let Some(fusion) = &self.fusion_concat_proj {
                let curr_embed = curr_embed.i((.., 0, ..))?;
                let concat = Tensor::cat(&[&lm_hidden, &curr_embed], D::Minus1)?;
                fusion.forward(&concat)?
            } else {
                lm_hidden.add(&curr_embed.i((.., 0, ..))?)?
            };
            residual_hidden = self
                .residual_lm
                .forward_with_cache(&curr_residual_input, position_id)?
                .squeeze(1)?;
        }
        let pred_seq = Tensor::cat(&pred_feat_seq, 1)?; // (b, t, p, d)
        let (b, _, _, d) = pred_seq.dims4()?;
        let feat_pred = pred_seq
            .permute((0, 3, 1, 2))?
            .reshape((b, d, ()))?
            .contiguous()?;
        self.clear_kv_cache();

        let decode_audio = audio_vae
            .decode(&feat_pred.to_dtype(DType::F32)?, None)?
            .squeeze(1)?;
        let decode_audio_len = decode_audio.dim(D::Minus1)? - 640 - 640;
        let decode_audio = decode_audio.narrow(D::Minus1, 640, decode_audio_len)?;
        Ok(decode_audio)
    }

    pub fn inference_stream(
        &mut self,
        text: Tensor,
        audio_feat: Option<Tensor>,
        audio_mask: Option<Tensor>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        audio_vae: &AudioVAE,
    ) -> Result<impl Stream<Item = Result<Tensor, anyhow::Error>>> {
        let (b, t) = text.dims2()?;
        let scale_emb = if self.config.lm_config.use_mup {
            self.config.lm_config.scale_emb
        } else {
            1.0
        };
        let text_embed = self
            .base_lm
            .embed_tokens
            .as_ref()
            .unwrap()
            .forward(&text)?
            .affine(scale_emb as f64, 0.0)?;
        let (combined_embed, mut prefix_feat_cond, feat_embed) = if let Some(audio_feat) =
            &audio_feat
            && let Some(audio_mask) = &audio_mask
        {
            let audio_feat = audio_feat.to_dtype(self.dtype)?;
            let audio_t = audio_feat.dim(1)?;
            let feat_embed = self.feat_encoder.forward(&audio_feat)?; // [b, audio_t, h_feat]
            let feat_embed = self.enc_to_lm_proj.forward(&feat_embed)?.squeeze(0)?;
            let embeds = masked_scatter_dim0(&text_embed, &feat_embed, audio_mask)?;
            let prefix_feat_cond = audio_feat.i((.., audio_t - 1, ..))?;
            (embeds, prefix_feat_cond, Some(feat_embed))
        } else {
            let prefix_feat_cond = Tensor::zeros(
                (b, self.patch_size, self.latent_dim),
                self.dtype,
                &self.device,
            )?;
            (text_embed, prefix_feat_cond, None)
        };

        let streaming_prefix_len = 4usize;
        // 流式处理固定4个结果使得VAE decode结果正常
        let mut pred_feat_seq = Vec::with_capacity(streaming_prefix_len);
        // 流式处理添加prompt块
        if let Some(audio_feat) = &audio_feat
            && let Some(audio_mask) = &audio_mask
        {
            let audio_len = audio_mask.dim(1)?;
            if audio_mask
                .narrow(1, audio_len - 1, 1)?
                .squeeze(0)?
                .squeeze(0)?
                .to_scalar::<u32>()?
                == 1
            {
                let audio_len = audio_mask.sum_all()?.to_scalar::<u32>()? as usize;
                let context_len = audio_len.min(streaming_prefix_len - 1);
                let start = audio_feat.dim(1)? - context_len;
                let last_feat = audio_feat
                    .narrow(1, start, context_len)?
                    .to_dtype(self.dtype)?;
                pred_feat_seq.push(last_feat);
            }
        }
        let mut position_id = 0;
        let mut seq_len = t;
        let enc_outputs = self
            .base_lm
            .forward_with_cache(&combined_embed, position_id)?;

        let (mut lm_hidden, input_embeds) = if let Some(_) = &audio_feat
            && let Some(audio_mask) = &audio_mask
            && let Some(feat_embed) = feat_embed
        {
            let fsq_emb = self.fsq_layer.forward(&enc_outputs)?;
            let audio_mask_broadcast = audio_mask
                .unsqueeze(D::Minus1)?
                .broadcast_as(fsq_emb.shape())?;
            let enc_outputs = audio_mask_broadcast.where_cond(&fsq_emb, &enc_outputs)?;
            let lm_hidden = enc_outputs.i((.., t - 1, ..))?;
            let input_embeds = if let Some(fusion) = &self.fusion_concat_proj {
                let feat = enc_outputs.zeros_like()?;
                let feat = masked_scatter_dim0(&feat, &feat_embed, audio_mask)?;
                let concat = Tensor::cat(&[&enc_outputs, &feat], D::Minus1)?;
                fusion.forward(&concat)?
            } else {
                let feat = enc_outputs.zeros_like()?;
                let feat = masked_scatter_dim0(&feat, &feat_embed, audio_mask)?;
                enc_outputs.add(&feat)?
            };
            (lm_hidden, input_embeds)
        } else {
            let lm_hidden = enc_outputs.i((.., t - 1, ..))?;
            let input_embeds = if let Some(fusion) = &self.fusion_concat_proj {
                let feat = enc_outputs.zeros_like()?;
                let concat = Tensor::cat(&[&enc_outputs, &feat], D::Minus1)?;
                fusion.forward(&concat)?
            } else {
                enc_outputs
            };
            (lm_hidden, input_embeds)
        };
        let residual_enc_outputs = self
            .residual_lm
            .forward_with_cache(&input_embeds, position_id)?;
        let mut residual_hidden = residual_enc_outputs.i((.., t - 1, ..))?;
        let stream = stream! {
            let mut first_flag = true;
            for i in 0..max_len {
                let dit_hidden_1 = self.lm_to_dit_proj.forward(&lm_hidden)?; // [b, h_dit]
                let dit_hidden_2 = self.res_to_dit_proj.forward(&residual_hidden)?; // [b, h_dit]
                // let dit_hidden = dit_hidden_1.add(&dit_hidden_2)?;
                let dit_hidden = if self.fusion_concat_proj.is_some() {
                    Tensor::cat(&[&dit_hidden_1, &dit_hidden_2], D::Minus1)?
                } else {
                    dit_hidden_1.add(&dit_hidden_2)?
                };
                let cond = prefix_feat_cond.transpose(1, 2)?.contiguous()?;
                let pred_feat = self
                    .feat_decoder
                    .forward(
                        &dit_hidden,
                        inference_timesteps,
                        self.patch_size,
                        &cond,
                        1.0,
                        cfg_value,
                        1.0,
                        true,
                    )?
                    .transpose(1, 2)?; // [b, p, d]
                let curr_embed = self.feat_encoder.forward(&pred_feat.unsqueeze(1)?)?; // [b, 1, c]
                let curr_embed = self.enc_to_lm_proj.forward(&curr_embed)?;
                // 保持容量不超过最大值
                if pred_feat_seq.len() == streaming_prefix_len {
                    pred_feat_seq.remove(0);
                }
                pred_feat_seq.push(pred_feat.unsqueeze(1)?);
                prefix_feat_cond = pred_feat;
                let stop_flag = self.stop_proj.forward(&lm_hidden)?.silu()?;
                let stop_flag = self
                    .stop_head
                    .forward(&stop_flag)?
                    .argmax(D::Minus1)?
                    .i(0)?
                    .to_scalar::<u32>()?;

                let pred_feat_chunk = Tensor::cat(&pred_feat_seq, 1)?;
                let (b, _, _, d) = pred_feat_chunk.dims4()?;
                let feat_pred = pred_feat_chunk.permute((0, 3, 1, 2))?
                    .reshape((b, d, ()))?
                    .contiguous()?;
                let mut decode_audio = audio_vae
                    .decode(&feat_pred.to_dtype(DType::F32)?, None)?
                    .squeeze(1)?;
                // 只取当前帧结果
                let decode_start = decode_audio.dim(D::Minus1)? - self.decode_patch_len;
                decode_audio = decode_audio.narrow(D::Minus1, decode_start, self.decode_patch_len)?;
                if i > min_len && stop_flag == 1 {
                    // 最后一段去除噪音
                    let decode_audio_len = decode_audio.dim(D::Minus1)? - 640;
                    decode_audio = decode_audio.narrow(D::Minus1, 0, decode_audio_len)?;
                    yield Ok(decode_audio);
                    break;
                }
                if first_flag {
                    // 去除初始噪音
                    let decode_audio_len = decode_audio.dim(D::Minus1)? - 640;
                    decode_audio = decode_audio.narrow(D::Minus1, 640, decode_audio_len)?;
                    first_flag = false;
                }
                yield Ok(decode_audio);
                position_id += seq_len;
                seq_len = 1;
                lm_hidden = self
                    .base_lm
                    .forward_with_cache(&curr_embed.i((.., 0, ..))?, position_id)?
                    .squeeze(1)?;
                lm_hidden = self.fsq_layer.forward(&lm_hidden)?;
                let curr_residual_input = if let Some(fusion) = &self.fusion_concat_proj {
                    let curr_embed = curr_embed.i((.., 0, ..))?;
                    let concat = Tensor::cat(&[&lm_hidden, &curr_embed], D::Minus1)?;
                    fusion.forward(&concat)?
                } else {
                    lm_hidden.add(&curr_embed.i((.., 0, ..))?)?
                };
                residual_hidden = self
                    .residual_lm
                    .forward_with_cache(&curr_residual_input, position_id)?
                    .squeeze(1)?;

            }
            self.clear_kv_cache();
        };
        Ok(stream)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_lm.clear_kv_cache();
        self.residual_lm.clear_kv_cache();
    }
}
