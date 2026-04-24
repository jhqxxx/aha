use std::collections::HashMap;

use crate::{
    models::voxcpm::{audio_vae::AudioVAE, tokenizer::SingleChineseTokenizer},
    utils::audio_utils::load_audio_with_resample,
};
use anyhow::Result;
use candle_core::{D, Device, IndexOp, Tensor};

pub struct VoxCPMProcessor {
    sample_rate: usize,
    chunk_size: usize,
    patch_size: usize,
    audio_start_token: u32,
    ref_audio_start_token: u32,
    ref_audio_end_token: u32,
    device: Device,
}

impl VoxCPMProcessor {
    pub fn new(sample_rate: usize, chunk_size: usize, patch_size: usize, device: Device) -> Self {
        Self {
            sample_rate,
            chunk_size,
            patch_size,
            audio_start_token: 101,
            ref_audio_start_token: 103,
            ref_audio_end_token: 104,
            device,
        }
    }

    pub fn build_prompt_cache(
        &mut self,
        prompt_text: String,
        prompt_wav_path: String,
        tokenizer: &SingleChineseTokenizer,
        audio_vae: &AudioVAE,
    ) -> Result<HashMap<String, Tensor>> {
        let (text_token, _) = tokenizer.encode_tensor(prompt_text, &self.device)?;
        let mut audio =
            load_audio_with_resample(&prompt_wav_path, &self.device, Some(self.sample_rate))?;
        let patch_len = self.patch_size * self.chunk_size;
        if audio.dim(1)? % patch_len != 0 {
            audio = audio.pad_with_zeros(D::Minus1, 0, patch_len - audio.dim(1)? % patch_len)?;
        }
        let audio_feat = audio_vae.encode(&audio, Some(self.sample_rate))?;
        let audio_feat = audio_feat
            .reshape((audio_vae.latent_dim, (), self.patch_size))?
            .permute((1, 2, 0))?;
        let dim0 = audio_feat.dim(0)? - 1;
        let audio_feat = audio_feat.i(..dim0)?;
        let mut hashmap = HashMap::new();
        hashmap.insert("text_token".to_string(), text_token);
        hashmap.insert("audio_feat".to_string(), audio_feat);
        Ok(hashmap)
    }

    pub fn processor(
        &self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        tokenizer: &SingleChineseTokenizer,
        audio_vae: &AudioVAE,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        let text = if let Some(prompt_text) = &prompt_text {
            prompt_text.clone() + &target_text
        } else {
            target_text
        };
        let (text_token, _) = tokenizer.encode_tensor(text, &self.device)?;
        let audio_start = Tensor::new(vec![self.audio_start_token], &self.device)?;
        let mut text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;

        let (audio_feat, audio_mask) = if let Some(path) = prompt_wav_path {
            let mut audio = load_audio_with_resample(&path, &self.device, Some(self.sample_rate))?;
            let patch_len = self.patch_size * self.chunk_size;
            if audio.dim(1)? % patch_len != 0 {
                audio =
                    audio.pad_with_zeros(D::Minus1, patch_len - audio.dim(1)? % patch_len, 0)?;
            }
            let audio_feat = audio_vae.encode(&audio, Some(self.sample_rate))?;
            let audio_feat = audio_feat
                .reshape((audio_vae.latent_dim, (), self.patch_size))?
                .permute((1, 2, 0))?;
            let text_length = text_token.dim(0)?;
            let audio_length = audio_feat.dim(0)?;
            let audio_mask = if prompt_text.is_some() {
                let text_pad_token =
                    Tensor::zeros(audio_length, candle_core::DType::U32, &self.device)?;
                text_token = Tensor::cat(&[text_token, text_pad_token], D::Minus1)?;
                let mask = Tensor::cat(
                    &[
                        Tensor::zeros(text_length, candle_core::DType::U32, &self.device)?,
                        Tensor::ones(audio_length, candle_core::DType::U32, &self.device)?,
                    ],
                    D::Minus1,
                )?
                .unsqueeze(0)?;
                Some(mask)
            } else {
                let ref_start = Tensor::new(vec![self.ref_audio_start_token], &self.device)?;
                let ref_end = Tensor::new(vec![self.ref_audio_end_token], &self.device)?;
                let ref_token = Tensor::zeros(audio_length, candle_core::DType::U32, &self.device)?;
                text_token = Tensor::cat(&[&ref_start, &ref_token, &ref_end, &text_token], 0)?;
                let mask = Tensor::cat(
                    &[
                        Tensor::new(vec![0u32], &self.device)?,
                        Tensor::ones(audio_length, candle_core::DType::U32, &self.device)?,
                        Tensor::new(vec![0u32], &self.device)?,
                        Tensor::zeros(text_length, candle_core::DType::U32, &self.device)?,
                    ],
                    D::Minus1,
                )?
                .unsqueeze(0)?;
                Some(mask)
            };
            let audio_feat = audio_feat.unsqueeze(0)?;
            (Some(audio_feat), audio_mask)
        } else {
            (None, None)
        };
        let text_token = text_token.unsqueeze(0)?;
        Ok((text_token, audio_feat, audio_mask))
    }

    pub fn processor_use_cache(
        &self,
        target_text: String,
        prompt_cache: &HashMap<String, Tensor>,
        tokenizer: &SingleChineseTokenizer,
    ) -> Result<(Tensor, Option<Tensor>, Option<Tensor>)> {
        let (target_text_token, _) = tokenizer.encode_tensor(target_text, &self.device)?;
        let text_token = match prompt_cache.get("text_token") {
            Some(token) => Tensor::cat(&[token, &target_text_token], 0)?,
            None => target_text_token,
        };
        let audio_start = Tensor::new(vec![self.audio_start_token], &self.device)?;
        let mut text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
        let text_length = text_token.dim(0)?;
        let (audio_length, audio_feat) = match prompt_cache.get("audio_feat") {
            Some(feat) => (feat.dim(0)?, Some(feat.clone().unsqueeze(0)?)),
            None => (0, None),
        };
        let audio_mask = if audio_length > 0 {
            let text_pad_token =
                Tensor::zeros(audio_length, candle_core::DType::U32, &self.device)?;
            text_token = Tensor::cat(&[text_token, text_pad_token], D::Minus1)?;
            let mask = Tensor::cat(
                &[
                    Tensor::zeros(text_length, candle_core::DType::U32, &self.device)?,
                    Tensor::ones(audio_length, candle_core::DType::U32, &self.device)?,
                ],
                D::Minus1,
            )?
            .unsqueeze(0)?;
            Some(mask)
        } else {
            None
        };
        let text_token = text_token.unsqueeze(0)?;
        Ok((text_token, audio_feat, audio_mask))
    }
}
