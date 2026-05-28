use anyhow::{Result, anyhow};
use base64::{Engine, prelude::BASE64_STANDARD};
use candle_core::{DType, Device, Tensor, pickle::read_all_with_key};
use candle_nn::VarBuilder;
use rocket::futures::{Stream, stream};
use std::collections::HashMap;

use crate::{
    models::{
        GenerateModel,
        voxcpm::{
            audio_vae::AudioVAE,
            config::{AudioVaeConfig, VoxCPMConfig},
            tokenizer::SingleChineseTokenizer,
        },
        voxcpm_refact::{model::VoxCPMModelRefact, processor::VoxCPMProcessor},
    },
    params::chat::{ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse},
    utils::{
        audio_utils::{extract_audio_url, get_audio_wav_u8},
        extract_metadata_value, extract_user_text, find_type_files, get_device, get_dtype,
        response_utils::build_audio_completion_response,
    },
};

pub struct VoxCPMGenerateRefact {
    model: VoxCPMModelRefact,
    tokenizer: SingleChineseTokenizer,
    audio_vae: AudioVAE,
    processor: VoxCPMProcessor,
    prompt_cache: Option<HashMap<String, Tensor>>,
    out_sample_rate: usize,
    model_name: String,
}

impl VoxCPMGenerateRefact {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let device = &get_device(device);
        let config_path = path.to_string() + "/config.json";
        let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let model_list = find_type_files(path, "pth")?;
        let mut dict_to_hashmap = HashMap::new();
        let mut vae_dtype = candle_core::DType::F32;
        for m in model_list {
            let dict = read_all_with_key(m, Some("state_dict"))?;
            vae_dtype = dict[0].1.dtype();
            for (k, v) in dict {
                dict_to_hashmap.insert(k, v);
            }
        }
        let vb_vae = VarBuilder::from_tensors(dict_to_hashmap, vae_dtype, device);
        let audio_config = match config.audio_vae_config.clone() {
            Some(config) => config,
            None => AudioVaeConfig {
                encoder_dim: 128,
                encoder_rates: vec![2, 5, 8, 8],
                latent_dim: 64,
                decoder_dim: 1536,
                decoder_rates: vec![8, 8, 5, 2],
                sample_rate: 16000,
                out_sample_rate: None,
                sr_bin_boundaries: None,
            },
        };
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("VoxCPM")
            .to_string();
        let audio_vae = AudioVAE::new(
            vb_vae,
            audio_config.encoder_dim,
            audio_config.encoder_rates.clone(),
            Some(audio_config.latent_dim),
            audio_config.decoder_dim,
            audio_config.decoder_rates.clone(),
            audio_config.sample_rate,
            audio_config
                .out_sample_rate
                .unwrap_or(audio_config.sample_rate),
            audio_config.sr_bin_boundaries,
            Some("scale_bias".to_string()),
            // Some(128),
            // Some(false),
        )?;
        let decode_chunk_size = audio_config.decoder_rates.iter().product();
        let processor = VoxCPMProcessor::new(
            audio_vae.sample_rate,
            audio_vae.chunk_size,
            config.patch_size,
            device.clone(),
        );

        let cfg_dtype = config.dtype.as_str();
        let m_dtype = get_dtype(dtype, cfg_dtype);

        let model_list = find_type_files(path, "bin")?;
        // voxcpm0.5B模型文件是.bin类型， OpenBMB/VoxCPM1.5模型文件是.safetensors类型
        let vb_voxcpm = if model_list.is_empty() {
            let model_list = find_type_files(path, "safetensors")?;
            unsafe { VarBuilder::from_mmaped_safetensors(&model_list, m_dtype, device)? }
        } else {
            dict_to_hashmap = HashMap::new();
            let cfg_dtype = config.dtype.as_str();
            let m_dtype = get_dtype(dtype, cfg_dtype);
            for m in model_list {
                let dict = read_all_with_key(m, Some("state_dict"))?;
                for (k, v) in dict {
                    // println!("key: {}, tensor shape: {:?}", k, v);
                    dict_to_hashmap.insert(k, v);
                }
            }
            VarBuilder::from_tensors(dict_to_hashmap, m_dtype, device)
        };
        let tokenizer = SingleChineseTokenizer::new(path)?;
        let model =
            VoxCPMModelRefact::new(vb_voxcpm, config, audio_vae.latent_dim, decode_chunk_size)?;
        let out_sample_rate = audio_config
            .out_sample_rate
            .unwrap_or(audio_config.sample_rate);
        Ok(Self {
            model,
            tokenizer,
            audio_vae,
            processor,
            prompt_cache: None,
            out_sample_rate,
            model_name,
        })
    }

    pub fn sample_rate(&self) -> usize {
        self.out_sample_rate
    }

    pub fn inference(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        let (text_token, audio_feat, audio_mask) = self.processor.processor(
            target_text,
            prompt_text,
            prompt_wav_path,
            &self.tokenizer,
            &self.audio_vae,
        )?;
        let target_text_length = if let Some(mask) = &audio_mask {
            text_token.dim(1)? - (mask.sum_all()?.to_scalar::<u32>()? as usize)
        } else {
            text_token.dim(1)?
        };
        let max_len = if retry_badcase {
            (target_text_length as f64 * retry_badcase_ratio_threshold + 10.0) as usize
        } else {
            max_len
        };
        let audio = self.model.inference(
            &text_token,
            audio_feat.as_ref(),
            audio_mask.as_ref(),
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
            &self.audio_vae,
        )?;
        self.model.clear_kv_cache();
        Ok(audio)
    }

    pub fn generate_with_prompt_simple(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
    ) -> Result<Tensor> {
        let audio = self.inference(
            target_text,
            prompt_text,
            prompt_wav_path,
            2,
            1000,
            10,
            2.0,
            false,
            6.0,
        )?;
        Ok(audio)
    }
    pub fn generate_simple(&mut self, target_text: String) -> Result<Tensor> {
        let audio = self.inference(target_text, None, None, 2, 100, 10, 2.0, false, 6.0)?;
        Ok(audio)
    }

    pub fn build_prompt_cache(
        &mut self,
        prompt_text: String,
        prompt_wav_path: String,
    ) -> Result<()> {
        let prompt_cache = self.processor.build_prompt_cache(
            prompt_text,
            prompt_wav_path,
            &self.tokenizer,
            &self.audio_vae,
        )?;
        self.prompt_cache = Some(prompt_cache);
        Ok(())
    }

    pub fn generate_use_prompt_cache(
        &mut self,
        target_text: String,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        let audio = match &self.prompt_cache {
            Some(cache) => {
                let (text_token, audio_feat, audio_mask) =
                    self.processor
                        .processor_use_cache(target_text, cache, &self.tokenizer)?;
                let target_text_length = if let Some(mask) = &audio_mask {
                    text_token.dim(1)? - (mask.sum_all()?.to_scalar::<u32>()? as usize)
                } else {
                    text_token.dim(1)?
                };
                let max_len = if retry_badcase {
                    (target_text_length as f64 * retry_badcase_ratio_threshold + 10.0) as usize
                } else {
                    max_len
                };
                self.model.inference(
                    &text_token,
                    audio_feat.as_ref(),
                    audio_mask.as_ref(),
                    min_len,
                    max_len,
                    inference_timesteps,
                    cfg_value,
                    &self.audio_vae,
                )?
            }
            None => {
                return Err(anyhow!("need prompt_cache"));
            }
        };
        self.model.clear_kv_cache();
        Ok(audio)
    }

    pub fn generate_stream_use_prompt_cache(
        &mut self,
        target_text: String,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<impl Stream<Item = Result<Tensor, anyhow::Error>>> {
        match &self.prompt_cache {
            Some(cache) => {
                let (text_token, audio_feat, audio_mask) =
                    self.processor
                        .processor_use_cache(target_text, cache, &self.tokenizer)?;
                let target_text_length = if let Some(mask) = &audio_mask {
                    text_token.dim(1)? - (mask.sum_all()?.to_scalar::<u32>()? as usize)
                } else {
                    text_token.dim(1)?
                };
                let max_len = if retry_badcase {
                    (target_text_length as f64 * retry_badcase_ratio_threshold + 10.0) as usize
                } else {
                    max_len
                };
                self.model.inference_stream(
                    text_token,
                    audio_feat,
                    audio_mask,
                    min_len,
                    max_len,
                    inference_timesteps,
                    cfg_value,
                    &self.audio_vae,
                )
            }
            None => Err(anyhow!("need prompt_cache")),
        }
    }
}

impl GenerateModel for VoxCPMGenerateRefact {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let prompt_text = extract_metadata_value::<String>(&mes.metadata, "prompt_text");
        let control_instruction =
            extract_metadata_value::<String>(&mes.metadata, "control_instruction");
        let min_len = extract_metadata_value::<usize>(&mes.metadata, "min_len").unwrap_or(2);
        let max_len = extract_metadata_value::<usize>(&mes.metadata, "max_len").unwrap_or(4096);
        let inference_timesteps =
            extract_metadata_value::<usize>(&mes.metadata, "inference_timesteps").unwrap_or(10);
        let cfg_value = extract_metadata_value::<f64>(&mes.metadata, "cfg_value").unwrap_or(2.0);
        let retry_badcase_ratio_threshold =
            extract_metadata_value::<f64>(&mes.metadata, "retry_badcase_ratio_threshold")
                .unwrap_or(6.0);

        let prompt_wav = extract_audio_url(&mes);
        let prompt_wav_path = if !prompt_wav.is_empty() {
            Some(prompt_wav[0].clone())
        } else {
            None
        };
        if !self.model_name.contains("2") && prompt_wav_path.is_some() && prompt_text.is_none() {
            return Err(anyhow!(
                "reference mode is only supported with VoxCPM2 models"
            ));
        }
        let mut target_text = extract_user_text(&mes)?;
        if let Some(instruction) = control_instruction
            && self.model_name.contains("2")
        {
            target_text = format!("({instruction}){target_text}");
        }
        let audio = self
            .inference(
                target_text,
                prompt_text,
                prompt_wav_path,
                min_len,
                max_len,
                inference_timesteps,
                cfg_value,
                true,
                retry_badcase_ratio_threshold,
            )
            .inspect_err(|_| {
                self.model.clear_kv_cache();
            })?;
        let wav_u8 = get_audio_wav_u8(&audio, self.out_sample_rate as u32)?;
        // let wave_u8_str = String::from_utf8(wav_u8)?;
        let base64_audio = BASE64_STANDARD.encode(wav_u8);
        let response = build_audio_completion_response(&base64_audio, &self.model_name);
        self.model.clear_kv_cache();
        Ok(response)
    }
    #[allow(unused_variables)]
    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let error_stream = stream::once(async {
            Err(anyhow::anyhow!(format!(
                "{} model not support stream",
                self.model_name
            ))) as Result<ChatCompletionChunkResponse, anyhow::Error>
        });

        Ok(Box::new(Box::pin(error_stream)))
    }
}
