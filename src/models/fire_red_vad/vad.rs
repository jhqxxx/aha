use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    models::{
        common::modules::VadFrameResult,
        fire_red_vad::{
            config::{DetectModelConfig, FireRedVadConfig},
            model::DetectModel,
            processor::{AudioFeat, VadPostprocessor},
        },
    },
    utils::{
        audio_utils::{resample_audio_from_bytes, resample_audio_from_vec_f32},
        find_type_files, get_device,
        tensor_utils::split_tensor_with_size,
    },
};

#[derive(Debug)]
pub struct VadResult {
    pub dur: f32,
    pub timestamps: Vec<(f32, f32)>,
    pub model_name: String,
    pub mode: String,
}

pub struct FireRedVad {
    audio_feat: AudioFeat,
    vad_model: DetectModel,
    vad_postprocessor: VadPostprocessor,
    model_name: String,
    device: Device,
    cfg: FireRedVadConfig,
    caches: Option<Vec<Tensor>>,
    frame_length_sample: usize,
    speech_cache: Vec<Tensor>,
}

impl FireRedVad {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let device = get_device(device);
        let audio_feat = AudioFeat::new(path, &device)?;
        let model_list = find_type_files(path, "safetensors")?;
        let dtype = dtype.unwrap_or(DType::F32);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("VAD")
            .to_string();
        let (model_cfg, cfg) = if model_name.to_lowercase().contains("stream") {
            (
                DetectModelConfig::default_stream_vad(),
                FireRedVadConfig::default_stream_vad(),
            )
        } else if model_name.to_lowercase().contains("aed") {
            (
                DetectModelConfig::default_aed(),
                FireRedVadConfig::default_aed(),
            ) // TODO: aed
        } else {
            (
                DetectModelConfig::default_vad(),
                FireRedVadConfig::default_vad(),
            )
        };
        let vad_model = DetectModel::new(vb, model_cfg)?;
        let vad_postprocessor = VadPostprocessor::new(&cfg);
        Ok(Self {
            audio_feat,
            vad_model,
            vad_postprocessor,
            model_name,
            device,
            cfg,
            caches: None,
            frame_length_sample: 400,
            speech_cache: vec![],
        })
    }

    pub fn detect_frame(&mut self, audio_frame: &Tensor) -> Result<Option<VadFrameResult>> {
        if audio_frame.dim(0)? < self.frame_length_sample {
            return Err(anyhow!(
                "Expected {} samples, got {}",
                self.frame_length_sample,
                audio_frame.dim(0)?
            ));
        }
        let feats = self.audio_feat.extract(audio_frame)?;
        let (probs, caches) = self
            .vad_model
            .forward(&feats.unsqueeze(0)?, self.caches.as_ref())?;
        self.caches = Some(caches);
        let probs = probs.squeeze(D::Minus1)?.squeeze(0)?;
        let binary_preds = self
            .vad_postprocessor
            .process_thresh(&probs)?
            .to_dtype(DType::U32)?;
        let preds_sum = binary_preds.sum_all()?.to_scalar::<u32>()?;
        let probs_len = probs.dim(0)?;
        // 输入数据中 is_speech > 0.1, 认为这帧数据可用
        let final_data = if preds_sum as f32 > probs_len as f32 * 0.1 {
            // 通过最后10个数据，判断说话是否结束
            let last_10_preds_sum = binary_preds
                .narrow(0, probs_len - 10, 10)?
                .sum_all()?
                .to_scalar::<u32>()?;
            // 10个数据中，至少8个是 speech, 认为说话没有结束，缓存数据，等待下一帧
            if last_10_preds_sum >= 8 {
                self.speech_cache.push(audio_frame.clone());
                None
            } else {
                // 否则认为此次说话结束，结合缓存数据，一起返回
                let data = if self.speech_cache.is_empty() {
                    // 缓存数据为空，直接返回
                    audio_frame.clone()
                } else {
                    // 缓存数据不为空，cat数据
                    self.speech_cache.push(audio_frame.clone());
                    let audio_frame = Tensor::cat(&self.speech_cache, 0)?;
                    self.speech_cache = vec![]; // 清空缓存
                    audio_frame
                };
                Some(data)
            }
        } else {
            // 认为这帧数据不可用，是否有缓存数据
            if self.speech_cache.is_empty() {
                // 没有返回None
                None
            } else {
                // 有缓存，返回缓存数据
                let data = Some(Tensor::cat(&self.speech_cache, 0)?);
                self.speech_cache.clear(); // 清空缓存
                data
            }
        };

        if final_data.is_none() {
            Ok(None)
        } else {
            Ok(Some(VadFrameResult {
                is_speech: true,
                is_i16: true,
                orig_audio: final_data,
                model_name: self.model_name.clone(),
                mode: "speech".to_string(),
            }))
        }
    }

    pub fn detect_frame_f32(
        &mut self,
        audio_vec_f32: Vec<f32>,
        channels: usize,
        orig_sr: Option<usize>,
    ) -> Result<Option<VadFrameResult>> {
        if !self.model_name.to_lowercase().contains("stream") {
            return Err(anyhow!("only stream model support detect_frame"));
        }
        let audio_frame = resample_audio_from_vec_f32(
            audio_vec_f32,
            &self.device,
            channels,
            orig_sr,
            Some(16000),
            true,
        )?
        .squeeze(0)?;
        self.detect_frame(&audio_frame)
    }

    pub fn detect_frame_bytes(&mut self, audio_bytes: Vec<u8>) -> Result<Option<VadFrameResult>> {
        if !self.model_name.to_lowercase().contains("stream") {
            return Err(anyhow!("only stream model support detect_frame"));
        }
        let audio_frame =
            resample_audio_from_bytes(audio_bytes, &self.device, Some(16000), true)?.squeeze(0)?;
        self.detect_frame(&audio_frame)
    }

    pub fn detect_file(&self, audio_path: &str) -> Result<VadResult> {
        let (feats, dur) = self.audio_feat.extract_file(audio_path, &self.device)?;
        let probs = if feats.dim(0)? <= self.cfg.chunk_max_frame {
            let (probs, _) = self.vad_model.forward(&feats.unsqueeze(0)?, None)?;
            probs
        } else {
            let mut chunk_probs = vec![];
            let chunks = split_tensor_with_size(&feats, self.cfg.chunk_max_frame, 0usize)?;
            for chunk in chunks.iter() {
                let (chunk_prob, _) = self.vad_model.forward(&chunk.unsqueeze(0)?, None)?;
                chunk_probs.push(chunk_prob);
            }
            Tensor::cat(&chunk_probs, 1)?
        };
        let probs = if self.model_name.to_lowercase().contains("aed") {
            // only care speech
            probs
                .squeeze(0)?
                .narrow(D::Minus1, 0, 1)?
                .squeeze(D::Minus1)?
        } else {
            probs.squeeze(0)?.squeeze(D::Minus1)?
        };
        let segments = self.vad_postprocessor.process(&probs, dur)?;
        let res = VadResult {
            dur,
            timestamps: segments,
            model_name: self.model_name.clone(),
            mode: "speech".to_string(),
        };
        Ok(res)
    }

    pub fn reset(&mut self) {
        self.caches = None;
    }
}
