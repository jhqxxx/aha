use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};

use crate::{
    models::{
        feature_extractor::feature_extraction_whisper::WhisperFeatureExtractor,
        glm_asr_nano::config::GlmAsrNanoProcessorConfig,
    },
    utils::{audio_utils::extract_audios, tensor_utils::split_tensor},
};

pub struct GlmAsrNanoProcessor {
    sampling_rate: usize,
    chunk_length: usize,
    n_samples: usize,
    // n_fft: usize,
    // window: Tensor,
    // mel_filters: Tensor,
    hop_length: usize,
    audio_token: String,
    // audio_token_id: u32,
    max_audio_len: usize,
    // default_transcription_prompt: String,
    device: Device,
    whisper_feature_extrator: WhisperFeatureExtractor,
}

impl GlmAsrNanoProcessor {
    pub fn new(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let path = path.to_string();
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let processor_config_path = path.to_string() + "/processor_config.json";
        assert!(
            std::path::Path::new(&processor_config_path).exists(),
            "processor_config.json not exists in model path"
        );
        let processor_cfg: GlmAsrNanoProcessorConfig =
            serde_json::from_slice(&std::fs::read(processor_config_path)?)?;
        let audio_token = processor_cfg.audio_token.clone();
        // let audio_token_id = 59260u32;
        let max_audio_len = processor_cfg.max_audio_len;
        // let default_transcription_prompt = processor_cfg.default_transcription_prompt.clone();
        let sampling_rate = processor_cfg.feature_extractor.sampling_rate;
        let chunk_length = processor_cfg.feature_extractor.chunk_length;
        let n_samples = processor_cfg.feature_extractor.n_samples;
        // let n_fft = processor_cfg.feature_extractor.n_fft;
        let hop_length = processor_cfg.feature_extractor.hop_length;
        // let window = create_hann_window(n_fft, dtype, device)?;
        // let window = window.unsqueeze(0)?.unsqueeze(0)?;
        // let mel_filters = mel_filter_bank(
        //     1 + n_fft / 2,
        //     processor_cfg.feature_extractor.feature_size,
        //     0.0,
        //     8000.0,
        //     sampling_rate as f32,
        //     Some("slaney"),
        //     crate::utils::audio_utils::MelScale::Slaney,
        //     false,
        //     device,
        // )?
        // .t()?;
        let whisper_feature_extrator = WhisperFeatureExtractor::new(
            processor_cfg.feature_extractor.feature_size,
            processor_cfg.feature_extractor.hop_length,
            // processor_cfg.feature_extractor.chunk_length,
            processor_cfg.feature_extractor.n_fft,
            processor_cfg.feature_extractor.dither,
            // processor_cfg.feature_extractor.padding_value,
            processor_cfg.feature_extractor.sampling_rate,
            device,
        )?;
        Ok(Self {
            sampling_rate,
            chunk_length,
            n_samples,
            // n_fft,
            // window,
            // mel_filters,
            hop_length,
            audio_token,
            // audio_token_id,
            max_audio_len,
            // default_transcription_prompt,
            device: device.clone(),
            whisper_feature_extrator,
        })
    }

    // pub fn extract_fbank_features(&self, waveform: &Tensor) -> Result<Tensor> {
    //     let pad = self.n_fft / 2;
    //     let waveform = pad_reflect_last_dim(waveform, (pad, pad))?;
    //     let (_, samples) = waveform.dims2()?;

    //     // // (bs, n_frames, n_fft)
    //     // let frames = extract_frames(&waveform, self.n_fft, self.hop_length)?;
    //     // // 应用汉明窗口
    //     // let result = frames.broadcast_mul(&self.window)?;
    //     // // 傅立叶变换
    //     // let magnitudes = apply_stft(&result)?.transpose(D::Minus1, D::Minus2)?;
    //     let magnitudes = torch_stft(&waveform, self.n_fft, self.hop_length, &self.window)?
    //         .transpose(D::Minus1, D::Minus2)?;
    //     let n_frames = (samples - self.n_fft) / self.hop_length + 1;
    //     let magnitudes = magnitudes.narrow(D::Minus1, 0, n_frames - 1)?;
    //     let mel_spec = self.mel_filters.broadcast_matmul(&magnitudes)?;
    //     let mel_spec = mel_spec.clamp(1e-10f32, f32::INFINITY)?;
    //     // let ln_spec = mel_spec.log()?;
    //     // let log10_spec = ln_spec.broadcast_div(&Tensor::new(f32::ln(10.0), mel_spec.device())?)?;
    //     let log10_spec = log10(&mel_spec)?;
    //     let max_val = log10_spec.max_all()?.affine(1.0, -8.0)?;
    //     let log10_spec = log10_spec.broadcast_maximum(&max_val)?;
    //     let log_spec = log10_spec.affine(1.0, 4.0)?.affine(1.0 / 4.0, 0.0)?;
    //     Ok(log_spec)
    // }

    pub fn feature_extractor(&self, raw_speech: Vec<Tensor>) -> Result<(Tensor, Tensor)> {
        let mut pad_audio = vec![];
        let mut input_features_mask = vec![];
        for audio in raw_speech {
            let audio_len = audio.dim(0)?;
            let pad_num = self.n_samples - audio_len;

            let audio_pad = if pad_num > 0 {
                audio.pad_with_zeros(0, 0, pad_num)?
            } else {
                audio
            };
            // (n_samples) -> (1, n_samples)
            let audio_pad = audio_pad.unsqueeze(0)?;
            pad_audio.push(audio_pad);
            let mut mask = vec![1u32; audio_len];
            if pad_num > 0 {
                mask.extend_from_slice(&vec![0u32; pad_num]);
            }
            input_features_mask.push(mask);
        }
        let input_features = Tensor::cat(&pad_audio, 0)?;
        let input_features_mask = Tensor::new(input_features_mask, input_features.device())?;
        // let input_features = self.extract_fbank_features(&input_features)?;
        let (input_features, _) =
            self.whisper_feature_extrator
                .call(&input_features, self.sampling_rate, false)?;
        let (_, audio_len) = input_features_mask.dims2()?;
        let mask_idx: Vec<u32> = (0..audio_len)
            .step_by(self.hop_length)
            .map(|i| i as u32)
            .collect();
        let mask_idx = Tensor::new(mask_idx, &self.device)?;
        let input_features_mask = input_features_mask.index_select(&mask_idx, D::Minus1)?;
        Ok((input_features, input_features_mask))
    }

    pub fn process_audio(&self, audios: Vec<Tensor>) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let window_size = self.sampling_rate * self.chunk_length;
        let max_windows = self.max_audio_len / self.chunk_length;
        let mut per_sample_windows = vec![];
        let mut flat_chunks = vec![];
        for audio_el in audios {
            let audio_el = if audio_el.rank() == 2 {
                audio_el.squeeze(0)?
            } else {
                audio_el
            };
            let n_samples = audio_el.dim(0)?;
            let n_win = ((n_samples + window_size - 1) / window_size).max(1);
            let n_win = if n_win > max_windows {
                max_windows
            } else {
                n_win
            };
            per_sample_windows.push(n_win);
            let time_cap = (n_win * window_size).min(n_samples);
            for i in 0..n_win {
                let start = i * window_size;
                let end = ((i + 1) * window_size).min(time_cap);
                flat_chunks.push(audio_el.i(start..end)?);
            }
        }
        let (input_features, input_features_mask) = self.feature_extractor(flat_chunks)?;
        Ok((input_features, input_features_mask, per_sample_windows))
    }

    pub fn get_audio_token_length(&self, audio_lens: Vec<u32>) -> Result<Vec<u32>> {
        let merge_factor = 4;
        let audio_lens = audio_lens
            .iter()
            .map(|i| (i + 2 - 3) + 1) // (pad=1, ks=3, stride=1)
            .collect::<Vec<u32>>()
            .iter()
            .map(|i| (i + 2 - 3) / 2 + 1) // (pad=1, ks=3, stride=2)
            .collect::<Vec<u32>>();
        let num_tokens = audio_lens
            .iter()
            .map(|i| (i - merge_factor) / merge_factor + 1)
            .collect::<Vec<u32>>();
        Ok(num_tokens)
    }

    pub fn process_info(
        &self,
        mes: &ChatCompletionParameters,
        render_text: &str,
    ) -> Result<(Tensor, Vec<u32>, String)> {
        let audio_tensors = extract_audios(mes, &self.device, Some(self.sampling_rate))?;
        let (input_features, input_features_mask, per_sample_windows) =
            self.process_audio(audio_tensors)?;
        let audio_lengths = input_features_mask.sum(D::Minus1)?;
        let audio_vec = split_tensor(&audio_lengths, &per_sample_windows, 0)?;
        let audio_vec: Vec<u32> = audio_vec
            .iter()
            .map(|t| t.sum_all().unwrap().to_scalar::<u32>().unwrap())
            .collect();

        let audio_token_lengths = self.get_audio_token_length(audio_vec)?;
        let mut text = render_text.to_string();
        for audio_len in audio_token_lengths.clone() {
            let replace = "<|placeholder|>".repeat(audio_len as usize);
            text = text.replacen(&self.audio_token, &replace, 1);
        }
        text = text.replace("<|placeholder|>", &self.audio_token);
        Ok((input_features, audio_token_lengths, text))
    }
}
