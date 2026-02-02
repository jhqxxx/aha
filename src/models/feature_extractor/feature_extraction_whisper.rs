use anyhow::Result;
use candle_core::{D, Device, Tensor};

use crate::utils::{
    audio_utils::{create_hann_window, mel_filter_bank, torch_stft},
    tensor_utils::{log10, pad_reflect_last_dim},
};

pub struct WhisperFeatureExtractor {
    feature_size: usize,
    hop_length: usize,
    chunk_length: usize,
    n_samples: usize,
    n_fft: usize,
    dither: f64,
    padding_value: f32,
    sampling_rate: usize,
    mel_filters: Tensor,
    window: Tensor,
}

impl WhisperFeatureExtractor {
    pub fn new(
        feature_size: usize,
        hop_length: usize,
        chunk_length: usize,
        n_fft: usize,
        dither: f64,
        padding_value: f32,
        sampling_rate: usize,
        device: &Device,
    ) -> Result<Self> {
        let window = create_hann_window(n_fft, candle_core::DType::F32, device)?;
        let window = window.unsqueeze(0)?.unsqueeze(0)?;
        let mel_filters = mel_filter_bank(
            1 + n_fft / 2,
            feature_size,
            0.0,
            8000.0,
            sampling_rate as f32,
            Some("slaney"),
            crate::utils::audio_utils::MelScale::Slaney,
            false,
            device,
        )?
        .t()?;
        let n_samples = chunk_length * sampling_rate;
        Ok(Self {
            feature_size,
            hop_length,
            chunk_length,
            n_samples,
            n_fft,
            dither,
            padding_value,
            sampling_rate,
            mel_filters,
            window,
        })
    }

    pub fn call(
        &self,
        raw_speech: &Tensor,
        sampling_rate: usize,
        // do_normalize: bool,
        return_attention_mask: bool,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // raw_speech: 重采样后的音频，shape: (bs, raw_len)
        // sampling_rate: 音频采样率，验证是否与模型的预处理采样率一致
        if sampling_rate != self.sampling_rate {
            return Err(anyhow::anyhow!(
                "The model feature extractor was trained sampling rate {} not equal to audio sample rate {}",
                self.sampling_rate,
                sampling_rate
            ));
        }

        let input_features = self.extract_fbank_features(raw_speech)?;
        let mask_len = input_features.dim(2)?;
        let mask = if return_attention_mask {
            let mask = Tensor::new(1u32, input_features.device())?.broadcast_as((1, mask_len))?;
            Some(mask)
        } else {
            None
        };
        Ok((input_features, mask))
    }

    pub fn extract_fbank_features(&self, waveform: &Tensor) -> Result<Tensor> {
        let mut waveform = waveform.clone();
        if self.dither != 0.0 {
            waveform = waveform.add(&waveform.randn_like(0.0, 1.0)?.affine(self.dither, 0.0)?)?;
        }
        let pad = self.n_fft / 2;
        let waveform = pad_reflect_last_dim(&waveform, (pad, pad))?;
        let (_, samples) = waveform.dims2()?;

        let magnitudes = torch_stft(&waveform, self.n_fft, self.hop_length, &self.window)?
            .transpose(D::Minus1, D::Minus2)?;
        let n_frames = (samples - self.n_fft) / self.hop_length + 1;
        let magnitudes = magnitudes.narrow(D::Minus1, 0, n_frames - 1)?;
        let mel_spec = self.mel_filters.broadcast_matmul(&magnitudes)?;
        let mel_spec = mel_spec.clamp(1e-10f32, f32::INFINITY)?;
        // let ln_spec = mel_spec.log()?;
        // let log10_spec = ln_spec.broadcast_div(&Tensor::new(f32::ln(10.0), mel_spec.device())?)?;
        let log10_spec = log10(&mel_spec)?;
        let max_val = log10_spec.max_all()?.affine(1.0, -8.0)?;
        let log10_spec = log10_spec.broadcast_maximum(&max_val)?;
        let log_spec = log10_spec.affine(1.0, 4.0)?.affine(1.0 / 4.0, 0.0)?;
        Ok(log_spec)
    }
}
