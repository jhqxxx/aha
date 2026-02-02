use anyhow::Result;
use candle_core::{D, Device, Tensor};

use crate::utils::{
    audio_utils::{create_povey_window, mel_filter_bank, spectrogram},
    tensor_utils::{PaddingSide, z_score_normalize},
};

pub struct SeamlessM4TFeatureExtractor {
    // feature_size: usize,
    num_mel_bins: usize,
    padding_side: PaddingSide,
    padding_value: f32,
    sampling_rate: usize,
    stride: usize,
    mel_filters: Tensor,
    window: Tensor,
}

impl SeamlessM4TFeatureExtractor {
    pub fn new(
        // feature_size: usize,
        num_mel_bins: usize,
        padding_side: PaddingSide,
        padding_value: f32,
        sampling_rate: usize,
        stride: usize,
        device: &Device,
    ) -> Result<Self> {
        let mel_filters = mel_filter_bank(
            257,
            num_mel_bins,
            20.0,
            (sampling_rate / 2) as f32,
            sampling_rate as f32,
            None,
            crate::utils::audio_utils::MelScale::Kaldi,
            true,
            device,
        )?;
        let window = create_povey_window(400, candle_core::DType::F32, device)?;
        Ok(Self {
            // feature_size,
            num_mel_bins,
            padding_side,
            padding_value,
            sampling_rate,
            stride,
            mel_filters,
            window,
        })
    }

    pub fn call(
        &self,
        raw_speech: &Tensor,
        sampling_rate: usize,
        do_normalize_per_mel_bins: bool,
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
        let waveform = raw_speech.affine(32768.0, 0.0)?;
        // println!("waveform: {}", waveform);
        // println!("self.mel_filters: {}", self.mel_filters);
        // println!("self.window: {}", self.window);
        let mut features = spectrogram(
            &waveform,
            &self.window,
            400,
            160,
            512,
            Some(2.0),
            false,
            0.97,
            Some(&self.mel_filters),
            Some("log"),
            1.192092955078125e-07,
            true,
        )?
        .transpose(D::Minus1, D::Minus2)?;
        if do_normalize_per_mel_bins {
            features = z_score_normalize(&features, 1)?;
        }
        let n_frame = features.dim(1)?;
        let mask_1 = n_frame / self.stride;
        let pad_len = n_frame % self.stride;
        if pad_len > 0 {
            let pad = Tensor::new(self.padding_value, features.device())?.broadcast_as((
                1,
                pad_len,
                self.num_mel_bins,
            ))?;
            match self.padding_side {
                PaddingSide::Left => features = Tensor::cat(&[pad, features], 1)?,
                PaddingSide::Right => features = Tensor::cat(&[features, pad], 1)?,
            }
        }
        let (bs, num_frames, dim) = features.dims3()?;
        let n_frames_stride = num_frames / self.stride;
        let features = features.reshape((bs, n_frames_stride, dim * self.stride))?;
        let mask_0 = n_frames_stride - mask_1;
        let mask = if return_attention_mask {
            let mut mask = Tensor::new(1u32, features.device())?.broadcast_as((1, mask_1))?;
            if mask_0 > 0 {
                let mask_pad = Tensor::new(0u32, features.device())?.broadcast_as((1, mask_0))?;
                match self.padding_side {
                    PaddingSide::Left => {
                        mask = Tensor::cat(&[mask_pad, mask], D::Minus1)?;
                    }
                    PaddingSide::Right => {
                        mask = Tensor::cat(&[mask, mask_pad], D::Minus1)?;
                    }
                }
            }
            Some(mask)
        } else {
            None
        };
        Ok((features, mask))
    }
}
