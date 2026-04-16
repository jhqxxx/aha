use std::cmp::min;

use anyhow::{Result, anyhow};
use candle_core::{D, Device, IndexOp, Tensor};
use kaldi_native_fbank::{
    FbankComputer, FbankOptions,
    window::{Window, extract_window, num_frames},
};

use crate::{
    models::{
        common::modules::native_conv1d,
        fire_red_vad::config::{CMVNData, FireRedVadConfig},
    },
    utils::{audio_utils::load_audio_with_resample, tensor_utils::apply_threshold},
};
pub struct CMVN {
    dim: usize,
    means: Tensor,
    inverse_std_variances: Tensor,
}

impl CMVN {
    pub fn new(path: &str, device: &Device) -> Result<Self> {
        let cmvn_path = path.to_string() + "/cmvn.json";
        assert!(
            std::path::Path::new(&cmvn_path).exists(),
            "cmvn path file not exists"
        );
        let cmvn: CMVNData = serde_json::from_slice(&std::fs::read(cmvn_path)?)?;
        let cmvn_data = Tensor::new(cmvn.cmvn, device)?;
        assert_eq!(cmvn_data.rank(), 2);
        assert_eq!(cmvn_data.dim(0)?, 2);
        let (_, dim) = cmvn_data.dims2()?;
        let dim = dim - 1;
        let count = cmvn_data.i((0, dim))?.to_scalar::<f32>()?;
        assert!(count >= 1.0);
        let floor = 1e-20f32;
        let means = cmvn_data.i((0, 0..dim))?.affine(1.0 / count as f64, 0.0)?;
        let variance = cmvn_data
            .i((1, 0..dim))?
            .affine(1.0 / count as f64, 0.0)?
            .sub(&means.powf(2.0)?)?
            .clamp(floor, f32::MAX)?;
        let inverse_std_variances = (1.0 / variance.sqrt()?)?;
        Ok(Self {
            dim,
            means,
            inverse_std_variances,
        })
    }

    pub fn call(&self, xs: &Tensor) -> Result<Tensor> {
        assert_eq!(xs.dim(D::Minus1)?, self.dim, "CMVN dim mismatch");
        let xs = xs.broadcast_sub(&self.means)?;
        let xs = xs.broadcast_mul(&self.inverse_std_variances)?;
        Ok(xs)
    }
}

pub struct KaldifeatFbank {
    opts: FbankOptions,
    win: Window,
}

impl KaldifeatFbank {
    pub fn new(num_mel_bins: usize, dither: f32) -> Result<Self> {
        let mut opts = FbankOptions::default();
        opts.frame_opts.samp_freq = 16000.0;
        opts.frame_opts.frame_length_ms = 25.0;
        opts.frame_opts.frame_shift_ms = 10.0;
        opts.frame_opts.dither = dither;
        opts.frame_opts.snip_edges = true;
        opts.mel_opts.num_bins = num_mel_bins;
        opts.mel_opts.debug_mel = false;
        opts.use_energy = false;
        let win = Window::new(&opts.frame_opts)
            .ok_or("window new error")
            .map_err(|e| anyhow!("fbank comput err: {e}"))?;
        Ok(Self { opts, win })
    }

    pub fn call(&self, wav_tensor: &Tensor) -> Result<Tensor> {
        let mut comp =
            FbankComputer::new(self.opts.clone()).map_err(|e| anyhow!("fbank comput err: {e}"))?;

        let padded = self.opts.frame_opts.padded_window_size();
        let wave = if wav_tensor.rank() == 1 {
            wav_tensor.to_vec1::<f32>()?
        } else if wav_tensor.rank() == 2 {
            wav_tensor.squeeze(0)?.to_vec1::<f32>()?
        } else {
            return Err(anyhow!("not support wav dim: {}", wav_tensor.rank()));
        };
        let mut feats = vec![];
        let mut window_buf = vec![0.0; padded];
        let frame_len = num_frames(wave.len(), &self.opts.frame_opts, true);
        for frame in 0..frame_len {
            let raw_log_energy = extract_window(
                0,
                &wave,
                frame,
                &self.opts.frame_opts,
                Some(&self.win),
                &mut window_buf,
            )
            .unwrap();
            let mut feat = vec![0.0; comp.dim()];
            comp.compute(raw_log_energy, 1.0, &mut window_buf, &mut feat);
            feats.push(feat);
        }
        let feats = Tensor::new(feats, wav_tensor.device())?;
        Ok(feats)
    }
}

pub struct AudioFeat {
    cmvn: CMVN,
    fbank: KaldifeatFbank,
}

impl AudioFeat {
    pub fn new(path: &str, device: &Device) -> Result<Self> {
        let cmvn = CMVN::new(path, device)?;
        let fbank = KaldifeatFbank::new(80, 0.0)?;
        Ok(Self { cmvn, fbank })
    }

    pub fn extract(&self, wave_tensor: &Tensor) -> Result<Tensor> {
        let fbank = self.fbank.call(wave_tensor)?;
        let fbank = self.cmvn.call(&fbank)?;
        Ok(fbank)
    }

    pub fn extract_file(&self, audio_path: &str, device: &Device) -> Result<(Tensor, f32)> {
        let wave_tensor =
            load_audio_with_resample(audio_path, device, Some(16000), true)?.squeeze(0)?;
        let dur = wave_tensor.dim(0)? as f32 / 16000.0;
        let fbank = self.extract(&wave_tensor)?;
        Ok((fbank, dur))
    }
}

pub enum VadState {
    SILENCE,
    POSSIBLESPEECH,
    SPEECH,
    POSSIBLESILENCE,
}

pub struct VadPostprocessor {
    pub smooth_window_size: usize,
    pub prob_threshold: f32,
    pub pad_start_frame: usize,
    pub min_speech_frame: usize,
    pub max_speech_frame: usize,
    pub min_silence_frame: usize,
    pub merge_silence_frame: usize,
    pub extend_speech_frame: usize,
    pub frame_shift_s: f32,
    pub frame_cnt: usize,
    pub state: VadState,
}

impl VadPostprocessor {
    pub fn new(cfg: &FireRedVadConfig) -> Self {
        Self {
            smooth_window_size: cfg.smooth_window_size,
            prob_threshold: cfg.speech_threshold,
            pad_start_frame: cfg.pad_start_frame,
            min_speech_frame: cfg.min_speech_frame,
            max_speech_frame: cfg.max_speech_frame,
            min_silence_frame: cfg.min_silence_frame,
            merge_silence_frame: cfg.merge_silence_frame,
            extend_speech_frame: cfg.extend_speech_frame,
            frame_shift_s: 0.01,
            frame_cnt: 0,
            state: VadState::SILENCE,
        }
    }

    pub fn reset(&mut self) {
        self.frame_cnt = 0;
    }

    pub fn process_one(&self, probs: f32) -> Result<bool> {
        // TODO： 状态管理
        let is_speech = probs >= self.prob_threshold;
        Ok(is_speech)
    }

    pub fn process_thresh(&self, raw_probs: &Tensor) -> Result<Tensor> {
        let smoothed_probs = self.smooth_prob(raw_probs)?;
        let binary_preds = apply_threshold(&smoothed_probs, self.prob_threshold)?;
        Ok(binary_preds)
    }

    pub fn process(&self, raw_probs: &Tensor, dur: f32) -> Result<Vec<(f32, f32)>> {
        let binary_preds = self.process_thresh(raw_probs)?;
        self.decision_to_segment(&binary_preds, dur)
    }

    pub fn decision_to_segment(&self, decisions: &Tensor, dur: f32) -> Result<Vec<(f32, f32)>> {
        let mut segments = vec![];
        let mut speech_start = -1;
        let decisions = decisions.to_vec1::<u8>()?;
        for (t, &flag) in decisions.iter().enumerate() {
            if flag == 1 && speech_start == -1 {
                speech_start = t as i32;
            } else if flag == 0 && speech_start != -1 {
                segments.push((
                    speech_start as f32 * self.frame_shift_s,
                    t as f32 * self.frame_shift_s,
                ));
                speech_start = -1;
            }
        }
        if speech_start != -1 {
            let t = decisions.len() - 1;
            let end_time = dur.min(t as f32 * self.frame_shift_s);
            segments.push((speech_start as f32 * self.frame_shift_s, end_time));
        }
        Ok(segments)
    }

    fn smooth_prob(&self, probs: &Tensor) -> Result<Tensor> {
        if self.smooth_window_size <= 1 {
            Ok(probs.clone())
        } else {
            let kernel_value = 1.0 / self.smooth_window_size as f32;
            let probs_len = probs.dim(0)?;
            let weight = Tensor::new(vec![kernel_value; self.smooth_window_size], probs.device())?;
            let mut moothed = native_conv1d(probs, &weight, "full")?.i(0..probs_len)?;
            let mean_len = min(self.smooth_window_size - 1, probs_len);
            let mut mean_vec = vec![];
            for i in 0..mean_len {
                let mean = probs.i(0..i + 1)?.mean(0)?.to_scalar::<f32>()?;
                mean_vec.push(mean);
            }
            let means = Tensor::new(mean_vec, probs.device())?;
            moothed = moothed.slice_assign(&[(0..mean_len)], &means)?;
            Ok(moothed)
        }
    }
}
