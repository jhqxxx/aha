use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor, pickle::read_all_with_key};
use candle_nn::VarBuilder;

use crate::{
    models::{
        campplus::CAMPPlus,
        feature_extractor::seamless_m4t_feature_extractor::SeamlessM4TFeatureExtractor,
        index_tts2::config::{IndexTTS2Config, PreprocessParams},
        mask_gct::model::RepCodec,
        w2v_bert_2_0::model::W2VBert2_0Model,
    },
    utils::{
        audio_utils::{
            create_hann_window, extract_audio_url, get_waveform_and_window_properties, kaldi_fbank,
            kaldi_get_mel_banks, load_audio, mel_filter_bank, resample_simple, torch_stft,
        },
        get_vb_model_path,
        tensor_utils::pad_reflect_last_dim,
    },
};

pub struct IndexTTS2Processor {
    device: Device,
    max_audio_length_seconds: usize,
    feature_extractor: SeamlessM4TFeatureExtractor,
    semantic_model: W2VBert2_0Model,
    semantic_mean: Tensor,
    semantic_std: Tensor,
    semantic_codec: RepCodec,
    s2mel_filters: Tensor,
    s2mel_windows: Tensor,
    s2mel_preprocess_params: PreprocessParams,
    window_shift: usize,
    window_size: usize,
    padded_window_size: usize,
    mel_energies: Tensor,
    campplus_model: CAMPPlus,
}

impl IndexTTS2Processor {
    pub fn new(
        path: &str,
        save_dir: &str,
        config: &IndexTTS2Config,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let feature_extractor = SeamlessM4TFeatureExtractor::new(
            // 80,
            80,
            crate::utils::tensor_utils::PaddingSide::Right,
            1.0,
            16000,
            2,
            device,
        )?;
        let w2vbert2_path = save_dir.to_string() + "/facebook/w2v-bert-2.0";
        let semantic_model = W2VBert2_0Model::init(&w2vbert2_path, device, dtype)?;
        let semantic_mean_var_path = path.to_string() + "/" + &config.w2v_stat;
        let dict = read_all_with_key(semantic_mean_var_path, None)?;
        let mut semantic_mean = Tensor::new(0.0, device)?.to_dtype(dtype)?;
        let mut semantic_std = Tensor::new(1.0, device)?.to_dtype(dtype)?;
        for (k, v) in dict {
            if k.eq("mean") {
                semantic_mean = v.to_device(device)?.to_dtype(dtype)?;
            } else if k.eq("var") {
                semantic_std = v.to_device(device)?.to_dtype(dtype)?.sqrt()?;
            }
        }

        let semantic_codec_path =
            save_dir.to_string() + "/amphion/MaskGCT/semantic_codec/model.safetensors";
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[semantic_codec_path], dtype, &device)? };
        let semantic_codec = RepCodec::new(vb, &config.semantic_codec)?;
        let s2mel_filters = mel_filter_bank(
            config.s2mel.preprocess_params.spect_params.n_fft / 2 + 1,
            config.s2mel.preprocess_params.spect_params.n_mels,
            config.s2mel.preprocess_params.spect_params.fmin as f32,
            config
                .s2mel
                .preprocess_params
                .spect_params
                .fmax
                .unwrap_or(config.s2mel.preprocess_params.sr / 2) as f32,
            config.s2mel.preprocess_params.sr as f32,
            Some("slaney"),
            crate::utils::audio_utils::MelScale::Slaney,
            false,
            device,
        )?
        .t()?;
        let s2mel_windows = create_hann_window(
            config.s2mel.preprocess_params.spect_params.win_length,
            dtype,
            device,
        )?;
        let (window_shift, window_size, padded_window_size) =
            get_waveform_and_window_properties(16000, 10.0, 25.0, true)?;
        let (mel_energies, _) =
            kaldi_get_mel_banks(80, padded_window_size, 16000 as f32, 20.0, 0.0, device)?;
        let mel_energies = mel_energies.pad_with_zeros(D::Minus1, 0, 1)?.t()?;
        let campplus_model_path = save_dir.to_string()
            + "/iic/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin";
        let campplus_vb = get_vb_model_path(campplus_model_path, dtype, device.clone(), None)?;
        let campplus_model = CAMPPlus::new(campplus_vb, 80, 192, 32, 4, 128)?;
        Ok(Self {
            device: device.clone(),
            max_audio_length_seconds: 15,
            feature_extractor,
            semantic_model,
            semantic_mean,
            semantic_std,
            semantic_codec,
            s2mel_filters,
            s2mel_windows,
            s2mel_preprocess_params: config.s2mel.preprocess_params.clone(),
            window_shift,
            window_size,
            padded_window_size,
            mel_energies,
            campplus_model,
        })
    }

    pub fn cut_audio(&self, audio: &Tensor, sr: usize) -> Result<(Tensor, usize)> {
        let max_audio_samples = self.max_audio_length_seconds * sr;
        let audio_lens = audio.dim(1)?;
        let audio = if audio_lens > max_audio_samples {
            audio.i((.., 0..max_audio_samples))?
        } else {
            audio.clone()
        };
        Ok((audio, sr))
    }

    pub fn extract_audio_and_cut(
        &self,
        mes: &ChatCompletionParameters,
        device: &Device,
    ) -> Result<(Tensor, usize)> {
        let audio_url_vec = extract_audio_url(mes);
        let (audio, sr) = load_audio(&audio_url_vec[0], device)?;
        self.cut_audio(&audio, sr)
    }

    pub fn get_emb(
        &self,
        input_features: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let output =
            self.semantic_model
                .forward(input_features, attention_mask, Some(17), false)?;
        let feature = &output.specify_layer_id_hidden_state.unwrap();
        let feature = feature
            .broadcast_sub(&self.semantic_mean)?
            .broadcast_div(&self.semantic_std)?;
        Ok(feature)
    }

    pub fn s2mel_spectrogram(&self, waveform: &Tensor) -> Result<Tensor> {
        let pad = (self.s2mel_preprocess_params.spect_params.n_fft
            - self.s2mel_preprocess_params.spect_params.hop_length)
            / 2;
        let pad_audio_22k = pad_reflect_last_dim(&waveform, (pad, pad))?;
        let spec = torch_stft(
            &pad_audio_22k,
            self.s2mel_preprocess_params.spect_params.n_fft,
            self.s2mel_preprocess_params.spect_params.hop_length,
            &self.s2mel_windows,
        )?
        .transpose(1, 2)?;
        let spec = self.s2mel_filters.broadcast_matmul(&spec)?;
        let spec = spec.clamp(1e-5, f64::INFINITY)?.log()?;
        Ok(spec)
    }
    pub fn process_info(&self, mes: &ChatCompletionParameters) -> Result<()> {
        let (audio, sr) = self.extract_audio_and_cut(mes, &self.device)?;
        let audio_22k = resample_simple(&audio, sr as i64, 22050)?;
        let audio_16k = resample_simple(&audio, sr as i64, 16000)?;
        let (audio_16k_features, audio_16k_mask) =
            self.feature_extractor.call(&audio_16k, 16000, true, true)?;
        let spk_cond_emb = self.get_emb(&audio_16k_features, audio_16k_mask.as_ref())?;
        let (_, s_ref) = self.semantic_codec.quantize(&spk_cond_emb)?;
        let ref_mel = self.s2mel_spectrogram(&audio_22k)?;
        let feat = kaldi_fbank(
            &audio_16k,
            &self.mel_energies,
            self.window_shift,
            self.window_size,
            self.padded_window_size,
            0.0,
        )?;
        let feat = feat.broadcast_sub(&feat.mean_keepdim(1)?)?;
        let style = self.campplus_model.forward(&feat)?;
        println!("style: {}", style);
        Ok(())
    }
}
