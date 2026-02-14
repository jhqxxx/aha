use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::{
    models::feature_extractor::{
        config::FeatureExtractor, feature_extraction_whisper::WhisperFeatureExtractor,
    },
    tokenizer::TokenizerModel,
    utils::{
        audio_utils::{extract_audios, split_audio_into_chunks},
        capitalize_first_letter,
        tensor_utils::float_range_normalize,
    },
};

pub struct Qwen3AsrProcessor {
    device: Device,
    sample_rate: usize,
    support_language: Vec<String>,
    max_asr_input_seconds: f32,
    whisper_feature_extracor: WhisperFeatureExtractor,
    audio_token: String,
}

impl Qwen3AsrProcessor {
    pub fn new(device: &Device, config: &FeatureExtractor) -> Result<Self> {
        let support_language: Vec<String> = vec![
            "Chinese",
            "English",
            "Cantonese",
            "Arabic",
            "German",
            "French",
            "Spanish",
            "Portuguese",
            "Indonesian",
            "Italian",
            "Korean",
            "Russian",
            "Thai",
            "Vietnamese",
            "Japanese",
            "Turkish",
            "Hindi",
            "Malay",
            "Dutch",
            "Swedish",
            "Danish",
            "Finnish",
            "Polish",
            "Czech",
            "Filipino",
            "Persian",
            "Greek",
            "Romanian",
            "Hungarian",
            "Macedonian",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let whisper_feature_extracor = WhisperFeatureExtractor::new(
            config.feature_size,
            config.hop_length,
            // config.chunk_length,
            config.n_fft,
            config.dither,
            // config.padding_value,
            config.sampling_rate,
            device,
        )?;
        Ok(Self {
            device: device.clone(),
            sample_rate: 16000,
            support_language,
            max_asr_input_seconds: 1200.0,
            whisper_feature_extracor,
            audio_token: "<|audio_pad|>".to_string(),
        })
    }

    pub fn process_audio(&self, mes: &ChatCompletionParameters) -> Result<Vec<Tensor>> {
        let audio_tensors = extract_audios(mes, &self.device, Some(self.sample_rate))?;
        audio_tensors.iter().map(float_range_normalize).collect()
    }

    pub fn validate_language(&self, lang: &String) -> bool {
        self.support_language.contains(lang)
    }

    fn replace_special_tokens(&self, text: &str, token_len: usize) -> String {
        let replace = "<|audio_placeholder|>".repeat(token_len);
        let text = text.replacen(&self.audio_token, &replace, 1);
        text.replace("<|audio_placeholder|>", &self.audio_token)
    }

    pub fn process_info(
        &self,
        mes: &ChatCompletionParameters,
        render: &str,
        tokenizer: &TokenizerModel,
    ) -> Result<Vec<AudioData>> {
        let audio_count = render
            .matches("<|audio_start|><|audio_pad|><|audio_end|>")
            .count();
        let mut render = if audio_count > 1 {
            render.replace(
                &"<|audio_start|><|audio_pad|><|audio_end|>".repeat(audio_count),
                "<|audio_start|><|audio_pad|><|audio_end|>",
            )
        } else {
            render.to_string()
        };
        if let Some(map) = &mes.metadata
            && map.contains_key("language")
        {
            let lang = map.get("language").unwrap();
            let lang = capitalize_first_letter(lang);
            if self.validate_language(&lang) {
                render = format!("{}language {}'<asr_text>'", render, lang);
            }
        }
        let audio_tensors = self.process_audio(mes)?;
        let audio_len = audio_tensors.len();
        if audio_len != audio_count {
            return Err(anyhow::anyhow!("audio_pad num != audio num"));
        }
        let mut split_wavs = vec![];
        for wav in audio_tensors.iter() {
            let wavs = split_audio_into_chunks(wav, self.sample_rate, self.max_asr_input_seconds)?;
            split_wavs.extend_from_slice(&wavs);
        }
        let mut audio_datas = vec![];
        for wav in split_wavs.iter() {
            let (input_features, _) =
                self.whisper_feature_extracor
                    .call(wav, self.sample_rate, false)?;
            let audio_len = input_features.dim(2)?;
            let output_len = get_feat_extract_output_lengths(audio_len);
            let text = self.replace_special_tokens(&render, output_len);
            let input_ids = tokenizer.text_encode(text, &self.device)?;
            let input_features = input_features.squeeze(0)?;
            let audio = AudioData {
                input_features,
                input_ids,
            };
            audio_datas.push(audio);
        }
        Ok(audio_datas)
    }
}

pub struct AudioData {
    pub input_features: Tensor,
    pub input_ids: Tensor,
}

pub fn get_feat_extract_output_lengths(audio_len: usize) -> usize {
    let input_len_leave = audio_len % 100;
    if input_len_leave > 0 {
        let feat_lengths = (input_len_leave - 1) / 2 + 1;
        ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1 + (audio_len / 100) * 13
    } else {
        (audio_len / 100) * 13
    }
}
