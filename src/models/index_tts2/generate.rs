use aha_openai_dive::v1::resources::chat::{ChatCompletionParameters, ChatCompletionResponse};
use anyhow::{Result, anyhow};
use base64::{Engine, prelude::BASE64_STANDARD};
use candle_core::{DType, Device};
use sentencepiece::SentencePieceProcessor;

use crate::{
    models::index_tts2::{
        config::IndexTTS2Config, model::IndexTTS2Model, utils::tokenize_by_cjk_char,
    },
    tokenizer::sentencepiece_encode,
    utils::{
        audio_utils::get_audio_wav_u8, build_audio_completion_response, extract_user_text,
        get_default_save_dir, get_device,
    },
};

pub struct IndexTTS2Generate {
    tokenizer: SentencePieceProcessor,
    // config: IndexTTS2Config,
    model: IndexTTS2Model,
    device: Device,
    sample_rate: u32,
    model_name: String,
}

#[allow(unused)]
impl IndexTTS2Generate {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let config_path = path.to_string() + "/config.yaml";
        let save_dir = get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
        let config: IndexTTS2Config = serde_yaml::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let bpe_path = path.to_string() + "/bpe.model";
        let tokenizer = SentencePieceProcessor::open(bpe_path)
            .map_err(|e| anyhow!(format!("load bpe,model file error:{}", e)))?;
        let model = IndexTTS2Model::new(path, &save_dir, &config, &device)?;
        Ok(Self {
            tokenizer,
            // config,
            model,
            device,
            sample_rate: 22050,
            model_name: "index-tts2".to_string(),
        })
    }

    pub fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let text = extract_user_text(&mes)?;
        let text = tokenize_by_cjk_char(&text, true);
        let input_ids = sentencepiece_encode(&text, &self.tokenizer, &self.device)?;

        let audio = self.model.forward(&input_ids, &mes)?;
        let wav_u8 = get_audio_wav_u8(&audio, self.sample_rate)?;
        let base64_audio = BASE64_STANDARD.encode(wav_u8);
        let response = build_audio_completion_response(&base64_audio, &self.model_name);
        Ok(response)
    }
}
