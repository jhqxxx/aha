use std::collections::HashMap;

use crate::{
    models::{
        moss_audio_tokenizer_nano::{MossAudioTokenizer, config::MossAudioTokenizerConfig},
        moss_tts_nano::{
            config::MossTTSConfig,
            model::{MossTTSMode, MossTTSModel},
            processor::MossTTSProcessor,
        },
    },
    utils::{find_type_files, get_device, get_dtype},
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, pickle::read_all_with_key};
use candle_nn::VarBuilder;
use sentencepiece::SentencePieceProcessor;

pub struct MossTTSGenerate {
    pub audio_tokenizer: MossAudioTokenizer,
    pub text_tokenizer: SentencePieceProcessor,
    pub processor: MossTTSProcessor,
    pub model: MossTTSModel,
    pub device: Device,
}

impl MossTTSGenerate {
    pub fn init(
        tts_path: &str,
        audio_tokenizer_path: &str,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let audio_tokenizer_config_path = audio_tokenizer_path.to_string() + "/config.json";
        let audio_tokenizer_cfg: MossAudioTokenizerConfig =
            serde_json::from_slice(&std::fs::read(audio_tokenizer_config_path)?)?;
        let model_list = find_type_files(audio_tokenizer_path, "safetensors")?;
        let audio_dtype = get_dtype(dtype, &audio_tokenizer_cfg.dtype);
        let device = get_device(device);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, audio_dtype, &device)? };
        let audio_tokenizer = MossAudioTokenizer::new(vb, &audio_tokenizer_cfg)?;
        let text_tokenizer_path = tts_path.to_string() + "/tokenizer.model";
        let text_tokenizer = SentencePieceProcessor::open(text_tokenizer_path)
            .map_err(|e| anyhow!(format!("load bpe.model file error:{}", e)))?;
        let tts_cfg_path = tts_path.to_string() + "/config.json";
        let tts_cfg: MossTTSConfig = serde_json::from_slice(&std::fs::read(tts_cfg_path)?)?;
        let processor = MossTTSProcessor::new(
            &tts_cfg,
            audio_tokenizer_cfg.sample_rate,
            audio_tokenizer_cfg.number_channels,
            &text_tokenizer,
        )?;
        let model_list = find_type_files(tts_path, "bin")?;
        let mut dict_to_hashmap = HashMap::new();
        let m_dtype = get_dtype(dtype, "bfloat16");
        for m in model_list {
            let dict = read_all_with_key(m, None)?;
            for (k, v) in dict {
                dict_to_hashmap.insert(k, v);
            }
        }
        let vb = VarBuilder::from_tensors(dict_to_hashmap, m_dtype, &device);
        let model = MossTTSModel::new(vb, &tts_cfg)?;

        Ok(Self {
            audio_tokenizer,
            text_tokenizer,
            processor,
            model,
            device,
        })
    }

    pub fn generate(
        &mut self,
        text: &str,
        prompt_audio_path: Option<&str>,
        prompt_text: Option<&str>,
        mode: Option<MossTTSMode>,
    ) -> Result<()> {
        let mode = self.processor.resolved_mode(
            mode,
            prompt_text.is_some(),
            prompt_audio_path.is_some(),
        )?;
        let input_ids = self.processor.build_inference_input_ids(
            text,
            prompt_audio_path,
            prompt_text,
            mode.clone(),
            &self.audio_tokenizer,
            &self.text_tokenizer,
            &self.device,
        )?;
        self.model.generate(&input_ids, &self.audio_tokenizer)?;
        Ok(())
    }
}
