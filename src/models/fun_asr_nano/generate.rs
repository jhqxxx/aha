use std::collections::HashMap;

use crate::{
    models::common::{
        MultiModalData,
        generate::{GenerationDataProvider, PrepareData},
    },
    params::chat::ChatCompletionParameters,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, pickle::read_all_with_key};
use candle_nn::VarBuilder;

use crate::{
    models::{
        fun_asr_nano::{
            config::FunASRNanoConfig, model::FunAsrNanoModel, processor::FunAsrNanoProcessor,
        },
        qwen3::config::{Qwen3Config, Qwen3GenerationConfig},
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub struct FunAsrNanoGenerateModel {
    tokenizer: TokenizerModel,
    processor: FunAsrNanoProcessor,
    model: FunAsrNanoModel,
    device: Device,
    dtype: DType,
    generation_config: Qwen3GenerationConfig,
    model_name: String,
}

impl FunAsrNanoGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let llm_config_path = path.to_string() + "/Qwen3-0.6B";
        let tokenizer = TokenizerModel::init(&llm_config_path)?;
        let generation_config_path = llm_config_path.clone() + "/generation_config.json";
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let config_path = llm_config_path + "/config.json";
        let llm_cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let config_path = path.to_string() + "/config.yaml";
        let cfg: FunASRNanoConfig = serde_yaml::from_slice(&std::fs::read(config_path)?)?;
        let cfg_dtype = cfg.llm_conf.llm_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let processor = FunAsrNanoProcessor::new(&cfg.frontend_conf, &device)?;
        let model_list = find_type_files(path, "pt")?;
        let mut dict_to_hashmap = HashMap::new();
        for m in model_list {
            let dict = match read_all_with_key(m.clone(), Some("state_dict")) {
                Ok(dict) => dict,
                Err(e) => {
                    println!(
                        "model read_all_with_key {} get state_dict err: {}, use None try again",
                        &m, e
                    );
                    match read_all_with_key(m.clone(), None) {
                        Ok(dict) => dict,
                        Err(e) => {
                            return Err(anyhow!(format!(
                                "model read_all_with_key({}, None): e: {}",
                                &m, e
                            )));
                        }
                    }
                }
            };
            for (k, v) in dict {
                dict_to_hashmap.insert(k, v);
            }
        }
        let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &device);
        let model =
            FunAsrNanoModel::new(vb, &cfg, &llm_cfg, generation_config.eos_token_id.clone())?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("fun-asr-nano")
            .to_string();
        Ok(Self {
            tokenizer,
            processor,
            model,
            device,
            dtype,
            generation_config,
            model_name,
        })
    }
}

impl GenerationDataProvider for FunAsrNanoGenerateModel {
    fn get_temperature(&self, req_temp: Option<f32>) -> Option<f32> {
        Some(req_temp.unwrap_or(self.generation_config.temperature))
    }

    fn get_top_p(&self, req_top_p: Option<f32>) -> Option<f32> {
        Some(req_top_p.unwrap_or(self.generation_config.top_p))
    }

    fn get_top_k(&self, top_k: Option<usize>) -> Option<usize> {
        Some(top_k.unwrap_or(self.generation_config.top_k))
    }
    fn get_data(&self, mes: &ChatCompletionParameters) -> Result<PrepareData> {
        let (speech, fbank_mask, input_ids) = self.processor.process_info(mes, &self.tokenizer)?;
        let speech = speech.to_dtype(self.dtype)?;
        let data_vec = vec![speech.into(), fbank_mask.into()];
        let multi_model_data = MultiModalData::new(data_vec);

        Ok(PrepareData {
            in_reasoning: false,
            input_ids,
            multi_model_data,
        })
    }
}

crate::impl_generate_model!(FunAsrNanoGenerateModel);
