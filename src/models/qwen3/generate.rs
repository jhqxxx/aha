use crate::models::common::generate::{GenerationDataProvider, PrepareData};

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::models::qwen3::config::{Qwen3Config, Qwen3GenerationConfig};
use crate::models::qwen3::model::Qwen3Model;
use crate::utils::{find_type_files, get_device, get_dtype};
use crate::{chat_template::ChatTemplate, tokenizer::TokenizerModel};

pub struct Qwen3GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    model: Qwen3Model,
    device: Device,
    generation_config: Qwen3GenerationConfig,
    model_name: String,
}

impl<'a> Qwen3GenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let model = Qwen3Model::new(&cfg, vb, generation_config.eos_token_id.clone())?;

        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3")
            .to_string();
        Ok(Qwen3GenerateModel {
            chat_template,
            tokenizer,
            model,
            device: device.clone(),
            generation_config,
            model_name,
        })
    }
}

impl<'a> GenerationDataProvider for Qwen3GenerateModel<'a> {
    fn get_temperature(&self, req_temp: Option<f32>) -> Option<f32> {
        Some(req_temp.unwrap_or(self.generation_config.temperature))
    }

    fn get_top_p(&self, req_top_p: Option<f32>) -> Option<f32> {
        Some(req_top_p.unwrap_or(self.generation_config.top_p))
    }

    fn get_top_k(&self, top_k: Option<usize>) -> Option<usize> {
        Some(top_k.unwrap_or(self.generation_config.top_k))
    }

    fn get_data(&self, mes: &crate::params::chat::ChatCompletionParameters) -> Result<PrepareData> {
        let mes_render = self.chat_template.apply_chat_template(mes)?;
        let in_reasoning = self.is_in_reasoning(&mes_render);
        let input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let multi_model_data = self.get_multi_model_data();
        Ok(PrepareData {
            in_reasoning,
            input_ids,
            multi_model_data,
        })
    }
}

crate::impl_generate_model!(Qwen3GenerateModel<'a>);
