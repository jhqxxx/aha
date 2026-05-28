use crate::models::common::generate::{GenerationDataProvider, PrepareData};

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::models::minicpm4::config::MiniCPM4Config;
use crate::models::minicpm4::model::MiniCPMModel;
use crate::utils::{find_type_files, get_device, get_dtype};
use crate::{chat_template::ChatTemplate, tokenizer::TokenizerModel};

pub struct MiniCPM4GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    model: MiniCPMModel,
    device: Device,
    model_name: String,
}

impl<'a> MiniCPM4GenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: MiniCPM4Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let model = MiniCPMModel::new(vb, cfg)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("minicpm4")
            .to_string();
        Ok(MiniCPM4GenerateModel {
            chat_template,
            tokenizer,
            model,
            device: device.clone(),
            model_name,
        })
    }
}

impl<'a> GenerationDataProvider for MiniCPM4GenerateModel<'a> {
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

crate::impl_generate_model!(MiniCPM4GenerateModel<'a>);
