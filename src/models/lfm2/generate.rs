use crate::models::common::generate::{GenerationDataProvider, PrepareData};
use crate::{
    chat_template::ChatTemplate,
    models::lfm2::{
        config::{Lfm2Config, Lfm2GenerateConfig},
        model::Lfm2Model,
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub struct Lfm2GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    device: Device,
    model: Lfm2Model,
    model_name: String,
}
impl<'a> Lfm2GenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let device = get_device(device);
        let gen_cfg_path = path.to_string() + "/generation_config.json";
        let gen_cfg: Lfm2GenerateConfig = serde_json::from_slice(&std::fs::read(gen_cfg_path)?)?;
        let cfg_path = path.to_string() + "/config.json";
        let cfg: Lfm2Config = serde_json::from_slice(&std::fs::read(cfg_path)?)?;
        let model_path = find_type_files(path, "safetensors")?;
        let cfg_dtype = if let Some(dtype) = &cfg.dtype {
            dtype.clone()
        } else if let Some(dtype) = &cfg.torch_dtype {
            dtype.clone()
        } else {
            "bfloat16".to_string()
        };
        let dtype = get_dtype(dtype, &cfg_dtype);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_path, dtype, &device)? };
        let eos_ids = vec![gen_cfg.eos_token_id];
        let model = Lfm2Model::new(vb, &cfg, eos_ids)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("lfm2")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            device,
            model,
            model_name,
        })
    }
}

impl<'a> GenerationDataProvider for Lfm2GenerateModel<'a> {
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

crate::impl_generate_model!(Lfm2GenerateModel<'a>);
