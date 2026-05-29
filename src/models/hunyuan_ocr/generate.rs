use crate::{
    models::common::{
        MultiModalData,
        generate::{GenerationDataProvider, PrepareData},
    },
    params::chat::ChatCompletionParameters,
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::{
    chat_template::ChatTemplate,
    models::hunyuan_ocr::{
        config::{HunYuanVLConfig, HunyuanOCRGenerationConfig},
        model::HunyuanVLModel,
        processor::HunyuanVLProcessor,
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub struct HunyuanOCRGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: HunyuanVLProcessor,
    model: HunyuanVLModel,
    device: Device,
    generation_config: HunyuanOCRGenerationConfig,
    model_name: String,
}

impl<'a> HunyuanOCRGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: HunYuanVLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = HunyuanVLProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: HunyuanOCRGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let model = HunyuanVLModel::new(vb, cfg.clone(), generation_config.eos_token_id.clone())?;

        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("hunyuan_ocr")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            model,
            device,
            generation_config,
            model_name,
        })
    }
}

impl<'a> GenerationDataProvider for HunyuanOCRGenerateModel<'a> {
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
        let mes_render = self.chat_template.apply_chat_template(mes)?;
        let data = self
            .pre_processor
            .process_info(mes, &self.tokenizer, &mes_render)?;
        let input_ids = data.input_ids;
        let data_vec = vec![
            data.pixel_values,
            data.image_grid_thw,
            data.image_mask.into(),
            data.position_ids.into(),
        ];
        let multi_model_data = MultiModalData::new(data_vec);
        Ok(PrepareData {
            in_reasoning: false,
            input_ids,
            multi_model_data,
        })
    }
}

crate::impl_generate_model!(HunyuanOCRGenerateModel<'a>);
