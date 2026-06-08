use crate::{
    models::common::{
        MultiModalData,
        generate::{GenerationDataProvider, PrepareData},
    },
    params::chat::ChatCompletionParameters,
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    chat_template::ChatTemplate,
    models::{
        qwen3::config::Qwen3GenerationConfig,
        qwen3vl::{config::Qwen3VLConfig, model::Qwen3VLModel, processor::Qwen3VLProcessor},
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub struct Qwen3VLGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: Qwen3VLProcessor,
    model: Qwen3VLModel,
    device: Device,
    generation_config: Qwen3GenerationConfig,
    model_name: String,
}

impl<'a> Qwen3VLGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen3VLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen3VLProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let model = Qwen3VLModel::new(cfg, vb, generation_config.eos_token_id.clone())?;

        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3vl")
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

impl<'a> GenerationDataProvider for Qwen3VLGenerateModel<'a> {
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
        let in_reasoning = self.is_in_reasoning(&mes_render);
        let input = self.pre_processor.process_info(mes, &mes_render)?;
        let input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.device)?;
        let seq_len = input_ids.dim(1)?;
        let cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?;
        let data_vec = vec![
            input.pixel_values,
            input.image_grid_thw,
            input.pixel_values_video,
            input.video_grid_thw,
            cache_position.into(),
        ];
        let multi_model_data = MultiModalData::new(data_vec);
        Ok(PrepareData {
            in_reasoning,
            input_ids,
            multi_model_data,
        })
    }
}

crate::impl_generate_model!(Qwen3VLGenerateModel<'a>);
