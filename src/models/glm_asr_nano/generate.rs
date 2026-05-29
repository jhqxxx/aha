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
    models::glm_asr_nano::{
        config::GlmAsrNanoConfig, model::GlmAsrNanoModel, processor::GlmAsrNanoProcessor,
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub struct GlmAsrNanoGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: GlmAsrNanoProcessor,
    model: GlmAsrNanoModel,
    device: Device,
    dtype: DType,
    model_name: String,
}

impl<'a> GlmAsrNanoGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let device = get_device(device);
        let processor = GlmAsrNanoProcessor::new(path, &device, DType::F32)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: GlmAsrNanoConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let cfg_dtype = cfg.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let eos_ids = vec![59246u32, 59253, 59255];
        let model = GlmAsrNanoModel::new(vb, cfg, eos_ids)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("glm-asr-nano")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            processor,
            model,
            device,
            dtype,
            model_name,
        })
    }
}

impl<'a> GenerationDataProvider for GlmAsrNanoGenerateModel<'a> {
    fn get_data(&self, mes: &ChatCompletionParameters) -> Result<PrepareData> {
        let render_text: String = self.chat_template.apply_chat_template(mes)?;
        let (input_features, audio_token_lengths, replace_text) =
            self.processor.process_info(mes, &render_text)?;
        let input_ids = self.tokenizer.text_encode(replace_text, &self.device)?;
        let input_features = input_features.to_dtype(self.dtype)?;
        let audio_token_lengths = Tensor::new(audio_token_lengths, &self.device)?;
        let data_vec = vec![input_features.into(), audio_token_lengths.into()];
        let multi_model_data = MultiModalData::new(data_vec);
        Ok(PrepareData {
            in_reasoning: false,
            input_ids,
            multi_model_data,
        })
    }
}

crate::impl_generate_model!(GlmAsrNanoGenerateModel<'a>);
