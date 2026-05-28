use crate::models::common::{
    MultiModalData,
    generate::{GenerationDataProvider, PrepareData},
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::{
    models::deepseek_ocr::{
        config::DeepseekOCRConfig, model::DeepseekOCRModel, processor::DeepseekOCRProcessor,
    },
    tokenizer::TokenizerModel,
    utils::{extract_metadata_value, find_type_files, get_device, get_dtype},
};

pub struct DeepseekOCRGenerateModel {
    tokenizer: TokenizerModel,
    processor: DeepseekOCRProcessor,
    model: DeepseekOCRModel,
    device: Device,
    size: Vec<u32>,
    model_name: String,
    version: usize,
}

impl DeepseekOCRGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: DeepseekOCRConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let cfg_dtype = cfg.language_config.torch_dtype.clone();
        let device = &get_device(device);
        let dtype = get_dtype(dtype, &cfg_dtype);
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("deepseek-ai/DeepSeek-OCR")
            .to_string();
        let version = if model_name.contains("2") || cfg.vision_config.width.qwen2_0_5b.is_some() {
            2usize
        } else {
            1usize
        };
        let processor = DeepseekOCRProcessor::new(device, dtype, version)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let model = DeepseekOCRModel::new(vb, cfg, version)?;
        let size = vec![512u32, 640, 1024, 1280];

        Ok(Self {
            tokenizer,
            processor,
            model,
            device: device.clone(),
            size,
            model_name: model_name.to_string(),
            version,
        })
    }
}

impl GenerationDataProvider for DeepseekOCRGenerateModel {
    fn get_data(&self, mes: &crate::params::chat::ChatCompletionParameters) -> Result<PrepareData> {
        let base_size = extract_metadata_value::<u32>(&mes.metadata, "base_size").unwrap_or(640);
        let base_size = if self.size.contains(&base_size) {
            base_size
        } else {
            640
        };
        let image_size = extract_metadata_value::<u32>(&mes.metadata, "image_size").unwrap_or(640);
        let image_size = if self.size.contains(&image_size) {
            image_size
        } else {
            640
        };
        let base_size = if self.version == 2 { 1024 } else { base_size };
        let image_size = if self.version == 2 { 768 } else { image_size };
        let crop_mode = extract_metadata_value::<bool>(&mes.metadata, "crop_mode").unwrap_or(false);
        let (input_ids, images_ori, image_crop, images_seq_mask, images_spatial_crop_t) = self
            .processor
            .process_info(mes, &self.tokenizer, base_size, image_size, crop_mode)?;
        let data_vec = vec![
            Some(images_ori),
            Some(image_crop),
            Some(images_seq_mask),
            Some(images_spatial_crop_t),
        ];
        let multi_model_data = MultiModalData::new(data_vec);
        Ok(PrepareData {
            in_reasoning: false,
            input_ids,
            multi_model_data,
        })
    }
}

crate::impl_generate_model!(DeepseekOCRGenerateModel);
