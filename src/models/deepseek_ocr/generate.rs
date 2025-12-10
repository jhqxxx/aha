use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::{
    models::{
        GenerateModel,
        deepseek_ocr::{
            config::DeepseekOCRConfig, model::DeepseekOCRModel, processor::DeepseekOCRProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct DeepseekOCRGenerateModel {
    tokenizer: TokenizerModel,
    processor: DeepseekOCRProcessor,
    deepseekocr_model: DeepseekOCRModel,
    bos_token_id: u32,
    eos_token_id: u32,
    device: Device,
    size: Vec<u32>,
    model_name: String,
}

impl DeepseekOCRGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: DeepseekOCRConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let cfg_dtype = cfg.language_config.torch_dtype.clone();
        let device = &get_device(device);
        let dtype = get_dtype(dtype, &cfg_dtype);
        let processor = DeepseekOCRProcessor::new(device, dtype)?;
        let eos_token_id = cfg.eos_token_id;
        let bos_token_id = cfg.bos_token_id;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let deepseekocr_model = DeepseekOCRModel::new(vb, cfg)?;
        let size = vec![512u32, 640, 1024, 1280];
        Ok(Self {
            tokenizer,
            processor,
            deepseekocr_model,
            bos_token_id,
            eos_token_id,
            device: device.clone(),
            size,
            model_name: "deepseek-ocr".to_string(),
        })
    }
}

impl GenerateModel for DeepseekOCRGenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let base_size = if let Some(map) = &mes.metadata
            && map.contains_key("base_size")
        {
            let size = map.get("base_size").unwrap();
            let size = size.parse::<u32>().unwrap_or(640);
            if self.size.contains(&size) { size } else { 640 }
        } else {
            640
        };
        let image_size = if let Some(map) = &mes.metadata
            && map.contains_key("image_size")
        {
            let size = map.get("image_size").unwrap();
            let size = size.parse::<u32>().unwrap_or(640);
            if self.size.contains(&size) { size } else { 640 }
        } else {
            640
        };
        let crop_mode = if let Some(map) = &mes.metadata
            && map.contains_key("crop_mode")
        {
            let size = map.get("crop_mode").unwrap();
            size.parse::<bool>().unwrap_or(false)
        } else {
            false
        };
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let (mut input_ids, images_ori, image_crop, images_seq_mask, images_spatial_crop_t) = self
            .processor
            .process_info(&mes, &self.tokenizer, base_size, image_size, crop_mode)?;
        let mut images_ori = Some(&images_ori);
        let mut image_crop = Some(&image_crop);
        let mut images_seq_mask = Some(&images_seq_mask);
        let mut images_spatial_crop_t = Some(&images_spatial_crop_t);
        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            let logits = self.deepseekocr_model.forward(
                &input_ids,
                images_ori,
                image_crop,
                images_seq_mask,
                images_spatial_crop_t,
                seqlen_offset,
            )?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.bos_token_id || next_token == self.eos_token_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            images_ori = None;
            image_crop = None;
            images_seq_mask = None;
            images_spatial_crop_t = None;
        }
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        self.deepseekocr_model.clear_kv_cache();
        let response = build_completion_response(res, &self.model_name, Some(num_token));
        Ok(response)
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let base_size = if let Some(map) = &mes.metadata
            && map.contains_key("base_size")
        {
            let size = map.get("base_size").unwrap();
            let size = size.parse::<u32>().unwrap_or(640);
            if self.size.contains(&size) { size } else { 640 }
        } else {
            640
        };
        let image_size = if let Some(map) = &mes.metadata
            && map.contains_key("image_size")
        {
            let size = map.get("image_size").unwrap();
            let size = size.parse::<u32>().unwrap_or(640);
            if self.size.contains(&size) { size } else { 640 }
        } else {
            640
        };
        let crop_mode = if let Some(map) = &mes.metadata
            && map.contains_key("crop_mode")
        {
            let size = map.get("crop_mode").unwrap();
            size.parse::<bool>().unwrap_or(false)
        } else {
            false
        };
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let (mut input_ids, images_ori, image_crop, images_seq_mask, images_spatial_crop_t) = self
            .processor
            .process_info(&mes, &self.tokenizer, base_size, image_size, crop_mode)?;

        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut images_ori = Some(&images_ori);
            let mut image_crop = Some(&image_crop);
            let mut images_seq_mask = Some(&images_seq_mask);
            let mut images_spatial_crop_t = Some(&images_spatial_crop_t);
            for _ in 0..sample_len {
                let logits = self.deepseekocr_model.forward(
                    &input_ids,
                    images_ori,
                    image_crop,
                    images_seq_mask,
                    images_spatial_crop_t,
                    seqlen_offset,
                )?;
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let next_token = logit_processor.sample(&logits)?;
                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);
                let decoded_token = self.tokenizer.token_decode(decode_ids).map_err(|e| anyhow!(format!("stream decode error{}", e)))?;
                if decoded_token.contains("�") {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                    images_ori = None;
                    image_crop = None;
                    images_seq_mask = None;
                    images_spatial_crop_t = None;
                    continue;
                }
                error_tokens.clear();
                let chunk = build_completion_chunk_response(decoded_token, &self.model_name, None, None);
                yield Ok(chunk);
                if next_token == self.bos_token_id || next_token == self.eos_token_id {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                images_ori = None;
                image_crop = None;
                images_seq_mask = None;
                images_spatial_crop_t = None;
            }
            self.deepseekocr_model.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
