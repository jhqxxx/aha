//! GLM-OCR Inference and Generation
use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::{
    // chat_template::ChatTemplate,
    models::{
        GenerateModel,
        glm_ocr::{
            config::{GlmOcrConfig, GlmOcrGenerationConfig},
            model::GlmOcrModel,
            processor::GlmOcrProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, extract_user_text,
        find_type_files, get_device, get_dtype, get_logit_processor, img_utils::extract_image_url,
    },
};

pub struct GlmOcrGenerateModel {
    // chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: GlmOcrProcessor,
    model: GlmOcrModel,
    device: Device,
    eos_token_ids: Vec<u32>,
    model_name: String,
    image_token_id: u32,
    image_start_token_id: u32,
    image_end_token_id: u32,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
}

impl GlmOcrGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        // let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: GlmOcrConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let processor = GlmOcrProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = GlmOcrModel::new(vb, cfg.clone())?;
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: GlmOcrGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("glm-ocr")
            .to_string();
        Ok(Self {
            // chat_template,
            tokenizer,
            processor,
            model,
            device,
            eos_token_ids: generation_config.eos_token_id.clone(),
            model_name,
            image_token_id: cfg.image_token_id,
            image_start_token_id: cfg.image_start_token_id,
            image_end_token_id: cfg.image_end_token_id,
            patch_size: cfg.vision_config.patch_size,
            temporal_patch_size: cfg.vision_config.temporal_patch_size,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
        })
    }
}

impl GenerateModel for GlmOcrGenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);

        // Extract image path and prompt from messages
        let image_urls = extract_image_url(&mes);
        let image_path = image_urls
            .first()
            .ok_or_else(|| anyhow!("No image provided"))?;

        // Get prompt text from messages
        let mut prompt = extract_user_text(&mes)?;
        if prompt.chars().count() == 0 {
            prompt = "Extract all text from this image.".to_string()
        }

        let processed = self.processor.process_info(
            image_path,
            &prompt,
            &self.tokenizer,
            self.image_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
        )?;

        let mut input_ids = processed.input_ids;
        let pixel_values = Some(processed.pixel_values);
        let image_grid_thw = Some(processed.grid_thw);
        let image_mask = Some(processed.image_mask);
        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let prompt_tokens = seq_len as u32;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(512);

        for _ in 0..sample_len {
            let is_first_pass = seqlen_offset == 0;
            let logits = self.model.forward(
                &input_ids,
                if is_first_pass {
                    pixel_values.as_ref()
                } else {
                    None
                },
                if is_first_pass {
                    image_grid_thw.as_ref()
                } else {
                    None
                },
                if is_first_pass {
                    image_mask.as_ref()
                } else {
                    None
                },
                seqlen_offset,
            )?;
            let logits = logits.i((0, seq_len - 1, ..))?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;

            generate.push(next_token);
            if self.eos_token_ids.contains(&next_token) {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }

        self.model.clear_kv_cache();
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        let response =
            build_completion_response(res, &self.model_name, Some(num_token), Some(prompt_tokens));
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
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);

        // Extract image path and prompt from messages
        let image_urls = extract_image_url(&mes);
        let image_path = image_urls
            .first()
            .ok_or_else(|| anyhow!("No image provided"))?;

        // Get prompt text from messages
        let mut prompt = extract_user_text(&mes)?;
        if prompt.chars().count() == 0 {
            prompt = "Extract all text from this image.".to_string()
        }

        let processed = self.processor.process_info(
            image_path,
            &prompt,
            &self.tokenizer,
            self.image_token_id,
            self.image_start_token_id,
            self.image_end_token_id,
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
        )?;

        let mut input_ids = processed.input_ids;
        let pixel_values = Some(processed.pixel_values);
        let image_grid_thw = Some(processed.grid_thw);
        let image_mask = Some(processed.image_mask);
        let mut seqlen_offset = 0;
        let mut seq_len = input_ids.dim(1)?;
        let sample_len = mes.max_tokens.unwrap_or(512);

        let stream = stream! {
            let mut generated: Vec<u32> = Vec::new();
            let mut error_tokens = Vec::new();
            for _ in 0..sample_len {
                let is_first_pass = seqlen_offset == 0;
                let logits = self.model.forward(
                    &input_ids,
                    if is_first_pass { pixel_values.as_ref() } else { None },
                    if is_first_pass { image_grid_thw.as_ref() } else { None },
                    if is_first_pass { image_mask.as_ref() } else { None },
                    seqlen_offset,
                ).map_err(|e| anyhow!(format!("forward error: {e}")))?;
                let logits = logits.i((0, seq_len - 1, ..)).map_err(|e| anyhow!(format!("index error: {e}")))?.to_dtype(DType::F32).map_err(|e| anyhow!(format!("dtype error: {e}")))?;

                let next_token = logit_processor.sample(&logits).map_err(|e| anyhow!(format!("sample error: {e}")))?;
                generated.push(next_token);

                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);

                let decoded_token = self.tokenizer.token_decode(decode_ids).map_err(|e| anyhow!(format!("decode error: {e}")))?;
                if decoded_token.contains("�") {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device).map_err(|e| anyhow!(format!("tensor error: {e}")))?;
                    continue;
                }
                error_tokens.clear();

                let chunk = build_completion_chunk_response(decoded_token, &self.model_name, None, None);
                yield Ok(chunk);

                if self.eos_token_ids.contains(&next_token) {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device).map_err(|e| anyhow!(format!("tensor error: {e}")))?;
            }
            self.model.clear_kv_cache();
        };

        Ok(Box::new(Box::pin(stream)))
    }
}
