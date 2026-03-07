//! GLM-OCR Inference and Generation
use anyhow::{Result, anyhow};
use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::utils::apply_repeat_penalty;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        glm_ocr::{
            config::{GlmOcrConfig, GlmOcrGenerationConfig},
            model::GlmOcrModel,
            processor::GlmOcrProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::img_utils::extract_image_url,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct GlmOcrGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: GlmOcrProcessor,
    model: GlmOcrModel,
    device: Device,
    eos_token_ids: Vec<u32>,
    generation_config: GlmOcrGenerationConfig,
    model_name: String,
    image_token_id: u32,
    image_start_token_id: u32,
    image_end_token_id: u32,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
}

impl<'a> GlmOcrGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let mut cfg: GlmOcrConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;

        if cfg.projector_config.hidden_size == 0 {
            cfg.projector_config.hidden_size = 1536;
        }
        if cfg.projector_config.num_queries == 0 {
            cfg.projector_config.num_queries = 256;
        }

        // Apply rope_parameters if present (config.json nests these under rope_parameters)
        if let Some(ref rope_params) = cfg.text_config.rope_parameters {
            if cfg.text_config.rope_theta == 0.0 {
                cfg.text_config.rope_theta = rope_params.rope_theta;
            }
            if cfg.text_config.mrope_section.is_empty() {
                cfg.text_config.mrope_section = rope_params.mrope_section.clone();
            }
            if cfg.text_config.rope_type.is_empty() {
                cfg.text_config.rope_type = rope_params.rope_type.clone();
            }
            if cfg.text_config.partial_rotary_factor == 0.0 {
                cfg.text_config.partial_rotary_factor = rope_params.partial_rotary_factor;
            }
        }
        // Fallback rope_theta if still 0
        if cfg.text_config.rope_theta == 0.0 {
            cfg.text_config.rope_theta = 10000.0;
        }

        // Collect all EOS token IDs (config may have one or multiple)
        let mut eos_token_ids: Vec<u32> = Vec::new();
        if cfg.eos_token_id != 0 {
            eos_token_ids.push(cfg.eos_token_id);
        }
        if let Some(ref eos_val) = cfg.text_config.eos_token_id {
            match eos_val {
                serde_json::Value::Number(n) => {
                    if let Some(id) = n.as_u64() {
                        let id = id as u32;
                        if !eos_token_ids.contains(&id) {
                            eos_token_ids.push(id);
                        }
                    }
                }
                serde_json::Value::Array(arr) => {
                    for v in arr {
                        if let Some(id) = v.as_u64() {
                            let id = id as u32;
                            if !eos_token_ids.contains(&id) {
                                eos_token_ids.push(id);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if eos_token_ids.is_empty() {
            eos_token_ids.push(59246); // GLM-OCR default
        }

        let device = get_device(device);
        let cfg_dtype = if cfg.torch_dtype.is_empty() {
            "bfloat16"
        } else {
            &cfg.torch_dtype
        };
        let dtype = get_dtype(dtype, cfg_dtype);
        // Vision encoder has ops unsupported in F16 on CPU; use F32 for CPU
        let dtype = if matches!(device, Device::Cpu) && !matches!(dtype, DType::F32 | DType::F64) {
            DType::F32
        } else {
            dtype
        };

        let processor = GlmOcrProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = GlmOcrModel::new(vb, cfg.clone())?;
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: GlmOcrGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;

        Ok(Self {
            chat_template,
            tokenizer,
            processor,
            model,
            device,
            eos_token_ids,
            generation_config,
            model_name: "glm-ocr".to_string(),
            image_token_id: cfg.image_token_id,
            image_start_token_id: cfg.image_start_token_id,
            image_end_token_id: cfg.image_end_token_id,
            patch_size: cfg.vision_config.patch_size,
            temporal_patch_size: cfg.vision_config.temporal_patch_size,
            spatial_merge_size: cfg.vision_config.spatial_merge_size,
        })
    }
}

impl<'a> GenerateModel for GlmOcrGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        // Check if sampling is enabled - if do_sample is false, use greedy decoding (temperature = None)
        let do_sample = mes.temperature.is_some() || self.generation_config.do_sample;
        let temperature = if !do_sample {
            None  // Greedy decoding
        } else {
            match mes.temperature {
                None => Some(self.generation_config.temperature),
                Some(tem) => Some(tem),
            }
        };
        let top_p = match mes.top_p {
            None => Some(self.generation_config.top_p),
            Some(top_p) => Some(top_p),
        };
        let top_k = Some(self.generation_config.top_k);
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor =
            get_logit_processor(temperature, top_p, top_k, seed);

        // Extract image path and prompt from messages
        let image_urls = extract_image_url(&mes);
        let image_path = image_urls
            .first()
            .ok_or_else(|| anyhow!("No image provided"))?;

        // Get prompt text from messages
        let prompt = extract_text_from_messages(&mes).unwrap_or_else(|| "Extract all text from this image.".to_string());

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
            let logits = if self.generation_config.repetition_penalty != 1.0 {
                apply_repeat_penalty(&logits, self.generation_config.repetition_penalty, &generate)?
            } else {
                logits
            };
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
        // Check if sampling is enabled - if do_sample is false, use greedy decoding (temperature = None)
        let do_sample = mes.temperature.is_some() || self.generation_config.do_sample;
        let temperature = if !do_sample {
            None  // Greedy decoding
        } else {
            match mes.temperature {
                None => Some(self.generation_config.temperature),
                Some(tem) => Some(tem),
            }
        };
        let top_p = match mes.top_p {
            None => Some(self.generation_config.top_p),
            Some(top_p) => Some(top_p),
        };
        let top_k = Some(self.generation_config.top_k);
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor =
            get_logit_processor(temperature, top_p, top_k, seed);

        // Extract image path and prompt from messages
        let image_urls = extract_image_url(&mes);
        let image_path = image_urls
            .first()
            .ok_or_else(|| anyhow!("No image provided"))?;

        // Get prompt text from messages
        let prompt = extract_text_from_messages(&mes).unwrap_or_else(|| "Extract all text from this image.".to_string());

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
                let logits = if self.generation_config.repetition_penalty != 1.0 {
                    apply_repeat_penalty(&logits, self.generation_config.repetition_penalty, &generated).map_err(|e| anyhow!(format!("repeat penalty error: {e}")))?
                } else {
                    logits
                };
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

/// Extract text content from chat messages
fn extract_text_from_messages(mes: &ChatCompletionParameters) -> Option<String> {
    use aha_openai_dive::v1::resources::chat::{
        ChatMessage, ChatMessageContent, ChatMessageContentPart,
    };
    for msg in &mes.messages {
        if let ChatMessage::User { content, .. } = msg {
            match content {
                ChatMessageContent::Text(text) => return Some(text.clone()),
                ChatMessageContent::ContentPart(parts) => {
                    for part in parts {
                        if let ChatMessageContentPart::Text(text_part) = part {
                            return Some(text_part.text.clone());
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}
