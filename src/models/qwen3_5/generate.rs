use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse, ChatMessage,
    ChatMessageContent, ChatMessageContentPart,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        artifact::{ArtifactKind, LoadSpec},
        common::{
            gguf::{load_gguf_file, load_text_bootstrap_from_gguf},
            onnx::resolve_tokenizer_dir,
        },
        qwen3_5::{config::Qwen3_5Config, model::Qwen3_5Model, onnx::Qwen3_5OnnxBackend},
        qwen3vl::processor::Qwen3VLProcessor,
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

fn contains_unsupported_onnx_multimodal_content(mes: &ChatCompletionParameters) -> bool {
    mes.messages.iter().any(|chat_mes| {
        if let ChatMessage::User { content, .. } = chat_mes {
            match content {
                ChatMessageContent::Text(_) => false,
                ChatMessageContent::ContentPart(part_vec) => part_vec.iter().any(|part| {
                    matches!(
                        part,
                        ChatMessageContentPart::Audio(_) | ChatMessageContentPart::Video(_)
                    )
                }),
                _ => false,
            }
        } else {
            false
        }
    })
}

fn contains_image_content(mes: &ChatCompletionParameters) -> bool {
    mes.messages.iter().any(|chat_mes| {
        if let ChatMessage::User { content, .. } = chat_mes {
            match content {
                ChatMessageContent::Text(_) => false,
                ChatMessageContent::ContentPart(part_vec) => part_vec
                    .iter()
                    .any(|part| matches!(part, ChatMessageContentPart::Image(_))),
                _ => false,
            }
        } else {
            false
        }
    })
}

pub struct Qwen3_5GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: Option<Qwen3VLProcessor>,
    qwen3_5: Option<Qwen3_5Model>,
    onnx_backend: Option<Qwen3_5OnnxBackend>,
    device: Device,
    eos_token_id: u32,
    image_token_id: u32,
    model_name: String,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl<'a> Qwen3_5GenerateModel<'a> {
    pub fn init_from_spec(
        spec: &LoadSpec,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        match spec.resolved_artifact() {
            ArtifactKind::Safetensors => {
                let path =
                    spec.paths.weight_dir.as_deref().ok_or_else(|| {
                        anyhow!("weight_path is required for qwen3.5 safetensors")
                    })?;
                Self::init(path, device, dtype)
            }
            ArtifactKind::Gguf => {
                let gguf = spec
                    .paths
                    .gguf_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("gguf_path is required for qwen3.5 gguf"))?;
                Self::init_from_gguf(gguf, spec.paths.mmproj_path.as_deref(), device)
            }
            ArtifactKind::Onnx => {
                let onnx_path = spec
                    .paths
                    .onnx_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("onnx_path is required for qwen3.5 onnx"))?;
                Self::init_from_onnx(onnx_path, spec.paths.tokenizer_dir.as_deref())
            }
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
        }
    }

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let model_name = path
            .split("/")
            .collect::<Vec<&str>>()
            .pop()
            .unwrap_or("qwen3.5");
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen3_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen3VLProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let eos_token_id = cfg.text_config.eos_token_id;
        let image_token_id = cfg.image_token_id;
        let qwen3_5 = Qwen3_5Model::new_from_vb(vb, cfg)?;

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor: Some(pre_processor),
            qwen3_5: Some(qwen3_5),
            onnx_backend: None,
            device,
            eos_token_id,
            image_token_id,
            model_name: model_name.to_string(),
            repeat_penalty: 1.01,
            repeat_last_n: 64,
        })
    }

    pub fn init_from_gguf(
        model_file: &str,
        mmproj_file: Option<&str>,
        device: Option<&Device>,
    ) -> Result<Self> {
        if !model_file.contains("Qwen3.5") || !model_file.ends_with("gguf") {
            return Err(anyhow!("Qwen3.5 gguf model file name illigal {model_file}"));
        }
        if let Some(mmproj) = mmproj_file
            && (!mmproj.contains("mmproj") || !mmproj.ends_with("gguf"))
        {
            return Err(anyhow!("Qwen3.5 mmproj_file name illigal {model_file}"));
        }

        let device = get_device(device);
        let mut model_gguf = load_gguf_file(model_file, &device)?;
        let bootstrap =
            load_text_bootstrap_from_gguf(model_file, Some(false), Some(false), Some(false))?;
        let chat_template_str = bootstrap.chat_template.ok_or_else(|| {
            anyhow!("tokenizer.chat_template metadata is missing in {model_file}")
        })?;
        let chat_template = ChatTemplate::str_init(&chat_template_str)?;
        let tokenizer = bootstrap.tokenizer;
        let (pre_processor, mut mmproj_gguf) = if let Some(mmproj_f) = mmproj_file {
            let mmproj_gguf = load_gguf_file(mmproj_f, &device)?;
            let processor = Qwen3VLProcessor::new_qwen3_5_default(&device, DType::F32)?;
            (Some(processor), Some(mmproj_gguf))
        } else {
            (None, None)
        };

        let eos_token_id = bootstrap.eos_token_id.unwrap_or(
            model_gguf
                .get_matedata("tokenizer.ggml.eos_token_id")?
                .to_u32()?,
        );
        let qwen3_5 = Qwen3_5Model::new_from_gguf(&mut model_gguf, mmproj_gguf.as_mut(), &device)?;
        let stem = std::path::Path::new(model_file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3.5");
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_5: Some(qwen3_5),
            onnx_backend: None,
            device,
            eos_token_id,
            image_token_id: 248056,
            model_name: stem.to_string(),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        })
    }

    pub fn init_from_onnx(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        let tokenizer_dir = resolve_tokenizer_dir(
            onnx_path,
            tokenizer_dir,
            &["tokenizer.json", "config.json", "chat_template.jinja"],
        )?;
        let base_path = tokenizer_dir.to_string_lossy().to_string();
        let chat_template = ChatTemplate::init(&base_path)?;
        let tokenizer = TokenizerModel::init(&base_path)?;
        let config_path = tokenizer_dir.join("config.json");
        let cfg: Qwen3_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let model_name = tokenizer_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("qwen3.5")
            .to_string();
        let onnx_backend = Qwen3_5OnnxBackend::load(onnx_path)?;
        let pre_processor = Qwen3VLProcessor::new_qwen3_5_default(&Device::Cpu, DType::F32)?;

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor: Some(pre_processor),
            qwen3_5: None,
            onnx_backend: Some(onnx_backend),
            device: Device::Cpu,
            eos_token_id: cfg.text_config.eos_token_id,
            image_token_id: cfg.image_token_id,
            model_name,
            repeat_penalty: 1.01,
            repeat_last_n: 64,
        })
    }

    fn clear_runtime_cache(&mut self) {
        if let Some(model) = self.qwen3_5.as_mut() {
            model.clear_cache();
        }
        if let Some(onnx_backend) = self.onnx_backend.as_mut() {
            onnx_backend.clear_cache();
        }
    }
}

impl<'a> GenerateModel for Qwen3_5GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        if self.onnx_backend.is_some() {
            if contains_unsupported_onnx_multimodal_content(&mes) {
                return Err(anyhow!(
                    "qwen3.5 onnx backend currently supports text/image input; audio/video are not supported"
                ));
            }
            let seed = mes.seed.unwrap_or(32768) as u64;
            let temperature = mes.temperature.unwrap_or(0.4);
            let top_p = mes.top_p.unwrap_or(0.95);
            let mut logit_processor =
                get_logit_processor(temperature.into(), top_p.into(), Some(20), seed);
            let mes_render = self.chat_template.apply_chat_template(&mes)?;
            let (mes_text, mut pixel_values, mut image_grid_thw, pixel_values_video, _) =
                if let Some(processor) = &self.pre_processor {
                    let input = processor.process_info(&mes, &mes_render)?;
                    (
                        input.replace_text,
                        input.pixel_values,
                        input.image_grid_thw,
                        input.pixel_values_video,
                        input.video_grid_thw,
                    )
                } else {
                    (mes_render, None, None, None, None)
                };
            if pixel_values_video.is_some() {
                return Err(anyhow!(
                    "qwen3.5 onnx backend currently does not support video multimodal input"
                ));
            }
            if contains_image_content(&mes)
                && !self
                    .onnx_backend
                    .as_ref()
                    .map(|backend| backend.supports_vision())
                    .unwrap_or(false)
            {
                return Err(anyhow!(
                    "qwen3.5 onnx backend does not include vision_encoder component"
                ));
            }
            let mut current_ids = self.tokenizer.text_encode_vec(mes_text, true)?;
            let prompt_tokens = current_ids.len() as u32;
            let mut position_start = 0usize;
            let mut generate = Vec::new();
            let sample_len = mes.max_tokens.unwrap_or(1024);
            for _ in 0..sample_len {
                let backend = self
                    .onnx_backend
                    .as_mut()
                    .ok_or_else(|| anyhow!("qwen3.5 onnx runtime is not initialized"))?;
                let logits = backend.forward_logits(
                    &current_ids,
                    position_start,
                    pixel_values.as_ref(),
                    image_grid_thw.as_ref(),
                    Some(self.image_token_id),
                )?;
                let vocab = logits.len();
                let logits = Tensor::from_vec(logits, vocab, &self.device)?;
                let logits = if self.repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = generate.len().saturating_sub(self.repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        self.repeat_penalty,
                        &generate[start_at..],
                    )?
                };
                let next_token = logit_processor.sample(&logits)?;
                generate.push(next_token);
                if next_token == self.eos_token_id {
                    break;
                }
                position_start += current_ids.len();
                current_ids = vec![next_token];
                pixel_values = None;
                image_grid_thw = None;
            }
            let completion_tokens = generate.len() as u32;
            let res = self.tokenizer.token_decode(generate)?;
            self.clear_runtime_cache();
            return Ok(build_completion_response(
                res,
                &self.model_name,
                Some(completion_tokens),
                Some(prompt_tokens),
            ));
        }

        let qwen3_5 = self
            .qwen3_5
            .as_mut()
            .ok_or_else(|| anyhow!("qwen3.5 native runtime is not initialized"))?;
        let seed = mes.seed.unwrap_or(32768) as u64;
        let temperature = mes.temperature.unwrap_or(0.4);
        let top_p = mes.top_p.unwrap_or(0.95);
        let mut logit_processor =
            get_logit_processor(temperature.into(), top_p.into(), Some(20), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (mes_text, pixel_values, image_grid_thw, pixel_values_video, video_grid_thw) =
            if let Some(processor) = &self.pre_processor {
                let input = processor.process_info(&mes, &mes_render)?;
                (
                    input.replace_text,
                    input.pixel_values,
                    input.image_grid_thw,
                    input.pixel_values_video,
                    input.video_grid_thw,
                )
            } else {
                (mes_render, None, None, None, None)
            };
        let mut input_ids = self.tokenizer.text_encode(mes_text, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let prompt_tokens = seq_len as u32;
        let mut seqlen_offset = 0;
        let mut pixel_values = pixel_values.as_ref();
        let image_grid_thw = image_grid_thw.as_ref();
        let mut pixel_values_video = pixel_values_video.as_ref();
        let video_grid_thw = video_grid_thw.as_ref();
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            let logits = qwen3_5.forward(
                &input_ids,
                pixel_values,
                image_grid_thw,
                pixel_values_video,
                video_grid_thw,
                seqlen_offset,
            )?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = generate.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &generate[start_at..],
                )?
            };
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            pixel_values = None;
            pixel_values_video = None;
        }
        let completion_tokens = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        self.clear_runtime_cache();
        let response = build_completion_response(
            res,
            &self.model_name,
            Some(completion_tokens),
            Some(prompt_tokens),
        );
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
        if self.onnx_backend.is_some() {
            if contains_unsupported_onnx_multimodal_content(&mes) {
                return Err(anyhow!(
                    "qwen3.5 onnx backend currently supports text/image input; audio/video are not supported"
                ));
            }
            if contains_image_content(&mes)
                && !self
                    .onnx_backend
                    .as_ref()
                    .map(|backend| backend.supports_vision())
                    .unwrap_or(false)
            {
                return Err(anyhow!(
                    "qwen3.5 onnx backend does not include vision_encoder component"
                ));
            }
            let mes_render = self.chat_template.apply_chat_template(&mes)?;
            let (mes_text, pixel_values, image_grid_thw, pixel_values_video, _) =
                if let Some(processor) = &self.pre_processor {
                    let input = processor.process_info(&mes, &mes_render)?;
                    (
                        input.replace_text,
                        input.pixel_values,
                        input.image_grid_thw,
                        input.pixel_values_video,
                        input.video_grid_thw,
                    )
                } else {
                    (mes_render, None, None, None, None)
                };
            if pixel_values_video.is_some() {
                return Err(anyhow!(
                    "qwen3.5 onnx backend currently does not support video multimodal input"
                ));
            }
            let onnx_backend = self
                .onnx_backend
                .as_mut()
                .ok_or_else(|| anyhow!("qwen3.5 onnx runtime is not initialized"))?;
            let tokenizer = &self.tokenizer;
            let model_name = self.model_name.clone();
            let repeat_penalty = self.repeat_penalty;
            let repeat_last_n = self.repeat_last_n;
            let eos_token_id = self.eos_token_id;
            let image_token_id = self.image_token_id;
            let device = self.device.clone();
            let seed = mes.seed.unwrap_or(34562) as u64;
            let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
            let mut current_ids = self.tokenizer.text_encode_vec(mes_text, true)?;
            let mut position_start = 0usize;
            let sample_len = mes.max_tokens.unwrap_or(1024);
            let stream = stream! {
                let mut error_tokens = Vec::new();
                let mut tool_call_id = None;
                let mut tool_call_content = String::new();
                let mut generate = Vec::new();
                let mut pixel_values = pixel_values;
                let mut image_grid_thw = image_grid_thw;
                for _ in 0..sample_len {
                    let logits = onnx_backend.forward_logits(
                        &current_ids,
                        position_start,
                        pixel_values.as_ref(),
                        image_grid_thw.as_ref(),
                        Some(image_token_id),
                    )?;
                    let vocab = logits.len();
                    let logits = Tensor::from_vec(logits, vocab, &device)?;
                    let logits = if repeat_penalty == 1. {
                        logits
                    } else {
                        let start_at = generate.len().saturating_sub(repeat_last_n);
                        candle_transformers::utils::apply_repeat_penalty(
                            &logits,
                            repeat_penalty,
                            &generate[start_at..],
                        )?
                    };
                    let next_token = logit_processor.sample(&logits)?;
                    generate.push(next_token);
                    let mut decode_ids = Vec::new();
                    if !error_tokens.is_empty() {
                        decode_ids.extend_from_slice(&error_tokens);
                    }
                    decode_ids.push(next_token);
                    let decoded_token = tokenizer.token_decode(decode_ids)
                        .map_err(|e| anyhow!(format!("stream decode error{e}")))?;
                    if decoded_token.contains("�") {
                        error_tokens.push(next_token);
                        if error_tokens.len() > 3 {
                            error_tokens.clear();
                        }
                        position_start += current_ids.len();
                        current_ids = vec![next_token];
                        pixel_values = None;
                        image_grid_thw = None;
                        continue;
                    }
                    error_tokens.clear();
                    match decoded_token.as_str() {
                        "<tool_call>" => {
                            tool_call_id = Some(uuid::Uuid::new_v4().to_string());
                            position_start += current_ids.len();
                            current_ids = vec![next_token];
                            pixel_values = None;
                            image_grid_thw = None;
                            continue;
                        }
                        "</tool_call>" => {
                            let chunk = build_completion_chunk_response(
                                decoded_token,
                                &model_name,
                                tool_call_id.clone(),
                                Some(tool_call_content.clone())
                            );
                            tool_call_id = None;
                            tool_call_content = String::new();
                            yield Ok(chunk);
                        }
                        _ => {
                            if tool_call_id.is_some() {
                                tool_call_content.push_str(&decoded_token);
                                position_start += current_ids.len();
                                current_ids = vec![next_token];
                                pixel_values = None;
                                image_grid_thw = None;
                                continue;
                            } else {
                                let chunk = build_completion_chunk_response(
                                    decoded_token,
                                    &model_name,
                                    None,
                                    None
                                );
                                yield Ok(chunk);
                            }
                        }
                    }
                    if next_token == eos_token_id {
                        break;
                    }
                    position_start += current_ids.len();
                    current_ids = vec![next_token];
                    pixel_values = None;
                    image_grid_thw = None;
                }
                onnx_backend.clear_cache();
            };
            return Ok(Box::new(Box::pin(stream)));
        }

        let qwen3_5 = self
            .qwen3_5
            .as_mut()
            .ok_or_else(|| anyhow!("qwen3.5 native runtime is not initialized"))?;
        let tokenizer = &self.tokenizer;
        let model_name = self.model_name.clone();
        let repeat_penalty = self.repeat_penalty;
        let repeat_last_n = self.repeat_last_n;
        let eos_token_id = self.eos_token_id;
        let device = self.device.clone();
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (mes_text, pixel_values, image_grid_thw, pixel_values_video, video_grid_thw) =
            if let Some(processor) = &self.pre_processor {
                let input = processor.process_info(&mes, &mes_render)?;
                (
                    input.replace_text,
                    input.pixel_values,
                    input.image_grid_thw,
                    input.pixel_values_video,
                    input.video_grid_thw,
                )
            } else {
                (mes_render, None, None, None, None)
            };
        let mut input_ids = self.tokenizer.text_encode(mes_text, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut pixel_values = pixel_values.as_ref();
            let image_grid_thw = image_grid_thw.as_ref();
            let mut pixel_values_video = pixel_values_video.as_ref();
            let video_grid_thw = video_grid_thw.as_ref();
            let mut tool_call_id = None;
            let mut tool_call_content = String::new();
            let mut generate = Vec::new();
            for _ in 0..sample_len {
                let logits = qwen3_5.forward(
                    &input_ids,
                    pixel_values,
                    image_grid_thw,
                    pixel_values_video,
                    video_grid_thw,
                    seqlen_offset,
                )?;
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let logits = if repeat_penalty == 1. {
                    logits
                } else {
                    let start_at = generate.len().saturating_sub(repeat_last_n);
                    candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        repeat_penalty,
                        &generate[start_at..],
                    )?
                };
                let next_token = logit_processor.sample(&logits)?;
                generate.push(next_token);
                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);
                let decoded_token = tokenizer.token_decode(decode_ids).map_err(|e| anyhow!(format!("stream decode error{e}")))?;
                if decoded_token.contains("�") {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    input_ids = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
                    pixel_values = None;
                    pixel_values_video = None;
                    continue;
                }
                error_tokens.clear();
                match decoded_token.as_str() {
                    "<tool_call>" => {
                        tool_call_id = Some(uuid::Uuid::new_v4().to_string());
                        seqlen_offset += seq_len;
                        seq_len = 1;
                        input_ids = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
                        pixel_values = None;
                        pixel_values_video = None;
                        continue;
                    }
                    "</tool_call>" => {
                        let chunk = build_completion_chunk_response(
                            decoded_token,
                            &model_name,
                            tool_call_id.clone(),
                            Some(tool_call_content.clone())
                        );
                        tool_call_id = None;
                        tool_call_content = String::new();
                        yield Ok(chunk);
                    }
                    _ => {
                        if tool_call_id.is_some() {
                            tool_call_content.push_str(&decoded_token);
                            seqlen_offset += seq_len;
                            seq_len = 1;
                            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
                            pixel_values = None;
                            pixel_values_video = None;
                            continue;
                        } else {
                            let chunk = build_completion_chunk_response(
                                decoded_token,
                                &model_name,
                                None,
                                None
                            );
                            yield Ok(chunk);
                        }
                    }
                }
                if next_token == eos_token_id {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &device)?;
                pixel_values = None;
                pixel_values_video = None;
            }
            qwen3_5.clear_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
