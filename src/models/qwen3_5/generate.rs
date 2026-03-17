use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        common::gguf::Gguf,
        qwen3_5::{config::Qwen3_5Config, model::Qwen3_5Model},
        qwen3vl::processor::Qwen3VLProcessor,
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct Qwen3_5GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: Option<Qwen3VLProcessor>,
    qwen3_5: Qwen3_5Model,
    device: Device,
    eos_token_id: u32,
    model_name: String,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl<'a> Qwen3_5GenerateModel<'a> {
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
        let qwen3_5 = Qwen3_5Model::new_from_vb(vb, cfg)?;

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor: Some(pre_processor),
            qwen3_5,
            device,
            eos_token_id,
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

        let mut reader = std::fs::File::open(model_file)?;
        let content = gguf_file::Content::read(&mut reader)?;
        let device = get_device(device);
        let mut model_gguf = Gguf::new(content, reader, device.clone());

        let chat_template_str = model_gguf
            .get_matedata("tokenizer.chat_template")?
            .to_string()?
            .clone();
        let chat_template = ChatTemplate::str_init(&chat_template_str)?;
        let tokenizer = model_gguf.build_tokenizer(Some(false), Some(false), Some(false))?;
        let (pre_processor, mut mmproj_gguf) = if let Some(mmproj_f) = mmproj_file {
            let mut reader = std::fs::File::open(mmproj_f)?;
            let content = gguf_file::Content::read(&mut reader)?;
            let mmproj_gguf = Gguf::new(content, reader, device.clone());
            let processor = Qwen3VLProcessor::new_qwen3_5_default(&device, DType::F32)?;
            (Some(processor), Some(mmproj_gguf))
        } else {
            (None, None)
        };

        let eos_token_id = model_gguf
            .get_matedata("tokenizer.ggml.eos_token_id")?
            .to_u32()?;
        let qwen3_5 = Qwen3_5Model::new_from_gguf(&mut model_gguf, mmproj_gguf.as_mut(), &device)?;
        let stem = std::path::Path::new(model_file)
            .file_stem() // 获取文件名主干（不含扩展名）
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3.5");
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_5,
            device,
            // eos_token_id: 248044,
            eos_token_id,
            model_name: stem.to_string(),
            repeat_penalty: 1.1,
            repeat_last_n: 64,
        })
    }
}

impl<'a> GenerateModel for Qwen3_5GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = mes.seed.unwrap_or(32768) as u64;
        let temperature = mes.temperature.unwrap_or(0.4);
        let top_p = mes.top_p.unwrap_or(0.95);
        let mut logit_processor =
            get_logit_processor(temperature.into(), top_p.into(), Some(20), seed);
        // let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
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
            let logits = self.qwen3_5.forward(
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
        self.qwen3_5.clear_cache();
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
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        // let input = self.pre_processor.process_info(&mes, &mes_render)?;
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
                let logits = self.qwen3_5.forward(
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
                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty() {
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);
                let decoded_token = self.tokenizer.token_decode(decode_ids).map_err(|e| anyhow!(format!("stream decode error{e}")))?;
                if decoded_token.contains("�") {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                    pixel_values = None;
                    pixel_values_video = None;
                    continue;
                }
                error_tokens.clear();
                // 处理特殊标记和工具调用
                match decoded_token.as_str() {
                    "<tool_call>" => {
                        // 开始工具调用
                        tool_call_id = Some(uuid::Uuid::new_v4().to_string());
                        seqlen_offset += seq_len;
                        seq_len = 1;
                        input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                        pixel_values = None;
                        pixel_values_video = None;
                        continue;
                    }
                    "</tool_call>" => {
                        // 结束工具调用
                        let chunk = build_completion_chunk_response(
                            decoded_token,
                            &self.model_name,
                            tool_call_id.clone(),
                            Some(tool_call_content.clone())
                        );
                        tool_call_id = None;
                        tool_call_content = String::new();
                        yield Ok(chunk);
                    }
                    _ => {
                        if tool_call_id.is_some() {
                            // 在工具调用过程中，收集工具调用内容
                            tool_call_content.push_str(&decoded_token);
                            seqlen_offset += seq_len;
                            seq_len = 1;
                            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                            pixel_values = None;
                            pixel_values_video = None;
                            continue;
                        } else {
                            // 正常文本输出
                            let chunk = build_completion_chunk_response(
                                decoded_token,
                                &self.model_name,
                                None,
                                None
                            );
                            yield Ok(chunk);
                        }
                    }
                }
                if next_token == self.eos_token_id {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                pixel_values = None;
                pixel_values_video = None;
            }
            self.qwen3_5.clear_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
