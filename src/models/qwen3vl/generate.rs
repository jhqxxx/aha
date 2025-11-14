use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
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
        qwen3vl::{
            config::{Qwen3VLConfig, Qwen3VLGenerationConfig},
            model::Qwen3VLModel,
            processor::Qwen3VLProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct Qwen3VLGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: Qwen3VLProcessor,
    qwen3_vl: Qwen3VLModel,
    device: Device,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3VLGenerationConfig,
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
        let vb = vb.pp("model");
        let qwen3_vl = Qwen3VLModel::new(cfg, vb)?;
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: Qwen3VLGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_vl,
            device,
            eos_token_id1: generation_config.eos_token_id[0] as u32,
            eos_token_id2: generation_config.eos_token_id[1] as u32,
            generation_config,
        })
    }
}

impl<'a> GenerateModel for Qwen3VLGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let temperature = match mes.temperature {
            None => self.generation_config.temperature,
            Some(tem) => tem,
        };
        let top_p = match mes.top_p {
            None => self.generation_config.top_p,
            Some(top_p) => top_p,
        };
        let top_k = self.generation_config.top_k;
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let mut input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let mut pixel_values = input.pixel_values.as_ref();
        let image_grid_thw = input.image_grid_thw.as_ref();
        let mut pixel_values_video = input.pixel_values_video.as_ref();
        let video_grid_thw = input.video_grid_thw.as_ref();
        let mut cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            let logits = self.qwen3_vl.forward(
                &input_ids,
                pixel_values,
                image_grid_thw,
                pixel_values_video,
                video_grid_thw,
                Some(&cache_position),
                seqlen_offset,
            )?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
            pixel_values = None;
            pixel_values_video = None;
        }
        let res = self.tokenizer.token_decode(generate)?;
        self.qwen3_vl.clear_kv_cache();
        let response = build_completion_response(res, "qwen3vl");
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
        let temperature = match mes.temperature {
            None => self.generation_config.temperature,
            Some(tem) => tem,
        };
        let top_p = match mes.top_p {
            None => self.generation_config.top_p,
            Some(top_p) => top_p,
        };
        let top_k = self.generation_config.top_k;
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let mut input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let pixel_values = input.pixel_values.clone();
        let image_grid_thw = input.image_grid_thw.clone();
        let pixel_values_video = input.pixel_values_video.clone();
        let video_grid_thw = input.video_grid_thw.clone();
        let mut cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut pixel_values = pixel_values.as_ref();
            let image_grid_thw = image_grid_thw.as_ref();
            let mut pixel_values_video = pixel_values_video.as_ref();
            let video_grid_thw = video_grid_thw.as_ref();
            for _ in 0..sample_len {
            let logits = self.qwen3_vl.forward(
                &input_ids,
                pixel_values,
                image_grid_thw,
                pixel_values_video,
                video_grid_thw,
                Some(&cache_position),
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
                    cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
                    pixel_values = None;
                    pixel_values_video = None;
                    continue;
                }
                error_tokens.clear();
                let chunk = build_completion_chunk_response(decoded_token, "qwen3vl", None, None);
                yield Ok(chunk);
                if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
                pixel_values = None;
                pixel_values_video = None;
            }
            self.qwen3_vl.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
