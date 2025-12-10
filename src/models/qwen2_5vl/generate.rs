use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::models::qwen2_5vl::config::Qwen2_5VLConfig;
use crate::utils::{
    build_completion_chunk_response, build_completion_response, find_type_files, get_device,
    get_dtype, get_logit_processor,
};
use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        qwen2_5vl::{model::Qwen2_5VLModel, processor::Qwen2_5VLProcessor},
    },
    tokenizer::TokenizerModel,
};

pub struct Qwen2_5VLGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: Qwen2_5VLProcessor,
    qwen2_5_vl: Qwen2_5VLModel,
    device: Device,
    endoftext_id: u32,
    im_end_id: u32,
    model_name: String,
}

impl<'a> Qwen2_5VLGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen2_5VLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen2_5VLProcessor::new(device, dtype)?;
        let endoftext_id = cfg.bos_token_id;
        let im_end_id = cfg.eos_token_id;
        // let model_list = find_safetensors_files(&path)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let qwen2_5_vl = Qwen2_5VLModel::new(cfg, vb)?;

        Ok(Qwen2_5VLGenerateModel {
            chat_template,
            tokenizer,
            pre_processor,
            qwen2_5_vl,
            device: device.clone(),
            endoftext_id,
            im_end_id,
            model_name: "qwen2.5vl".to_string(),
        })
    }
}

impl<'a> GenerateModel for Qwen2_5VLGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
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
        let second_per_grid_ts = input.second_per_grid_ts.clone();

        let mut mask = Tensor::ones_like(&input_ids)?;
        let mut cache_position = Tensor::ones_like(&input_ids.i(0)?)?
            .to_dtype(candle_core::DType::F64)?
            .cumsum(D::Minus1)?
            .to_dtype(candle_core::DType::U32)?
            .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;

        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            let logits = self.qwen2_5_vl.forward(
                &input_ids,
                pixel_values,
                image_grid_thw,
                pixel_values_video,
                video_grid_thw,
                &mask,
                Some(&cache_position),
                seqlen_offset,
                second_per_grid_ts.clone(),
            )?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.endoftext_id || next_token == self.im_end_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            let appendd_mask = Tensor::ones((1, 1), mask.dtype(), &self.device)?;
            mask = Tensor::cat(&[mask, appendd_mask], 1)?;
            cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
            pixel_values = None;
            pixel_values_video = None;
        }
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        self.qwen2_5_vl.clear_kv_cache();
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
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
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
        let second_per_grid_ts = input.second_per_grid_ts.clone();

        let mut mask = Tensor::ones_like(&input_ids)?;
        let mut cache_position = Tensor::ones_like(&input_ids.i(0)?)?
            .to_dtype(candle_core::DType::F64)?
            .cumsum(D::Minus1)?
            .to_dtype(candle_core::DType::U32)?
            .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;

        let sample_len = mes.max_tokens.unwrap_or(512);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut pixel_values = pixel_values.as_ref();
            let image_grid_thw = image_grid_thw.as_ref();
            let mut pixel_values_video = pixel_values_video.as_ref();
            let video_grid_thw = video_grid_thw.as_ref();
            let mut tool_call_id = None;
            let mut tool_call_content = String::new();
            for _ in 0..sample_len {
                let logits = self.qwen2_5_vl.forward(
                    &input_ids,
                    pixel_values,
                    image_grid_thw,
                    pixel_values_video,
                    video_grid_thw,
                    &mask,
                    Some(&cache_position),
                    seqlen_offset,
                    second_per_grid_ts.clone(),
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
                    let appendd_mask = Tensor::ones((1, 1), mask.dtype(), &self.device)?;
                    mask = Tensor::cat(&[mask, appendd_mask], 1)?;
                    cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
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
                        cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
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
                            cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
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
                // let chunk = build_completion_chunk_response(decoded_token, &self.model_name, None, None);
                // yield Ok(chunk);
                if next_token == self.endoftext_id || next_token == self.im_end_id {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                let appendd_mask = Tensor::ones((1, 1), mask.dtype(), &self.device)?;
                mask = Tensor::cat(&[mask, appendd_mask], 1)?;
                cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
                pixel_values = None;
                pixel_values_video = None;
            }
            self.qwen2_5_vl.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
