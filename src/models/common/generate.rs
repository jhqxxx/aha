use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use rocket::async_stream::stream;
use rocket::futures::Stream;
use std::time::Instant;

use crate::{
    models::common::{InferenceModel, MultiModalData},
    params::chat::{ChatCompletionChunkResponse, ChatCompletionResponse},
    tokenizer::TokenizerModel,
    utils::response_utils::{
        build_chunk_response_with_reasoning, build_chunk_response_with_usage,
        build_completion_chunk_response, build_completion_response_with_time,
    },
};
pub fn get_logit_processor(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: u64,
) -> LogitsProcessor {
    let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
    match top_k {
        None => LogitsProcessor::new(
            seed,
            temperature.map(|temp| temp as f64),
            top_p.map(|tp| tp as f64),
        ),
        Some(k) => {
            let sampling = match temperature {
                None => Sampling::ArgMax,
                Some(temperature) => match top_p {
                    None => Sampling::TopK {
                        k,
                        temperature: temperature as f64,
                    },
                    Some(p) => Sampling::TopKThenTopP {
                        k,
                        p: p as f64,
                        temperature: temperature as f64,
                    },
                },
            };
            LogitsProcessor::from_sampling(seed, sampling)
        }
    }
}

pub struct GenerationContext {
    pub logit_processor: LogitsProcessor,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seqlen_offset: usize,
    pub seq_len: usize,
    pub sample_len: u32,
    pub device: Device,
}

impl GenerationContext {
    pub fn new(
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        repeat_penalty: Option<f32>,
        repeat_last_n: Option<usize>,
        seed: u64,
        initial_seq_len: usize,
        max_tokens: u32,
        device: Device,
    ) -> Self {
        Self {
            logit_processor: get_logit_processor(temperature, top_p, top_k, seed),
            repeat_penalty: repeat_penalty.unwrap_or(1.0),
            repeat_last_n: repeat_last_n.unwrap_or(64),
            seqlen_offset: 0,
            seq_len: initial_seq_len,
            sample_len: max_tokens,
            device,
        }
    }

    pub fn prepare_for_next_token(&mut self, token: u32) -> Result<Tensor> {
        self.update_status();
        self.create_input_ids(token)
    }

    fn update_status(&mut self) {
        self.seqlen_offset += self.seq_len;
        self.seq_len = 1;
    }

    fn create_input_ids(&self, token: u32) -> Result<Tensor> {
        Ok(Tensor::from_vec(vec![token], (1, 1), &self.device)?)
    }
}

/// 采样辅助函数
fn sample_and_push(
    ctx: &mut GenerationContext,
    logits: &Tensor,
    generated: &mut Vec<u32>,
) -> Result<u32> {
    let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
    // 重复惩罚
    let logits = if ctx.repeat_penalty == 1. || ctx.repeat_last_n == 0 {
        logits
    } else {
        let start_at = generated.len().saturating_sub(ctx.repeat_last_n);
        candle_transformers::utils::apply_repeat_penalty(
            &logits,
            ctx.repeat_penalty,
            &generated[start_at..],
        )?
    };
    let token = ctx.logit_processor.sample(&logits)?;
    generated.push(token);
    Ok(token)
}
pub fn generate_generic_text<M: InferenceModel>(
    model: &mut M,
    tokenizer: &TokenizerModel,
    input_ids: Tensor,
    data: MultiModalData,
    ctx: &mut GenerationContext,
) -> Result<String> {
    let mut generated = Vec::new();
    let eos_ids = model.stop_token_ids();
    let logits = model.forward_initial(&input_ids, ctx.seqlen_offset, data)?;
    let next_token = sample_and_push(ctx, &logits, &mut generated)?;
    let mut input_ids = ctx.prepare_for_next_token(next_token)?;

    // 自回归循环
    for _ in 1..ctx.sample_len {
        let logits = model.forward_step(&input_ids, ctx.seqlen_offset)?;
        let next_token = sample_and_push(ctx, &logits, &mut generated)?;

        if eos_ids.contains(&next_token) {
            break;
        }
        input_ids = ctx.prepare_for_next_token(next_token)?;
    }
    model.clear_cache();
    let text = tokenizer.token_decode(generated)?;
    Ok(text)
}

pub fn generate_generic<M: InferenceModel>(
    model: &mut M,
    tokenizer: &TokenizerModel,
    input_ids: Tensor,
    data: MultiModalData,
    ctx: &mut GenerationContext,
    model_name: &str,
) -> Result<ChatCompletionResponse> {
    let prompt_tokens = ctx.seq_len as u32;
    let mut generated = Vec::new();
    let eos_ids = model.stop_token_ids();
    let i_start = Instant::now();
    let logits = model.forward_initial(&input_ids, ctx.seqlen_offset, data)?;
    let next_token = sample_and_push(ctx, &logits, &mut generated)?;
    let i_duration = i_start.elapsed();
    let prompt_secs = i_duration.as_secs_f64();
    let mut input_ids = ctx.prepare_for_next_token(next_token)?;

    // 自回归循环
    let i_start = Instant::now();
    for _ in 1..ctx.sample_len {
        let logits = model.forward_step(&input_ids, ctx.seqlen_offset)?;
        let next_token = sample_and_push(ctx, &logits, &mut generated)?;

        if eos_ids.contains(&next_token) {
            break;
        }
        input_ids = ctx.prepare_for_next_token(next_token)?;
    }
    let i_duration = i_start.elapsed();
    let completion_secs = i_duration.as_secs_f64();

    model.clear_cache();

    let num_tokens = generated.len() as u32;
    let text = tokenizer.token_decode(generated)?;
    Ok(build_completion_response_with_time(
        text,
        model_name,
        Some(num_tokens),
        Some(completion_secs),
        Some(prompt_tokens),
        Some(prompt_secs),
    ))
}

pub fn generate_stream_generic_text<M: InferenceModel>(
    model: &mut M,
    tokenizer: &TokenizerModel,
    input_ids: Tensor,
    data: MultiModalData,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: u64,
    max_tokens: u32,
    device: &Device,
) -> Result<impl Stream<Item = Result<String, anyhow::Error>>> {
    let mut ctx = GenerationContext::new(
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        repeat_last_n,
        seed,
        input_ids.dim(1)?,
        max_tokens,
        device.clone(),
    );
    let mut error_tokens = Vec::new();
    let eos_ids = model.stop_token_ids();
    let stream = stream! {
        let mut input_ids = input_ids;
        let mut generated = Vec::new();
        for _ in 0..ctx.sample_len {
            let logits = if ctx.seqlen_offset == 0 {
                model.forward_initial(&input_ids, ctx.seqlen_offset, data.clone())

            } else {
                model.forward_step(&input_ids, ctx.seqlen_offset)
            }?;
            let next_token = sample_and_push(&mut ctx, &logits, &mut generated)?;

            // 解码（处理�的累积）
            let decode_ids = if error_tokens.is_empty() {
                vec![next_token]
            } else {
                let mut ids = error_tokens.clone();
                ids.push(next_token);
                ids
            };

            let decoded = tokenizer.token_decode(decode_ids)?;

            if decoded.contains("�") {
                error_tokens.push(next_token);
                if error_tokens.len() > 3 {
                    error_tokens.clear();
                }
                input_ids = ctx.prepare_for_next_token(next_token)?;
                continue;
            }
            error_tokens.clear();
            yield Ok(decoded);
            if eos_ids.contains(&next_token) {
                break;
            }
            input_ids = ctx.prepare_for_next_token(next_token)?;
        }
        model.clear_cache();
    };
    Ok(stream)
}

pub fn generate_stream_generic<M: InferenceModel>(
    model: &mut M,
    tokenizer: &TokenizerModel,
    input_ids: Tensor,
    data: MultiModalData,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: u64,
    max_tokens: u32,
    in_reasoning: bool,
    device: &Device,
    model_name: &str,
) -> Result<impl Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>> {
    let mut ctx = GenerationContext::new(
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        repeat_last_n,
        seed,
        input_ids.dim(1)?,
        max_tokens,
        device.clone(),
    );
    let prompt_tokens = ctx.seq_len as u32;
    let mut prompt_secs = 0.0f64;
    let mut completion_tokens = 0u32;
    let mut completion_secs = 0.0f64;
    let mut error_tokens = Vec::new();
    let eos_ids = model.stop_token_ids();
    let stream = stream! {
        let mut input_ids = input_ids;
        let mut tool_call_id = None;
        let mut tool_call_content = String::new();
        let mut in_reasoning = in_reasoning;
        let mut generated = Vec::new();
        for _ in 0..ctx.sample_len {
            let i_start = Instant::now();
            let logits = if ctx.seqlen_offset == 0 {
                model.forward_initial(&input_ids, ctx.seqlen_offset, data.clone())

            } else {
                model.forward_step(&input_ids, ctx.seqlen_offset)
            }?;
            let next_token = sample_and_push(&mut ctx, &logits, &mut generated)?;
            completion_tokens += 1;
            let i_duration = i_start.elapsed();
            if ctx.seqlen_offset == 0 {
                prompt_secs += i_duration.as_secs_f64();
            } else {
                completion_secs += i_duration.as_secs_f64();
            };

            // 解码（处理�的累积）
            let decode_ids = if error_tokens.is_empty() {
                vec![next_token]
            } else {
                let mut ids = error_tokens.clone();
                ids.push(next_token);
                ids
            };

            let decoded = tokenizer.token_decode(decode_ids)?;

            if decoded.contains("�") {
                error_tokens.push(next_token);
                if error_tokens.len() > 3 {
                    error_tokens.clear();
                }
                input_ids = ctx.prepare_for_next_token(next_token)?;
                continue;
            }
            error_tokens.clear();
            if decoded.eq("<think>") {
                in_reasoning = true;
                input_ids = ctx.prepare_for_next_token(next_token)?;
                continue;
            }
            if decoded.eq("</think>") {
                in_reasoning = false;
                input_ids = ctx.prepare_for_next_token(next_token)?;
                continue;
            }

            // 处理特殊标记和工具调用
            match decoded.as_str() {
                "<tool_call>" => {
                    // 开始工具调用
                    tool_call_id = Some(uuid::Uuid::new_v4().to_string());
                    input_ids = ctx.prepare_for_next_token(next_token)?;
                    continue;
                }
                "</tool_call>" => {
                    // 结束工具调用
                    let chunk = build_completion_chunk_response(
                        decoded,
                        model_name,
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
                        tool_call_content.push_str(&decoded);
                        input_ids = ctx.prepare_for_next_token(next_token)?;
                        continue;
                    } else {

                        // 正常文本输出
                        let chunk = if in_reasoning {
                            build_chunk_response_with_reasoning(decoded, model_name)
                        } else {
                            build_completion_chunk_response(
                            decoded, model_name,
                            None,
                            None
                        )};
                        yield Ok(chunk);
                    }
                }
            }
            if eos_ids.contains(&next_token) {
                yield Ok(build_chunk_response_with_usage(model_name, completion_tokens.into(), completion_secs.into(), prompt_tokens.into(), prompt_secs.into()));
                break;
            }
            input_ids = ctx.prepare_for_next_token(next_token)?;
        }
        model.clear_cache();
    };
    Ok(stream)
}
