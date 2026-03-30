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
        glm_asr_nano::{
            config::GlmAsrNanoConfig, model::GlmAsrNanoModel, processor::GlmAsrNanoProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct GlmAsrNanoGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: GlmAsrNanoProcessor,
    glm_asr_nano: GlmAsrNanoModel,
    device: Device,
    dtype: DType,
    eos_token_id1: u32,
    eos_token_id2: u32,
    eos_token_id3: u32,
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
        let glm_asr_nano = GlmAsrNanoModel::new(vb, cfg)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("glm-asr-nano")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            processor,
            glm_asr_nano,
            device,
            dtype,
            eos_token_id1: 59246,
            eos_token_id2: 59253,
            eos_token_id3: 59255,
            model_name,
        })
    }
}

impl<'a> GenerateModel for GlmAsrNanoGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let render_text: String = self.chat_template.apply_chat_template(&mes)?;
        let (input_features, audio_token_lengths, replace_text) =
            self.processor.process_info(&mes, &render_text)?;
        let mut input_ids = self.tokenizer.text_encode(replace_text, &self.device)?;
        let mut input_features = Some(input_features.to_dtype(self.dtype)?);
        let mut audio_token_lengths = Some(audio_token_lengths);
        let mut seq_len = input_ids.dim(1)?;
        let prompt_tokens = seq_len as u32;
        let mut seqlen_offset = 0;
        let mut generate: Vec<u32> = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            let logits = self.glm_asr_nano.forward(
                input_features.as_ref(),
                audio_token_lengths.as_ref(),
                &input_ids,
                seqlen_offset,
            )?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id1
                || next_token == self.eos_token_id2
                || next_token == self.eos_token_id3
            {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            input_features = None;
            audio_token_lengths = None;
        }
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        self.glm_asr_nano.clear_kv_cache();
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
        let render_text = self.chat_template.apply_chat_template(&mes)?;
        let (input_features, audio_token_lengths, replace_text) =
            self.processor.process_info(&mes, &render_text)?;
        let input_ids = self.tokenizer.text_encode(replace_text, &self.device)?;

        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut input_features = Some(input_features.to_dtype(self.dtype)?);
            let mut audio_token_lengths = Some(audio_token_lengths);
            let mut input_ids = input_ids;
            for _ in 0..sample_len {
                let logits =
                    self.glm_asr_nano
                        .forward(input_features.as_ref(), audio_token_lengths.as_ref(), &input_ids, seqlen_offset)?;
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let next_token = logit_processor.sample(&logits)?;
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
                    input_features = None;
                    audio_token_lengths = None;
                    continue;
                }
                error_tokens.clear();
                let chunk = build_completion_chunk_response(decoded_token, &self.model_name, None, None);
                yield Ok(chunk);
                if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 || next_token == self.eos_token_id3{
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                input_features = None;
                audio_token_lengths = None;
            }
            self.glm_asr_nano.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
