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
        feature_extractor::config::FeatureExtractor,
        qwen3_asr::{
            config::{Qwen3ASRConfig, Qwen3ASRGenerationConfig},
            model::Qwen3ASRModel,
            processor::Qwen3AsrProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct Qwen3AsrGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: Qwen3AsrProcessor,
    qwen3_asr: Qwen3ASRModel,
    device: Device,
    dtype: DType,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3ASRGenerationConfig,
    model_name: String,
}

impl<'a> Qwen3AsrGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: Qwen3ASRGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let device = get_device(device);
        let preprocess_config_path = path.to_string() + "/preprocessor_config.json";
        let preprocess_config: FeatureExtractor =
            serde_json::from_slice(&std::fs::read(preprocess_config_path)?)?;
        let processor = Qwen3AsrProcessor::new(&device, &preprocess_config)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen3ASRConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let cfg_dtype = cfg.thinker_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let qwen3_asr = Qwen3ASRModel::new(vb, &cfg)?;

        Ok(Self {
            chat_template,
            tokenizer,
            processor,
            qwen3_asr,
            device,
            dtype,
            eos_token_id1: generation_config.eos_token_id[0] as u32,
            eos_token_id2: generation_config.eos_token_id[1] as u32,
            generation_config,
            model_name: "qwen3-asr".to_string(),
        })
    }
}

impl<'a> GenerateModel for Qwen3AsrGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let temperature = match mes.temperature {
            None => self.generation_config.temperature,
            Some(tem) => tem,
        };
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor = get_logit_processor(Some(temperature), mes.top_p, None, seed);
        let render_text = self.chat_template.apply_chat_template(&mes)?;
        let audio_datas = self
            .processor
            .process_info(&mes, &render_text, &self.tokenizer)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let mut generate = Vec::new();
        for data in audio_datas.iter() {
            let mut input_ids = data.input_ids.clone();
            let mut input_features = Some(data.input_features.clone().to_dtype(self.dtype)?);
            let mut seq_len = input_ids.dim(1)?;
            let mut seqlen_offset = 0;
            for _ in 0..sample_len {
                let logits =
                    self.qwen3_asr
                        .forward(&input_ids, seqlen_offset, input_features.as_ref())?;
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let next_token = logit_processor.sample(&logits)?;
                generate.push(next_token);
                if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                input_features = None;
            }
            self.qwen3_asr.clear_kv_cache();
        }
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
        let temperature = match mes.temperature {
            None => self.generation_config.temperature,
            Some(tem) => tem,
        };
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor = get_logit_processor(Some(temperature), mes.top_p, None, seed);
        let render_text = self.chat_template.apply_chat_template(&mes)?;
        let audio_datas = self
            .processor
            .process_info(&mes, &render_text, &self.tokenizer)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            for data in audio_datas.iter() {
                let mut input_ids = data.input_ids.clone();
                let mut input_features = Some(data.input_features.clone().to_dtype(self.dtype)?);
                let mut seq_len = input_ids.dim(1)?;
                let mut seqlen_offset = 0;
                for _ in 0..sample_len {
                    let logits =
                        self.qwen3_asr
                            .forward(&input_ids, seqlen_offset, input_features.as_ref())?;
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
                        continue;
                    }
                    error_tokens.clear();
                    let chunk = build_completion_chunk_response(decoded_token, &self.model_name, None, None);
                    yield Ok(chunk);
                    if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                        break;
                    }
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                    input_features = None;
                }
                self.qwen3_asr.clear_kv_cache();
            }
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
