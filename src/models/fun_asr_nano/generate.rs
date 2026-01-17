use std::collections::HashMap;

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, pickle::read_all_with_key};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::{
    models::{
        GenerateModel,
        fun_asr_nano::{
            config::FunASRNanoConfig, model::FunAsrNanoModel, processor::FunAsrNanoProcessor,
        },
        qwen3::config::{Qwen3Config, Qwen3GenerationConfig},
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct FunAsrNanoGenerateModel {
    tokenizer: TokenizerModel,
    processor: FunAsrNanoProcessor,
    fun_asr_nano: FunAsrNanoModel,
    device: Device,
    dtype: DType,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3GenerationConfig,
    model_name: String,
}

impl FunAsrNanoGenerateModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let llm_config_path = path.to_string() + "/Qwen3-0.6B";
        let tokenizer = TokenizerModel::init(&llm_config_path)?;
        let generation_config_path = llm_config_path.clone() + "/generation_config.json";
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let config_path = llm_config_path + "/config.json";
        let llm_cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let config_path = path.to_string() + "/config.yaml";
        let cfg: FunASRNanoConfig = serde_yaml::from_slice(&std::fs::read(config_path)?)?;
        let cfg_dtype = cfg.llm_conf.llm_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let processor = FunAsrNanoProcessor::new(&cfg.frontend_conf, &device)?;
        let model_list = find_type_files(path, "pt")?;
        let mut dict_to_hashmap = HashMap::new();
        for m in model_list {
            let dict = match read_all_with_key(m.clone(), Some("state_dict")) {
                Ok(dict) => dict,
                Err(e) => {
                    println!(
                        "model read_all_with_key {} get state_dict err: {}, use None try again",
                        &m, e
                    );
                    match read_all_with_key(m.clone(), None) {
                        Ok(dict) => dict,
                        Err(e) => {
                            return Err(anyhow!(format!(
                                "model read_all_with_key({}, None): e: {}",
                                &m, e
                            )));
                        }
                    }
                }
            };
            for (k, v) in dict {
                dict_to_hashmap.insert(k, v);
            }
        }
        let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &device);
        let fun_asr_nano = FunAsrNanoModel::new(vb, &cfg, &llm_cfg)?;
        Ok(Self {
            tokenizer,
            processor,
            fun_asr_nano,
            device,
            dtype,
            eos_token_id1: generation_config.eos_token_id[0] as u32,
            eos_token_id2: generation_config.eos_token_id[1] as u32,
            generation_config,
            model_name: "fun-asr-nano".to_string(),
        })
    }
}

impl GenerateModel for FunAsrNanoGenerateModel {
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
        let (speech, fbank_mask, mut input_ids) =
            self.processor.process_info(&mes, &self.tokenizer)?;
        let mut speech = Some(speech.to_dtype(self.dtype)?);
        let mut fbank_mask = Some(&fbank_mask);
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            let logits = self.fun_asr_nano.forward(
                &input_ids,
                speech.as_ref(),
                fbank_mask,
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
            speech = None;
            fbank_mask = None;
        }
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        self.fun_asr_nano.clear_kv_cache();
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
        let (speech, fbank_mask, input_ids) = self.processor.process_info(&mes, &self.tokenizer)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut speech = Some(speech.to_dtype(self.dtype)?);
            let mut fbank_mask = Some(&fbank_mask);
            let mut input_ids = input_ids;
            for _ in 0..sample_len {
                let logits = self.fun_asr_nano.forward(
                    &input_ids,
                    speech.as_ref(),
                    fbank_mask,
                    seqlen_offset,
                )?;
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
                    speech = None;
                    fbank_mask = None;
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
                speech = None;
                fbank_mask = None;
            }
            self.fun_asr_nano.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
