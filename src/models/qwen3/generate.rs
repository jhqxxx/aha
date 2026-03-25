use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor, quantized::gguf_file};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3Model;
use rocket::async_stream::stream;
use rocket::futures::Stream;
use std::path::Path;

use crate::models::qwen3::config::{Qwen3Config, Qwen3GenerationConfig};
use crate::models::qwen3::model::Qwen3Model;
use crate::models::qwen3::onnx::Qwen3OnnxBackend;
// use crate::models::GenerateStream;
use crate::utils::{
    build_completion_chunk_response, build_completion_response, find_type_files, get_device,
    get_dtype, get_logit_processor,
};
use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        artifact::{ArtifactKind, LoadSpec},
        common::{gguf::load_text_bootstrap_from_gguf, onnx::resolve_tokenizer_dir},
    },
    tokenizer::TokenizerModel,
};

enum Qwen3Runtime {
    Safetensors(Qwen3Model),
    Gguf(QuantizedQwen3Model),
    Onnx(Qwen3OnnxBackend),
}

pub struct Qwen3GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    runtime: Qwen3Runtime,
    device: Device,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: Qwen3GenerationConfig,
    model_name: String,
}

impl<'a> Qwen3GenerateModel<'a> {
    pub fn init_from_spec(
        spec: &LoadSpec,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        match spec.resolved_artifact() {
            ArtifactKind::Safetensors => {
                let path = spec
                    .paths
                    .weight_dir
                    .as_deref()
                    .ok_or_else(|| anyhow!("weight_path is required for qwen3 safetensors"))?;
                Self::init(path, device, dtype)
            }
            ArtifactKind::Gguf => {
                let gguf = spec
                    .paths
                    .gguf_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("gguf_path is required for qwen3 gguf"))?;
                Self::init_from_gguf(gguf, device)
            }
            ArtifactKind::Onnx => {
                let onnx_path = spec
                    .paths
                    .onnx_path
                    .as_deref()
                    .ok_or_else(|| anyhow!("onnx_path is required for qwen3 onnx"))?;
                Self::init_from_onnx(onnx_path, spec.paths.tokenizer_dir.as_deref())
            }
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
        }
    }

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let qwen3 = Qwen3Model::new(&cfg, vb)?;
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let (eos_token_id1, eos_token_id2) = resolve_eos_ids(&generation_config, cfg.eos_token_id);
        let model_name = Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("qwen3")
            .to_string();

        Ok(Qwen3GenerateModel {
            chat_template,
            tokenizer,
            runtime: Qwen3Runtime::Safetensors(qwen3),
            device: device.clone(),
            eos_token_id1,
            eos_token_id2,
            generation_config,
            model_name,
        })
    }

    pub fn init_from_gguf(model_file: &str, device: Option<&Device>) -> Result<Self> {
        if !model_file.ends_with(".gguf") {
            return Err(anyhow!(
                "qwen3 gguf model path must end with .gguf: {model_file}"
            ));
        }

        let device = get_device(device);
        let bootstrap =
            load_text_bootstrap_from_gguf(model_file, Some(false), Some(false), Some(false))?;
        let chat_template = if let Some(chat_template_str) = bootstrap.chat_template {
            ChatTemplate::str_init(&chat_template_str)?
        } else {
            let parent = Path::new(model_file)
                .parent()
                .and_then(|path| path.to_str())
                .ok_or_else(|| anyhow!("cannot resolve gguf parent directory for {model_file}"))?;
            ChatTemplate::init(parent)?
        };
        let eos_token_id = bootstrap.eos_token_id.unwrap_or(151_645);
        let generation_config = default_generation_config(eos_token_id);
        let (eos_token_id1, eos_token_id2) = resolve_eos_ids(&generation_config, eos_token_id);

        let mut reader = std::fs::File::open(model_file)?;
        let content = gguf_file::Content::read(&mut reader)?;
        let qwen3 = QuantizedQwen3Model::from_gguf(content, &mut reader, &device)?;
        let model_name = Path::new(model_file)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("qwen3")
            .to_string();

        Ok(Self {
            chat_template,
            tokenizer: bootstrap.tokenizer,
            runtime: Qwen3Runtime::Gguf(qwen3),
            device,
            eos_token_id1,
            eos_token_id2,
            generation_config,
            model_name,
        })
    }

    pub fn init_from_onnx(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        let tokenizer_dir =
            resolve_tokenizer_dir(onnx_path, tokenizer_dir, &["tokenizer.json", "config.json"])?;
        let base_path = tokenizer_dir.to_string_lossy().to_string();
        let chat_template = ChatTemplate::init(&base_path)?;
        let tokenizer = TokenizerModel::init(&base_path)?;
        let config_path = tokenizer_dir.join("config.json");
        let cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;

        let generation_config_path = tokenizer_dir.join("generation_config.json");
        let generation_config: Qwen3GenerationConfig = if generation_config_path.exists() {
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?
        } else {
            default_generation_config(cfg.eos_token_id)
        };
        let (eos_token_id1, eos_token_id2) = resolve_eos_ids(&generation_config, cfg.eos_token_id);
        let model_name = tokenizer_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("qwen3")
            .to_string();
        let backend = Qwen3OnnxBackend::load(onnx_path)?;

        Ok(Self {
            chat_template,
            tokenizer,
            runtime: Qwen3Runtime::Onnx(backend),
            device: Device::Cpu,
            eos_token_id1,
            eos_token_id2,
            generation_config,
            model_name,
        })
    }

    fn forward_logits(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let logits = match &mut self.runtime {
            Qwen3Runtime::Safetensors(model) => model
                .forward(Some(input_ids), None, seqlen_offset)?
                .squeeze(0)?
                .squeeze(0)?,
            Qwen3Runtime::Gguf(model) => model.forward(input_ids, seqlen_offset)?.squeeze(0)?,
            Qwen3Runtime::Onnx(_) => {
                return Err(anyhow!("onnx runtime should use vector input path"));
            }
        };
        logits.to_dtype(DType::F32).map_err(Into::into)
    }

    fn clear_runtime_cache(&mut self) {
        match &mut self.runtime {
            Qwen3Runtime::Safetensors(model) => model.clear_kv_cache(),
            Qwen3Runtime::Gguf(model) => model.clear_kv_cache(),
            Qwen3Runtime::Onnx(backend) => backend.clear_cache(),
        }
    }

    fn generate_with_onnx(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<ChatCompletionResponse> {
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut current_ids = self.tokenizer.text_encode_vec(mes_render, true)?;
        let prompt_tokens = current_ids.len() as u32;
        let mut position_start = 0usize;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);
        let onnx_backend = match &mut self.runtime {
            Qwen3Runtime::Onnx(backend) => backend,
            _ => return Err(anyhow!("qwen3 onnx runtime is not initialized")),
        };
        for _ in 0..sample_len {
            let logits = onnx_backend.forward_logits(&current_ids, position_start)?;
            let vocab_size = logits.len();
            let logits = Tensor::from_vec(logits, vocab_size, &self.device)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                break;
            }
            position_start += current_ids.len();
            current_ids = vec![next_token];
        }
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        onnx_backend.clear_cache();
        Ok(build_completion_response(
            res,
            &self.model_name,
            Some(num_token),
            Some(prompt_tokens),
        ))
    }

    fn generate_stream_with_onnx(
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
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let tokenizer = &self.tokenizer;
        let model_name = self.model_name.clone();
        let eos_token_id1 = self.eos_token_id1;
        let eos_token_id2 = self.eos_token_id2;
        let device = self.device.clone();
        let mut current_ids = tokenizer.text_encode_vec(mes_render, true)?;
        let mut position_start = 0usize;
        let sample_len = mes.max_tokens.unwrap_or(512);
        let onnx_backend = match &mut self.runtime {
            Qwen3Runtime::Onnx(backend) => backend,
            _ => return Err(anyhow!("qwen3 onnx runtime is not initialized")),
        };
        let stream = stream! {
            let mut error_tokens = Vec::new();
            for _ in 0..sample_len {
                let logits = onnx_backend.forward_logits(&current_ids, position_start)?;
                let vocab_size = logits.len();
                let logits = Tensor::from_vec(logits, vocab_size, &device)?;
                let next_token = logit_processor.sample(&logits)?;
                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty(){
                    decode_ids.extend_from_slice(&error_tokens);
                }
                decode_ids.push(next_token);
                let decoded_token = tokenizer.token_decode(decode_ids).map_err(|e| anyhow!(format!("stream decode error{e}")))?;
                if decoded_token.contains("�") {
                    error_tokens.push(next_token);
                    if error_tokens.len() > 3 {
                        error_tokens.clear();
                    }
                    position_start += current_ids.len();
                    current_ids = vec![next_token];
                    continue;
                }
                error_tokens.clear();
                let chunk = build_completion_chunk_response(decoded_token, &model_name, None, None);
                yield Ok(chunk);
                if next_token == eos_token_id1 || next_token == eos_token_id2 {
                    break;
                }
                position_start += current_ids.len();
                current_ids = vec![next_token];
            }
            onnx_backend.clear_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}

impl<'a> GenerateModel for Qwen3GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        if matches!(self.runtime, Qwen3Runtime::Onnx(_)) {
            return self.generate_with_onnx(mes);
        }

        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);

        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        // let enable_thinking = extract_metadata_value::<bool>(&mes.metadata, "enable_thinking");
        // let mes_render = self
        //     .chat_template
        //     .apply_chat_temp_think(&mes, enable_thinking)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let prompt_tokens = seq_len as u32;
        let mut seqlen_offset = 0;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);
        for _ in 0..sample_len {
            let logits = self.forward_logits(&input_ids, seqlen_offset)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.eos_token_id1 || next_token == self.eos_token_id2 {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        self.clear_runtime_cache();
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
        if matches!(self.runtime, Qwen3Runtime::Onnx(_)) {
            return self.generate_stream_with_onnx(mes);
        }

        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        // let enable_thinking = extract_metadata_value::<bool>(&mes.metadata, "enable_thinking");
        // let mes_render = self
        //     .chat_template
        //     .apply_chat_temp_think(&mes, enable_thinking)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(512);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            for _ in 0..sample_len {
                let logits = self.forward_logits(&input_ids, seqlen_offset)?;
                let next_token = logit_processor.sample(&logits)?;
                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty(){
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

            }
            self.clear_runtime_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}

fn default_generation_config(eos_token_id: u32) -> Qwen3GenerationConfig {
    Qwen3GenerationConfig {
        bos_token_id: eos_token_id as usize,
        pad_token_id: eos_token_id as usize,
        do_sample: true,
        eos_token_id: vec![eos_token_id as usize],
        top_p: 0.95,
        top_k: 20,
        temperature: 0.4,
        repetition_penalty: 1.0,
    }
}

fn resolve_eos_ids(generation_config: &Qwen3GenerationConfig, fallback: u32) -> (u32, u32) {
    let eos_token_id1 = generation_config
        .eos_token_id
        .first()
        .copied()
        .unwrap_or(fallback as usize) as u32;
    let eos_token_id2 = generation_config
        .eos_token_id
        .get(1)
        .copied()
        .unwrap_or(eos_token_id1 as usize) as u32;
    (eos_token_id1, eos_token_id2)
}
