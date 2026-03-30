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
        hunyuan_ocr::{
            config::{HunYuanVLConfig, HunyuanOCRGenerationConfig},
            model::HunyuanVLModel,
            processor::HunyuanVLProcessor,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_chunk_response, build_completion_response, find_type_files, get_device,
        get_dtype, get_logit_processor,
    },
};

pub struct HunyuanOCRGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: HunyuanVLProcessor,
    hunyuan_vl: HunyuanVLModel,
    device: Device,
    eos_token_id1: u32,
    eos_token_id2: u32,
    generation_config: HunyuanOCRGenerationConfig,
    model_name: String,
}

impl<'a> HunyuanOCRGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: HunYuanVLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = HunyuanVLProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let hunyuan_vl = HunyuanVLModel::new(vb, cfg.clone())?;
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: HunyuanOCRGenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let model_name = std::path::Path::new(path)
            .file_name() 
            .and_then(|s| s.to_str())
            .unwrap_or("hunyuan_ocr")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            hunyuan_vl,
            device,
            eos_token_id1: generation_config.eos_token_id[0] as u32,
            eos_token_id2: generation_config.eos_token_id[1] as u32,
            generation_config,
            model_name,
        })
    }
}

impl<'a> GenerateModel for HunyuanOCRGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let data = self
            .pre_processor
            .process_info(&mes, &self.tokenizer, &mes_render)?;
        let mut input_ids = data.input_ids;
        let mut position_ids = Some(&data.position_ids);
        let mut image_mask = Some(&data.image_mask);
        let mut pixel_values = data.pixel_values;
        let mut image_grid_thw = data.image_grid_thw;
        let mut seq_len = input_ids.dim(1)?;
        let prompt_tokens = seq_len as u32;
        let mut seqlen_offset = 0;
        let mut generate: Vec<u32> = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            let logits = self.hunyuan_vl.forward(
                &input_ids,
                pixel_values.as_ref(),
                image_grid_thw.as_ref(),
                image_mask,
                position_ids,
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
            position_ids = None;
            image_mask = None;
            pixel_values = None;
            image_grid_thw = None;
        }
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        self.hunyuan_vl.clear_kv_cache();
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
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mut logit_processor =
            get_logit_processor(Some(temperature), Some(top_p), Some(top_k), seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let data = self
            .pre_processor
            .process_info(&mes, &self.tokenizer, &mes_render)?;

        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut input_ids = data.input_ids;
            let mut position_ids = Some(&data.position_ids);
            let mut image_mask = Some(&data.image_mask);
            let mut pixel_values = data.pixel_values;
            let mut image_grid_thw = data.image_grid_thw;
            let mut seq_len = input_ids.dim(1)?;
            for _ in 0..sample_len {
            let logits = self.hunyuan_vl.forward(
                &input_ids,
                pixel_values.as_ref(),
                image_grid_thw.as_ref(),
                image_mask,
                position_ids,
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
                    position_ids = None;
                    image_mask = None;
                    pixel_values = None;
                    image_grid_thw = None;
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
                position_ids = None;
                image_mask = None;
                pixel_values = None;
                image_grid_thw = None;
            }
            self.hunyuan_vl.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
