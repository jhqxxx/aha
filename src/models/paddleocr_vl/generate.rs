use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::models::paddleocr_vl::config::{PaddleOCRVLConfig, PaddleOCRVLPreprocessorConfig};
use crate::models::paddleocr_vl::model::PaddleOCRVLModel;
use crate::models::paddleocr_vl::processor::PaddleOCRVLProcessor;
use crate::utils::tensor_utils::get_equal_mask;
use crate::utils::{
    build_completion_chunk_response, build_completion_response, find_type_files, get_device,
    get_dtype, get_logit_processor,
};
use crate::{chat_template::ChatTemplate, models::GenerateModel, tokenizer::TokenizerModel};

pub struct PaddleOCRVLGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: PaddleOCRVLProcessor,
    paddleocr_vl: PaddleOCRVLModel,
    cfg: PaddleOCRVLConfig,
    device: Device,
    end_token_id: u32,
    model_name: String,
}

impl<'a> PaddleOCRVLGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: PaddleOCRVLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let processor_cfg_path = path.to_string() + "/preprocessor_config.json";
        let processor_cfg: PaddleOCRVLPreprocessorConfig =
            serde_json::from_slice(&std::fs::read(processor_cfg_path)?)?;
        let pre_processor = PaddleOCRVLProcessor::new(processor_cfg, device, dtype)?;
        let end_token_id = 2;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let paddleocr_vl = PaddleOCRVLModel::new(cfg.clone(), vb)?;

        Ok(PaddleOCRVLGenerateModel {
            chat_template,
            tokenizer,
            pre_processor,
            paddleocr_vl,
            cfg,
            device: device.clone(),
            end_token_id,
            model_name: "paddleocr_vl".to_string(),
        })
    }
}

impl<'a> GenerateModel for PaddleOCRVLGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (replace_text, mut pixel_values, mut image_grid_thw) =
            self.pre_processor.process_info(&mes, &mes_render)?;
        let mut input_ids = self.tokenizer.text_encode(replace_text, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let image_mask = get_equal_mask(&input_ids, self.cfg.image_token_id)?;

        let mut cache_position = Tensor::ones_like(&input_ids.i(0)?)?
            .to_dtype(candle_core::DType::F64)?
            .cumsum(D::Minus1)?
            .to_dtype(candle_core::DType::U32)?
            .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;

        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            let logits = self.paddleocr_vl.forward(
                &input_ids,
                pixel_values.as_ref(),
                image_grid_thw.as_ref(),
                &image_mask,
                Some(&cache_position),
                seqlen_offset,
            )?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.end_token_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
            pixel_values = None;
            image_grid_thw = None;
        }
        let num_token = generate.len() as u32;
        let res = self.tokenizer.token_decode(generate)?;
        self.paddleocr_vl.clear_kv_cache();
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
        let (replace_text, pixel_values, image_grid_thw) =
            self.pre_processor.process_info(&mes, &mes_render)?;
        let mut input_ids = self.tokenizer.text_encode(replace_text, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let image_mask = get_equal_mask(&input_ids, self.cfg.image_token_id)?;

        let mut cache_position = Tensor::ones_like(&input_ids.i(0)?)?
            .to_dtype(candle_core::DType::F64)?
            .cumsum(D::Minus1)?
            .to_dtype(candle_core::DType::U32)?
            .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;

        let sample_len = mes.max_tokens.unwrap_or(1024);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            let mut pixel_values = pixel_values.as_ref();
            let mut image_grid_thw = image_grid_thw.as_ref();
            for _ in 0..sample_len {
                let logits = self.paddleocr_vl.forward(
                    &input_ids,
                    pixel_values,
                    image_grid_thw,
                    &image_mask,
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
                    image_grid_thw = None;
                    continue;
                }
                error_tokens.clear();
                let chunk = build_completion_chunk_response(decoded_token, &self.model_name, None, None);
                yield Ok(chunk);
                if next_token == self.end_token_id {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
                cache_position = Tensor::from_vec(vec![seqlen_offset as u32], 1, &self.device)?;
                pixel_values = None;
                image_grid_thw = None;
            }
            self.paddleocr_vl.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
