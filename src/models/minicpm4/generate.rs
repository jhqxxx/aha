use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;
use rocket::futures::Stream;

use crate::models::minicpm4::config::MiniCPM4Config;
use crate::models::minicpm4::model::MiniCPMModel;
// use crate::models::GenerateStream;
use crate::utils::{
    build_completion_chunk_response, build_completion_response, find_type_files, get_device,
    get_dtype, get_logit_processor,
};
use crate::{chat_template::ChatTemplate, models::GenerateModel, tokenizer::TokenizerModel};

pub struct MiniCPMGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    minicpm: MiniCPMModel,
    device: Device,
    endoftext_id: u32,
    im_end_id: u32,
}

impl<'a> MiniCPMGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: MiniCPM4Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = &get_device(device);
        let cfg_dtype = cfg.torch_dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let endoftext_id = cfg.eos_token_id[0];
        let im_end_id = cfg.eos_token_id[1];
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let minicpm = MiniCPMModel::new(vb, cfg)?;

        Ok(MiniCPMGenerateModel {
            chat_template,
            tokenizer,
            minicpm,
            device: device.clone(),
            endoftext_id,
            im_end_id,
        })
    }
}

impl<'a> GenerateModel for MiniCPMGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = match mes.seed {
            None => 34562u64,
            Some(s) => s as u64,
        };
        let mut logit_processor = get_logit_processor(mes.temperature, mes.top_p, None, seed);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let mut generate = Vec::new();
        let sample_len = mes.max_tokens.unwrap_or(2048);
        for _ in 0..sample_len {
            let logits = self.minicpm.forward_with_cache(&input_ids, seqlen_offset)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            generate.push(next_token);
            if next_token == self.endoftext_id || next_token == self.im_end_id {
                break;
            }
            seqlen_offset += seq_len;
            seq_len = 1;
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
        }
        let res = self.tokenizer.token_decode(generate)?;
        self.minicpm.clear_kv_cache();
        let response = build_completion_response(res, "minicpm");
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
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(512);
        let stream = stream! {
            let mut error_tokens = Vec::new();
            for _ in 0..sample_len {
                let logits = self.minicpm.forward_with_cache(
                    &input_ids,
                    seqlen_offset,
                )?;
                let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
                let next_token = logit_processor.sample(&logits)?;
                let mut decode_ids = Vec::new();
                if !error_tokens.is_empty(){
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
                    continue;
                }
                error_tokens.clear();
                let chunk = build_completion_chunk_response(decoded_token, "minicpm", None, None);
                yield Ok(chunk);
                if next_token == self.endoftext_id || next_token == self.im_end_id {
                    break;
                }
                seqlen_offset += seq_len;
                seq_len = 1;
                input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;

            }
            self.minicpm.clear_kv_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
