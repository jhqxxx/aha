use crate::utils::build_completion_chunk_response;
use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        lfm2::{
            config::{Lfm2Config, Lfm2GenerateConfig},
            model::Lfm2Model,
        },
    },
    tokenizer::TokenizerModel,
    utils::{
        build_completion_response, find_type_files, get_device, get_dtype, get_logit_processor,
    },
};
use aha_openai_dive::v1::resources::chat::{ChatCompletionParameters, ChatCompletionResponse};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rocket::async_stream::stream;

pub struct Lfm2GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    device: Device,
    model: Lfm2Model,
    eos_token_id: u32,
    model_name: String,
}
impl<'a> Lfm2GenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let device = get_device(device);
        let gen_cfg_path = path.to_string() + "/generation_config.json";
        let gen_cfg: Lfm2GenerateConfig = serde_json::from_slice(&std::fs::read(gen_cfg_path)?)?;
        let cfg_path = path.to_string() + "/config.json";
        let cfg: Lfm2Config = serde_json::from_slice(&std::fs::read(cfg_path)?)?;
        let model_path = find_type_files(path, "safetensors")?;
        let cfg_dtype = if let Some(dtype) = &cfg.dtype {
            dtype.clone()
        } else if let Some(dtype) = &cfg.torch_dtype {
            dtype.clone()
        } else {
            "bfloat16".to_string()
        };
        let dtype = get_dtype(dtype, &cfg_dtype);
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_path, dtype, &device)? };
        let model = Lfm2Model::new(vb, &cfg)?;
        let eos_token_id = gen_cfg.eos_token_id;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("lfm2")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            device,
            model,
            eos_token_id,
            model_name,
        })
    }
}

impl<'a> GenerateModel for Lfm2GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut logits = get_logit_processor(
            mes.temperature,
            mes.top_p,
            None,
            mes.seed.unwrap_or(34562) as u64,
        );
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let prompt_tokens = seq_len as u32;
        let mut seqlen_offset = 0;
        let mut generate = vec![];
        let sample_len = mes.max_tokens.unwrap_or(1024);
        for _ in 0..sample_len {
            let logit = self.model.forward(&input_ids, seqlen_offset)?;
            let logit = logit.squeeze(0)?.squeeze(0)?;
            let next_token = logits.sample(&logit)?;
            generate.push(next_token);
            if next_token == self.eos_token_id {
                break;
            }
            input_ids = Tensor::new(vec![next_token], &self.device)?.unsqueeze(0)?;
            seqlen_offset += seq_len;
            seq_len = 1;
        }
        self.model.clear_cache();
        let completion_tokens = generate.len() as u32;
        let decode = self.tokenizer.token_decode(generate)?;
        let mes = build_completion_response(
            decode,
            &self.model_name,
            Some(completion_tokens),
            Some(prompt_tokens),
        );
        Ok(mes)
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn rocket::futures::Stream<
                    Item = Result<
                        aha_openai_dive::v1::resources::chat::ChatCompletionChunkResponse,
                        anyhow::Error,
                    >,
                > + Send
                + Unpin
                + '_,
        >,
    > {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let mut logits = get_logit_processor(
            mes.temperature,
            mes.top_p,
            None,
            mes.seed.unwrap_or(34562) as u64,
        );
        let mut input_ids = self.tokenizer.text_encode(mes_render, &self.device)?;
        let mut seq_len = input_ids.dim(1)?;
        let mut seqlen_offset = 0;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let stream = stream! {
            let mut err_tokens = vec![];
            for _ in 0..sample_len {
                let logit = self.model.forward(&input_ids, seqlen_offset)?;
                let logit = logit.squeeze(0)?.squeeze(0)?;
                let next_token = logits.sample(&logit)?;
                let mut decode_ids = vec![];
                if !err_tokens.is_empty() {
                    decode_ids.extend_from_slice(&err_tokens);
                }
                decode_ids.push(next_token);
                let decode = self.tokenizer.token_decode(decode_ids)?;
                if decode.contains("�") {
                    err_tokens.push(next_token);
                    if err_tokens.len() > 3 {
                        err_tokens.clear();
                    }
                    input_ids = Tensor::new(vec![next_token], &self.device)?.unsqueeze(0)?;
                    seqlen_offset += seq_len;
                    seq_len = 1;
                    continue;
                }
                err_tokens.clear();
                let chunk = build_completion_chunk_response(decode, &self.model_name, None, None);
                yield Ok(chunk);
                if next_token == self.eos_token_id {
                    break;
                }
                input_ids = Tensor::new(vec![next_token], &self.device)?.unsqueeze(0)?;
                seqlen_offset += seq_len;
                seq_len = 1;
            }
            self.model.clear_cache();
        };
        Ok(Box::new(Box::pin(stream)))
    }
}
