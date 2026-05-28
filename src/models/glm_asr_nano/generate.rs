use crate::{
    models::common::{
        MultiModalData,
        generate::{GenerationContext, generate_generic, generate_stream_generic},
    },
    params::chat::{ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse},
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
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
    utils::{find_type_files, get_device, get_dtype},
};

pub struct GlmAsrNanoGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    processor: GlmAsrNanoProcessor,
    model: GlmAsrNanoModel,
    device: Device,
    dtype: DType,
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
        let eos_ids = vec![59246u32, 59253, 59255];
        let model = GlmAsrNanoModel::new(vb, cfg, eos_ids)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("glm-asr-nano")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            processor,
            model,
            device,
            dtype,
            model_name,
        })
    }
}

impl<'a> GenerateModel for GlmAsrNanoGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = mes.seed.unwrap_or(34562) as u64;
        let render_text: String = self.chat_template.apply_chat_template(&mes)?;
        let (input_features, audio_token_lengths, replace_text) =
            self.processor.process_info(&mes, &render_text)?;
        let input_ids = self.tokenizer.text_encode(replace_text, &self.device)?;
        let input_features = input_features.to_dtype(self.dtype)?;
        let audio_token_lengths = Tensor::new(audio_token_lengths, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let mut ctx = GenerationContext::new(
            mes.temperature,
            mes.top_p,
            mes.top_k,
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            input_ids.dim(1)?,
            sample_len,
            self.device.clone(),
        );

        let data_vec = vec![input_features.into(), audio_token_lengths.into()];
        let data = MultiModalData::new(data_vec);
        generate_generic(
            &mut self.model,
            &self.tokenizer,
            input_ids,
            data,
            &mut ctx,
            &self.model_name,
        )
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
        let render_text = self.chat_template.apply_chat_template(&mes)?;
        let (input_features, audio_token_lengths, replace_text) =
            self.processor.process_info(&mes, &render_text)?;
        let input_ids = self.tokenizer.text_encode(replace_text, &self.device)?;
        let input_features = input_features.to_dtype(self.dtype)?;
        let audio_token_lengths = Tensor::new(audio_token_lengths, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let data_vec = vec![input_features.into(), audio_token_lengths.into()];
        let data = MultiModalData::new(data_vec);
        let stream = generate_stream_generic(
            &mut self.model,
            &self.tokenizer,
            input_ids,
            data,
            mes.temperature,
            mes.top_p,
            None,
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            sample_len,
            false,
            &self.device,
            &self.model_name,
        )?;
        Ok(Box::new(Box::pin(stream)))
    }
}
