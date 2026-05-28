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
        qwen3::config::Qwen3GenerationConfig,
        qwen3vl::{config::Qwen3VLConfig, model::Qwen3VLModel, processor::Qwen3VLProcessor},
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub struct Qwen3VLGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: Qwen3VLProcessor,
    model: Qwen3VLModel,
    device: Device,
    generation_config: Qwen3GenerationConfig,
    model_name: String,
}

impl<'a> Qwen3VLGenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen3VLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen3VLProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let generation_config_path = path.to_string() + "/generation_config.json";
        let generation_config: Qwen3GenerationConfig =
            serde_json::from_slice(&std::fs::read(generation_config_path)?)?;
        let model = Qwen3VLModel::new(cfg, vb, generation_config.eos_token_id.clone())?;

        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3vl")
            .to_string();
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            model,
            device,
            generation_config,
            model_name,
        })
    }
}

impl<'a> GenerateModel for Qwen3VLGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.device)?;
        let seq_len = input_ids.dim(1)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let mut ctx = GenerationContext::new(
            temperature.into(),
            top_p.into(),
            top_k.into(),
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            input_ids.dim(1)?,
            sample_len,
            self.device.clone(),
        );
        let cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?;
        let data_vec = vec![
            input.pixel_values,
            input.image_grid_thw,
            input.pixel_values_video,
            input.video_grid_thw,
            cache_position.into(),
        ];
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
        let temperature = mes
            .temperature
            .unwrap_or(self.generation_config.temperature);
        let top_p = mes.top_p.unwrap_or(self.generation_config.top_p);
        let top_k = self.generation_config.top_k;
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let in_reasoning = mes_render.ends_with("<think>\n");
        let input = self.pre_processor.process_info(&mes, &mes_render)?;
        let input_ids = self
            .tokenizer
            .text_encode(input.replace_text.clone(), &self.device)?;
        let seq_len = input_ids.dim(1)?;
        let cache_position = Tensor::arange(0u32, seq_len as u32, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let data_vec = vec![
            input.pixel_values,
            input.image_grid_thw,
            input.pixel_values_video,
            input.video_grid_thw,
            cache_position.into(),
        ];
        let data = MultiModalData::new(data_vec);
        let seed = mes.seed.unwrap_or(34562) as u64;
        let stream = generate_stream_generic(
            &mut self.model,
            &self.tokenizer,
            input_ids,
            data,
            temperature.into(),
            top_p.into(),
            top_k.into(),
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            sample_len,
            in_reasoning,
            &self.device,
            &self.model_name,
        )?;
        Ok(Box::new(Box::pin(stream)))
    }
}
