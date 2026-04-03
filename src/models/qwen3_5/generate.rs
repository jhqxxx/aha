use crate::{
    models::common::{
        MultiModalData,
        generate::{GenerationContext, generate_generic, generate_stream_generic},
    },
    params::chat::{ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse},
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, quantized::gguf_file};
use candle_nn::VarBuilder;
use rocket::futures::Stream;

use crate::{
    chat_template::ChatTemplate,
    models::{
        GenerateModel,
        common::gguf::Gguf,
        qwen3_5::{config::Qwen3_5Config, model::Qwen3_5Model},
        qwen3vl::processor::Qwen3VLProcessor,
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub struct Qwen3_5GenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: Option<Qwen3VLProcessor>,
    qwen3_5: Qwen3_5Model,
    device: Device,
    model_name: String,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl<'a> Qwen3_5GenerateModel<'a> {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3.5");
        let chat_template = ChatTemplate::init(path)?;
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen3_5Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let cfg_dtype = cfg.text_config.dtype.as_str();
        let dtype = get_dtype(dtype, cfg_dtype);
        let pre_processor = Qwen3VLProcessor::new(path, &device, dtype)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let eos_ids = vec![cfg.text_config.eos_token_id];
        let qwen3_5 = Qwen3_5Model::new_from_vb(vb, cfg, eos_ids)?;

        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor: Some(pre_processor),
            qwen3_5,
            device,
            model_name: model_name.to_string(),
            repeat_penalty: 1.0,
            repeat_last_n: 64,
        })
    }

    pub fn init_from_gguf(
        model_file: &str,
        mmproj_file: Option<&str>,
        device: Option<&Device>,
    ) -> Result<Self> {
        if !model_file.contains("Qwen3.5") || !model_file.ends_with("gguf") {
            return Err(anyhow!("Qwen3.5 gguf model file name illigal {model_file}"));
        }
        if let Some(mmproj) = mmproj_file
            && (!mmproj.contains("mmproj") || !mmproj.ends_with("gguf"))
        {
            return Err(anyhow!("Qwen3.5 mmproj_file name illigal {model_file}"));
        }

        let mut reader = std::fs::File::open(model_file)?;
        let content = gguf_file::Content::read(&mut reader)?;
        let device = get_device(device);
        let mut model_gguf = Gguf::new(content, reader, device.clone());

        let chat_template_str = model_gguf
            .get_matedata("tokenizer.chat_template")?
            .to_string()?
            .clone();
        let chat_template = ChatTemplate::str_init(&chat_template_str)?;
        let tokenizer = model_gguf.build_tokenizer(Some(false), Some(false), Some(false))?;
        let (pre_processor, mut mmproj_gguf) = if let Some(mmproj_f) = mmproj_file {
            let mut reader = std::fs::File::open(mmproj_f)?;
            let content = gguf_file::Content::read(&mut reader)?;
            let mmproj_gguf = Gguf::new(content, reader, device.clone());
            let processor = Qwen3VLProcessor::new_qwen3_5_default(&device, DType::F32)?;
            (Some(processor), Some(mmproj_gguf))
        } else {
            (None, None)
        };

        let eos_token_id = model_gguf
            .get_matedata("tokenizer.ggml.eos_token_id")?
            .to_u32()?;
        let eos_ids = vec![eos_token_id];
        let qwen3_5 =
            Qwen3_5Model::new_from_gguf(&mut model_gguf, mmproj_gguf.as_mut(), &device, eos_ids)?;
        let stem = std::path::Path::new(model_file)
            .file_stem() // 获取文件名主干（不含扩展名）
            .and_then(|s| s.to_str())
            .unwrap_or("qwen3.5");
        Ok(Self {
            chat_template,
            tokenizer,
            pre_processor,
            qwen3_5,
            device,
            model_name: stem.to_string(),
            repeat_penalty: 1.2,
            repeat_last_n: 64,
        })
    }
}

impl<'a> GenerateModel for Qwen3_5GenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let seed = mes.seed.unwrap_or(32768) as u64;
        let temperature = mes.temperature.unwrap_or(0.4);
        let top_p = mes.top_p.unwrap_or(0.95);
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (mes_text, pixel_values, image_grid_thw, pixel_values_video, video_grid_thw) =
            if let Some(processor) = &self.pre_processor {
                let input = processor.process_info(&mes, &mes_render)?;
                (
                    input.replace_text,
                    input.pixel_values,
                    input.image_grid_thw,
                    input.pixel_values_video,
                    input.video_grid_thw,
                )
            } else {
                (mes_render, None, None, None, None)
            };
        let input_ids = self.tokenizer.text_encode(mes_text, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let mut ctx = GenerationContext::new(
            temperature.into(),
            top_p.into(),
            Some(20),
            self.repeat_penalty.into(),
            self.repeat_last_n.into(),
            seed,
            input_ids.dim(1)?,
            sample_len,
            self.device.clone(),
        );
        let data_vec = vec![
            pixel_values,
            image_grid_thw,
            pixel_values_video,
            video_grid_thw,
        ];
        let data = MultiModalData::new(data_vec);
        generate_generic(
            &mut self.qwen3_5,
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
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let in_reasoning = mes_render.ends_with("<think>\n");
        let (mes_text, pixel_values, image_grid_thw, pixel_values_video, video_grid_thw) =
            if let Some(processor) = &self.pre_processor {
                let input = processor.process_info(&mes, &mes_render)?;
                (
                    input.replace_text,
                    input.pixel_values,
                    input.image_grid_thw,
                    input.pixel_values_video,
                    input.video_grid_thw,
                )
            } else {
                (mes_render, None, None, None, None)
            };
        let input_ids = self.tokenizer.text_encode(mes_text, &self.device)?;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let data_vec = vec![
            pixel_values,
            image_grid_thw,
            pixel_values_video,
            video_grid_thw,
        ];
        let data = MultiModalData::new(data_vec);
        let seed = mes.seed.unwrap_or(34562) as u64;
        let stream = generate_stream_generic(
            &mut self.qwen3_5,
            &self.tokenizer,
            input_ids,
            data,
            mes.temperature,
            mes.top_p,
            None,
            self.repeat_penalty.into(),
            self.repeat_last_n.into(),
            seed,
            sample_len,
            in_reasoning,
            &self.device,
            &self.model_name,
        )?;
        Ok(Box::new(Box::pin(stream)))
    }
}
