use crate::models::common::MultiModalData;
use crate::models::common::generate::{
    GenerationContext, generate_generic, generate_stream_generic,
};
use crate::params::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::Result;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use rocket::futures::Stream;

use crate::models::paddleocr_vl::config::{PaddleOCRVLConfig, PaddleOCRVLPreprocessorConfig};
use crate::models::paddleocr_vl::model::PaddleOCRVLModel;
use crate::models::paddleocr_vl::processor::PaddleOCRVLProcessor;
use crate::utils::tensor_utils::get_equal_mask;
use crate::utils::{find_type_files, get_device, get_dtype};
use crate::{chat_template::ChatTemplate, models::GenerateModel, tokenizer::TokenizerModel};

pub struct PaddleOCRVLGenerateModel<'a> {
    chat_template: ChatTemplate<'a>,
    tokenizer: TokenizerModel,
    pre_processor: PaddleOCRVLProcessor,
    model: PaddleOCRVLModel,
    cfg: PaddleOCRVLConfig,
    device: Device,
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
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        let model = PaddleOCRVLModel::new(cfg.clone(), vb, vec![2])?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("paddleocr_vl")
            .to_string();
        Ok(PaddleOCRVLGenerateModel {
            chat_template,
            tokenizer,
            pre_processor,
            model,
            cfg,
            device: device.clone(),
            model_name,
        })
    }
}

impl<'a> GenerateModel for PaddleOCRVLGenerateModel<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (replace_text, pixel_values, image_grid_thw) =
            self.pre_processor.process_info(&mes, &mes_render)?;
        let input_ids = self.tokenizer.text_encode(replace_text, &self.device)?;
        let image_mask = get_equal_mask(&input_ids, self.cfg.image_token_id)?;

        let cache_position = Tensor::ones_like(&input_ids.i(0)?)?
            .to_dtype(candle_core::DType::F64)?
            .cumsum(D::Minus1)?
            .to_dtype(candle_core::DType::U32)?
            .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;
        let seed = mes.seed.unwrap_or(34562) as u64;
        let sample_len = mes.max_tokens.unwrap_or(1024);
        let mut ctx = GenerationContext::new(
            mes.temperature,
            mes.top_p,
            None,
            mes.repeat_penalty,
            mes.repeat_last_n,
            seed,
            input_ids.dim(1)?,
            sample_len,
            self.device.clone(),
        );
        let data_vec = vec![
            pixel_values,
            image_grid_thw,
            image_mask.into(),
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
        let mes_render = self.chat_template.apply_chat_template(&mes)?;
        let (replace_text, pixel_values, image_grid_thw) =
            self.pre_processor.process_info(&mes, &mes_render)?;
        let input_ids = self.tokenizer.text_encode(replace_text, &self.device)?;
        let image_mask = get_equal_mask(&input_ids, self.cfg.image_token_id)?;

        let cache_position = Tensor::ones_like(&input_ids.i(0)?)?
            .to_dtype(candle_core::DType::F64)?
            .cumsum(D::Minus1)?
            .to_dtype(candle_core::DType::U32)?
            .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;

        let sample_len = mes.max_tokens.unwrap_or(1024);
        let data_vec = vec![
            pixel_values,
            image_grid_thw,
            image_mask.into(),
            cache_position.into(),
        ];
        let data = MultiModalData::new(data_vec);
        let seed = mes.seed.unwrap_or(34562) as u64;
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
