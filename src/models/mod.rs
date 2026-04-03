pub mod bigvgan;
pub mod campplus;
pub mod common;
pub mod deepseek_ocr;
pub mod feature_extractor;
pub mod fun_asr_nano;
pub mod glm_asr_nano;
pub mod glm_ocr;
pub mod hunyuan_ocr;
pub mod lfm2;
pub mod lfm2vl;
pub mod mask_gct;
pub mod minicpm4;
pub mod paddleocr_vl;
pub mod qwen2;
pub mod qwen2_5vl;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_asr;
pub mod qwen3vl;
pub mod rmbg2_0;
pub mod voxcpm;
pub mod w2v_bert_2_0;

use crate::{
    models::common::model_mapping::WhichModel,
    params::chat::{ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse},
};
use anyhow::{Result, anyhow};
use rocket::futures::Stream;

use crate::models::{
    deepseek_ocr::generate::DeepseekOCRGenerateModel,
    fun_asr_nano::generate::FunAsrNanoGenerateModel,
    glm_asr_nano::generate::GlmAsrNanoGenerateModel, glm_ocr::generate::GlmOcrGenerateModel,
    hunyuan_ocr::generate::HunyuanOCRGenerateModel, lfm2::generate::Lfm2GenerateModel,
    lfm2vl::generate::Lfm2VLGenerateModel, minicpm4::generate::MiniCPMGenerateModel,
    paddleocr_vl::generate::PaddleOCRVLGenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel,
    qwen3::generate::Qwen3GenerateModel, qwen3_5::generate::Qwen3_5GenerateModel,
    qwen3_asr::generate::Qwen3AsrGenerateModel, qwen3vl::generate::Qwen3VLGenerateModel,
    rmbg2_0::generate::RMBG2_0Model, voxcpm::generate::VoxCPMGenerate,
};

pub trait GenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse>;
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
    >;
}

pub enum ModelInstance<'a> {
    MiniCPM4(MiniCPMGenerateModel<'a>),
    Lfm2(Lfm2GenerateModel<'a>),
    Lfm2VL(Lfm2VLGenerateModel<'a>),
    Qwen2_5VL(Qwen2_5VLGenerateModel<'a>),
    Qwen3(Qwen3GenerateModel<'a>),
    Qwen3_5(Qwen3_5GenerateModel<'a>),
    Qwen3ASR(Qwen3AsrGenerateModel<'a>),
    Qwen3VL(Box<Qwen3VLGenerateModel<'a>>),
    DeepSeekOCR(DeepseekOCRGenerateModel),
    HunyuanOCR(HunyuanOCRGenerateModel<'a>),
    PaddleOCRVL(Box<PaddleOCRVLGenerateModel<'a>>),
    RMBG2_0(Box<RMBG2_0Model>),
    VoxCPM(Box<VoxCPMGenerate>),
    GlmASRNano(GlmAsrNanoGenerateModel<'a>),
    FunASRNano(FunAsrNanoGenerateModel),
    GlmOCR(GlmOcrGenerateModel),
}

impl<'a> GenerateModel for ModelInstance<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        match self {
            ModelInstance::MiniCPM4(model) => model.generate(mes),
            ModelInstance::Lfm2(model) => model.generate(mes),
            ModelInstance::Lfm2VL(model) => model.generate(mes),
            ModelInstance::Qwen2_5VL(model) => model.generate(mes),
            ModelInstance::Qwen3(model) => model.generate(mes),
            ModelInstance::Qwen3_5(model) => model.generate(mes),
            ModelInstance::Qwen3ASR(model) => model.generate(mes),
            ModelInstance::Qwen3VL(model) => model.generate(mes),
            ModelInstance::DeepSeekOCR(model) => model.generate(mes),
            ModelInstance::HunyuanOCR(model) => model.generate(mes),
            ModelInstance::PaddleOCRVL(model) => model.generate(mes),
            ModelInstance::RMBG2_0(model) => model.generate(mes),
            ModelInstance::VoxCPM(model) => model.generate(mes),
            ModelInstance::GlmASRNano(model) => model.generate(mes),
            ModelInstance::FunASRNano(model) => model.generate(mes),
            ModelInstance::GlmOCR(model) => model.generate(mes),
        }
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
        match self {
            ModelInstance::MiniCPM4(model) => model.generate_stream(mes),
            ModelInstance::Lfm2(model) => model.generate_stream(mes),
            ModelInstance::Lfm2VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen2_5VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen3(model) => model.generate_stream(mes),
            ModelInstance::Qwen3_5(model) => model.generate_stream(mes),
            ModelInstance::Qwen3VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen3ASR(model) => model.generate_stream(mes),
            ModelInstance::DeepSeekOCR(model) => model.generate_stream(mes),
            ModelInstance::HunyuanOCR(model) => model.generate_stream(mes),
            ModelInstance::PaddleOCRVL(model) => model.generate_stream(mes),
            ModelInstance::RMBG2_0(model) => model.generate_stream(mes),
            ModelInstance::VoxCPM(model) => model.generate_stream(mes),
            ModelInstance::GlmASRNano(model) => model.generate_stream(mes),
            ModelInstance::FunASRNano(model) => model.generate_stream(mes),
            ModelInstance::GlmOCR(model) => model.generate_stream(mes),
        }
    }
}

#[allow(unused)]
pub fn load_gguf_model<'a>(
    model_type: WhichModel,
    config_path: Option<&str>, // 有些gguf未包含模型其他配置，需额外指定
    gguf_path: &str,
    mmproj_path: Option<&str>,
) -> Result<ModelInstance<'a>> {
    let model = match model_type {
        WhichModel::Qwen3_5Gguf => {
            let model = Qwen3_5GenerateModel::init_from_gguf(gguf_path, mmproj_path, None)?;
            ModelInstance::Qwen3_5(model)
        }
        _ => {
            let model_id = model_type.as_string();
            return Err(anyhow!("model id {model_id} is not gguf model"));
        }
    };
    Ok(model)
}

pub fn load_model<'a>(model_type: WhichModel, path: &str) -> Result<ModelInstance<'a>> {
    let model = match model_type {
        WhichModel::MiniCPM4_0_5B => {
            let model = MiniCPMGenerateModel::init(path, None, None)?;
            ModelInstance::MiniCPM4(model)
        }
        WhichModel::LFM2_1_2B => {
            let model = Lfm2GenerateModel::init(path, None, None)?;
            ModelInstance::Lfm2(model)
        }
        WhichModel::LFM2_5_1_2BInstruct => {
            let model = Lfm2GenerateModel::init(path, None, None)?;
            ModelInstance::Lfm2(model)
        }
        WhichModel::LFM2_5VL1_6B => {
            let model = Lfm2VLGenerateModel::init(path, None, None)?;
            ModelInstance::Lfm2VL(model)
        }
        WhichModel::LFM2VL1_6B => {
            let model = Lfm2VLGenerateModel::init(path, None, None)?;
            ModelInstance::Lfm2VL(model)
        }
        WhichModel::Qwen2_5VL3B => {
            let model = Qwen2_5VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen2_5VL(model)
        }
        WhichModel::Qwen2_5VL7B => {
            let model = Qwen2_5VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen2_5VL(model)
        }
        WhichModel::Qwen3_0_6B => {
            let model = Qwen3GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3(model)
        }
        WhichModel::Qwen3_1_7B => {
            let model = Qwen3GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3(model)
        }
        WhichModel::Qwen3_4B => {
            let model = Qwen3GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3(model)
        }
        WhichModel::Qwen3_5_0_8B => {
            let model = Qwen3_5GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3_5_2B => {
            let model = Qwen3_5GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3_5_4B => {
            let model = Qwen3_5GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3_5_9B => {
            let model = Qwen3_5GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3ASR0_6B => {
            let model = Qwen3AsrGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3ASR(model)
        }
        WhichModel::Qwen3ASR1_7B => {
            let model = Qwen3AsrGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3ASR(model)
        }
        WhichModel::Qwen3VL2B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::Qwen3VL4B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::Qwen3VL8B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::Qwen3VL32B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::DeepSeekOCR => {
            let model = DeepseekOCRGenerateModel::init(path, None, None)?;
            ModelInstance::DeepSeekOCR(model)
        }
        WhichModel::DeepSeekOCR2 => {
            let model = DeepseekOCRGenerateModel::init(path, None, None)?;
            ModelInstance::DeepSeekOCR(model)
        }
        WhichModel::HunyuanOCR => {
            let model = HunyuanOCRGenerateModel::init(path, None, None)?;
            ModelInstance::HunyuanOCR(model)
        }
        WhichModel::PaddleOCRVL => {
            let model = PaddleOCRVLGenerateModel::init(path, None, None)?;
            ModelInstance::PaddleOCRVL(Box::new(model))
        }
        WhichModel::PaddleOCRVL1_5 => {
            let model = PaddleOCRVLGenerateModel::init(path, None, None)?;
            ModelInstance::PaddleOCRVL(Box::new(model))
        }
        WhichModel::RMBG2_0 => {
            let model = RMBG2_0Model::init(path, None, None)?;
            ModelInstance::RMBG2_0(Box::new(model))
        }
        WhichModel::VoxCPM => {
            let model = VoxCPMGenerate::init(path, None, None)?;
            ModelInstance::VoxCPM(Box::new(model))
        }
        WhichModel::VoxCPM1_5 => {
            let model = VoxCPMGenerate::init(path, None, None)?;
            ModelInstance::VoxCPM(Box::new(model))
        }
        WhichModel::GlmASRNano2512 => {
            let model = GlmAsrNanoGenerateModel::init(path, None, None)?;
            ModelInstance::GlmASRNano(model)
        }
        WhichModel::FunASRNano2512 => {
            let model = FunAsrNanoGenerateModel::init(path, None, None)?;
            ModelInstance::FunASRNano(model)
        }
        WhichModel::GlmOCR => {
            let model = GlmOcrGenerateModel::init(path, None, None)?;
            ModelInstance::GlmOCR(model)
        }
        _ => {
            let model_id = model_type.as_string();
            return Err(anyhow!("model id {model_id} is not safetensor model"));
        }
    };
    Ok(model)
}
