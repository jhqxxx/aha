pub mod common;
pub mod deepseek_ocr;
pub mod fun_asr_nano;
pub mod glm_asr_nano;
pub mod hunyuan_ocr;
pub mod minicpm4;
pub mod paddleocr_vl;
pub mod qwen2_5vl;
pub mod qwen3;
pub mod qwen3vl;
pub mod rmbg2_0;
pub mod voxcpm;

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::Result;
use rocket::futures::Stream;

use crate::models::{
    deepseek_ocr::generate::DeepseekOCRGenerateModel,
    fun_asr_nano::generate::FunAsrNanoGenerateModel,
    glm_asr_nano::generate::GlmAsrNanoGenerateModel,
    hunyuan_ocr::generate::HunyuanOCRGenerateModel, minicpm4::generate::MiniCPMGenerateModel,
    paddleocr_vl::generate::PaddleOCRVLGenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel,
    qwen3::generate::Qwen3GenerateModel, qwen3vl::generate::Qwen3VLGenerateModel,
    rmbg2_0::generate::RMBG2_0Model, voxcpm::generate::VoxCPMGenerate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum WhichModel {
    #[value(name = "minicpm4-0.5b", hide = true)]
    MiniCPM4_0_5B,
    #[value(name = "qwen2.5vl-3b", hide = true)]
    Qwen2_5vl3B,
    #[value(name = "qwen2.5vl-7b", hide = true)]
    Qwen2_5vl7B,
    #[value(name = "qwen3-0.6b", hide = true)]
    Qwen3_0_6B,
    #[value(name = "qwen3vl-2b", hide = true)]
    Qwen3vl2B,
    #[value(name = "qwen3vl-4b", hide = true)]
    Qwen3vl4B,
    #[value(name = "qwen3vl-8b", hide = true)]
    Qwen3vl8B,
    #[value(name = "qwen3vl-32b", hide = true)]
    Qwen3vl32B,
    #[value(name = "deepseek-ocr", hide = true)]
    DeepSeekOCR,
    #[value(name = "hunyuan-ocr", hide = true)]
    HunyuanOCR,
    #[value(name = "paddleocr-vl", hide = true)]
    PaddleOCRVL,
    #[value(name = "RMBG2.0", hide = true)]
    RMBG2_0,
    #[value(name = "voxcpm", hide = true)]
    VoxCPM,
    #[value(name = "voxcpm1.5", hide = true)]
    VoxCPM1_5,
    #[value(name = "glm-asr-nano-2512", hide = true)]
    GlmASRNano2512,
    #[value(name = "fun-asr-nano-2512", hide = true)]
    FunASRNano2512,
}

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
    Qwen2_5VL(Qwen2_5VLGenerateModel<'a>),
    Qwen3(Qwen3GenerateModel<'a>),
    Qwen3VL(Qwen3VLGenerateModel<'a>),
    DeepSeekOCR(DeepseekOCRGenerateModel),
    HunyuanOCR(HunyuanOCRGenerateModel<'a>),
    PaddleOCRVL(Box<PaddleOCRVLGenerateModel<'a>>),
    RMBG2_0(Box<RMBG2_0Model>),
    VoxCPM(Box<VoxCPMGenerate>),
    GlmASRNano(GlmAsrNanoGenerateModel<'a>),
    FunASRNano(FunAsrNanoGenerateModel),
}

impl<'a> GenerateModel for ModelInstance<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        match self {
            ModelInstance::MiniCPM4(model) => model.generate(mes),
            ModelInstance::Qwen2_5VL(model) => model.generate(mes),
            ModelInstance::Qwen3(model) => model.generate(mes),
            ModelInstance::Qwen3VL(model) => model.generate(mes),
            ModelInstance::DeepSeekOCR(model) => model.generate(mes),
            ModelInstance::HunyuanOCR(model) => model.generate(mes),
            ModelInstance::PaddleOCRVL(model) => model.generate(mes),
            ModelInstance::RMBG2_0(model) => model.generate(mes),
            ModelInstance::VoxCPM(model) => model.generate(mes),
            ModelInstance::GlmASRNano(model) => model.generate(mes),
            ModelInstance::FunASRNano(model) => model.generate(mes),
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
            ModelInstance::Qwen2_5VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen3(model) => model.generate_stream(mes),
            ModelInstance::Qwen3VL(model) => model.generate_stream(mes),
            ModelInstance::DeepSeekOCR(model) => model.generate_stream(mes),
            ModelInstance::HunyuanOCR(model) => model.generate_stream(mes),
            ModelInstance::PaddleOCRVL(model) => model.generate_stream(mes),
            ModelInstance::RMBG2_0(model) => model.generate_stream(mes),
            ModelInstance::VoxCPM(model) => model.generate_stream(mes),
            ModelInstance::GlmASRNano(model) => model.generate_stream(mes),
            ModelInstance::FunASRNano(model) => model.generate_stream(mes),
        }
    }
}

pub fn load_model(model_type: WhichModel, path: &str) -> Result<ModelInstance<'_>> {
    let model = match model_type {
        WhichModel::MiniCPM4_0_5B => {
            let model = MiniCPMGenerateModel::init(path, None, None)?;
            ModelInstance::MiniCPM4(model)
        }
        WhichModel::Qwen2_5vl3B => {
            let model = Qwen2_5VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen2_5VL(model)
        }
        WhichModel::Qwen2_5vl7B => {
            let model = Qwen2_5VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen2_5VL(model)
        }
        WhichModel::Qwen3_0_6B => {
            let model = Qwen3GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3(model)
        }
        WhichModel::Qwen3vl2B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(model)
        }
        WhichModel::Qwen3vl4B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(model)
        }
        WhichModel::Qwen3vl8B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(model)
        }
        WhichModel::Qwen3vl32B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(model)
        }
        WhichModel::DeepSeekOCR => {
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
    };
    Ok(model)
}
