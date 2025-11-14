pub mod common;
pub mod minicpm4;
pub mod qwen2_5vl;
pub mod qwen3vl;
pub mod voxcpm;

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::Result;
use rocket::futures::Stream;

use crate::models::{
    minicpm4::generate::MiniCPMGenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel,
    qwen3vl::generate::Qwen3VLGenerateModel,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum WhichModel {
    #[value(name = "minicpm4-0.5b")]
    MiniCPM4_0_5B,
    #[value(name = "qwen2.5vl-3b")]
    Qwen2_5vl3B,
    #[value(name = "qwen3vl-2b")]
    Qwen3vl2B,
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
    Qwen3VL(Qwen3VLGenerateModel<'a>),
}

impl<'a> GenerateModel for ModelInstance<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        match self {
            ModelInstance::MiniCPM4(model) => model.generate(mes),
            ModelInstance::Qwen2_5VL(model) => model.generate(mes),
            ModelInstance::Qwen3VL(model) => model.generate(mes),
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
            ModelInstance::Qwen3VL(model) => model.generate_stream(mes),
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
        WhichModel::Qwen3vl2B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(model)
        }
    };
    Ok(model)
}
