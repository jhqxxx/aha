pub mod bigvgan;
pub mod campplus;
pub mod common;
pub mod deepseek_ocr;
pub mod feature_extractor;
pub mod fun_asr_nano;
pub mod glm_asr_nano;
pub mod glm_ocr;
pub mod hunyuan_ocr;
pub mod mask_gct;
pub mod minicpm4;
pub mod paddleocr_vl;
pub mod qwen2;
pub mod qwen2_5vl;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_asr;
pub mod qwen3_embedding;
pub mod qwen3_reranker;
pub mod qwen3vl;
pub mod rmbg2_0;
pub mod voxcpm;
pub mod w2v_bert_2_0;

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::{Result, anyhow};
use rocket::futures::Stream;

use crate::models::{
    deepseek_ocr::generate::DeepseekOCRGenerateModel,
    fun_asr_nano::generate::FunAsrNanoGenerateModel,
    glm_asr_nano::generate::GlmAsrNanoGenerateModel, glm_ocr::generate::GlmOcrGenerateModel,
    hunyuan_ocr::generate::HunyuanOCRGenerateModel, minicpm4::generate::MiniCPMGenerateModel,
    paddleocr_vl::generate::PaddleOCRVLGenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel,
    qwen3::generate::Qwen3GenerateModel, qwen3_5::generate::Qwen3_5GenerateModel,
    qwen3_asr::generate::Qwen3AsrGenerateModel, qwen3_embedding::generate::Qwen3EmbeddingModel,
    qwen3_reranker::generate::Qwen3RerankerModel, qwen3vl::generate::Qwen3VLGenerateModel,
    rmbg2_0::generate::RMBG2_0Model, voxcpm::generate::VoxCPMGenerate,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelArtifactFormat {
    Safetensors,
    Gguf,
    Onnx,
}

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
    #[value(name = "qwen3-embedding-0.6b", hide = true)]
    Qwen3Embedding0_6B,
    #[value(name = "qwen3-embedding-4b", hide = true)]
    Qwen3Embedding4B,
    #[value(name = "qwen3-embedding-8b", hide = true)]
    Qwen3Embedding8B,
    #[value(name = "qwen3-reranker-0.6b", hide = true)]
    Qwen3Reranker0_6B,
    #[value(name = "qwen3-reranker-4b", hide = true)]
    Qwen3Reranker4B,
    #[value(name = "qwen3-reranker-8b", hide = true)]
    Qwen3Reranker8B,
    #[value(name = "qwen3.5-0.8b", hide = true)]
    Qwen3_5_0_8B,
    #[value(name = "qwen3.5-2b", hide = true)]
    Qwen3_5_2B,
    #[value(name = "qwen3.5-4b", hide = true)]
    Qwen3_5_4B,
    #[value(name = "qwen3.5-9b", hide = true)]
    Qwen3_5_9B,
    #[value(
        name = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2",
        hide = true
    )]
    Qwen3_5_9BClaude46OpusReasoningDistilledV2,
    #[value(name = "qwen3.5-gguf", hide = true)]
    Qwen3_5Gguf,
    #[value(
        name = "qwen3.5-4b-claude-4.6-opus-reasoning-distilled-v2-gguf",
        hide = true
    )]
    Qwen3_5_4BClaude46OpusReasoningDistilledV2Gguf,
    #[value(
        name = "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2-gguf",
        hide = true
    )]
    Qwen3_5_9BClaude46OpusReasoningDistilledV2Gguf,
    #[value(name = "qwen3.5-0.8b-unsloth-gguf", hide = true)]
    Qwen3_5_0_8BUnslothGguf,
    #[value(name = "qwen3.5-2b-unsloth-gguf", hide = true)]
    Qwen3_5_2BUnslothGguf,
    #[value(name = "qwen3.5-4b-unsloth-gguf", hide = true)]
    Qwen3_5_4BUnslothGguf,
    #[value(name = "qwen3.5-0.8b-lmstudio-gguf", hide = true)]
    Qwen3_5_0_8BLmstudioGguf,
    #[value(name = "qwen3.5-2b-lmstudio-gguf", hide = true)]
    Qwen3_5_2BLmstudioGguf,
    #[value(name = "qwen3.5-4b-lmstudio-gguf", hide = true)]
    Qwen3_5_4BLmstudioGguf,
    #[value(name = "qwen3asr-0.6b", hide = true)]
    Qwen3ASR0_6B,
    #[value(name = "qwen3asr-1.7b", hide = true)]
    Qwen3ASR1_7B,
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
    #[value(name = "deepseek-ocr2", hide = true)]
    DeepSeekOCR2,
    #[value(name = "hunyuan-ocr", hide = true)]
    HunyuanOCR,
    #[value(name = "paddleocr-vl", hide = true)]
    PaddleOCRVL,
    #[value(name = "paddleocr-vl1.5", hide = true)]
    PaddleOCRVL1_5,
    #[value(name = "rmbg2.0")]
    RMBG2_0,
    #[value(name = "voxcpm", hide = true)]
    VoxCPM,
    #[value(name = "voxcpm1.5", hide = true)]
    VoxCPM1_5,
    #[value(name = "glm-asr-nano-2512", hide = true)]
    GlmASRNano2512,
    #[value(name = "fun-asr-nano-2512", hide = true)]
    FunASRNano2512,
    #[value(name = "glm-ocr", hide = true)]
    GlmOCR,
}

pub const LISTED_MODELS: &[WhichModel] = &[
    WhichModel::MiniCPM4_0_5B,
    WhichModel::Qwen2_5vl3B,
    WhichModel::Qwen2_5vl7B,
    WhichModel::Qwen3_0_6B,
    WhichModel::Qwen3Embedding0_6B,
    WhichModel::Qwen3Embedding4B,
    WhichModel::Qwen3Embedding8B,
    WhichModel::Qwen3Reranker0_6B,
    WhichModel::Qwen3Reranker4B,
    WhichModel::Qwen3Reranker8B,
    WhichModel::Qwen3_5_0_8B,
    WhichModel::Qwen3_5_2B,
    WhichModel::Qwen3_5_4B,
    WhichModel::Qwen3_5_9B,
    WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2,
    WhichModel::Qwen3_5_4BClaude46OpusReasoningDistilledV2Gguf,
    WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2Gguf,
    WhichModel::Qwen3_5_0_8BUnslothGguf,
    WhichModel::Qwen3_5_2BUnslothGguf,
    WhichModel::Qwen3_5_4BUnslothGguf,
    WhichModel::Qwen3_5_0_8BLmstudioGguf,
    WhichModel::Qwen3_5_2BLmstudioGguf,
    WhichModel::Qwen3_5_4BLmstudioGguf,
    WhichModel::Qwen3ASR0_6B,
    WhichModel::Qwen3ASR1_7B,
    WhichModel::Qwen3vl2B,
    WhichModel::Qwen3vl4B,
    WhichModel::Qwen3vl8B,
    WhichModel::Qwen3vl32B,
    WhichModel::DeepSeekOCR,
    WhichModel::DeepSeekOCR2,
    WhichModel::HunyuanOCR,
    WhichModel::PaddleOCRVL,
    WhichModel::PaddleOCRVL1_5,
    WhichModel::RMBG2_0,
    WhichModel::VoxCPM,
    WhichModel::VoxCPM1_5,
    WhichModel::GlmASRNano2512,
    WhichModel::FunASRNano2512,
    WhichModel::GlmOCR,
];

impl WhichModel {
    pub fn artifact_format(self) -> ModelArtifactFormat {
        match self {
            WhichModel::Qwen3_5Gguf
            | WhichModel::Qwen3_5_4BClaude46OpusReasoningDistilledV2Gguf
            | WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2Gguf
            | WhichModel::Qwen3_5_0_8BUnslothGguf
            | WhichModel::Qwen3_5_2BUnslothGguf
            | WhichModel::Qwen3_5_4BUnslothGguf
            | WhichModel::Qwen3_5_0_8BLmstudioGguf
            | WhichModel::Qwen3_5_2BLmstudioGguf
            | WhichModel::Qwen3_5_4BLmstudioGguf => ModelArtifactFormat::Gguf,
            _ => ModelArtifactFormat::Safetensors,
        }
    }

    pub fn is_download_managed(self) -> bool {
        !matches!(
            self.artifact_format(),
            ModelArtifactFormat::Gguf | ModelArtifactFormat::Onnx
        )
    }

    pub fn openai_model_id(self) -> &'static str {
        match self {
            WhichModel::MiniCPM4_0_5B => "minicpm4-0.5b",
            WhichModel::Qwen2_5vl3B => "qwen2.5vl-3b",
            WhichModel::Qwen2_5vl7B => "qwen2.5vl-7b",
            WhichModel::Qwen3_0_6B => "qwen3-0.6b",
            WhichModel::Qwen3Embedding0_6B => "qwen3-embedding-0.6b",
            WhichModel::Qwen3Embedding4B => "qwen3-embedding-4b",
            WhichModel::Qwen3Embedding8B => "qwen3-embedding-8b",
            WhichModel::Qwen3Reranker0_6B => "qwen3-reranker-0.6b",
            WhichModel::Qwen3Reranker4B => "qwen3-reranker-4b",
            WhichModel::Qwen3Reranker8B => "qwen3-reranker-8b",
            WhichModel::Qwen3_5_0_8B => "qwen3.5-0.8b",
            WhichModel::Qwen3_5_2B => "qwen3.5-2b",
            WhichModel::Qwen3_5_4B => "qwen3.5-4b",
            WhichModel::Qwen3_5_9B => "qwen3.5-9b",
            WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2 => {
                "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2"
            }
            WhichModel::Qwen3_5Gguf => "qwen3.5-gguf",
            WhichModel::Qwen3_5_4BClaude46OpusReasoningDistilledV2Gguf => {
                "qwen3.5-4b-claude-4.6-opus-reasoning-distilled-v2-gguf"
            }
            WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2Gguf => {
                "qwen3.5-9b-claude-4.6-opus-reasoning-distilled-v2-gguf"
            }
            WhichModel::Qwen3_5_0_8BUnslothGguf => "qwen3.5-0.8b-unsloth-gguf",
            WhichModel::Qwen3_5_2BUnslothGguf => "qwen3.5-2b-unsloth-gguf",
            WhichModel::Qwen3_5_4BUnslothGguf => "qwen3.5-4b-unsloth-gguf",
            WhichModel::Qwen3_5_0_8BLmstudioGguf => "qwen3.5-0.8b-lmstudio-gguf",
            WhichModel::Qwen3_5_2BLmstudioGguf => "qwen3.5-2b-lmstudio-gguf",
            WhichModel::Qwen3_5_4BLmstudioGguf => "qwen3.5-4b-lmstudio-gguf",
            WhichModel::Qwen3ASR0_6B => "qwen3asr-0.6b",
            WhichModel::Qwen3ASR1_7B => "qwen3asr-1.7b",
            WhichModel::Qwen3vl2B => "qwen3vl-2b",
            WhichModel::Qwen3vl4B => "qwen3vl-4b",
            WhichModel::Qwen3vl8B => "qwen3vl-8b",
            WhichModel::Qwen3vl32B => "qwen3vl-32b",
            WhichModel::DeepSeekOCR => "deepseek-ocr",
            WhichModel::DeepSeekOCR2 => "deepseek-ocr2",
            WhichModel::HunyuanOCR => "hunyuan-ocr",
            WhichModel::PaddleOCRVL => "paddleocr-vl",
            WhichModel::PaddleOCRVL1_5 => "paddleocr-vl1.5",
            WhichModel::RMBG2_0 => "rmbg2.0",
            WhichModel::VoxCPM => "voxcpm",
            WhichModel::VoxCPM1_5 => "voxcpm1.5",
            WhichModel::GlmASRNano2512 => "glm-asr-nano-2512",
            WhichModel::FunASRNano2512 => "fun-asr-nano-2512",
            WhichModel::GlmOCR => "glm-ocr",
        }
    }

    pub fn owner(self) -> &'static str {
        match self {
            WhichModel::MiniCPM4_0_5B => "OpenBMB",
            WhichModel::Qwen2_5vl3B | WhichModel::Qwen2_5vl7B => "Qwen",
            WhichModel::Qwen3_0_6B
            | WhichModel::Qwen3Embedding0_6B
            | WhichModel::Qwen3Embedding4B
            | WhichModel::Qwen3Embedding8B
            | WhichModel::Qwen3Reranker0_6B
            | WhichModel::Qwen3Reranker4B
            | WhichModel::Qwen3Reranker8B
            | WhichModel::Qwen3ASR0_6B
            | WhichModel::Qwen3ASR1_7B => "Qwen",
            WhichModel::Qwen3vl2B
            | WhichModel::Qwen3vl4B
            | WhichModel::Qwen3vl8B
            | WhichModel::Qwen3vl32B
            | WhichModel::Qwen3_5Gguf => "Qwen",
            WhichModel::Qwen3_5_0_8B
            | WhichModel::Qwen3_5_2B
            | WhichModel::Qwen3_5_4B
            | WhichModel::Qwen3_5_9B => "Qwen",
            WhichModel::Qwen3_5_0_8BUnslothGguf
            | WhichModel::Qwen3_5_2BUnslothGguf
            | WhichModel::Qwen3_5_4BUnslothGguf => "unsloth",
            WhichModel::Qwen3_5_0_8BLmstudioGguf
            | WhichModel::Qwen3_5_2BLmstudioGguf
            | WhichModel::Qwen3_5_4BLmstudioGguf => "lmstudio-community",
            WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2
            | WhichModel::Qwen3_5_4BClaude46OpusReasoningDistilledV2Gguf
            | WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2Gguf => "Jackrong",
            WhichModel::DeepSeekOCR | WhichModel::DeepSeekOCR2 => "deepseek-ai",
            WhichModel::HunyuanOCR => "Tencent-Hunyuan",
            WhichModel::PaddleOCRVL | WhichModel::PaddleOCRVL1_5 => "PaddlePaddle",
            WhichModel::RMBG2_0 => "AI-ModelScope",
            WhichModel::VoxCPM | WhichModel::VoxCPM1_5 => "OpenBMB",
            WhichModel::GlmASRNano2512 | WhichModel::GlmOCR => "ZhipuAI",
            WhichModel::FunASRNano2512 => "FunAudioLLM",
        }
    }

    /// Get the ModelScope model ID for this model variant
    pub fn model_id(self) -> &'static str {
        match self {
            WhichModel::MiniCPM4_0_5B => "OpenBMB/MiniCPM4-0.5B",
            WhichModel::Qwen2_5vl3B => "Qwen/Qwen2.5-VL-3B-Instruct",
            WhichModel::Qwen2_5vl7B => "Qwen/Qwen2.5-VL-7B-Instruct",
            WhichModel::Qwen3_0_6B => "Qwen/Qwen3-0.6B",
            WhichModel::Qwen3Embedding0_6B => "Qwen/Qwen3-Embedding-0.6B",
            WhichModel::Qwen3Embedding4B => "Qwen/Qwen3-Embedding-4B",
            WhichModel::Qwen3Embedding8B => "Qwen/Qwen3-Embedding-8B",
            WhichModel::Qwen3Reranker0_6B => "Qwen/Qwen3-Reranker-0.6B",
            WhichModel::Qwen3Reranker4B => "Qwen/Qwen3-Reranker-4B",
            WhichModel::Qwen3Reranker8B => "Qwen/Qwen3-Reranker-8B",
            WhichModel::Qwen3_5_0_8B => "Qwen/Qwen3.5-0.8B",
            WhichModel::Qwen3_5_2B => "Qwen/Qwen3.5-2B",
            WhichModel::Qwen3_5_4B => "Qwen/Qwen3.5-4B",
            WhichModel::Qwen3_5_9B => "Qwen/Qwen3.5-9B",
            WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2 => {
                "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2"
            }
            WhichModel::Qwen3_5Gguf => "GGUF",
            WhichModel::Qwen3_5_4BClaude46OpusReasoningDistilledV2Gguf => {
                "Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF"
            }
            WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2Gguf => {
                "Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF"
            }
            WhichModel::Qwen3_5_0_8BUnslothGguf => "unsloth/Qwen3.5-0.8B-GGUF",
            WhichModel::Qwen3_5_2BUnslothGguf => "unsloth/Qwen3.5-2B-GGUF",
            WhichModel::Qwen3_5_4BUnslothGguf => "unsloth/Qwen3.5-4B-GGUF",
            WhichModel::Qwen3_5_0_8BLmstudioGguf => "lmstudio-community/Qwen3.5-0.8B-GGUF",
            WhichModel::Qwen3_5_2BLmstudioGguf => "lmstudio-community/Qwen3.5-2B-GGUF",
            WhichModel::Qwen3_5_4BLmstudioGguf => "lmstudio-community/Qwen3.5-4B-GGUF",
            WhichModel::Qwen3ASR0_6B => "Qwen/Qwen3-ASR-0.6B",
            WhichModel::Qwen3ASR1_7B => "Qwen/Qwen3-ASR-1.7B",
            WhichModel::Qwen3vl2B => "Qwen/Qwen3-VL-2B-Instruct",
            WhichModel::Qwen3vl4B => "Qwen/Qwen3-VL-4B-Instruct",
            WhichModel::Qwen3vl8B => "Qwen/Qwen3-VL-8B-Instruct",
            WhichModel::Qwen3vl32B => "Qwen/Qwen3-VL-32B-Instruct",
            WhichModel::DeepSeekOCR => "deepseek-ai/DeepSeek-OCR",
            WhichModel::DeepSeekOCR2 => "deepseek-ai/DeepSeek-OCR-2",
            WhichModel::HunyuanOCR => "Tencent-Hunyuan/HunyuanOCR",
            WhichModel::PaddleOCRVL => "PaddlePaddle/PaddleOCR-VL",
            WhichModel::PaddleOCRVL1_5 => "PaddlePaddle/PaddleOCR-VL-1.5",
            WhichModel::RMBG2_0 => "AI-ModelScope/RMBG-2.0",
            WhichModel::VoxCPM => "OpenBMB/VoxCPM-0.5B",
            WhichModel::VoxCPM1_5 => "OpenBMB/VoxCPM1.5",
            WhichModel::GlmASRNano2512 => "ZhipuAI/GLM-ASR-Nano-2512",
            WhichModel::FunASRNano2512 => "FunAudioLLM/Fun-ASR-Nano-2512",
            WhichModel::GlmOCR => "ZhipuAI/GLM-OCR",
        }
    }

    /// Get the model type category for this model variant
    pub fn model_type(self) -> &'static str {
        match self {
            // LLM models
            WhichModel::MiniCPM4_0_5B | WhichModel::Qwen3_0_6B => "llm",
            WhichModel::Qwen3Embedding0_6B
            | WhichModel::Qwen3Embedding4B
            | WhichModel::Qwen3Embedding8B => "embedding",
            WhichModel::Qwen3Reranker0_6B
            | WhichModel::Qwen3Reranker4B
            | WhichModel::Qwen3Reranker8B => "reranker",
            WhichModel::Qwen2_5vl3B
            | WhichModel::Qwen2_5vl7B
            | WhichModel::Qwen3vl2B
            | WhichModel::Qwen3vl4B
            | WhichModel::Qwen3vl8B
            | WhichModel::Qwen3vl32B
            | WhichModel::Qwen3_5_0_8B
            | WhichModel::Qwen3_5_2B
            | WhichModel::Qwen3_5_4B
            | WhichModel::Qwen3_5_9B
            | WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2
            | WhichModel::Qwen3_5Gguf => "vlm",
            WhichModel::Qwen3_5_4BClaude46OpusReasoningDistilledV2Gguf
            | WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2Gguf
            | WhichModel::Qwen3_5_0_8BUnslothGguf
            | WhichModel::Qwen3_5_2BUnslothGguf
            | WhichModel::Qwen3_5_4BUnslothGguf
            | WhichModel::Qwen3_5_0_8BLmstudioGguf
            | WhichModel::Qwen3_5_2BLmstudioGguf
            | WhichModel::Qwen3_5_4BLmstudioGguf => "vlm",
            // OCR models
            WhichModel::DeepSeekOCR
            | WhichModel::DeepSeekOCR2
            | WhichModel::HunyuanOCR
            | WhichModel::GlmOCR
            | WhichModel::PaddleOCRVL
            | WhichModel::PaddleOCRVL1_5 => "ocr",
            // ASR models
            WhichModel::Qwen3ASR0_6B
            | WhichModel::Qwen3ASR1_7B
            | WhichModel::GlmASRNano2512
            | WhichModel::FunASRNano2512 => "asr",
            // Image models
            WhichModel::RMBG2_0 | WhichModel::VoxCPM | WhichModel::VoxCPM1_5 => "image",
        }
    }
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
    Qwen3Embedding(Qwen3EmbeddingModel),
    Qwen3Reranker(Qwen3RerankerModel),
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
            ModelInstance::Qwen2_5VL(model) => model.generate(mes),
            ModelInstance::Qwen3(model) => model.generate(mes),
            ModelInstance::Qwen3Embedding(_) => {
                Err(anyhow!("embedding model does not support chat completions"))
            }
            ModelInstance::Qwen3Reranker(_) => {
                Err(anyhow!("reranker model does not support chat completions"))
            }
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
            ModelInstance::Qwen2_5VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen3(model) => model.generate_stream(mes),
            ModelInstance::Qwen3Embedding(_) => Err(anyhow!(
                "embedding model does not support streaming chat completions"
            )),
            ModelInstance::Qwen3Reranker(_) => Err(anyhow!(
                "reranker model does not support streaming chat completions"
            )),
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

impl<'a> ModelInstance<'a> {
    pub fn embedding(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            ModelInstance::Qwen3Embedding(model) => model.embed(input),
            _ => Err(anyhow!("current model does not support embeddings")),
        }
    }

    pub fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        match self {
            ModelInstance::Qwen3Reranker(model) => model.rerank(query, documents),
            _ => Err(anyhow!("current model does not support reranking")),
        }
    }
}

pub fn load_model<'a>(
    model_type: WhichModel,
    path: &str,
    gguf: Option<&str>,
    mmproj: Option<&str>,
) -> Result<ModelInstance<'a>> {
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
        WhichModel::Qwen3Embedding0_6B
        | WhichModel::Qwen3Embedding4B
        | WhichModel::Qwen3Embedding8B => {
            let model = Qwen3EmbeddingModel::init(path, None, None)?;
            ModelInstance::Qwen3Embedding(model)
        }
        WhichModel::Qwen3Reranker0_6B
        | WhichModel::Qwen3Reranker4B
        | WhichModel::Qwen3Reranker8B => {
            let model = Qwen3RerankerModel::init(path, None, None)?;
            ModelInstance::Qwen3Reranker(model)
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
        WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2 => {
            let model = Qwen3_5GenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3_5Gguf
        | WhichModel::Qwen3_5_4BClaude46OpusReasoningDistilledV2Gguf
        | WhichModel::Qwen3_5_9BClaude46OpusReasoningDistilledV2Gguf
        | WhichModel::Qwen3_5_0_8BUnslothGguf
        | WhichModel::Qwen3_5_2BUnslothGguf
        | WhichModel::Qwen3_5_4BUnslothGguf
        | WhichModel::Qwen3_5_0_8BLmstudioGguf
        | WhichModel::Qwen3_5_2BLmstudioGguf
        | WhichModel::Qwen3_5_4BLmstudioGguf => {
            if gguf.is_none() {
                return Err(anyhow!("Qwen3_5Gguf gguf model path is required"));
            }
            let gguf = gguf.unwrap();
            let model = Qwen3_5GenerateModel::init_from_gguf(gguf, mmproj, None)?;
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
        WhichModel::Qwen3vl2B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::Qwen3vl4B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::Qwen3vl8B => {
            let model = Qwen3VLGenerateModel::init(path, None, None)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::Qwen3vl32B => {
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
    };
    Ok(model)
}
