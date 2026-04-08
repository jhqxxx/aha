use clap::ValueEnum;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum WhichModel {
    #[value(name = "sentence-transformers/all-MiniLM-L6-v2")]
    AllMiniLML6V2,
    #[value(name = "LiquidAI/LFM2-1.2B")]
    LFM2_1_2B,
    #[value(name = "LiquidAI/LFM2.5-1.2B-Instruct")]
    LFM2_5_1_2BInstruct,
    #[value(name = "LiquidAI/LFM2.5-VL-1.6B")]
    LFM2_5VL1_6B,
    #[value(name = "LiquidAI/LFM2-VL-1.6B")]
    LFM2VL1_6B,
    #[value(name = "OpenBMB/MiniCPM4-0.5B")]
    MiniCPM4_0_5B,
    #[value(name = "Qwen/Qwen2.5-VL-3B-Instruct")]
    Qwen2_5VL3B,
    #[value(name = "Qwen/Qwen2.5-VL-7B-Instruct")]
    Qwen2_5VL7B,
    #[value(name = "Qwen/Qwen3-0.6B")]
    Qwen3_0_6B,
    #[value(name = "Qwen/Qwen3-1.7B")]
    Qwen3_1_7B,
    #[value(name = "Qwen/Qwen3-4B")]
    Qwen3_4B,
    #[value(name = "Qwen/Qwen3.5-0.8B")]
    Qwen3_5_0_8B,
    #[value(name = "Qwen/Qwen3.5-2B")]
    Qwen3_5_2B,
    #[value(name = "Qwen/Qwen3.5-4B")]
    Qwen3_5_4B,
    #[value(name = "Qwen/Qwen3.5-9B")]
    Qwen3_5_9B,
    #[value(name = "qwen3.5-gguf")] // todo
    Qwen3_5Gguf,
    #[value(name = "Qwen/Qwen3-ASR-0.6B")]
    Qwen3ASR0_6B,
    #[value(name = "Qwen/Qwen3-ASR-1.7B")]
    Qwen3ASR1_7B,
    #[value(name = "Qwen/Qwen3-Embedding-0.6B")]
    Qwen3Embedding0_6B,
    #[value(name = "Qwen/Qwen3-Embedding-4B")]
    Qwen3Embedding4B,
    #[value(name = "Qwen/Qwen3-Embedding-8B")]
    Qwen3Embedding8B,
    #[value(name = "Qwen/Qwen3-Reranker-0.6B")]
    Qwen3Reranker0_6B,
    #[value(name = "Qwen/Qwen3-Reranker-4B")]
    Qwen3Reranker4B,
    #[value(name = "Qwen/Qwen3-Reranker-8B")]
    Qwen3Reranker8B,
    #[value(name = "Qwen/Qwen3-VL-2B-Instruct")]
    Qwen3VL2B,
    #[value(name = "Qwen/Qwen3-VL-4B-Instruct")]
    Qwen3VL4B,
    #[value(name = "Qwen/Qwen3-VL-8B-Instruct")]
    Qwen3VL8B,
    #[value(name = "Qwen/Qwen3-VL-32B-Instruct")]
    Qwen3VL32B,
    #[value(name = "deepseek-ai/DeepSeek-OCR")]
    DeepSeekOCR,
    #[value(name = "deepseek-ai/DeepSeek-OCR-2")]
    DeepSeekOCR2,
    #[value(name = "Tencent-Hunyuan/HunyuanOCR")]
    HunyuanOCR,
    #[value(name = "PaddlePaddle/PaddleOCR-VL")]
    PaddleOCRVL,
    #[value(name = "PaddlePaddle/PaddleOCR-VL-1.5")]
    PaddleOCRVL1_5,
    #[value(name = "AI-ModelScope/RMBG-2.0")]
    RMBG2_0,
    #[value(name = "OpenBMB/VoxCPM-0.5B")]
    VoxCPM,
    #[value(name = "OpenBMB/VoxCPM1.5")]
    VoxCPM1_5,
    #[value(name = "OpenBMB/VoxCPM2")]
    VoxCPM2,
    #[value(name = "ZhipuAI/GLM-ASR-Nano-2512")]
    GlmASRNano2512,
    #[value(name = "FunAudioLLM/Fun-ASR-Nano-2512")]
    FunASRNano2512,
    #[value(name = "ZhipuAI/GLM-OCR")]
    GlmOCR,
}

impl WhichModel {
    /// Get the model ID for this model variant
    pub fn as_string(&self) -> String {
        self.to_possible_value()
            .expect("not exists")
            .get_name()
            .to_string()
    }

    /// Checks if the model is in GGUF format
    ///
    /// Returns true if the model ID contains "gguf", false otherwise
    pub fn is_gguf(&self) -> bool {
        let model_id = self.as_string();
        model_id.to_lowercase().contains("gguf")
    }

    /// Checks if the model is in ONNX format
    ///
    /// Returns true if the model ID contains "onnx", false otherwise
    pub fn is_onnx(&self) -> bool {
        let model_id = self.as_string();
        model_id.to_lowercase().contains("onnx")
    }

    /// Get the WhichModel enum list
    pub fn model_list() -> Vec<Self> {
        WhichModel::value_variants().to_vec()
    }

    /// Extracts the model owner/organization from the model ID
    ///
    /// Splits the model ID string on '/' and returns the first part which typically represents
    /// the organization or user who owns the model in Hugging Face format (e.g., "Qwen" from "Qwen/Qwen3-0.6B")
    /// Returns "none" if the model ID doesn't contain a '/' separator
    pub fn model_owner(&self) -> String {
        let name = self.as_string();
        let names: Vec<&str> = name.split("/").collect();
        if names.len() < 2 {
            "none".to_string()
        } else {
            names.first().map_or("none", |&s| s).to_string()
        }
    }

    /// Get the model type category for this model variant
    pub fn model_type(self) -> &'static str {
        match self {
            // LLM models
            WhichModel::MiniCPM4_0_5B
            | WhichModel::Qwen3_0_6B
            | WhichModel::Qwen3_1_7B
            | WhichModel::Qwen3_4B
            | WhichModel::LFM2_1_2B
            | WhichModel::LFM2_5_1_2BInstruct => "llm",
            // VLM models
            WhichModel::Qwen2_5VL3B
            | WhichModel::Qwen2_5VL7B
            | WhichModel::Qwen3VL2B
            | WhichModel::Qwen3VL4B
            | WhichModel::Qwen3VL8B
            | WhichModel::Qwen3VL32B
            | WhichModel::Qwen3_5_0_8B
            | WhichModel::Qwen3_5_2B
            | WhichModel::Qwen3_5_4B
            | WhichModel::Qwen3_5_9B
            | WhichModel::Qwen3_5Gguf
            | WhichModel::LFM2_5VL1_6B
            | WhichModel::LFM2VL1_6B => "vlm",
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
            WhichModel::RMBG2_0 => "image",
            // TTS models
            WhichModel::VoxCPM | WhichModel::VoxCPM1_5 | WhichModel::VoxCPM2 => "tts",
            WhichModel::Qwen3Embedding0_6B
            | WhichModel::Qwen3Embedding4B
            | WhichModel::Qwen3Embedding8B
            | WhichModel::AllMiniLML6V2 => "embedding",
            WhichModel::Qwen3Reranker0_6B
            | WhichModel::Qwen3Reranker4B
            | WhichModel::Qwen3Reranker8B => "reranker",
        }
    }
}
