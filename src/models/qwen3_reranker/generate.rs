use anyhow::Result;
use candle_core::{DType, Device};

use crate::models::{
    artifact::{ArtifactKind, LoadSpec},
    qwen3_reranker::model::Qwen3RerankerBackend,
};

pub struct Qwen3RerankerModel {
    backend: Qwen3RerankerBackend,
}

impl Qwen3RerankerModel {
    pub fn init_from_spec(
        spec: &LoadSpec,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        match spec.resolved_artifact() {
            ArtifactKind::Safetensors => {
                let path = spec.paths.weight_dir.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("weight_path is required for qwen3 reranker safetensors")
                })?;
                Self::init(path, device, dtype)
            }
            ArtifactKind::Gguf => {
                let path = spec.paths.gguf_path.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("gguf_path is required for qwen3 reranker gguf")
                })?;
                Self::init_gguf(path, spec.paths.tokenizer_dir.as_deref())
            }
            ArtifactKind::Onnx => {
                let path = spec.paths.onnx_path.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("onnx_path is required for qwen3 reranker onnx")
                })?;
                Self::init_onnx(path, spec.paths.tokenizer_dir.as_deref())
            }
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
        }
    }

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let backend = Qwen3RerankerBackend::load(path, device, dtype)?;
        Ok(Self { backend })
    }

    pub fn init_onnx(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        let backend = Qwen3RerankerBackend::load_onnx(onnx_path, tokenizer_dir)?;
        Ok(Self { backend })
    }

    pub fn init_gguf(gguf_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        let backend = Qwen3RerankerBackend::load_gguf(gguf_path, tokenizer_dir)?;
        Ok(Self { backend })
    }

    pub fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        self.backend.rerank(query, documents)
    }
}
