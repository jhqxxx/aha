use anyhow::Result;
use candle_core::{DType, Device};

use crate::models::{
    all_minilm_l6_v2::model::AllMiniLML6V2Backend,
    artifact::{ArtifactKind, LoadSpec},
    common::retrieval::TextEmbeddingBackend,
};

pub struct AllMiniLML6V2Model {
    backend: AllMiniLML6V2Backend,
}

impl AllMiniLML6V2Model {
    pub fn init_from_spec(
        spec: &LoadSpec,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        match spec.resolved_artifact() {
            ArtifactKind::Safetensors => {
                let path = spec.paths.weight_dir.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("weight_path is required for all-minilm-l6-v2 safetensors")
                })?;
                Self::init(path, device, dtype)
            }
            ArtifactKind::Gguf => {
                let path = spec.paths.gguf_path.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("gguf_path is required for all-minilm-l6-v2 gguf")
                })?;
                Self::init_gguf(path, spec.paths.tokenizer_dir.as_deref(), device, dtype)
            }
            ArtifactKind::Onnx => {
                let path = spec.paths.onnx_path.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("onnx_path is required for all-minilm-l6-v2 onnx")
                })?;
                Self::init_onnx(path, spec.paths.tokenizer_dir.as_deref())
            }
            ArtifactKind::Auto => unreachable!("artifact kind should be resolved before init"),
        }
    }

    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let backend = AllMiniLML6V2Backend::load(path, device, dtype)?;
        Ok(Self { backend })
    }

    pub fn init_onnx(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        let backend = AllMiniLML6V2Backend::load_onnx(onnx_path, tokenizer_dir)?;
        Ok(Self { backend })
    }

    pub fn init_gguf(
        gguf_path: &str,
        tokenizer_dir: Option<&str>,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let backend = AllMiniLML6V2Backend::load_gguf(gguf_path, tokenizer_dir, device, dtype)?;
        Ok(Self { backend })
    }

    pub fn embed(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        self.backend.embed_texts(input)
    }
}

impl TextEmbeddingBackend for AllMiniLML6V2Model {
    fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        self.backend.embed_texts(input)
    }
}
