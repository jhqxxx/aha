use anyhow::Result;
use candle_core::{DType, Device};

use crate::models::{
    common::retrieval::TextEmbeddingBackend, qwen3_embedding::model::Qwen3EmbeddingBackend,
};

pub struct Qwen3EmbeddingModel {
    backend: Qwen3EmbeddingBackend,
}

impl Qwen3EmbeddingModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let backend = Qwen3EmbeddingBackend::load(path, device, dtype)?;
        Ok(Self { backend })
    }

    pub fn embed(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        self.backend.embed_texts(input)
    }
}

impl TextEmbeddingBackend for Qwen3EmbeddingModel {
    fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        self.backend.embed_texts(input)
    }
}
