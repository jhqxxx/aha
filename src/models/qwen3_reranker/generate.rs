use anyhow::Result;
use candle_core::{DType, Device};

use crate::models::qwen3_reranker::model::Qwen3RerankerBackend;

pub struct Qwen3RerankerModel {
    backend: Qwen3RerankerBackend,
}

impl Qwen3RerankerModel {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let backend = Qwen3RerankerBackend::load(path, device, dtype)?;
        Ok(Self { backend })
    }

    pub fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        self.backend.rerank(query, documents)
    }
}
