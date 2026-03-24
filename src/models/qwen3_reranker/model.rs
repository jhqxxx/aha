use anyhow::{Result, anyhow};
use candle_core::{DType, Device};

use crate::models::{
    common::retrieval::cosine_similarity,
    qwen3_embedding::generate::Qwen3EmbeddingModel,
    qwen3_reranker::config::{Qwen3RerankerConfig, Qwen3RerankerSimilarity},
};

pub struct Qwen3RerankerBackend {
    config: Qwen3RerankerConfig,
    embedding_backend: Qwen3EmbeddingModel,
}

impl Qwen3RerankerBackend {
    pub fn load(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let embedding_backend = Qwen3EmbeddingModel::init(path, device, dtype)?;
        Ok(Self {
            config: Qwen3RerankerConfig::default(),
            embedding_backend,
        })
    }

    pub fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        if query.trim().is_empty() {
            return Err(anyhow!("reranker query cannot be empty"));
        }
        if documents.is_empty() {
            return Err(anyhow!("reranker documents cannot be empty"));
        }

        let mut batch = Vec::with_capacity(documents.len() + 1);
        batch.push(query.to_string());
        batch.extend(documents.iter().cloned());

        let embeddings = self.embedding_backend.embed(&batch)?;
        let query_embedding = embeddings
            .first()
            .ok_or_else(|| anyhow!("failed to produce query embedding"))?;

        let mut scores = Vec::with_capacity(documents.len());
        for doc_embedding in embeddings.iter().skip(1) {
            scores.push(match self.config.similarity {
                Qwen3RerankerSimilarity::Cosine => {
                    cosine_similarity(query_embedding, doc_embedding)?
                }
            });
        }
        Ok(scores)
    }
}
