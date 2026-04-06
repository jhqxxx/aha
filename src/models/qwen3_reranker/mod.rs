use crate::models::{
    common::reranker::{RerankerSimilarity, TextRerank},
    qwen3_embedding::Qwen3Embedding,
};
use anyhow::Result;
use candle_core::{DType, Device};

pub struct Qwen3Reranker {
    embedding: Qwen3Embedding,
    similar: RerankerSimilarity,
}

impl Qwen3Reranker {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let embedding = Qwen3Embedding::init(path, device, dtype)?;
        Ok(Self {
            embedding,
            similar: RerankerSimilarity::Cosine,
        })
    }
}

impl TextRerank for Qwen3Reranker {
    fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        let query = self.embedding.embed_one(query)?.unsqueeze(0)?;
        let documents_matrix = self.embedding.embed_multi(documents)?;
        let score = self.similar.similar(&query, &documents_matrix, false)?;
        let score = score.squeeze(0)?.to_vec1::<f32>()?;
        Ok(score)
    }
}
