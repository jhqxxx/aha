use anyhow::Result;
use candle_core::Tensor;

use crate::models::common::modules::{cosine_similarity, cosine_similarity_no_l2};
pub trait TextRerank {
    fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>>;
}

pub enum RerankerSimilarity {
    Cosine,
}

impl RerankerSimilarity {
    pub fn similar(&self, query_vector: &Tensor, matrix: &Tensor, need_l2: bool) -> Result<Tensor> {
        match self {
            RerankerSimilarity::Cosine => {
                if need_l2 {
                    cosine_similarity(query_vector, matrix)
                } else {
                    cosine_similarity_no_l2(query_vector, matrix)
                }
            }
        }
    }
}
