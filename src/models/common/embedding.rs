use anyhow::Result;
use candle_core::Tensor;

use crate::models::common::modules::{
    l1_normalize, l2_normalize, max_abs_normalize, min_max_normalize, z_score_normalize,
};
pub trait TextEmbedding {
    fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>>;
}

pub enum NormalizeType {
    L1,
    L2,
    ZScore,
    MinMax,
    MaxAbs,
}

impl NormalizeType {
    pub fn normalize(&self, t: &Tensor, dim: usize) -> Result<Tensor> {
        match self {
            NormalizeType::L1 => l1_normalize(t, dim),
            NormalizeType::L2 => l2_normalize(t, dim),
            NormalizeType::ZScore => z_score_normalize(t, dim),
            NormalizeType::MinMax => min_max_normalize(t, dim),
            NormalizeType::MaxAbs => max_abs_normalize(t, dim),
        }
    }
}
