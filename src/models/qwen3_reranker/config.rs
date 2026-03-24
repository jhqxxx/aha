#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3RerankerSimilarity {
    Cosine,
}

#[derive(Debug, Clone)]
pub struct Qwen3RerankerConfig {
    pub similarity: Qwen3RerankerSimilarity,
}

impl Default for Qwen3RerankerConfig {
    fn default() -> Self {
        Self {
            similarity: Qwen3RerankerSimilarity::Cosine,
        }
    }
}
