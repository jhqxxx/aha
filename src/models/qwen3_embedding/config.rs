use anyhow::Result;

use crate::models::qwen3::config::Qwen3Config;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3EmbeddingPoolingStrategy {
    Mean,
}

#[derive(Debug, Clone)]
pub struct Qwen3EmbeddingConfig {
    pub base: Qwen3Config,
    pub pooling: Qwen3EmbeddingPoolingStrategy,
    pub normalize: bool,
}

impl Qwen3EmbeddingConfig {
    pub fn load(path: &str) -> Result<Self> {
        let config_path = format!("{path}/config.json");
        let base: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        Ok(Self {
            base,
            pooling: Qwen3EmbeddingPoolingStrategy::Mean,
            normalize: true,
        })
    }
}
