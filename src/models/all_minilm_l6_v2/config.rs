use std::path::Path;

use anyhow::Result;
use candle_transformers::models::bert::Config as BertConfig;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllMiniLML6V2PoolingStrategy {
    Cls,
    Mean,
    Max,
    MeanSqrtLen,
}

#[derive(Debug, Clone)]
pub struct AllMiniLML6V2Config {
    pub base: BertConfig,
    pub pooling: AllMiniLML6V2PoolingStrategy,
    pub normalize: bool,
    pub max_seq_length: usize,
    pub do_lower_case: bool,
}

#[derive(Debug, Deserialize)]
struct SentenceBertConfig {
    #[serde(default = "default_max_seq_length")]
    max_seq_length: usize,
    #[serde(default)]
    do_lower_case: bool,
}

#[derive(Debug, Deserialize)]
struct PoolingConfig {
    #[serde(default)]
    pooling_mode_cls_token: bool,
    #[serde(default)]
    pooling_mode_mean_tokens: bool,
    #[serde(default)]
    pooling_mode_max_tokens: bool,
    #[serde(default)]
    pooling_mode_mean_sqrt_len_tokens: bool,
}

#[derive(Debug, Deserialize)]
struct ModuleEntry {
    #[serde(default)]
    r#type: String,
}

fn default_max_seq_length() -> usize {
    256
}

impl AllMiniLML6V2Config {
    pub fn load(path: &str) -> Result<Self> {
        let config_path = Path::new(path).join("config.json");
        let base: BertConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;

        let sentence_bert_path = Path::new(path).join("sentence_bert_config.json");
        let sentence_bert = if sentence_bert_path.exists() {
            serde_json::from_slice::<SentenceBertConfig>(&std::fs::read(sentence_bert_path)?)?
        } else {
            SentenceBertConfig {
                max_seq_length: default_max_seq_length(),
                do_lower_case: false,
            }
        };

        let pooling_path = Path::new(path).join("1_Pooling").join("config.json");
        let pooling = if pooling_path.exists() {
            let cfg: PoolingConfig = serde_json::from_slice(&std::fs::read(pooling_path)?)?;
            if cfg.pooling_mode_mean_tokens {
                AllMiniLML6V2PoolingStrategy::Mean
            } else if cfg.pooling_mode_cls_token {
                AllMiniLML6V2PoolingStrategy::Cls
            } else if cfg.pooling_mode_max_tokens {
                AllMiniLML6V2PoolingStrategy::Max
            } else if cfg.pooling_mode_mean_sqrt_len_tokens {
                AllMiniLML6V2PoolingStrategy::MeanSqrtLen
            } else {
                AllMiniLML6V2PoolingStrategy::Mean
            }
        } else {
            AllMiniLML6V2PoolingStrategy::Mean
        };

        let modules_path = Path::new(path).join("modules.json");
        let normalize = if modules_path.exists() {
            let modules: Vec<ModuleEntry> = serde_json::from_slice(&std::fs::read(modules_path)?)?;
            modules
                .iter()
                .any(|module| module.r#type.ends_with(".Normalize"))
        } else {
            false
        };

        Ok(Self {
            base,
            pooling,
            normalize,
            max_seq_length: sentence_bert.max_seq_length,
            do_lower_case: sentence_bert.do_lower_case,
        })
    }
}
