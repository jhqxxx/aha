use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::{
    models::{
        common::retrieval::{l2_normalize, mean_pool},
        qwen3::model::Qwen3Model,
        qwen3_embedding::config::{Qwen3EmbeddingConfig, Qwen3EmbeddingPoolingStrategy},
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub struct Qwen3EmbeddingBackend {
    tokenizer: TokenizerModel,
    model: Qwen3Model,
    device: Device,
    pooling: Qwen3EmbeddingPoolingStrategy,
    normalize: bool,
}

impl Qwen3EmbeddingBackend {
    pub fn load(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let tokenizer = TokenizerModel::init(path)?;
        let cfg = Qwen3EmbeddingConfig::load(path)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, cfg.base.torch_dtype.as_str());
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = Qwen3Model::new(&cfg.base, vb)?;
        Ok(Self {
            tokenizer,
            model,
            device,
            pooling: cfg.pooling,
            normalize: cfg.normalize,
        })
    }

    pub fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(anyhow!("embedding input cannot be empty"));
        }
        let mut out = Vec::with_capacity(input.len());
        for text in input {
            out.push(self.embed_one(text)?);
            self.model.clear_kv_cache();
        }
        Ok(out)
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let input_ids = self.tokenizer.text_encode(text.to_string(), &self.device)?;
        let hidden = self
            .model
            .forward_hidden(Some(&input_ids), None, 0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let hidden_vec = hidden.to_vec2::<f32>()?;
        let mut pooled = match self.pooling {
            Qwen3EmbeddingPoolingStrategy::Mean => mean_pool(&hidden_vec)?,
        };
        if self.normalize {
            l2_normalize(&mut pooled);
        }
        Ok(pooled)
    }
}
