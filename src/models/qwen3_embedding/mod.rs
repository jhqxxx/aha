use crate::{
    models::{
        common::embedding::{NormalizeType, TextEmbedding},
        qwen3::{config::Qwen3Config, model::Qwen3Model},
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

pub struct Qwen3Embedding {
    tokenizer: TokenizerModel,
    model: Qwen3Model,
    device: Device,
    normalize: NormalizeType,
}

impl Qwen3Embedding {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: Qwen3Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, cfg.torch_dtype.as_str());
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = Qwen3Model::new(&cfg, vb, vec![])?;
        Ok(Self {
            tokenizer,
            model,
            device,
            normalize: NormalizeType::L2,
        })
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let input_ids = self.tokenizer.text_encode(text.to_string(), &self.device)?;
        let hidden = self
            .model
            .forward_hidden(Some(&input_ids), None, 0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let norm = self
            .normalize
            .normalize(&hidden, hidden.rank() - 1)?
            .squeeze(0)?;
        let norm = norm.to_vec1::<f32>()?;
        Ok(norm)
    }
}

impl TextEmbedding for Qwen3Embedding {
    fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
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
}
