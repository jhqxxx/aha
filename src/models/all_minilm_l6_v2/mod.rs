use crate::{
    models::common::embedding::{NormalizeType, TextEmbedding},
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};

pub struct AllMiniLML6V2Embedding {
    tokenizer: TokenizerModel,
    model: BertModel,
    device: Device,
    normalize: NormalizeType,
}

impl AllMiniLML6V2Embedding {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let tokenizer = TokenizerModel::init(path)?;
        let config_path = path.to_string() + "/config.json";
        let cfg: BertConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, "float32");
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = BertModel::load(vb, &cfg)?;
        Ok(Self {
            tokenizer,
            model,
            device,
            normalize: NormalizeType::L2,
        })
    }

    fn prepare_token_ids(&self, text: &str) -> Result<Vec<u32>> {
        let mut token_ids = self.tokenizer.text_encode_vec(text.to_string(), true)?;
        token_ids = token_ids
            .into_iter()
            .filter(|&x| x != 0)
            .collect::<Vec<u32>>();
        if token_ids.is_empty() {
            return Err(anyhow!("embedding tokenized input cannot be empty"));
        }
        Ok(token_ids)
    }
    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let token_ids = self.prepare_token_ids(text)?;
        let seq_len = token_ids.len();
        let input_ids = Tensor::from_slice(&token_ids, (1, seq_len), &self.device)?;
        let token_type_ids = Tensor::zeros((1, seq_len), DType::U32, &self.device)?;
        let attention_mask = Tensor::ones((1, seq_len), DType::U32, &self.device)?;
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?
            .to_dtype(DType::F32)?;
        let hidden = hidden.mean(1)?;
        let embed = self
            .normalize
            .normalize(&hidden, hidden.rank() - 1)?
            .squeeze(0)?;
        let embed = embed.to_vec1::<f32>()?;
        Ok(embed)
    }
}

impl TextEmbedding for AllMiniLML6V2Embedding {
    fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(anyhow!("embedding input cannot be empty"));
        }
        let mut out = Vec::with_capacity(input.len());
        for text in input {
            out.push(self.embed_one(text)?);
        }
        Ok(out)
    }
}
