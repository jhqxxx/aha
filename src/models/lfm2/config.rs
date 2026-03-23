use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct Lfm2Config {
    pub architectures: Vec<String>,
    pub block_auto_adjust_ff_dim: bool,
    pub block_dim: usize,
    pub block_ff_dim: usize,
    pub block_ffn_dim_multiplier: f64,
    pub block_mlp_init_scale: f64,
    pub block_multiple_of: usize,
    pub block_norm_eps: f64,
    pub block_out_init_scale: f64,
    pub block_use_swiglu: bool,
    pub block_use_xavier_init: bool,
    pub bos_token_id: u32,
    #[serde[rename="conv_L_cache"]]
    pub conv_l_cache: usize,
    pub conv_bias: bool,
    pub conv_dim: usize,
    pub conv_dim_out: Option<usize>,
    pub conv_use_xavier_init: bool,
    pub eos_token_id: u32,
    pub full_attn_idxs: Option<Vec<usize>>,
    pub layer_types: Option<Vec<String>>,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: Option<usize>,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub norm_eps: f64,
    pub num_attention_heads: usize,
    pub num_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pad_token_id: u32,
    pub rope_theta: f32,
    pub torch_dtype: Option<String>,
    pub dtype: Option<String>,
    pub use_cache: bool,
    pub use_pos_enc: bool,
    pub vocab_size: usize,
    pub tie_embedding: Option<bool>,
}

impl Lfm2Config {
    pub fn full_attn_idx2layer_type(&mut self) {
        if self.layer_types.is_none()
            && let Some(idx) = &self.full_attn_idxs
        {
            let mut layer_types = vec![];
            for i in 0..self.num_hidden_layers {
                if idx.contains(&i) {
                    layer_types.push("full_attention".to_string());
                } else {
                    layer_types.push("conv".to_string());
                }
            }
            self.layer_types = Some(layer_types);
        }
    }

    pub fn get_layer_types(&self) -> Result<Vec<String>> {
        if let Some(types) = &self.layer_types {
            Ok(types.clone())
        } else if let Some(idx) = &self.full_attn_idxs {
            let mut layer_types = vec![];
            for i in 0..self.num_hidden_layers {
                if idx.contains(&i) {
                    layer_types.push("full_attention".to_string());
                } else {
                    layer_types.push("conv".to_string());
                }
            }
            Ok(layer_types)
        } else {
            Err(anyhow!(
                "layer_types full_attn_idxs cannot be none at the same time"
            ))
        }
    }
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
pub struct Lfm2GenerateConfig {
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}
