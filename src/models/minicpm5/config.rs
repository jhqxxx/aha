use candle_nn::Activation;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MiniCPM5Config {
    pub bos_token_id: u32,
    pub eos_token_id: Vec<u32>,
    pub pad_token_id: u32,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub rope_scaling: Option<serde_json::Value>, // Using Value for null/complex scaling objects
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub use_cache: bool,
    pub vocab_size: usize,
}
