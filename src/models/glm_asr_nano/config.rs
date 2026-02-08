use candle_nn::Activation;
use serde::Deserialize;

use crate::models::feature_extractor::config::FeatureExtractor;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmAsrNanoProcessorConfig {
    pub audio_token: String,
    pub default_transcription_prompt: String,
    pub feature_extractor: FeatureExtractor,
    pub max_audio_len: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmAsrNanoConfig {
    pub audio_config: GlmAsrAudioConfig,
    pub audio_token_id: u32,
    pub dtype: String,
    pub hidden_size: usize,
    pub projector_hidden_act: Activation,
    pub text_config: GlmAsrTextConfig,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmAsrAudioConfig {
    pub attention_dropout: f64,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_mel_bins: usize,
    pub partial_rotary_factor: f64,
    pub rope_parameters: GlmAsrRopeParameters,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmAsrRopeParameters {
    pub partial_rotary_factor: f64,
    pub rope_theta: f32,
    pub rope_type: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmAsrTextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f64,
    pub eos_token_id: Vec<u32>,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f64,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub mlp_bias: bool,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pretraining_tp: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: GlmAsrTextRopeParameters,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GlmAsrTextRopeParameters {
    pub rope_theta: f32,
    pub rope_type: String,
}
