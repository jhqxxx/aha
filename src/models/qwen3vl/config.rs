use candle_nn::Activation;

use crate::models::qwen3::config::Qwen3Config;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Size {
    pub longest_edge: usize,
    pub shortest_edge: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct PreprocessorConfig {
    pub size: Size,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}

impl PreprocessorConfig {
    pub fn qwen3_5_img_default() -> Self {
        Self {
            size: Size {
                longest_edge: 16777216,
                shortest_edge: 65536,
            },
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
        }
    }

    pub fn qwen3_5_video_default() -> Self {
        Self {
            size: Size {
                longest_edge: 25165824,
                shortest_edge: 4096,
            },
            patch_size: 16,
            temporal_patch_size: 2,
            merge_size: 2,
            image_mean: vec![0.5, 0.5, 0.5],
            image_std: vec![0.5, 0.5, 0.5],
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeScaling {
    pub rope_type: String,
    pub mrope_section: Vec<usize>,
    pub mrope_interleaved: bool,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3VLTextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub bos_token_id: usize,
    pub dtype: String,
    pub eos_token_id: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_scaling: RopeScaling,
    pub rope_theta: f32,
    pub use_cache: bool,
    pub vocab_size: usize,
}

pub fn qwen3vl_text_config2qwen3_config(cfg: &Qwen3VLTextConfig) -> Qwen3Config {
    Qwen3Config {
        attention_bias: cfg.attention_bias,
        attention_dropout: cfg.attention_dropout as f64,
        bos_token_id: cfg.bos_token_id as u32,
        eos_token_id: cfg.eos_token_id as u32,
        head_dim: cfg.head_dim,
        hidden_act: cfg.hidden_act,
        hidden_size: cfg.hidden_size,
        initializer_range: cfg.initializer_range as f64,
        intermediate_size: cfg.intermediate_size,
        max_position_embeddings: cfg.max_position_embeddings,
        max_window_layers: 0,
        num_attention_heads: cfg.num_attention_heads,
        num_hidden_layers: cfg.num_hidden_layers,
        num_key_value_heads: cfg.num_key_value_heads,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_theta,
        tie_word_embeddings: true,
        torch_dtype: cfg.dtype.clone(),
        use_cache: cfg.use_cache,
        use_sliding_window: false,
        vocab_size: cfg.vocab_size,
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3VLVisionConfig {
    pub deepstack_visual_indexes: Vec<usize>,
    pub depth: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub in_channels: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3VLConfig {
    pub image_token_id: usize,
    pub text_config: Qwen3VLTextConfig,
    pub tie_word_embeddings: bool,
    pub video_token_id: usize,
    pub vision_config: Qwen3VLVisionConfig,
    pub vision_end_token_id: usize,
    pub vision_start_token_id: usize,
}
