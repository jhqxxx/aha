use candle_nn::Activation;

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
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3VLVisionConfig {
    pub deepstack_visual_indexes: Vec<usize>,
    pub depth: usize,
    pub hidden_act: String,
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

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen3VLGenerationConfig {
    pub bos_token_id: usize,
    pub pad_token_id: usize,
    pub do_sample: bool,
    pub eos_token_id: Vec<usize>,
    pub top_p: f32,
    pub top_k: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
}
