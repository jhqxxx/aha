use candle_nn::Activation;
use serde::Deserialize;

/// Vision encoder configuration for GLM-OCR.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrVisionConfig {
    #[serde(default)]
    pub model_type: String,
    /// Number of transformer layers (depth) in the vision encoder. Default: 24
    #[serde(default)]
    pub depth: usize,
    /// Dimensionality of the encoder layers and the pooler layer. Default: 1024
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Non-linear activation function in the encoder. Default: "silu"
    #[serde(default)]
    pub hidden_act: Activation,
    /// Whether to add bias to queries, keys and values. Default: true
    #[serde(default = "default_true")]
    pub attention_bias: bool,
    /// Dropout probability for attention weights. Default: 0.0
    #[serde(default)]
    pub attention_dropout: f64,
    /// Number of attention heads per layer. Default: 16
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    /// Number of input image channels. Default: 3
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    /// Input image resolution. Default: 336
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    /// Size of each image patch. Default: 14
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    /// Epsilon for RMS normalization layers. Default: 1e-5
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// Size used for merging spatial dimensions. Default: 2
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    /// Patch size along the temporal dimension (for video). Default: 2
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    /// Output hidden size of the vision model. Default: 1536
    #[serde(alias = "out_hidden_size", default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    /// Dimensionality of the feed-forward layer. Default: 4096
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    /// Std dev of truncated normal initializer for weight matrices. Default: 0.02
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,
    /// Base frequency for RoPE in vision encoder. Default: 10000.0
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_hidden_size() -> usize {
    1024
}
fn default_true() -> bool {
    true
}
fn default_num_heads() -> usize {
    16
}
fn default_in_channels() -> usize {
    3
}
fn default_image_size() -> usize {
    336
}
fn default_patch_size() -> usize {
    14
}
fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_spatial_merge_size() -> usize {
    2
}
fn default_temporal_patch_size() -> usize {
    2
}
fn default_out_hidden_size() -> usize {
    1536
}
fn default_intermediate_size() -> usize {
    4096
}
fn default_initializer_range() -> f64 {
    0.02
}

fn default_projector_hidden_size() -> usize {
    1536
}

/// Projector configuration for mapping vision features to LLM embedding space.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrProjectorConfig {
    /// Hidden size for the projector. Default: 1536
    #[serde(default = "default_projector_hidden_size")]
    pub hidden_size: usize,
    /// Activation function for the projector.
    #[serde(default)]
    pub projector_hidden_act: Activation,
    /// Number of query tokens for the projector. Default: 256
    #[serde(default = "default_num_queries")]
    pub num_queries: usize,
}

fn default_num_queries() -> usize {
    256
}

/// RoPE (Rotary Position Embedding) configuration parameters.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrRopeParameters {
    /// Type of RoPE scaling (e.g., "mrope" for multimodal).
    #[serde(default)]
    pub rope_type: String,
    /// Section sizes for M-RoPE (Multimodal RoPE) dimensions.
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    /// Fraction of head dim to apply rotary embedding to.
    #[serde(default)]
    pub partial_rotary_factor: f32,
    /// Base frequency for RoPE. Default: 10000.0
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

fn default_rope_theta() -> f32 {
    10000.0
}

/// Text decoder configuration for GLM-OCR.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrTextConfig {
    /// Vocabulary size. Defines the number of different tokens. Default: 59392
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    /// Dimension of the hidden representations. Default: 1024
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Dimension of the MLP representations. Default: 4608
    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,
    /// Number of hidden layers in the transformer decoder. Default: 16
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    /// Number of attention heads per layer. Default: 16
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    /// Number of key-value heads for Grouped Query Attention.
    /// If equal to num_attention_heads, uses MHA. If 1, uses MQA. Default: 8
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    /// Dimension of each attention head. Default: 128
    #[serde(default = "default_head_dim")]
    pub head_dim: Option<usize>,
    /// Maximum sequence length the model can handle. Default: 131072
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    /// Epsilon for RMS normalization layers. Default: 1e-5
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// Base frequency for RoPE positional embeddings.
    #[serde(default)]
    pub rope_theta: f32,
    /// Fraction of head dim to apply rotary embedding to.
    #[serde(default)]
    pub partial_rotary_factor: f32,
    /// Non-linear activation function in the decoder. Default: "silu"
    #[serde(default)]
    pub hidden_act: Activation,
    /// Whether to return key/value caches for faster generation. Default: true
    #[serde(default = "default_true")]
    pub use_cache: bool,
    /// Dropout ratio for attention probabilities. Default: 0.0
    #[serde(default)]
    pub attention_dropout: f64,
    /// Type of RoPE scaling.
    #[serde(default)]
    pub rope_type: String,
    /// Section sizes for M-RoPE dimensions.
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    /// Full RoPE configuration parameters.
    #[serde(default)]
    pub rope_parameters: Option<GlmOcrRopeParameters>,
    /// End-of-sequence token ID.
    #[serde(default)]
    pub eos_token_id: Option<serde_json::Value>,
}

fn default_vocab_size() -> usize {
    59392
}
fn default_num_hidden_layers() -> usize {
    16
}
fn default_num_attention_heads() -> usize {
    16
}
fn default_num_key_value_heads() -> usize {
    8
}
fn default_max_position_embeddings() -> usize {
    131072
}
fn default_text_intermediate_size() -> usize {
    4608
}
fn default_head_dim() -> Option<usize> {
    Some(128)
}

/// Top-level configuration for GLM-OCR multimodal model.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,
    /// Vision encoder configuration.
    #[serde(default)]
    pub vision_config: GlmOcrVisionConfig,
    /// Projector configuration for vision-to-text mapping.
    #[serde(default)]
    pub projector_config: GlmOcrProjectorConfig,
    /// Text decoder configuration.
    #[serde(default)]
    pub text_config: GlmOcrTextConfig,
    /// Token index to encode image prompts. Default: 59280
    #[serde(default = "default_image_token_id")]
    pub image_token_id: u32,
    /// Token index to encode video prompts. Default: 59281
    #[serde(default = "default_video_token_id")]
    pub video_token_id: u32,
    /// Token index marking start of image. Default: 59256
    #[serde(default = "default_image_start_token_id")]
    pub image_start_token_id: u32,
    /// Token index marking end of image. Default: 59257
    #[serde(default = "default_image_end_token_id")]
    pub image_end_token_id: u32,
    /// Token index marking start of video. Default: 59258
    #[serde(default = "default_video_start_token_id")]
    pub video_start_token_id: u32,
    /// Token index marking end of video. Default: 59259
    #[serde(default = "default_video_end_token_id")]
    pub video_end_token_id: u32,
    /// Beginning-of-sequence token ID.
    #[serde(default)]
    pub bos_token_id: u32,
    /// End-of-sequence token ID.
    #[serde(default)]
    pub eos_token_id: u32,
    /// Padding token ID.
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default)]
    pub torch_dtype: String,
}

fn default_image_token_id() -> u32 {
    59280
}
fn default_video_token_id() -> u32 {
    59281
}
fn default_image_start_token_id() -> u32 {
    59256
}
fn default_image_end_token_id() -> u32 {
    59257
}
fn default_video_start_token_id() -> u32 {
    59258
}
fn default_video_end_token_id() -> u32 {
    59259
}

/// Generation configuration for controlling text output.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrGenerationConfig {
    /// Beginning-of-sequence token ID.
    #[serde(default)]
    pub bos_token_id: usize,
    /// Padding token ID.
    #[serde(default)]
    pub pad_token_id: usize,
    /// Whether to use sampling (true) or greedy decoding (false). Default: true
    #[serde(default = "default_true")]
    pub do_sample: bool,
    /// End-of-sequence token ID(s) that stop generation.
    #[serde(default)]
    pub eos_token_id: Vec<usize>,
    /// Nucleus sampling probability threshold. Default: 0.9
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    /// Top-k tokens to consider for sampling. Default: 50
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Sampling temperature (higher = more random). Default: 0.7
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Penalty for repeating tokens. Default: 1.0
    #[serde(default = "default_repetition_penalty")]
    pub repetition_penalty: f32,
}

fn default_top_p() -> f32 {
    0.9
}
fn default_top_k() -> usize {
    50
}
fn default_temperature() -> f32 {
    0.7
}
fn default_repetition_penalty() -> f32 {
    1.0
}

/// Image preprocessor configuration.
#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
pub struct GlmOcrPreprocessorConfig {
    /// Mean values for image normalization (per channel).
    #[serde(default)]
    pub image_mean: Vec<f32>,
    /// Std dev values for image normalization (per channel).
    #[serde(default)]
    pub image_std: Vec<f32>,
    /// Shortest edge for dynamic image resizing. Default: 448
    #[serde(default)]
    pub size: Option<serde_json::Value>,
    /// Shortest edge length for resizing (min_pixels in Python).
    #[serde(default = "default_shortest_edge")]
    pub shortest_edge: usize,
    /// Longest edge for resizing (max_pixels in Python).
    #[serde(default = "default_longest_edge")]
    pub longest_edge: usize,
    /// Patch size for vision encoder.
    #[serde(default = "default_patch_size_14")]
    pub patch_size: Option<usize>,
    /// Merge size for spatial merge.
    #[serde(default = "default_merge_size")]
    pub merge_size: Option<usize>,
}

fn default_shortest_edge() -> usize {
    12544  // Python's default min_pixels
}

fn default_longest_edge() -> usize {
    9633792  // Python's default max_pixels
}

fn default_patch_size_14() -> Option<usize> {
    Some(14)
}

fn default_merge_size() -> Option<usize> {
    Some(2)
}
