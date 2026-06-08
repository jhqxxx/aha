use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct MossAudioTokenizerConfig {
    pub sample_rate: usize,
    pub sampling_rate: usize,
    pub downsample_rate: usize,
    pub causal_transformer_context_duration: f64,
    pub number_channels: usize,
    pub enable_channel_interleave: bool,
    pub compute_dtype: String,
    pub dtype: String,
    pub code_dim: usize,
    pub encoder_kwargs: Vec<MossAudioTokenizerModuleConfig>,
    pub decoder_kwargs: Vec<MossAudioTokenizerModuleConfig>,
    pub quantizer_type: String,
    pub quantizer_kwargs: MossAudioTokenizerQuantizerKwargs,
    pub reversed_decoder_kwargs: Vec<MossAudioTokenizerModuleConfig>,
}

#[derive(Debug, Deserialize)]
pub struct MossAudioTokenizerModuleConfig {
    pub module_type: String,
    pub patch_size: Option<usize>,
    pub causal: Option<bool>,
    pub context_duration: Option<f64>,
    pub conv_layout: Option<bool>,
    pub d_model: Option<usize>,
    pub dim_feedforward: Option<usize>,
    pub gating: Option<String>,
    pub input_dimension: Option<usize>,
    pub layer_scale: Option<f64>,
    pub max_period: Option<usize>,
    pub norm: Option<String>,
    pub num_heads: Option<usize>,
    pub num_layers: Option<usize>,
    pub output_dimension: Option<usize>,
    pub positional_embedding: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct MossAudioTokenizerQuantizerKwargs {
    pub codebook_dim: usize,
    pub codebook_loss_weight: f64,
    pub codebook_size: usize,
    pub commitment_loss_weight: f64,
    pub input_dim: usize,
    pub num_quantizers: usize,
    pub output_dim: usize,
    pub quantizer_dropout: f64,
    pub quantizer_type: String,
    pub rvq_dim: usize,
}
