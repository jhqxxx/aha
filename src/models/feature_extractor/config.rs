use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FeatureExtractor {
    pub chunk_length: usize,
    pub dither: f64,
    pub feature_size: usize,
    pub hop_length: usize,
    pub n_fft: usize,
    pub n_samples: usize,
    pub nb_max_frames: usize,
    pub padding_side: String,
    pub padding_value: f32,
    pub return_attention_mask: bool,
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: usize,
}

fn default_sampling_rate() -> usize {
    16000
}
