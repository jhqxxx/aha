#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct SemanticCodec {
    pub codebook_size: usize,
    pub hidden_size: usize,
    pub codebook_dim: usize,
    pub vocos_dim: usize,
    pub vocos_intermediate_dim: usize,
    pub vocos_num_layers: usize,
    #[serde(default = "default_num_quantizers")]
    pub num_quantizers: usize,
    #[serde(default = "default_downsample_scale")]
    pub downsample_scale: usize,
}

fn default_num_quantizers() -> usize {
    1
}

fn default_downsample_scale() -> usize {
    1
}
