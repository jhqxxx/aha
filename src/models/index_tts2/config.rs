use serde::{Deserialize, Deserializer};

use crate::models::mask_gct::config::SemanticCodec;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct IndexTTS2Config {
    pub dataset: Dataset,
    pub gpt: GptConfig,
    pub semantic_codec: SemanticCodec,
    pub s2mel: S2MelConfig,
    pub gpt_checkpoint: String,
    pub w2v_stat: String,
    pub s2mel_checkpoint: String,
    pub emo_matrix: String,
    pub spk_matrix: String,
    pub emo_num: Vec<usize>,
    pub qwen_emo_path: String,
    pub vocoder: Vocoder,
    pub version: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Dataset {
    pub bpe_model: String,
    pub sample_rate: usize,
    pub squeeze: bool,
    pub mel: Mel,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Mel {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub mel_fmin: usize,
    pub normalize: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct GptConfig {
    pub model_dim: usize,
    pub max_mel_tokens: usize,
    pub max_text_tokens: usize,
    pub heads: usize,
    pub use_mel_codes_as_input: bool,
    pub mel_length_compression: usize,
    pub layers: usize,
    pub number_text_tokens: usize,
    pub number_mel_codes: usize,
    pub start_mel_token: usize,
    pub stop_mel_token: usize,
    pub start_text_token: usize,
    pub stop_text_token: usize,
    pub train_solo_embeddings: bool,
    pub condition_type: String,
    pub condition_module: ConditionModule,
    pub emo_condition_module: EmoConditionModule,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ConditionModule {
    pub output_size: usize,
    pub linear_units: usize,
    pub attention_heads: usize,
    pub num_blocks: usize,
    pub input_layer: String,
    pub perceiver_mult: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct EmoConditionModule {
    pub output_size: usize,
    pub linear_units: usize,
    pub attention_heads: usize,
    pub num_blocks: usize,
    pub input_layer: String,
    pub perceiver_mult: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct S2MelConfig {
    pub preprocess_params: PreprocessParams,
    pub dit_type: String,
    pub reg_loss_type: String,
    pub style_encoder: StyleEncoder,
    pub length_regulator: LengthRegulator,
    #[serde(rename = "DiT")]
    pub di_t: DiTConfig,
    pub wavenet: WavenetConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PreprocessParams {
    pub sr: usize,
    pub spect_params: SpectParams,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SpectParams {
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub fmin: usize,
    #[serde(deserialize_with = "deserialize_optional_fmax")]
    pub fmax: Option<usize>,
}

fn deserialize_optional_fmax<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where D: Deserializer<'de> {
    let opt: Option<serde_json::Value> = Option::deserialize(deserializer)?;
    match opt {
        Some(serde_json::Value::String(s)) if s == "None" || s == "null" => Ok(None),
        Some(serde_json::Value::Number(n)) => {
            if let Some(n) = n.as_u64() {
                Ok(Some(n as usize))
            } else {
                Err(serde::de::Error::custom("Expected positive integer"))
            }
        }
        Some(_) => Err(serde::de::Error::custom("Expected number or 'None' string")),
        None => Ok(None),
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct StyleEncoder {
    pub dim: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LengthRegulator {
    pub channels: usize,
    pub is_discrete: bool,
    pub in_channels: usize,
    pub content_codebook_size: usize,
    pub sampling_ratios: Vec<usize>,
    pub vector_quantize: bool,
    pub n_codebooks: usize,
    pub quantizer_dropout: f32,
    pub f0_condition: bool,
    pub n_f0_bins: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DiTConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub depth: usize,
    pub class_dropout_prob: f32,
    pub block_size: usize,
    pub in_channels: usize,
    pub style_condition: bool,
    pub final_layer_type: String,
    pub target: String,
    pub content_dim: usize,
    pub content_codebook_size: usize,
    pub content_type: String,
    pub f0_condition: bool,
    pub n_f0_bins: usize,
    pub content_codebooks: usize,
    pub is_causal: bool,
    pub long_skip_connection: bool,
    pub zero_prompt_speech_token: bool,
    pub time_as_token: bool,
    pub style_as_token: bool,
    pub uvit_skip_connection: bool,
    pub add_resblock_in_transformer: bool,
}

pub struct DiTModelArgs {
    pub block_size: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub dim: usize,
    pub intermediate_size: usize,
    pub n_local_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub norm_eps: f64,
    pub has_cross_attention: bool,
    pub context_dim: usize,
    pub uvit_skip_connection: bool,
    pub time_as_token: bool,
}

impl DiTModelArgs {
    pub fn new_from_dit_config(config: &DiTConfig) -> Self {
        let hidden_dim = 4 * config.hidden_dim;
        let n_hidden = 2 * hidden_dim / 3;
        let intermediate_size = n_hidden + (256 - n_hidden % 256) % 256;
        Self {
            block_size: config.block_size,
            vocab_size: 1024,
            n_layer: config.depth,
            n_head: config.num_heads,
            dim: config.hidden_dim,
            intermediate_size,
            n_local_heads: config.num_heads,
            head_dim: config.hidden_dim / config.num_heads,
            rope_base: 10000.0,
            norm_eps: 1e-5,
            has_cross_attention: false,
            context_dim: 0,
            uvit_skip_connection: config.uvit_skip_connection,
            time_as_token: config.time_as_token,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct WavenetConfig {
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub kernel_size: usize,
    pub dilation_rate: usize,
    pub p_dropout: f32,
    pub style_condition: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Vocoder {
    pub r#type: String,
    pub name: String,
}
