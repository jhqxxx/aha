use candle_nn::Activation;

use crate::models::qwen2::Qwen2Config;

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_chans: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub spatial_patch_size: usize,
    pub window_size: usize,
    pub fullatt_block_indexes: Vec<usize>,
    pub tokens_per_second: usize,
    pub temporal_patch_size: usize,
}
#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeScaling {
    pub r#type: String,
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Qwen2_5VLConfig {
    pub attention_dropout: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub vision_start_token_id: usize,
    pub vision_end_token_id: usize,
    pub vision_token_id: usize,
    pub image_token_id: usize,
    pub video_token_id: usize,
    pub hidden_act: Activation,
    pub hidden_size: usize,
    pub initializer_range: f32,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub max_window_layers: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub tie_word_embeddings: bool,
    pub torch_dtype: String,
    pub use_sliding_window: bool,
    pub vision_config: VisionConfig,
    pub rope_scaling: RopeScaling,
    pub vocab_size: usize,
}

impl Qwen2_5VLConfig {
    pub fn to_qwen2cfg(&self) -> Qwen2Config {
        Qwen2Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            max_window_layers: self.max_window_layers,
            tie_word_embeddings: self.tie_word_embeddings,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            use_sliding_window: self.use_sliding_window,
            hidden_act: self.hidden_act,
        }
    }
}

pub struct VisionSetting {
    pub image_factor: u32,
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub max_ratio: u32,
    pub temporal_patch_size: usize,
    pub patch_size: usize,
    pub merge_size: usize,
    pub video_min_pixels: u32,
    pub video_max_pixels: u32,
    pub video_total_pixels: u32,
    pub frame_factor: u32,
    pub fps: f32,
    pub fps_min_frames: u32,
    pub fps_max_frames: u32,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}

impl Default for VisionSetting {
    fn default() -> Self {
        Self {
            image_factor: 28,
            min_pixels: 4 * 28 * 28,
            max_pixels: 16384 * 28 * 28,
            // max_pixels: 1000 * 28 * 28,
            max_ratio: 200,
            temporal_patch_size: 2,
            patch_size: 14,
            merge_size: 2,
            video_min_pixels: 128 * 28 * 28,
            video_max_pixels: 768 * 28 * 28,
            video_total_pixels: 24576 * 28 * 28,
            frame_factor: 2,
            fps: 2.0,
            fps_min_frames: 4,
            fps_max_frames: 768,
            image_mean: vec![0.48145466_f32, 0.4578275f32, 0.40821073f32],
            image_std: vec![0.26862954f32, 0.2613026f32, 0.2757771f32],
        }
    }
}
