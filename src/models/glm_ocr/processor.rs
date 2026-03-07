use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::config::GlmOcrPreprocessorConfig;
use crate::tokenizer::TokenizerModel;
use crate::utils::img_utils::get_image;

/// GLM-OCR Processor for image and text preprocessing.
///
/// Matches Python's Glm46VImageProcessor behavior:
/// - Uses smart_resize to compute target dimensions
/// - Outputs flattened patches format [num_patches, patch_dim]
pub struct GlmOcrProcessor {
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    shortest_edge: usize,  // min_pixels in Python
    longest_edge: usize,   // max_pixels in Python
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    device: Device,
    dtype: DType,
}

pub struct ProcessedImage {
    pub pixel_values: Tensor,  // Shape: [num_patches, patch_dim]
    pub grid_h: usize,
    pub grid_w: usize,
}

pub struct ProcessedInput {
    pub input_ids: Tensor,
    pub pixel_values: Tensor,  // Shape: [num_patches, patch_dim]
    pub image_mask: Tensor,
    pub grid_thw: Tensor,
}

impl GlmOcrProcessor {
    pub fn new(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let config_path = format!("{}/preprocessor_config.json", path);

        // Load preprocessor config
        let (image_mean, image_std, shortest_edge, longest_edge, patch_size, merge_size) =
            if std::path::Path::new(&config_path).exists() {
                let config: GlmOcrPreprocessorConfig = serde_json::from_slice(&std::fs::read(&config_path)?)?;
                // Parse size object - Python uses shortest_edge: 12544, longest_edge: 9633792
                let (shortest, longest) = if let Some(size_val) = &config.size {
                    if let Some(obj) = size_val.as_object() {
                        let s = obj.get("shortest_edge")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                            .unwrap_or(12544);
                        let l = obj.get("longest_edge")
                            .and_then(|v| v.as_u64())
                            .map(|v| v as usize)
                            .unwrap_or(9_633_792);
                        (s, l)
                    } else {
                        (12544, 9_633_792)
                    }
                } else {
                    (12544, 9_633_792)
                };
                let patch_size = config.patch_size.unwrap_or(14);
                let merge_size = config.merge_size.unwrap_or(2);
                (config.image_mean, config.image_std, shortest, longest, patch_size, merge_size)
            } else {
                (
                    vec![0.48145466, 0.4578275, 0.40821073],
                    vec![0.26862954, 0.26130258, 0.27577711],
                    12544,      // Python's min_pixels
                    9_633_792,  // Python's max_pixels
                    14,         // patch_size
                    2,          // merge_size
                )
            };

        Ok(Self {
            image_mean,
            image_std,
            shortest_edge,
            longest_edge,
            patch_size,
            merge_size,
            temporal_patch_size: 2,  // Fixed for images
            device: device.clone(),
            dtype,
        })
    }

    /// Python's smart_resize implementation
    /// Returns (resized_height, resized_width)
    fn smart_resize(&self, height: usize, width: usize) -> (usize, usize) {
        let factor = self.patch_size * self.merge_size;  // 28
        let temporal_factor = self.temporal_patch_size;  // 2
        
        // Ensure minimum size
        let mut h = height;
        let mut w = width;
        if h < factor || w < factor {
            let scale = (factor as f32 / h.min(w) as f32).max(1.0);
            h = (h as f32 * scale).round() as usize;
            w = (w as f32 * scale).round() as usize;
        }
        
        // Check aspect ratio constraint
        if w.max(h) as f32 / w.min(h) as f32 > 200.0 {
            // Would raise error in Python
            // For now, just proceed
        }
        
        // Round to nearest multiple of factor
        let h_bar = ((h + factor / 2) / factor) * factor;
        let w_bar = ((w + factor / 2) / factor) * factor;
        let t_bar = ((temporal_factor + temporal_factor / 2) / temporal_factor) * temporal_factor;
        
        // Check max_pixels constraint
        let max_pixels = self.longest_edge;
        let min_pixels = self.shortest_edge;
        
        let mut final_h = h_bar;
        let mut final_w = w_bar;
        
        if t_bar * h_bar * w_bar > max_pixels {
            let beta = ((h * w) as f32 / max_pixels as f32).sqrt();
            final_h = (factor.max((h as f32 / beta / factor as f32).floor() as usize * factor)) as usize;
            final_w = (factor.max((w as f32 / beta / factor as f32).floor() as usize * factor)) as usize;
        } else if t_bar * h_bar * w_bar < min_pixels {
            let beta = (min_pixels as f32 / (h * w) as f32).sqrt();
            final_h = ((h as f32 * beta / factor as f32).ceil() as usize * factor) as usize;
            final_w = ((w as f32 * beta / factor as f32).ceil() as usize * factor) as usize;
        }
        
        (final_h, final_w)
    }

    /// Process image for vision encoder.
    ///
    /// Matches Python's Glm46VImageProcessor._preprocess():
    /// 1. Resize using smart_resize
    /// 2. Normalize
    /// 3. Reshape into flattened patches [num_patches, patch_dim]
    ///
    /// Output format: [grid_t * grid_h * grid_w, channels * temporal_patch_size * patch_size * patch_size]
    /// For images: grid_t = 1, so [grid_h * grid_w, 3 * 2 * 14 * 14] = [num_patches, 1176]
    pub fn process_image(&self, image_path: &str) -> Result<ProcessedImage> {
        let img = get_image(image_path)?;
        let (orig_w, orig_h) = (img.width() as usize, img.height() as usize);

        // Use smart_resize to compute target dimensions
        let (target_h, target_w) = self.smart_resize(orig_h, orig_w);

        // Resize image
        let img = img.resize_exact(
            target_w as u32,
            target_h as u32,
            image::imageops::FilterType::Lanczos3,
        );

        // Convert to RGB and normalize
        let img = img.to_rgb8();
        let pixels: Vec<f32> = img
            .pixels()
            .flat_map(|p| {
                vec![
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            })
            .collect();

        let tensor = Tensor::from_vec(pixels, (target_h, target_w, 3), &self.device)?;
        
        let mean = Tensor::new(self.image_mean.clone(), &self.device)?.reshape((1, 1, 3))?;
        let std = Tensor::new(self.image_std.clone(), &self.device)?.reshape((1, 1, 3))?;
        let tensor = tensor.broadcast_sub(&mean)?.broadcast_div(&std)?;
        
        // Now reshape into flattened patches like Python
        let grid_h = target_h / self.patch_size;
        let grid_w = target_w / self.patch_size;
        let patch_size = self.patch_size;
        let channels = 3;
        let temporal_patch_size = 2; 
        
        let tensor = tensor.reshape((
            grid_h, patch_size,
            grid_w, patch_size,
            channels,
        ))?;

        let tensor = tensor.permute((0, 2, 4, 1, 3))?;

        let num_patches = grid_h * grid_w;
        let tensor = tensor.reshape((num_patches, channels, patch_size, patch_size))?;

        let tensor = tensor.unsqueeze(2)?;
        let tensor = tensor.repeat((1, 1, temporal_patch_size, 1, 1))?;

        let patch_dim = channels * temporal_patch_size * patch_size * patch_size;
        let tensor = tensor.reshape((num_patches, patch_dim))?;
        
        let tensor = tensor.to_dtype(self.dtype)?;
        
        Ok(ProcessedImage {
            pixel_values: tensor,
            grid_h,
            grid_w,
        })
    }

    /// Process image and text for multimodal input
    pub fn process_info(
        &self,
        image_path: &str,
        prompt: &str,
        tokenizer: &TokenizerModel,
        image_token_id: u32,
        image_start_token_id: u32,
        image_end_token_id: u32,
        _patch_size: usize,
        _temporal_patch_size: usize,
        spatial_merge_size: usize,
    ) -> Result<ProcessedInput> {
        let processed_image = self.process_image(image_path)?;
        let pixel_values = processed_image.pixel_values;
        let grid_h = processed_image.grid_h;
        let grid_w = processed_image.grid_w;

        // After spatial merge, each spatial_merge_size x spatial_merge_size block becomes 1 token
        let merged_h = grid_h / spatial_merge_size;
        let merged_w = grid_w / spatial_merge_size;
        let num_image_tokens = merged_h * merged_w;

        // GLM-OCR format: [gMASK] <sop> <|user|> \n <|begin_of_image|> <|image|>*N <|end_of_image|> text <|assistant|> \n
        // Special token IDs:
        //   59248 = [gMASK]
        //   59250 = <sop>
        //   59253 = <|user|>
        //   59256 = <|begin_of_image|>
        //   59280 = <|image|>
        //   59257 = <|end_of_image|>
        //   59254 = <|assistant|>

        // Build input_ids following Python format
        let mut input_ids_vec = Vec::new();

        // Header: [gMASK] <sop> <|user|> \n
        input_ids_vec.push(59248); // [gMASK]
        input_ids_vec.push(59250); // <sop>
        input_ids_vec.push(59253); // <|user|>
        input_ids_vec.push(10);    // newline

        // Image tokens: <|begin_of_image|> <|image|>*N <|end_of_image|>
        input_ids_vec.push(image_start_token_id); // <|begin_of_image|>
        for _ in 0..num_image_tokens {
            input_ids_vec.push(image_token_id); // <|image|>
        }
        input_ids_vec.push(image_end_token_id); // <|end_of_image|>

        // Text prompt (without special tokens - they're already added)
        let text_ids = tokenizer.text_encode_vec(prompt.to_string(), false)?;
        input_ids_vec.extend(text_ids);

        // Generation prompt: <|assistant|> \n
        input_ids_vec.push(59254); // <|assistant|>
        input_ids_vec.push(10);    // newline

        let input_ids = Tensor::from_vec(
            input_ids_vec.clone(),
            (1, input_ids_vec.len()),
            &self.device,
        )?;

        // Create image mask (1s at image token positions, 0 elsewhere)
        // Image tokens start after header (4 tokens) + start token (1 token) = index 5
        let mut image_mask_vec = vec![0u32; input_ids_vec.len()];
        let image_start_idx = 5; // After [gMASK, sop, user, newline, begin_image]
        for i in 0..num_image_tokens {
            image_mask_vec[image_start_idx + i] = 1;
        }
        let image_mask = Tensor::from_vec(image_mask_vec, (1, input_ids_vec.len()), &self.device)?;

        // Compute grid_thw for RoPE
        let grid_thw = Tensor::from_vec(
            vec![1u32, grid_h as u32, grid_w as u32],
            (3,),
            &self.device,
        )?;

        Ok(ProcessedInput {
            input_ids,
            pixel_values,
            image_mask,
            grid_thw,
        })
    }
}
