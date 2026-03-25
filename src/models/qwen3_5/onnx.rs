use anyhow::{Result, anyhow};
use candle_core::Tensor;

#[cfg(feature = "onnx-runtime")]
use std::path::{Path, PathBuf};

#[cfg(feature = "onnx-runtime")]
use half::f16;

#[cfg(feature = "onnx-runtime")]
use crate::models::common::onnx::create_session;

#[cfg(feature = "onnx-runtime")]
pub struct Qwen3_5OnnxCacheEntry {
    pub name: String,
    pub dims: Vec<i64>,
    pub data: Vec<f16>,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone)]
struct OnnxInputDescriptor {
    name: String,
    shape: Vec<i64>,
    kind: Option<OnnxTensorKind>,
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone, Copy)]
enum OnnxTensorKind {
    Bool,
    I32,
    I64,
    F16,
    F32,
}

#[cfg(feature = "onnx-runtime")]
fn map_tensor_kind(ty: ort::value::TensorElementType) -> Option<OnnxTensorKind> {
    match ty {
        ort::value::TensorElementType::Bool => Some(OnnxTensorKind::Bool),
        ort::value::TensorElementType::Int32 => Some(OnnxTensorKind::I32),
        ort::value::TensorElementType::Int64 => Some(OnnxTensorKind::I64),
        ort::value::TensorElementType::Float16 => Some(OnnxTensorKind::F16),
        ort::value::TensorElementType::Float32 => Some(OnnxTensorKind::F32),
        _ => None,
    }
}

#[cfg(feature = "onnx-runtime")]
pub struct Qwen3_5OnnxBackend {
    embed_session: ort::session::Session,
    decoder_session: ort::session::Session,
    vision_session: Option<ort::session::Session>,
    decoder_input_names: Vec<String>,
    cache_values: Vec<Qwen3_5OnnxCacheEntry>,
    vision_input_descriptors: Vec<OnnxInputDescriptor>,
    vision_output_names: Vec<String>,
}

#[cfg(feature = "onnx-runtime")]
fn find_onnx_component_file(path: &str, marker: &str) -> Result<PathBuf> {
    let model_path = Path::new(path);
    if !model_path.exists() {
        return Err(anyhow!("onnx model path not found: {}", path));
    }

    if model_path.is_file() {
        let file_name = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        if file_name.contains(marker) && file_name.ends_with(".onnx") {
            return Ok(model_path.to_path_buf());
        }
    }

    let search_root = if model_path.is_dir() {
        model_path.to_path_buf()
    } else {
        model_path
            .parent()
            .ok_or_else(|| anyhow!("onnx component parent directory not found for {}", path))?
            .to_path_buf()
    };

    let mut stack = vec![search_root];
    let mut matches = Vec::new();
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                stack.push(entry_path);
                continue;
            }
            let file_name = entry_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            if file_name.contains(marker) && file_name.ends_with(".onnx") {
                matches.push(entry_path);
            }
        }
    }
    matches.sort();
    matches.into_iter().next().ok_or_else(|| {
        anyhow!(
            "unable to locate onnx component {} under {}",
            marker,
            model_path.display()
        )
    })
}

#[cfg(feature = "onnx-runtime")]
impl Qwen3_5OnnxBackend {
    pub fn load(onnx_path: &str) -> Result<Self> {
        let embed_file = find_onnx_component_file(onnx_path, "embed_tokens")?;
        let decoder_file = find_onnx_component_file(onnx_path, "decoder_model_merged")?;
        let embed_bundle = create_session(&embed_file.to_string_lossy(), None)?;
        let decoder_bundle = create_session(&decoder_file.to_string_lossy(), None)?;

        let (vision_session, vision_input_descriptors, vision_output_names) =
            match find_onnx_component_file(onnx_path, "vision_encoder") {
                Ok(vision_file) => {
                    let bundle = create_session(&vision_file.to_string_lossy(), None)?;
                    let input_descriptors = bundle
                        .session
                        .inputs()
                        .iter()
                        .map(|input| {
                            let (shape, kind) = match input.dtype() {
                                ort::value::ValueType::Tensor { ty, shape, .. } => (
                                    shape.iter().copied().collect::<Vec<_>>(),
                                    map_tensor_kind(*ty),
                                ),
                                _ => (Vec::new(), None),
                            };
                            OnnxInputDescriptor {
                                name: input.name().to_string(),
                                shape,
                                kind,
                            }
                        })
                        .collect::<Vec<_>>();
                    (
                        Some(bundle.session),
                        input_descriptors,
                        bundle.output_names.clone(),
                    )
                }
                Err(_) => (None, Vec::new(), Vec::new()),
            };

        Ok(Self {
            embed_session: embed_bundle.session,
            decoder_session: decoder_bundle.session,
            vision_session,
            decoder_input_names: decoder_bundle.input_names,
            cache_values: Vec::new(),
            vision_input_descriptors,
            vision_output_names,
        })
    }

    pub fn clear_cache(&mut self) {
        self.cache_values.clear();
    }

    pub fn supports_vision(&self) -> bool {
        self.vision_session.is_some()
    }

    pub fn forward_logits(
        &mut self,
        input_ids: &[u32],
        position_start: usize,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        image_token_id: Option<u32>,
    ) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Err(anyhow!("qwen3.5 onnx input_ids cannot be empty"));
        }

        let (mut embed_data, hidden_size) = self.embed_input_ids(input_ids)?;
        if let Some(pixel_values) = pixel_values {
            let image_grid_thw = image_grid_thw
                .ok_or_else(|| anyhow!("qwen3.5 onnx image_grid_thw is required for vision"))?;
            let image_token_id =
                image_token_id.ok_or_else(|| anyhow!("qwen3.5 onnx image_token_id is missing"))?;
            self.apply_vision_embeds(
                &mut embed_data,
                hidden_size,
                input_ids,
                image_token_id,
                pixel_values,
                image_grid_thw,
            )?;
        }

        let mut decoder_inputs = Vec::with_capacity(self.decoder_input_names.len());
        decoder_inputs.push((
            "inputs_embeds".to_string(),
            ort::value::Tensor::from_array((
                vec![1_i64, input_ids.len() as i64, hidden_size as i64],
                embed_data,
            ))?
            .into_dyn(),
        ));
        decoder_inputs.push((
            "attention_mask".to_string(),
            ort::value::Tensor::from_array((
                vec![1_i64, self.attention_mask_len(input_ids.len())? as i64],
                self.build_attention_mask(input_ids.len())?,
            ))?
            .into_dyn(),
        ));
        decoder_inputs.push((
            "position_ids".to_string(),
            ort::value::Tensor::from_array((
                vec![3_i64, 1_i64, input_ids.len() as i64],
                self.build_position_ids(position_start, input_ids.len()),
            ))?
            .into_dyn(),
        ));

        self.append_cache_inputs(&mut decoder_inputs)?;

        let decoder_input_names = self.decoder_input_names.clone();
        let (logits, new_cache_values) = {
            let decoder_outputs = self.decoder_session.run(decoder_inputs)?;
            let logits_value = decoder_outputs
                .get("logits")
                .ok_or_else(|| anyhow!("qwen3.5 onnx output logits not found"))?;
            let (shape, logits_data) = logits_value.try_extract_tensor::<f16>()?;
            if shape.len() != 3 || shape[0] != 1 {
                return Err(anyhow!("unexpected qwen3.5 onnx logits shape: {}", shape));
            }
            let seq_len = shape[1] as usize;
            let vocab_size = shape[2] as usize;
            let start = seq_len
                .checked_sub(1)
                .ok_or_else(|| anyhow!("qwen3.5 onnx logits sequence is empty"))?
                * vocab_size;
            let logits = logits_data[start..start + vocab_size]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();

            let mut cache_values = Vec::new();
            for name in &decoder_input_names {
                match name.as_str() {
                    "inputs_embeds" | "attention_mask" | "position_ids" => {}
                    name if name.starts_with("past_conv.") => {
                        let present_name = name.replacen("past_", "present_", 1);
                        let value = decoder_outputs.get(&present_name).ok_or_else(|| {
                            anyhow!("missing qwen3.5 onnx output {}", present_name)
                        })?;
                        let (shape, data) = value.try_extract_tensor::<f16>()?;
                        cache_values.push(Qwen3_5OnnxCacheEntry {
                            name: name.to_string(),
                            dims: shape.iter().copied().collect::<Vec<_>>(),
                            data: data.to_vec(),
                        });
                    }
                    name if name.starts_with("past_recurrent.") => {
                        let present_name = name.replacen("past_", "present_", 1);
                        let value = decoder_outputs.get(&present_name).ok_or_else(|| {
                            anyhow!("missing qwen3.5 onnx output {}", present_name)
                        })?;
                        let (shape, data) = value.try_extract_tensor::<f16>()?;
                        cache_values.push(Qwen3_5OnnxCacheEntry {
                            name: name.to_string(),
                            dims: shape.iter().copied().collect::<Vec<_>>(),
                            data: data.to_vec(),
                        });
                    }
                    name if name.starts_with("past_key_values.") => {
                        let present_name = name.replace("past_key_values.", "present.");
                        let value = decoder_outputs.get(&present_name).ok_or_else(|| {
                            anyhow!("missing qwen3.5 onnx output {}", present_name)
                        })?;
                        let (shape, data) = value.try_extract_tensor::<f16>()?;
                        cache_values.push(Qwen3_5OnnxCacheEntry {
                            name: name.to_string(),
                            dims: shape.iter().copied().collect::<Vec<_>>(),
                            data: data.to_vec(),
                        });
                    }
                    other => {
                        return Err(anyhow!("unsupported qwen3.5 onnx decoder input: {}", other));
                    }
                }
            }

            (logits, cache_values)
        };

        self.cache_values = new_cache_values;
        Ok(logits)
    }

    fn embed_input_ids(&mut self, input_ids: &[u32]) -> Result<(Vec<f32>, usize)> {
        let embed_outputs = self.embed_session.run(vec![(
            "input_ids".to_string(),
            ort::value::Tensor::from_array((
                vec![1_i64, input_ids.len() as i64],
                input_ids.iter().map(|id| *id as i64).collect::<Vec<_>>(),
            ))?
            .into_dyn(),
        )])?;
        let embed_value = embed_outputs
            .get("inputs_embeds")
            .ok_or_else(|| anyhow!("qwen3.5 onnx output inputs_embeds not found"))?;

        if let Ok((shape, embed_data)) = embed_value.try_extract_tensor::<f32>() {
            if shape.len() != 3 || shape[0] != 1 {
                return Err(anyhow!(
                    "unexpected qwen3.5 onnx inputs_embeds shape: {}",
                    shape
                ));
            }
            return Ok((embed_data.to_vec(), shape[2] as usize));
        }
        if let Ok((shape, embed_data)) = embed_value.try_extract_tensor::<f16>() {
            if shape.len() != 3 || shape[0] != 1 {
                return Err(anyhow!(
                    "unexpected qwen3.5 onnx inputs_embeds shape: {}",
                    shape
                ));
            }
            return Ok((
                embed_data
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                shape[2] as usize,
            ));
        }

        Err(anyhow!(
            "qwen3.5 onnx inputs_embeds output must be f32/f16 tensor"
        ))
    }

    fn apply_vision_embeds(
        &mut self,
        embed_data: &mut [f32],
        hidden_size: usize,
        input_ids: &[u32],
        image_token_id: u32,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<()> {
        let (vision_embeds, vision_rows, vision_hidden) =
            self.run_vision_encoder(pixel_values, image_grid_thw)?;
        if vision_hidden != hidden_size {
            return Err(anyhow!(
                "qwen3.5 onnx vision hidden size mismatch: vision={}, text={}",
                vision_hidden,
                hidden_size
            ));
        }

        let image_positions = input_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, token)| (*token == image_token_id).then_some(idx))
            .collect::<Vec<_>>();
        if image_positions.len() != vision_rows {
            return Err(anyhow!(
                "qwen3.5 onnx image token/vision embed mismatch: image_tokens={}, vision_embeds={}",
                image_positions.len(),
                vision_rows
            ));
        }

        for (row_idx, token_idx) in image_positions.into_iter().enumerate() {
            let src_start = row_idx * hidden_size;
            let dst_start = token_idx * hidden_size;
            let src_end = src_start + hidden_size;
            let dst_end = dst_start + hidden_size;
            embed_data[dst_start..dst_end].copy_from_slice(&vision_embeds[src_start..src_end]);
        }
        Ok(())
    }

    fn run_vision_encoder(
        &mut self,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let pixel_values_shape = pixel_values
            .dims()
            .iter()
            .map(|dim| *dim as i64)
            .collect::<Vec<_>>();
        let pixel_values_data = pixel_values
            .flatten_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_vec1::<f32>()?;
        let image_grid_shape = image_grid_thw
            .dims()
            .iter()
            .map(|dim| *dim as i64)
            .collect::<Vec<_>>();
        let image_grid_data = image_grid_thw
            .flatten_all()?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|value| value as i64)
            .collect::<Vec<_>>();

        let vision_output_names = self.vision_output_names.clone();
        let mut vision_inputs = Vec::with_capacity(self.vision_input_descriptors.len());
        for desc in &self.vision_input_descriptors {
            let value = match desc.name.as_str() {
                "pixel_values" => match desc.kind {
                    Some(OnnxTensorKind::F16) => ort::value::Tensor::from_array((
                        pixel_values_shape.clone(),
                        pixel_values_data
                            .iter()
                            .map(|value| f16::from_f32(*value))
                            .collect::<Vec<_>>(),
                    ))?
                    .into_dyn(),
                    _ => ort::value::Tensor::from_array((
                        pixel_values_shape.clone(),
                        pixel_values_data.clone(),
                    ))?
                    .into_dyn(),
                },
                "image_grid_thw" => ort::value::Tensor::from_array((
                    image_grid_shape.clone(),
                    image_grid_data.clone(),
                ))?
                .into_dyn(),
                _ => build_zero_input(desc)?,
            };
            vision_inputs.push((desc.name.clone(), value));
        }

        let vision_session = self.vision_session.as_mut().ok_or_else(|| {
            anyhow!(
                "qwen3.5 onnx artifact does not include vision_encoder component for multimodal inference"
            )
        })?;
        let outputs = vision_session.run(vision_inputs)?;
        let output_value = outputs
            .get("image_embeds")
            .or_else(|| outputs.get("vision_embeds"))
            .or_else(|| {
                vision_output_names
                    .first()
                    .and_then(|name| outputs.get(name))
            })
            .ok_or_else(|| anyhow!("qwen3.5 onnx vision output not found"))?;

        if let Ok((shape, values)) = output_value.try_extract_tensor::<f32>() {
            let shape_vec = shape.iter().copied().collect::<Vec<_>>();
            return extract_vision_output(shape_vec.as_slice(), values.to_vec());
        }
        if let Ok((shape, values)) = output_value.try_extract_tensor::<f16>() {
            let shape_vec = shape.iter().copied().collect::<Vec<_>>();
            let values = values
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>();
            return extract_vision_output(shape_vec.as_slice(), values);
        }

        Err(anyhow!("qwen3.5 onnx vision output must be f32/f16 tensor"))
    }

    fn attention_mask_len(&self, current_seq_len: usize) -> Result<usize> {
        if self.cache_values.is_empty() {
            return Ok(current_seq_len + 1);
        }
        self.cache_values
            .iter()
            .find(|entry| entry.name.starts_with("past_key_values."))
            .map(|entry| entry.dims[2] as usize)
            .ok_or_else(|| anyhow!("qwen3.5 onnx full-attention cache not found"))
    }

    fn build_attention_mask(&self, current_seq_len: usize) -> Result<Vec<i64>> {
        let len = self.attention_mask_len(current_seq_len)?;
        if self.cache_values.is_empty() {
            let mut mask = Vec::with_capacity(len);
            mask.push(0_i64);
            mask.extend(std::iter::repeat_n(1_i64, current_seq_len));
            return Ok(mask);
        }
        Ok(vec![1_i64; len])
    }

    fn build_position_ids(&self, position_start: usize, seq_len: usize) -> Vec<i64> {
        let positions = (position_start..position_start + seq_len)
            .map(|idx| idx as i64)
            .collect::<Vec<_>>();
        let mut ids = Vec::with_capacity(seq_len * 3);
        for _ in 0..3 {
            ids.extend_from_slice(&positions);
        }
        ids
    }

    fn append_cache_inputs(
        &self,
        decoder_inputs: &mut Vec<(String, ort::value::DynValue)>,
    ) -> Result<()> {
        for name in &self.decoder_input_names {
            match name.as_str() {
                "inputs_embeds" | "attention_mask" | "position_ids" => {}
                name if name.starts_with("past_conv.") => {
                    if let Some(cache) = self.cache_values.iter().find(|entry| entry.name == name) {
                        decoder_inputs.push((
                            name.to_string(),
                            ort::value::Tensor::from_array((
                                cache.dims.clone(),
                                cache.data.clone(),
                            ))?
                            .into_dyn(),
                        ));
                    } else {
                        decoder_inputs.push((
                            name.to_string(),
                            ort::value::Tensor::from_array((
                                vec![1_i64, 6144_i64, 4_i64],
                                vec![f16::from_f32(0.0); 6144 * 4],
                            ))?
                            .into_dyn(),
                        ));
                    }
                }
                name if name.starts_with("past_recurrent.") => {
                    if let Some(cache) = self.cache_values.iter().find(|entry| entry.name == name) {
                        decoder_inputs.push((
                            name.to_string(),
                            ort::value::Tensor::from_array((
                                cache.dims.clone(),
                                cache.data.clone(),
                            ))?
                            .into_dyn(),
                        ));
                    } else {
                        decoder_inputs.push((
                            name.to_string(),
                            ort::value::Tensor::from_array((
                                vec![1_i64, 16_i64, 128_i64, 128_i64],
                                vec![f16::from_f32(0.0); 16 * 128 * 128],
                            ))?
                            .into_dyn(),
                        ));
                    }
                }
                name if name.starts_with("past_key_values.") => {
                    if let Some(cache) = self.cache_values.iter().find(|entry| entry.name == name) {
                        decoder_inputs.push((
                            name.to_string(),
                            ort::value::Tensor::from_array((
                                cache.dims.clone(),
                                cache.data.clone(),
                            ))?
                            .into_dyn(),
                        ));
                    } else {
                        decoder_inputs.push((
                            name.to_string(),
                            ort::value::Tensor::from_array((
                                vec![1_i64, 2_i64, 1_i64, 256_i64],
                                vec![f16::from_f32(0.0); 2 * 256],
                            ))?
                            .into_dyn(),
                        ));
                    }
                }
                other => {
                    return Err(anyhow!("unsupported qwen3.5 onnx decoder input: {}", other));
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "onnx-runtime")]
fn resolve_shape(shape: &[i64]) -> Vec<i64> {
    if shape.is_empty() {
        return vec![1_i64];
    }
    shape
        .iter()
        .map(|dim| if *dim < 0 { 1 } else { *dim })
        .collect::<Vec<_>>()
}

#[cfg(feature = "onnx-runtime")]
fn build_zero_input(desc: &OnnxInputDescriptor) -> Result<ort::value::DynValue> {
    let kind = desc.kind.ok_or_else(|| {
        anyhow!(
            "unsupported qwen3.5 onnx vision input dtype for {}",
            desc.name
        )
    })?;
    let shape = resolve_shape(&desc.shape);
    let elem_count = shape
        .iter()
        .fold(1_usize, |acc, dim| acc.saturating_mul(*dim as usize));
    let value = match kind {
        OnnxTensorKind::Bool => {
            ort::value::Tensor::from_array((shape, vec![false; elem_count]))?.into_dyn()
        }
        OnnxTensorKind::I32 => {
            ort::value::Tensor::from_array((shape, vec![0_i32; elem_count]))?.into_dyn()
        }
        OnnxTensorKind::I64 => {
            ort::value::Tensor::from_array((shape, vec![0_i64; elem_count]))?.into_dyn()
        }
        OnnxTensorKind::F16 => {
            ort::value::Tensor::from_array((shape, vec![f16::from_f32(0.0); elem_count]))?
                .into_dyn()
        }
        OnnxTensorKind::F32 => {
            ort::value::Tensor::from_array((shape, vec![0_f32; elem_count]))?.into_dyn()
        }
    };
    Ok(value)
}

#[cfg(feature = "onnx-runtime")]
fn extract_vision_output(shape: &[i64], values: Vec<f32>) -> Result<(Vec<f32>, usize, usize)> {
    match shape {
        [rows, hidden] => Ok((values, *rows as usize, *hidden as usize)),
        [1, rows, hidden] => Ok((values, *rows as usize, *hidden as usize)),
        _ => Err(anyhow!(
            "unexpected qwen3.5 onnx vision output shape: {:?}",
            shape
        )),
    }
}

#[cfg(not(feature = "onnx-runtime"))]
pub struct Qwen3_5OnnxBackend;

#[cfg(not(feature = "onnx-runtime"))]
impl Qwen3_5OnnxBackend {
    pub fn load(_onnx_path: &str) -> Result<Self> {
        Err(anyhow!(
            "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
        ))
    }

    pub fn clear_cache(&mut self) {}

    pub fn supports_vision(&self) -> bool {
        false
    }

    pub fn forward_logits(
        &mut self,
        _input_ids: &[u32],
        _position_start: usize,
        _pixel_values: Option<&Tensor>,
        _image_grid_thw: Option<&Tensor>,
        _image_token_id: Option<u32>,
    ) -> Result<Vec<f32>> {
        Err(anyhow!(
            "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
        ))
    }
}
