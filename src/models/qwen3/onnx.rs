use anyhow::{Result, anyhow};

#[cfg(feature = "onnx-runtime")]
use half::f16;
#[cfg(feature = "onnx-runtime")]
use ndarray::{Array, IxDyn};

use crate::models::common::onnx::create_session;

#[cfg_attr(not(feature = "onnx-runtime"), allow(dead_code))]
#[derive(Clone)]
struct OnnxInputDescriptor {
    name: String,
    shape: Vec<i64>,
    kind: Option<OnnxTensorKind>,
}

#[cfg_attr(not(feature = "onnx-runtime"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
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
#[derive(Clone)]
enum OnnxCacheData {
    F16(Vec<f16>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    Bool(Vec<bool>),
}

#[cfg(feature = "onnx-runtime")]
#[derive(Clone)]
struct OnnxCacheEntry {
    name: String,
    dims: Vec<i64>,
    data: OnnxCacheData,
}

#[cfg_attr(not(feature = "onnx-runtime"), allow(dead_code))]
pub struct Qwen3OnnxBackend {
    #[cfg(feature = "onnx-runtime")]
    session: ort::session::Session,
    #[cfg(not(feature = "onnx-runtime"))]
    _session: (),
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_descriptors: Vec<OnnxInputDescriptor>,
    #[cfg(feature = "onnx-runtime")]
    cache_entries: Vec<OnnxCacheEntry>,
    cache_input_names: Vec<String>,
}

impl Qwen3OnnxBackend {
    pub fn load(onnx_path: &str) -> Result<Self> {
        let bundle = create_session(onnx_path, None)?;
        #[cfg(feature = "onnx-runtime")]
        {
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
            let cache_input_names = bundle
                .input_names
                .iter()
                .filter(|name| is_cache_input_name(name))
                .cloned()
                .collect::<Vec<_>>();

            Ok(Self {
                session: bundle.session,
                input_names: bundle.input_names,
                output_names: bundle.output_names,
                input_descriptors,
                cache_entries: Vec::new(),
                cache_input_names,
            })
        }
        #[cfg(not(feature = "onnx-runtime"))]
        {
            let _ = bundle;
            Ok(Self {
                _session: (),
                input_names: Vec::new(),
                output_names: Vec::new(),
                input_descriptors: Vec::new(),
                cache_input_names: Vec::new(),
            })
        }
    }

    pub fn clear_cache(&mut self) {
        #[cfg(feature = "onnx-runtime")]
        self.cache_entries.clear();
    }

    #[cfg(feature = "onnx-runtime")]
    pub fn forward_logits(&mut self, input_ids: &[u32], position_start: usize) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            return Err(anyhow!("qwen3 onnx input_ids cannot be empty"));
        }

        let input_ids_i64 = input_ids.iter().map(|id| *id as i64).collect::<Vec<_>>();
        let seq_len = input_ids_i64.len();
        let total_attention_len = if self.cache_input_names.is_empty() {
            seq_len.max(1)
        } else {
            (position_start + seq_len).max(1)
        };
        let attention_mask = vec![1_i64; total_attention_len];
        let position_ids = (position_start..position_start + seq_len)
            .map(|idx| idx as i64)
            .collect::<Vec<_>>();

        let mut inputs = Vec::with_capacity(self.input_names.len());
        for desc in &self.input_descriptors {
            let value =
                self.build_input_value(desc, &input_ids_i64, &attention_mask, &position_ids)?;
            inputs.push((desc.name.clone(), value));
        }

        let outputs = self.session.run(inputs).map_err(|err| {
            let input_desc = self
                .input_descriptors
                .iter()
                .map(|desc| desc.name.clone())
                .collect::<Vec<_>>()
                .join(", ");
            anyhow!("failed to run qwen3 onnx session: {err}; inputs={input_desc}")
        })?;
        let output_names = self.output_names.clone();

        let logits_value = outputs
            .get("logits")
            .or_else(|| output_names.first().and_then(|name| outputs.get(name)))
            .ok_or_else(|| anyhow!("qwen3 onnx output logits not found"))?;
        let logits = if let Ok((shape, data)) = logits_value.try_extract_tensor::<f32>() {
            extract_last_token_logits(&shape, data)?
        } else if let Ok((shape, data)) = logits_value.try_extract_tensor::<f16>() {
            extract_last_token_logits_f16(&shape, data)?
        } else {
            return Err(anyhow!(
                "qwen3 onnx logits output must be f32 or f16 tensor"
            ));
        };

        let mut new_cache = Vec::new();
        for input_name in &self.cache_input_names {
            let Some(output_name) = resolve_present_output_name(input_name, &output_names) else {
                continue;
            };
            let Some(output_value) = outputs.get(&output_name) else {
                continue;
            };

            if let Ok((shape, data)) = output_value.try_extract_tensor::<f16>() {
                new_cache.push(OnnxCacheEntry {
                    name: input_name.clone(),
                    dims: shape.iter().copied().collect::<Vec<_>>(),
                    data: OnnxCacheData::F16(data.to_vec()),
                });
                continue;
            }
            if let Ok((shape, data)) = output_value.try_extract_tensor::<f32>() {
                new_cache.push(OnnxCacheEntry {
                    name: input_name.clone(),
                    dims: shape.iter().copied().collect::<Vec<_>>(),
                    data: OnnxCacheData::F32(data.to_vec()),
                });
                continue;
            }
            if let Ok((shape, data)) = output_value.try_extract_tensor::<i64>() {
                new_cache.push(OnnxCacheEntry {
                    name: input_name.clone(),
                    dims: shape.iter().copied().collect::<Vec<_>>(),
                    data: OnnxCacheData::I64(data.to_vec()),
                });
                continue;
            }
            if let Ok((shape, data)) = output_value.try_extract_tensor::<i32>() {
                new_cache.push(OnnxCacheEntry {
                    name: input_name.clone(),
                    dims: shape.iter().copied().collect::<Vec<_>>(),
                    data: OnnxCacheData::I32(data.to_vec()),
                });
                continue;
            }
            if let Ok((shape, data)) = output_value.try_extract_tensor::<bool>() {
                new_cache.push(OnnxCacheEntry {
                    name: input_name.clone(),
                    dims: shape.iter().copied().collect::<Vec<_>>(),
                    data: OnnxCacheData::Bool(data.to_vec()),
                });
            }
        }
        self.cache_entries = new_cache;
        Ok(logits)
    }

    #[cfg(not(feature = "onnx-runtime"))]
    pub fn forward_logits(
        &mut self,
        _input_ids: &[u32],
        _position_start: usize,
    ) -> Result<Vec<f32>> {
        Err(anyhow!(
            "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
        ))
    }

    #[cfg(feature = "onnx-runtime")]
    fn build_input_value(
        &self,
        desc: &OnnxInputDescriptor,
        input_ids: &[i64],
        attention_mask: &[i64],
        position_ids: &[i64],
    ) -> Result<ort::value::DynValue> {
        if is_cache_input_name(&desc.name)
            && let Some(entry) = self
                .cache_entries
                .iter()
                .find(|entry| entry.name == desc.name)
        {
            return build_cached_tensor(entry);
        }

        let shape = self.resolve_shape(desc, input_ids.len(), attention_mask.len(), position_ids);
        match desc.name.as_str() {
            "input_ids" => {
                return Ok(ort::value::Tensor::from_array((shape, input_ids.to_vec()))?.into_dyn());
            }
            "attention_mask" => {
                return Ok(
                    ort::value::Tensor::from_array((shape, attention_mask.to_vec()))?.into_dyn(),
                );
            }
            "position_ids" => {
                let ids = if shape.len() == 3 && shape[0] == 3 {
                    let mut out = Vec::with_capacity(position_ids.len() * 3);
                    for _ in 0..3 {
                        out.extend_from_slice(position_ids);
                    }
                    out
                } else {
                    position_ids.to_vec()
                };
                return Ok(ort::value::Tensor::from_array((shape, ids))?.into_dyn());
            }
            _ => {}
        }

        self.build_zero_tensor(desc, shape)
    }

    #[cfg(feature = "onnx-runtime")]
    fn resolve_shape(
        &self,
        desc: &OnnxInputDescriptor,
        seq_len: usize,
        attention_mask_len: usize,
        position_ids: &[i64],
    ) -> Vec<i64> {
        if desc.shape.is_empty() {
            return vec![1_i64];
        }
        let mut shape = desc.shape.clone();
        let shape_len = shape.len();
        for (idx, dim) in shape.iter_mut().enumerate() {
            if *dim >= 0 {
                continue;
            }
            *dim = match desc.name.as_str() {
                "input_ids" => {
                    if idx == 0 {
                        1
                    } else {
                        seq_len as i64
                    }
                }
                "attention_mask" => {
                    if idx == 0 {
                        1
                    } else {
                        attention_mask_len as i64
                    }
                }
                "position_ids" => {
                    if shape_len == 3 {
                        if idx == 0 {
                            3
                        } else if idx == 1 {
                            1
                        } else {
                            seq_len as i64
                        }
                    } else if idx == 0 {
                        1
                    } else {
                        seq_len as i64
                    }
                }
                _ if is_cache_input_name(&desc.name) => {
                    if idx == 0 {
                        1
                    } else if idx + 2 == shape_len {
                        position_ids.first().copied().unwrap_or(0)
                    } else {
                        1
                    }
                }
                _ => {
                    if idx == 0 {
                        1
                    } else {
                        seq_len as i64
                    }
                }
            };
        }
        shape
    }

    #[cfg(feature = "onnx-runtime")]
    fn build_zero_tensor(
        &self,
        desc: &OnnxInputDescriptor,
        shape: Vec<i64>,
    ) -> Result<ort::value::DynValue> {
        let kind = desc.kind.ok_or_else(|| {
            anyhow!(
                "unsupported qwen3 onnx input dtype for {} (shape={:?})",
                desc.name,
                shape
            )
        })?;
        let elem_count = shape.iter().try_fold(1_usize, |acc, dim| {
            if *dim < 0 {
                Err(anyhow!(
                    "cannot resolve dynamic qwen3 onnx shape for {}: {:?}",
                    desc.name,
                    shape
                ))
            } else {
                Ok(acc.saturating_mul(*dim as usize))
            }
        })?;
        let has_zero_dim = shape.contains(&0);
        let make_ndarray_dims = || {
            let dims = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
            IxDyn(&dims)
        };
        let value = match kind {
            OnnxTensorKind::Bool => {
                if has_zero_dim {
                    let arr = Array::from_shape_vec(make_ndarray_dims(), vec![false; elem_count])?;
                    ort::value::Tensor::from_array(arr)?.into_dyn()
                } else {
                    ort::value::Tensor::from_array((shape, vec![false; elem_count]))?.into_dyn()
                }
            }
            OnnxTensorKind::I32 => {
                if has_zero_dim {
                    let arr = Array::from_shape_vec(make_ndarray_dims(), vec![0_i32; elem_count])?;
                    ort::value::Tensor::from_array(arr)?.into_dyn()
                } else {
                    ort::value::Tensor::from_array((shape, vec![0_i32; elem_count]))?.into_dyn()
                }
            }
            OnnxTensorKind::I64 => {
                if has_zero_dim {
                    let arr = Array::from_shape_vec(make_ndarray_dims(), vec![0_i64; elem_count])?;
                    ort::value::Tensor::from_array(arr)?.into_dyn()
                } else {
                    ort::value::Tensor::from_array((shape, vec![0_i64; elem_count]))?.into_dyn()
                }
            }
            OnnxTensorKind::F16 => {
                if has_zero_dim {
                    let arr = Array::from_shape_vec(
                        make_ndarray_dims(),
                        vec![f16::from_f32(0.0); elem_count],
                    )?;
                    ort::value::Tensor::from_array(arr)?.into_dyn()
                } else {
                    ort::value::Tensor::from_array((shape, vec![f16::from_f32(0.0); elem_count]))?
                        .into_dyn()
                }
            }
            OnnxTensorKind::F32 => {
                if has_zero_dim {
                    let arr = Array::from_shape_vec(make_ndarray_dims(), vec![0_f32; elem_count])?;
                    ort::value::Tensor::from_array(arr)?.into_dyn()
                } else {
                    ort::value::Tensor::from_array((shape, vec![0_f32; elem_count]))?.into_dyn()
                }
            }
        };
        Ok(value)
    }
}

#[cfg(feature = "onnx-runtime")]
fn is_cache_input_name(name: &str) -> bool {
    name.starts_with("past_key_values.") || name.starts_with("past_")
}

#[cfg(not(feature = "onnx-runtime"))]
#[allow(dead_code)]
fn is_cache_input_name(_name: &str) -> bool {
    false
}

#[cfg(feature = "onnx-runtime")]
fn resolve_present_output_name(input_name: &str, output_names: &[String]) -> Option<String> {
    let candidates = [
        input_name.replace("past_key_values.", "present."),
        input_name.replace("past_key_values.", "present_key_values."),
        input_name.replacen("past_", "present_", 1),
        input_name.to_string(),
    ];
    candidates
        .into_iter()
        .find(|candidate| output_names.iter().any(|name| name == candidate))
}

#[cfg(feature = "onnx-runtime")]
fn build_cached_tensor(entry: &OnnxCacheEntry) -> Result<ort::value::DynValue> {
    let value = match &entry.data {
        OnnxCacheData::F16(data) => {
            ort::value::Tensor::from_array((entry.dims.clone(), data.clone()))?.into_dyn()
        }
        OnnxCacheData::F32(data) => {
            ort::value::Tensor::from_array((entry.dims.clone(), data.clone()))?.into_dyn()
        }
        OnnxCacheData::I64(data) => {
            ort::value::Tensor::from_array((entry.dims.clone(), data.clone()))?.into_dyn()
        }
        OnnxCacheData::I32(data) => {
            ort::value::Tensor::from_array((entry.dims.clone(), data.clone()))?.into_dyn()
        }
        OnnxCacheData::Bool(data) => {
            ort::value::Tensor::from_array((entry.dims.clone(), data.clone()))?.into_dyn()
        }
    };
    Ok(value)
}

#[cfg(feature = "onnx-runtime")]
fn extract_last_token_logits(shape: &[i64], data: &[f32]) -> Result<Vec<f32>> {
    match shape {
        [1, seq_len, vocab] => {
            let seq_len = *seq_len as usize;
            let vocab = *vocab as usize;
            if seq_len == 0 {
                return Err(anyhow!("qwen3 onnx logits sequence is empty"));
            }
            let start = (seq_len - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }
        [seq_len, vocab] => {
            let seq_len = *seq_len as usize;
            let vocab = *vocab as usize;
            if seq_len == 0 {
                return Err(anyhow!("qwen3 onnx logits sequence is empty"));
            }
            let start = (seq_len - 1) * vocab;
            Ok(data[start..start + vocab].to_vec())
        }
        [vocab] => Ok(data[..*vocab as usize].to_vec()),
        _ => Err(anyhow!("unexpected qwen3 onnx logits shape: {:?}", shape)),
    }
}

#[cfg(feature = "onnx-runtime")]
fn extract_last_token_logits_f16(shape: &[i64], data: &[f16]) -> Result<Vec<f32>> {
    match shape {
        [1, seq_len, vocab] => {
            let seq_len = *seq_len as usize;
            let vocab = *vocab as usize;
            if seq_len == 0 {
                return Err(anyhow!("qwen3 onnx logits sequence is empty"));
            }
            let start = (seq_len - 1) * vocab;
            Ok(data[start..start + vocab]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>())
        }
        [seq_len, vocab] => {
            let seq_len = *seq_len as usize;
            let vocab = *vocab as usize;
            if seq_len == 0 {
                return Err(anyhow!("qwen3 onnx logits sequence is empty"));
            }
            let start = (seq_len - 1) * vocab;
            Ok(data[start..start + vocab]
                .iter()
                .map(|value| value.to_f32())
                .collect::<Vec<_>>())
        }
        [vocab] => Ok(data[..*vocab as usize]
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>()),
        _ => Err(anyhow!("unexpected qwen3 onnx logits shape: {:?}", shape)),
    }
}
