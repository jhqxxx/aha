use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use candle_nn::{Embedding, Module, VarBuilder};
#[cfg(feature = "onnx-runtime")]
use half::f16;
#[cfg(feature = "onnx-runtime")]
use ndarray::{Array, IxDyn};
use std::path::{Path, PathBuf};

use crate::{
    models::{
        common::{
            gguf::{load_gguf_file, load_text_bootstrap_from_gguf},
            onnx::{create_session, resolve_tokenizer_dir},
            retrieval::{l2_normalize, mean_pool},
        },
        qwen3::model::Qwen3Model,
        qwen3_embedding::config::{Qwen3EmbeddingConfig, Qwen3EmbeddingPoolingStrategy},
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub enum Qwen3EmbeddingBackend {
    Safetensors(Qwen3EmbeddingSafetensorsBackend),
    Gguf(Qwen3EmbeddingGgufBackend),
    Onnx(Qwen3EmbeddingOnnxBackend),
}

impl Qwen3EmbeddingBackend {
    pub fn load(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        Ok(Self::Safetensors(Qwen3EmbeddingSafetensorsBackend::load(
            path, device, dtype,
        )?))
    }

    pub fn load_onnx(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        Ok(Self::Onnx(Qwen3EmbeddingOnnxBackend::load(
            onnx_path,
            tokenizer_dir,
        )?))
    }

    pub fn load_gguf(gguf_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        Ok(Self::Gguf(Qwen3EmbeddingGgufBackend::load(
            gguf_path,
            tokenizer_dir,
        )?))
    }

    pub fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            Self::Safetensors(backend) => backend.embed_texts(input),
            Self::Gguf(backend) => backend.embed_texts(input),
            Self::Onnx(backend) => backend.embed_texts(input),
        }
    }
}

pub struct Qwen3EmbeddingSafetensorsBackend {
    tokenizer: TokenizerModel,
    model: Qwen3Model,
    device: Device,
    pooling: Qwen3EmbeddingPoolingStrategy,
    normalize: bool,
}

impl Qwen3EmbeddingSafetensorsBackend {
    pub fn load(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let tokenizer = TokenizerModel::init(path)?;
        let cfg = Qwen3EmbeddingConfig::load(path)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, cfg.base.torch_dtype.as_str());
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = Qwen3Model::new(&cfg.base, vb)?;
        Ok(Self {
            tokenizer,
            model,
            device,
            pooling: cfg.pooling,
            normalize: cfg.normalize,
        })
    }

    pub fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(anyhow!("embedding input cannot be empty"));
        }
        let mut out = Vec::with_capacity(input.len());
        for text in input {
            out.push(self.embed_one(text)?);
            self.model.clear_kv_cache();
        }
        Ok(out)
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let input_ids = self.tokenizer.text_encode(text.to_string(), &self.device)?;
        let hidden = self
            .model
            .forward_hidden(Some(&input_ids), None, 0)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let hidden_vec = hidden.to_vec2::<f32>()?;
        let mut pooled = match self.pooling {
            Qwen3EmbeddingPoolingStrategy::Mean => mean_pool(&hidden_vec)?,
        };
        if self.normalize {
            l2_normalize(&mut pooled);
        }
        Ok(pooled)
    }
}

pub struct Qwen3EmbeddingGgufBackend {
    tokenizer: TokenizerModel,
    token_embedding: Embedding,
    device: Device,
    pooling: Qwen3EmbeddingPoolingStrategy,
    normalize: bool,
}

impl Qwen3EmbeddingGgufBackend {
    pub fn load(gguf_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        if !Path::new(gguf_path).exists() {
            return Err(anyhow!(
                "qwen3 embedding gguf file not found: {}",
                gguf_path
            ));
        }

        let bootstrap =
            load_text_bootstrap_from_gguf(gguf_path, Some(false), Some(false), Some(false))?;
        let device = Device::Cpu;
        let mut gguf = load_gguf_file(gguf_path, &device)?;
        let token_embd = gguf.tensor("token_embd.weight").map_err(|err| {
            anyhow!(
                "qwen3 embedding gguf is missing token_embd.weight in {}: {}",
                gguf_path,
                err
            )
        })?;
        let token_embd = token_embd.dequantize(&device)?.to_dtype(DType::F32)?;
        let hidden_size = token_embd.dim(1)?;
        let token_embedding = Embedding::new(token_embd, hidden_size);
        let (pooling, normalize) = resolve_gguf_embedding_behavior(gguf_path, tokenizer_dir);
        Ok(Self {
            tokenizer: bootstrap.tokenizer,
            token_embedding,
            device,
            pooling,
            normalize,
        })
    }

    pub fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(anyhow!("embedding input cannot be empty"));
        }
        let mut out = Vec::with_capacity(input.len());
        for text in input {
            out.push(self.embed_one(text)?);
        }
        Ok(out)
    }

    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let input_ids = self.tokenizer.text_encode(text.to_string(), &self.device)?;
        let hidden = self
            .token_embedding
            .forward(&input_ids)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let hidden_vec = hidden.to_vec2::<f32>()?;
        let mut pooled = match self.pooling {
            Qwen3EmbeddingPoolingStrategy::Mean => mean_pool(&hidden_vec)?,
        };
        if self.normalize {
            l2_normalize(&mut pooled);
        }
        Ok(pooled)
    }
}

fn resolve_gguf_embedding_behavior(
    gguf_path: &str,
    tokenizer_dir: Option<&str>,
) -> (Qwen3EmbeddingPoolingStrategy, bool) {
    let mut candidates = Vec::new();
    if let Some(path) = tokenizer_dir {
        candidates.push(PathBuf::from(path));
    }
    if let Some(parent) = Path::new(gguf_path).parent() {
        candidates.push(parent.to_path_buf());
        if let Some(parent_name) = parent.file_name().and_then(|name| name.to_str()) {
            for suffix in ["-GGUF", "-gguf"] {
                if let Some(base_name) = parent_name.strip_suffix(suffix) {
                    candidates.push(parent.with_file_name(base_name));
                }
            }
        }
        if let Some(grand) = parent.parent() {
            candidates.push(grand.to_path_buf());
        }
    }
    for candidate in candidates {
        let config_path = candidate.join("config.json");
        if !config_path.exists() {
            continue;
        }
        if let Ok(cfg) = Qwen3EmbeddingConfig::load(&candidate.to_string_lossy()) {
            return (cfg.pooling, cfg.normalize);
        }
    }
    (Qwen3EmbeddingPoolingStrategy::Mean, true)
}

#[cfg_attr(not(feature = "onnx-runtime"), allow(dead_code))]
pub struct Qwen3EmbeddingOnnxBackend {
    tokenizer: TokenizerModel,
    #[cfg(feature = "onnx-runtime")]
    session: ort::session::Session,
    #[cfg(not(feature = "onnx-runtime"))]
    _session: (),
    input_names: Vec<String>,
    output_names: Vec<String>,
    input_descriptors: Vec<OnnxInputDescriptor>,
    has_past_cache_inputs: bool,
    pooling: Qwen3EmbeddingPoolingStrategy,
    normalize: bool,
}

#[cfg_attr(not(feature = "onnx-runtime"), allow(dead_code))]
#[derive(Clone)]
struct OnnxInputDescriptor {
    name: String,
    shape: Vec<i64>,
    kind: Option<OnnxTensorKind>,
}

#[cfg_attr(not(feature = "onnx-runtime"), allow(dead_code))]
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

impl Qwen3EmbeddingOnnxBackend {
    pub fn load(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        let tokenizer_dir =
            resolve_tokenizer_dir(onnx_path, tokenizer_dir, &["tokenizer.json", "config.json"])?;
        let tokenizer = TokenizerModel::init(&tokenizer_dir.to_string_lossy())?;
        let cfg = Qwen3EmbeddingConfig::load(&tokenizer_dir.to_string_lossy())?;
        let bundle = create_session(onnx_path, None)?;
        #[cfg(feature = "onnx-runtime")]
        {
            let has_past_cache_inputs = bundle
                .session
                .inputs()
                .iter()
                .any(|input| input.name().starts_with("past_key_values."));
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

            Ok(Self {
                tokenizer,
                session: bundle.session,
                input_names: bundle.input_names,
                output_names: bundle.output_names,
                input_descriptors,
                has_past_cache_inputs,
                pooling: cfg.pooling,
                normalize: cfg.normalize,
            })
        }
        #[cfg(not(feature = "onnx-runtime"))]
        {
            let _ = bundle;
            Ok(Self {
                tokenizer,
                _session: (),
                input_names: Vec::new(),
                output_names: Vec::new(),
                input_descriptors: Vec::new(),
                has_past_cache_inputs: false,
                pooling: cfg.pooling,
                normalize: cfg.normalize,
            })
        }
    }

    pub fn embed_texts(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        if input.is_empty() {
            return Err(anyhow!("embedding input cannot be empty"));
        }
        let mut out = Vec::with_capacity(input.len());
        for text in input {
            out.push(self.embed_one(text)?);
        }
        Ok(out)
    }

    #[cfg(feature = "onnx-runtime")]
    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let token_ids = self.tokenizer.text_encode_vec(text.to_string(), true)?;
        let seq_len = token_ids.len();
        if seq_len == 0 {
            return Err(anyhow!("embedding tokenized input cannot be empty"));
        }

        let input_ids = token_ids.iter().map(|id| *id as i64).collect::<Vec<_>>();
        let attention_mask = vec![1_i64; seq_len];
        let position_ids = (0..seq_len as i64).collect::<Vec<_>>();

        let mut inputs = Vec::with_capacity(self.input_names.len());
        for desc in &self.input_descriptors {
            let value = self.build_input_value(desc, &input_ids, &attention_mask, &position_ids)?;
            inputs.push((desc.name.clone(), value));
        }

        let outputs = self.session.run(inputs).map_err(|err| {
            let desc = self
                .input_descriptors
                .iter()
                .map(|d| d.name.clone())
                .collect::<Vec<_>>()
                .join(", ");
            anyhow!("failed to run qwen3 embedding onnx session: {err}; inputs={desc}")
        })?;
        let hidden_value = outputs
            .get("last_hidden_state")
            .or_else(|| self.output_names.first().and_then(|name| outputs.get(name)))
            .ok_or_else(|| anyhow!("onnx output last_hidden_state not found"))?;
        let (shape, values) = hidden_value.try_extract_tensor::<f32>()?;
        if shape.len() != 3 || shape[0] != 1 {
            return Err(anyhow!("unexpected onnx hidden state shape: {}", shape));
        }
        let token_count = shape[1] as usize;
        let hidden_size = shape[2] as usize;
        let mut hidden = Vec::with_capacity(token_count);
        for chunk in values.chunks(hidden_size).take(token_count) {
            hidden.push(chunk.to_vec());
        }

        let mut pooled = match self.pooling {
            Qwen3EmbeddingPoolingStrategy::Mean => mean_pool(&hidden)?,
        };
        if self.normalize {
            l2_normalize(&mut pooled);
        }
        Ok(pooled)
    }

    #[cfg(feature = "onnx-runtime")]
    fn build_input_value(
        &self,
        desc: &OnnxInputDescriptor,
        input_ids: &[i64],
        attention_mask: &[i64],
        position_ids: &[i64],
    ) -> Result<ort::value::DynValue> {
        let shape = self.resolve_shape(desc, input_ids.len());
        match desc.name.as_str() {
            "input_ids" => {
                return Ok(ort::value::Tensor::from_array((shape, input_ids.to_vec()))?.into_dyn());
            }
            "attention_mask" => {
                let elem_count = shape.iter().fold(1_usize, |acc, dim| {
                    acc.saturating_mul((*dim).max(1) as usize)
                });
                let mut mask = vec![1_i64; elem_count];
                let keep = attention_mask.len().min(elem_count);
                let offset = elem_count.saturating_sub(keep);
                if self.has_past_cache_inputs && offset > 0 {
                    for value in mask.iter_mut().take(offset) {
                        *value = 0;
                    }
                }
                for (idx, value) in attention_mask.iter().take(keep).enumerate() {
                    mask[offset + idx] = *value;
                }
                return Ok(ort::value::Tensor::from_array((shape, mask))?.into_dyn());
            }
            "position_ids" => {
                let pos = if shape.len() == 3 && shape[0] == 3 {
                    let mut out = Vec::with_capacity(position_ids.len() * 3);
                    for _ in 0..3 {
                        out.extend_from_slice(position_ids);
                    }
                    out
                } else {
                    position_ids.to_vec()
                };
                return Ok(ort::value::Tensor::from_array((shape, pos))?.into_dyn());
            }
            _ => {}
        }

        self.build_zero_tensor(desc, shape)
    }

    #[cfg(feature = "onnx-runtime")]
    fn resolve_shape(&self, desc: &OnnxInputDescriptor, seq_len: usize) -> Vec<i64> {
        if desc.shape.is_empty() {
            return vec![1_i64];
        }
        desc.shape
            .iter()
            .enumerate()
            .map(|(idx, dim)| {
                if *dim >= 0 {
                    return *dim;
                }
                match desc.name.as_str() {
                    "input_ids" | "attention_mask" => {
                        if idx == 0 {
                            1
                        } else {
                            seq_len as i64
                        }
                    }
                    "position_ids" => {
                        if desc.shape.len() == 3 {
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
                    name if name.starts_with("past_key_values.") => {
                        if idx == 2 {
                            0
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
                }
            })
            .collect()
    }

    #[cfg(feature = "onnx-runtime")]
    fn build_zero_tensor(
        &self,
        desc: &OnnxInputDescriptor,
        shape: Vec<i64>,
    ) -> Result<ort::value::DynValue> {
        let kind = desc.kind.ok_or_else(|| {
            anyhow!(
                "unsupported qwen3 embedding onnx input dtype for {}",
                desc.name
            )
        })?;
        let elem_count = shape.iter().try_fold(1_usize, |acc, dim| {
            if *dim < 0 {
                Err(anyhow!(
                    "cannot resolve dynamic onnx shape for input {}: {:?}",
                    desc.name,
                    shape
                ))
            } else {
                Ok(acc.saturating_mul(*dim as usize))
            }
        })?;
        let has_zero_dim = shape.contains(&0);
        let make_ndarray = || {
            let dims = shape.iter().map(|d| *d as usize).collect::<Vec<_>>();
            IxDyn(&dims)
        };
        let value = match kind {
            OnnxTensorKind::Bool => {
                if has_zero_dim {
                    let arr = Array::from_shape_vec(make_ndarray(), vec![false; elem_count])?;
                    ort::value::Tensor::from_array(arr)?.into_dyn()
                } else {
                    ort::value::Tensor::from_array((shape, vec![false; elem_count]))?.into_dyn()
                }
            }
            OnnxTensorKind::I32 => {
                if has_zero_dim {
                    let arr = Array::from_shape_vec(make_ndarray(), vec![0_i32; elem_count])?;
                    ort::value::Tensor::from_array(arr)?.into_dyn()
                } else {
                    ort::value::Tensor::from_array((shape, vec![0_i32; elem_count]))?.into_dyn()
                }
            }
            OnnxTensorKind::I64 => {
                if has_zero_dim {
                    let arr = Array::from_shape_vec(make_ndarray(), vec![0_i64; elem_count])?;
                    ort::value::Tensor::from_array(arr)?.into_dyn()
                } else {
                    ort::value::Tensor::from_array((shape, vec![0_i64; elem_count]))?.into_dyn()
                }
            }
            OnnxTensorKind::F16 => {
                if has_zero_dim {
                    let arr = Array::from_shape_vec(
                        make_ndarray(),
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
                    let arr = Array::from_shape_vec(make_ndarray(), vec![0_f32; elem_count])?;
                    ort::value::Tensor::from_array(arr)?.into_dyn()
                } else {
                    ort::value::Tensor::from_array((shape, vec![0_f32; elem_count]))?.into_dyn()
                }
            }
        };
        Ok(value)
    }

    #[cfg(not(feature = "onnx-runtime"))]
    fn embed_one(&mut self, _text: &str) -> Result<Vec<f32>> {
        Err(anyhow!(
            "onnx runtime support is not enabled; rebuild with --features onnx-runtime"
        ))
    }
}
