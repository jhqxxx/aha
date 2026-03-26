use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::BertModel;
#[cfg(feature = "onnx-runtime")]
use half::f16;
#[cfg(feature = "onnx-runtime")]
use ndarray::{Array, IxDyn};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::{
    models::{
        all_minilm_l6_v2::config::{AllMiniLML6V2Config, AllMiniLML6V2PoolingStrategy},
        common::{
            gguf::load_gguf_file,
            onnx::{create_session, resolve_tokenizer_dir},
            retrieval::l2_normalize,
        },
    },
    tokenizer::TokenizerModel,
    utils::{find_type_files, get_device, get_dtype},
};

pub enum AllMiniLML6V2Backend {
    Safetensors(AllMiniLML6V2SafetensorsBackend),
    Gguf(AllMiniLML6V2GgufBackend),
    Onnx(AllMiniLML6V2OnnxBackend),
}

impl AllMiniLML6V2Backend {
    pub fn load(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        Ok(Self::Safetensors(AllMiniLML6V2SafetensorsBackend::load(
            path, device, dtype,
        )?))
    }

    pub fn load_onnx(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        Ok(Self::Onnx(AllMiniLML6V2OnnxBackend::load(
            onnx_path,
            tokenizer_dir,
        )?))
    }

    pub fn load_gguf(
        gguf_path: &str,
        tokenizer_dir: Option<&str>,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        Ok(Self::Gguf(AllMiniLML6V2GgufBackend::load(
            gguf_path,
            tokenizer_dir,
            device,
            dtype,
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

pub struct AllMiniLML6V2SafetensorsBackend {
    tokenizer: TokenizerModel,
    model: BertModel,
    device: Device,
    pooling: AllMiniLML6V2PoolingStrategy,
    normalize: bool,
    max_seq_length: usize,
    do_lower_case: bool,
}

impl AllMiniLML6V2SafetensorsBackend {
    pub fn load(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let tokenizer = TokenizerModel::init(path)?;
        let cfg = AllMiniLML6V2Config::load(path)?;
        let device = get_device(device);
        let dtype = get_dtype(dtype, "float32");
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = BertModel::load(vb, &cfg.base)?;
        Ok(Self {
            tokenizer,
            model,
            device,
            pooling: cfg.pooling,
            normalize: cfg.normalize,
            max_seq_length: cfg.max_seq_length,
            do_lower_case: cfg.do_lower_case,
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
        let token_ids = prepare_token_ids(
            &self.tokenizer,
            text,
            self.max_seq_length,
            self.do_lower_case,
        )?;
        let seq_len = token_ids.len();
        let input_ids = Tensor::from_slice(&token_ids, (1, seq_len), &self.device)?;
        let token_type_ids = Tensor::zeros((1, seq_len), DType::U32, &self.device)?;
        let attention_mask = Tensor::ones((1, seq_len), DType::U32, &self.device)?;
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let hidden_vec = hidden.to_vec2::<f32>()?;
        let mut pooled = pool_hidden_state(&hidden_vec, self.pooling)?;
        if self.normalize {
            l2_normalize(&mut pooled);
        }
        Ok(pooled)
    }
}

pub struct AllMiniLML6V2GgufBackend {
    tokenizer: TokenizerModel,
    model: BertModel,
    device: Device,
    pooling: AllMiniLML6V2PoolingStrategy,
    normalize: bool,
    max_seq_length: usize,
    do_lower_case: bool,
}

impl AllMiniLML6V2GgufBackend {
    pub fn load(
        gguf_path: &str,
        tokenizer_dir: Option<&str>,
        device: Option<&Device>,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let tokenizer_dir = resolve_all_minilm_tokenizer_dir(gguf_path, tokenizer_dir)?;
        let tokenizer = TokenizerModel::init(&tokenizer_dir.to_string_lossy())?;
        let cfg = AllMiniLML6V2Config::load(&tokenizer_dir.to_string_lossy())?;
        let device = get_device(device);
        let dtype = dtype.unwrap_or(DType::F32);
        let mut gguf = load_gguf_file(resolve_all_minilm_gguf_file(gguf_path)?.as_ref(), &device)?;
        let tensors = load_all_minilm_gguf_tensors(&mut gguf, &cfg, &device, dtype)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        let model = BertModel::load(vb, &cfg.base)?;
        Ok(Self {
            tokenizer,
            model,
            device,
            pooling: cfg.pooling,
            normalize: cfg.normalize,
            max_seq_length: cfg.max_seq_length,
            do_lower_case: cfg.do_lower_case,
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
        let token_ids = prepare_token_ids(
            &self.tokenizer,
            text,
            self.max_seq_length,
            self.do_lower_case,
        )?;
        let seq_len = token_ids.len();
        let input_ids = Tensor::from_slice(&token_ids, (1, seq_len), &self.device)?;
        let token_type_ids = Tensor::zeros((1, seq_len), DType::U32, &self.device)?;
        let attention_mask = Tensor::ones((1, seq_len), DType::U32, &self.device)?;
        let hidden = self
            .model
            .forward(&input_ids, &token_type_ids, Some(&attention_mask))?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let hidden_vec = hidden.to_vec2::<f32>()?;
        let mut pooled = pool_hidden_state(&hidden_vec, self.pooling)?;
        if self.normalize {
            l2_normalize(&mut pooled);
        }
        Ok(pooled)
    }
}

#[cfg_attr(not(feature = "onnx-runtime"), allow(dead_code))]
pub struct AllMiniLML6V2OnnxBackend {
    tokenizer: TokenizerModel,
    #[cfg(feature = "onnx-runtime")]
    session: ort::session::Session,
    #[cfg(not(feature = "onnx-runtime"))]
    _session: (),
    output_names: Vec<String>,
    input_descriptors: Vec<OnnxInputDescriptor>,
    pooling: AllMiniLML6V2PoolingStrategy,
    normalize: bool,
    max_seq_length: usize,
    do_lower_case: bool,
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

impl AllMiniLML6V2OnnxBackend {
    pub fn load(onnx_path: &str, tokenizer_dir: Option<&str>) -> Result<Self> {
        let tokenizer_dir =
            resolve_tokenizer_dir(onnx_path, tokenizer_dir, &["tokenizer.json", "config.json"])?;
        let tokenizer = TokenizerModel::init(&tokenizer_dir.to_string_lossy())?;
        let cfg = AllMiniLML6V2Config::load(&tokenizer_dir.to_string_lossy())?;
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
            Ok(Self {
                tokenizer,
                session: bundle.session,
                output_names: bundle.output_names,
                input_descriptors,
                pooling: cfg.pooling,
                normalize: cfg.normalize,
                max_seq_length: cfg.max_seq_length,
                do_lower_case: cfg.do_lower_case,
            })
        }
        #[cfg(not(feature = "onnx-runtime"))]
        {
            let _ = bundle;
            Ok(Self {
                tokenizer,
                _session: (),
                output_names: Vec::new(),
                input_descriptors: Vec::new(),
                pooling: cfg.pooling,
                normalize: cfg.normalize,
                max_seq_length: cfg.max_seq_length,
                do_lower_case: cfg.do_lower_case,
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
        let token_ids = prepare_token_ids(
            &self.tokenizer,
            text,
            self.max_seq_length,
            self.do_lower_case,
        )?;
        let seq_len = token_ids.len();
        let input_ids = token_ids.iter().map(|id| *id as i64).collect::<Vec<_>>();
        let attention_mask = vec![1_i64; seq_len];
        let token_type_ids = vec![0_i64; seq_len];

        let mut inputs = Vec::with_capacity(self.input_descriptors.len());
        for desc in &self.input_descriptors {
            let value =
                self.build_input_value(desc, &input_ids, &attention_mask, &token_type_ids)?;
            inputs.push((desc.name.clone(), value));
        }

        let outputs = self.session.run(inputs).map_err(|err| {
            let names = self
                .input_descriptors
                .iter()
                .map(|desc| desc.name.clone())
                .collect::<Vec<_>>()
                .join(", ");
            anyhow!("failed to run all-minilm-l6-v2 onnx session: {err}; inputs={names}")
        })?;

        let output_value = outputs
            .get("sentence_embedding")
            .or_else(|| outputs.get("last_hidden_state"))
            .or_else(|| outputs.get("token_embeddings"))
            .or_else(|| self.output_names.first().and_then(|name| outputs.get(name)))
            .ok_or_else(|| anyhow!("onnx output for all-minilm-l6-v2 not found"))?;

        let mut embedding = extract_embedding_output(output_value, seq_len, self.pooling)?;
        if self.normalize {
            l2_normalize(&mut embedding);
        }
        Ok(embedding)
    }

    #[cfg(feature = "onnx-runtime")]
    fn build_input_value(
        &self,
        desc: &OnnxInputDescriptor,
        input_ids: &[i64],
        attention_mask: &[i64],
        token_type_ids: &[i64],
    ) -> Result<ort::value::DynValue> {
        let shape = self.resolve_shape(desc, input_ids.len());
        match desc.name.as_str() {
            "input_ids" => return build_i64_like_input(desc, shape, input_ids),
            "attention_mask" => return build_i64_like_input(desc, shape, attention_mask),
            "token_type_ids" => return build_i64_like_input(desc, shape, token_type_ids),
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
                    *dim
                } else if idx == 0 {
                    1
                } else {
                    seq_len as i64
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
                "unsupported all-minilm-l6-v2 onnx input dtype for {}",
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
            let dims = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
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

fn prepare_token_ids(
    tokenizer: &TokenizerModel,
    text: &str,
    max_seq_length: usize,
    do_lower_case: bool,
) -> Result<Vec<u32>> {
    let text = if do_lower_case {
        text.to_lowercase()
    } else {
        text.to_string()
    };
    let mut token_ids = tokenizer.text_encode_vec(text, true)?;
    if token_ids.len() > max_seq_length {
        token_ids.truncate(max_seq_length);
        if let Some(sep_id) = tokenizer.tokenizer.token_to_id("[SEP]")
            && let Some(last) = token_ids.last_mut()
        {
            *last = sep_id;
        }
    }
    if token_ids.is_empty() {
        return Err(anyhow!("embedding tokenized input cannot be empty"));
    }
    Ok(token_ids)
}

fn resolve_all_minilm_gguf_file(path: &str) -> Result<String> {
    let model_path = Path::new(path);
    if !model_path.exists() {
        return Err(anyhow!("gguf model path not found: {}", path));
    }
    if model_path.is_file() {
        if model_path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return Ok(model_path.to_string_lossy().to_string());
        }
        return Err(anyhow!(
            "gguf model path does not point to a .gguf file: {}",
            path
        ));
    }

    let mut matches = std::fs::read_dir(model_path)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|candidate| {
            candidate.is_file()
                && candidate
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches
        .into_iter()
        .next()
        .map(|path| path.to_string_lossy().to_string())
        .ok_or_else(|| anyhow!("no .gguf file found in {}", model_path.display()))
}

fn resolve_all_minilm_tokenizer_dir(
    gguf_path: &str,
    tokenizer_dir: Option<&str>,
) -> Result<PathBuf> {
    fn has_required_files(path: &Path) -> bool {
        path.join("tokenizer.json").exists() && path.join("config.json").exists()
    }

    fn without_suffix(path: &Path, suffix: &str) -> Option<PathBuf> {
        let name = path.file_name()?.to_string_lossy();
        let base = name.strip_suffix(suffix)?;
        if base.is_empty() {
            None
        } else {
            Some(path.with_file_name(base))
        }
    }

    let mut candidates = Vec::new();
    if let Some(dir) = tokenizer_dir {
        candidates.push(PathBuf::from(dir));
    }

    let gguf_file = PathBuf::from(resolve_all_minilm_gguf_file(gguf_path)?);
    if let Some(parent) = gguf_file.parent() {
        candidates.push(parent.to_path_buf());
        if let Some(grand) = parent.parent() {
            candidates.push(grand.to_path_buf());
        }
        if let Some(sibling) = without_suffix(parent, "-Embedding-GGUF") {
            candidates.push(sibling);
        }
        if let Some(sibling) = without_suffix(parent, "-GGUF") {
            candidates.push(sibling);
        }
        if let Some(sibling) = without_suffix(parent, "-gguf") {
            candidates.push(sibling);
        }
    }

    let mut unique = Vec::new();
    for candidate in candidates {
        if !unique
            .iter()
            .any(|existing: &PathBuf| existing == &candidate)
        {
            unique.push(candidate);
        }
    }

    for candidate in unique {
        if has_required_files(&candidate) {
            return Ok(candidate);
        }
    }

    Err(anyhow!(
        "unable to infer tokenizer directory for gguf artifact {}; provide --tokenizer-dir",
        gguf_file.display()
    ))
}

fn load_all_minilm_gguf_tensors<R: std::io::Read + std::io::Seek>(
    gguf: &mut crate::models::common::gguf::Gguf<R>,
    cfg: &AllMiniLML6V2Config,
    device: &Device,
    dtype: DType,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::new();

    insert_gguf_tensor(
        gguf,
        &mut tensors,
        "token_embd.weight",
        "embeddings.word_embeddings.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        gguf,
        &mut tensors,
        "token_types.weight",
        "embeddings.token_type_embeddings.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        gguf,
        &mut tensors,
        "position_embd.weight",
        "embeddings.position_embeddings.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        gguf,
        &mut tensors,
        "token_embd_norm.weight",
        "embeddings.LayerNorm.weight",
        device,
        dtype,
    )?;
    insert_gguf_tensor(
        gguf,
        &mut tensors,
        "token_embd_norm.bias",
        "embeddings.LayerNorm.bias",
        device,
        dtype,
    )?;

    for layer_idx in 0..cfg.base.num_hidden_layers {
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_q.weight"),
            &format!("encoder.layer.{layer_idx}.attention.self.query.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_q.bias"),
            &format!("encoder.layer.{layer_idx}.attention.self.query.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_k.weight"),
            &format!("encoder.layer.{layer_idx}.attention.self.key.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_k.bias"),
            &format!("encoder.layer.{layer_idx}.attention.self.key.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_v.weight"),
            &format!("encoder.layer.{layer_idx}.attention.self.value.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_v.bias"),
            &format!("encoder.layer.{layer_idx}.attention.self.value.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_output.weight"),
            &format!("encoder.layer.{layer_idx}.attention.output.dense.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_output.bias"),
            &format!("encoder.layer.{layer_idx}.attention.output.dense.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_output_norm.weight"),
            &format!("encoder.layer.{layer_idx}.attention.output.LayerNorm.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.attn_output_norm.bias"),
            &format!("encoder.layer.{layer_idx}.attention.output.LayerNorm.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.ffn_up.weight"),
            &format!("encoder.layer.{layer_idx}.intermediate.dense.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.ffn_up.bias"),
            &format!("encoder.layer.{layer_idx}.intermediate.dense.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.ffn_down.weight"),
            &format!("encoder.layer.{layer_idx}.output.dense.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.ffn_down.bias"),
            &format!("encoder.layer.{layer_idx}.output.dense.bias"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.layer_output_norm.weight"),
            &format!("encoder.layer.{layer_idx}.output.LayerNorm.weight"),
            device,
            dtype,
        )?;
        insert_gguf_tensor(
            gguf,
            &mut tensors,
            &format!("blk.{layer_idx}.layer_output_norm.bias"),
            &format!("encoder.layer.{layer_idx}.output.LayerNorm.bias"),
            device,
            dtype,
        )?;
    }

    Ok(tensors)
}

fn insert_gguf_tensor<R: std::io::Read + std::io::Seek>(
    gguf: &mut crate::models::common::gguf::Gguf<R>,
    tensors: &mut HashMap<String, Tensor>,
    gguf_name: &str,
    bert_name: &str,
    device: &Device,
    dtype: DType,
) -> Result<()> {
    let tensor = gguf
        .get_dequantized(gguf_name)
        .map_err(|err| anyhow!("failed to load gguf tensor {}: {}", gguf_name, err))?
        .to_device(device)
        .map_err(|err| anyhow!("failed to move gguf tensor {}: {}", gguf_name, err))?;
    let tensor = tensor
        .to_dtype(dtype)
        .map_err(|err| anyhow!("failed to convert gguf tensor {}: {}", gguf_name, err))?;
    tensors.insert(bert_name.to_string(), tensor);
    Ok(())
}

fn pool_hidden_state(
    hidden: &[Vec<f32>],
    pooling: AllMiniLML6V2PoolingStrategy,
) -> Result<Vec<f32>> {
    let first = hidden
        .first()
        .ok_or_else(|| anyhow!("embedding hidden state is empty"))?;
    let width = first.len();
    for row in hidden {
        if row.len() != width {
            return Err(anyhow!("inconsistent embedding width in hidden state"));
        }
    }
    match pooling {
        AllMiniLML6V2PoolingStrategy::Cls => Ok(first.clone()),
        AllMiniLML6V2PoolingStrategy::Mean => {
            let mut pooled = vec![0.0f32; width];
            for row in hidden {
                for (idx, value) in row.iter().enumerate() {
                    pooled[idx] += *value;
                }
            }
            let inv = 1.0f32 / hidden.len() as f32;
            for value in &mut pooled {
                *value *= inv;
            }
            Ok(pooled)
        }
        AllMiniLML6V2PoolingStrategy::Max => {
            let mut pooled = first.clone();
            for row in hidden.iter().skip(1) {
                for (idx, value) in row.iter().enumerate() {
                    pooled[idx] = pooled[idx].max(*value);
                }
            }
            Ok(pooled)
        }
        AllMiniLML6V2PoolingStrategy::MeanSqrtLen => {
            let mut pooled = vec![0.0f32; width];
            for row in hidden {
                for (idx, value) in row.iter().enumerate() {
                    pooled[idx] += *value;
                }
            }
            let inv = 1.0f32 / (hidden.len() as f32).sqrt();
            for value in &mut pooled {
                *value *= inv;
            }
            Ok(pooled)
        }
    }
}

#[cfg(feature = "onnx-runtime")]
fn build_i64_like_input(
    desc: &OnnxInputDescriptor,
    shape: Vec<i64>,
    data: &[i64],
) -> Result<ort::value::DynValue> {
    let kind = desc.kind.ok_or_else(|| {
        anyhow!(
            "unsupported all-minilm-l6-v2 onnx input dtype for {}",
            desc.name
        )
    })?;
    match kind {
        OnnxTensorKind::I32 => Ok(ort::value::Tensor::from_array((
            shape,
            data.iter().map(|v| *v as i32).collect::<Vec<_>>(),
        ))?
        .into_dyn()),
        OnnxTensorKind::I64 => {
            Ok(ort::value::Tensor::from_array((shape, data.to_vec()))?.into_dyn())
        }
        OnnxTensorKind::Bool => Ok(ort::value::Tensor::from_array((
            shape,
            data.iter().map(|v| *v != 0).collect::<Vec<_>>(),
        ))?
        .into_dyn()),
        _ => Err(anyhow!(
            "unsupported integer-like onnx input dtype for {}",
            desc.name
        )),
    }
}

#[cfg(feature = "onnx-runtime")]
fn extract_embedding_output(
    value: &ort::value::DynValue,
    seq_len: usize,
    pooling: AllMiniLML6V2PoolingStrategy,
) -> Result<Vec<f32>> {
    if let Ok((shape, values)) = value.try_extract_tensor::<f32>() {
        let shape_vec = shape.iter().copied().collect::<Vec<_>>();
        return extract_embedding_from_shape(&shape_vec, values, seq_len, pooling);
    }
    if let Ok((shape, values)) = value.try_extract_tensor::<f16>() {
        let shape_vec = shape.iter().copied().collect::<Vec<_>>();
        let values = values
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        return extract_embedding_from_shape(&shape_vec, &values, seq_len, pooling);
    }
    Err(anyhow!(
        "all-minilm-l6-v2 onnx output must be a f32/f16 tensor"
    ))
}

#[cfg(feature = "onnx-runtime")]
fn extract_embedding_from_shape(
    shape: &[i64],
    values: &[f32],
    seq_len: usize,
    pooling: AllMiniLML6V2PoolingStrategy,
) -> Result<Vec<f32>> {
    match shape {
        [1, token_count, hidden_size] => {
            let token_count = *token_count as usize;
            let hidden_size = *hidden_size as usize;
            let effective_tokens = token_count.min(seq_len);
            let mut hidden = Vec::with_capacity(effective_tokens);
            for chunk in values.chunks(hidden_size).take(effective_tokens) {
                hidden.push(chunk.to_vec());
            }
            pool_hidden_state(&hidden, pooling)
        }
        [token_count, hidden_size] if *token_count as usize == seq_len => {
            let hidden_size = *hidden_size as usize;
            let mut hidden = Vec::with_capacity(seq_len);
            for chunk in values.chunks(hidden_size).take(seq_len) {
                hidden.push(chunk.to_vec());
            }
            pool_hidden_state(&hidden, pooling)
        }
        [1, hidden_size] => Ok(values.iter().take(*hidden_size as usize).copied().collect()),
        [hidden_size] => Ok(values.iter().take(*hidden_size as usize).copied().collect()),
        _ => Err(anyhow!(
            "unexpected all-minilm-l6-v2 onnx output shape: {:?}",
            shape
        )),
    }
}
