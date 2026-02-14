use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_transformers::models::deepseek2::SplitOp;

use crate::utils::tensor_utils::{index_select_2d, split_tensor};

pub fn compute_default_rope_parameters(dim: usize, base: f32) -> Vec<f32> {
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|i| 1.0_f32 / base.powf(i as f32 / dim as f32))
        .collect();
    inv_freq
}

pub fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let half_dim = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half_dim)?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
    let x2 = x2.affine(-1.0, 0.0)?;
    let rotate_x = Tensor::cat(&[&x2, &x1], D::Minus1)?.contiguous()?;
    Ok(rotate_x)
}

pub fn apply_multimodel_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: Vec<usize>,
) -> Result<(Tensor, Tensor)> {
    let mrope_section = mrope_section.repeat(2);
    let cos_select: Vec<Tensor> = cos
        .split(&mrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m)| m.i(i % 3).unwrap())
        .collect();
    let cos = Tensor::cat(&cos_select, D::Minus1)?
        .unsqueeze(1)?
        .contiguous()?;
    let sin_select: Vec<Tensor> = sin
        .split(&mrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m)| m.i(i % 3).unwrap())
        .collect();
    let sin = Tensor::cat(&sin_select, D::Minus1)?
        .unsqueeze(1)?
        .contiguous()?;
    let q_embed = q
        .broadcast_mul(&cos)?
        .add(&rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .add(&rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

pub fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // q, k -> (seq_len, num_heads, head_dim)
    // cos, sin -> (seq_len, head_dim) -> (seq_len, 1, head_dim)
    let cos = cos.unsqueeze(D::Minus2)?;
    let sin = sin.unsqueeze(D::Minus2)?;
    let cos = cos.to_dtype(q.dtype())?;
    let sin = sin.to_dtype(q.dtype())?;
    let q_embed = q
        .broadcast_mul(&cos)?
        .add(&rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .add(&rotate_half(k)?.broadcast_mul(&sin)?)?;
    Ok((q_embed, k_embed))
}

pub fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    tof32: bool,
) -> Result<(Tensor, Tensor)> {
    // sin/cos: to (bs, 1, seq_len, head_dim)
    // q/k: (bs, n_head, seq_len, head_dim)
    let mut cos = cos.clone();
    let mut sin = sin.clone();
    if cos.rank() == 2 {
        // (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    }
    if cos.rank() == 3 {
        // (bs, seq_len, head_dim) -> (bs, 1, seq_len, head_dim)
        cos = cos.unsqueeze(1)?;
        sin = sin.unsqueeze(1)?;
    }
    let orig_dtype = q.dtype();
    let q = if tof32 { &q.to_dtype(DType::F32)? } else { q };
    let k = if tof32 { &k.to_dtype(DType::F32)? } else { k };
    let cos = cos.to_dtype(q.dtype())?;
    let sin = sin.to_dtype(q.dtype())?;

    let q_embed = q
        .broadcast_mul(&cos)?
        .add(&rotate_half(q)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .add(&rotate_half(k)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    Ok((q_embed, k_embed))
}

pub fn glm_asr_apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    tof32: bool,
) -> Result<(Tensor, Tensor)> {
    // sin/cos: to (bs, 1, seq_len, head_dim/2)
    // q/k: (bs, n_head, seq_len, head_dim)
    let mut cos = cos.clone();
    let mut sin = sin.clone();
    if cos.rank() == 2 {
        // (seq_len, head_dim/2) -> (1, 1, seq_len, head_dim/2)
        cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    }
    if cos.rank() == 3 {
        // (bs, seq_len, head_dim/2) -> (bs, 1, seq_len, head_dim/2)
        cos = cos.unsqueeze(1)?;
        sin = sin.unsqueeze(1)?;
    }
    let orig_dtype = q.dtype();
    let q = if tof32 { &q.to_dtype(DType::F32)? } else { q };
    let k = if tof32 { &k.to_dtype(DType::F32)? } else { k };
    let cos = cos.to_dtype(q.dtype())?;
    let sin = sin.to_dtype(q.dtype())?;
    let rotary_dim = cos.dim(D::Minus1)?;
    let q_rot = q.narrow(D::Minus1, 0, rotary_dim)?;
    let q_pass = q.narrow(D::Minus1, rotary_dim, rotary_dim)?;
    let k_rot = k.narrow(D::Minus1, 0, rotary_dim)?;
    let k_pass = k.narrow(D::Minus1, rotary_dim, rotary_dim)?;

    let q_embed = q_rot
        .broadcast_mul(&cos)?
        .add(&rotate_half(&q_rot)?.broadcast_mul(&sin)?)?;
    let k_embed = k_rot
        .broadcast_mul(&cos)?
        .add(&rotate_half(&k_rot)?.broadcast_mul(&sin)?)?;
    let q_embed = Tensor::cat(&[q_embed, q_pass], D::Minus1)?.to_dtype(orig_dtype)?;
    let k_embed = Tensor::cat(&[k_embed, k_pass], D::Minus1)?.to_dtype(orig_dtype)?;
    Ok((q_embed, k_embed))
}

pub fn roformer_rotate(x: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let last_dim = dims
        .last()
        .ok_or(anyhow!("Input tensor must have at least one dimension"))?;
    if last_dim % 2 != 0 {
        return Err(anyhow!(
            "Last dimension size must be even, got {}",
            last_dim
        ));
    }
    let new_dims: Vec<usize> = dims[..dims.len() - 1]
        .iter()
        .copied()
        .chain([last_dim / 2, 2])
        .collect();
    let x_reshape = x.reshape(new_dims)?;
    let x_chunks = x_reshape.chunk(2, D::Minus1)?;
    let x1 = &x_chunks[0];
    let x2 = &x_chunks[1];
    // let x1 = x_reshape.narrow(D::Minus1, 0, 1)?;
    // let x2 = x_reshape.narrow(D::Minus1, 1, 1)?;
    let x2_neg = x2.affine(-1.0, 0.0)?;
    let rotate_x = Tensor::cat(&[&x2_neg, x1], D::Minus1)?;
    Ok(rotate_x.flatten(D::Minus2, D::Minus1)?)
}

pub fn apply_rotary_pos_emb_roformer(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    tof32: bool,
) -> Result<(Tensor, Tensor)> {
    let mut cos = cos.clone();
    let mut sin = sin.clone();
    if cos.rank() == 2 {
        // (seq_len, head_dim) -> (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    }
    if cos.rank() == 3 {
        // (bs, seq_len, head_dim) -> (bs, 1, seq_len, head_dim)
        cos = cos.unsqueeze(1)?;
        sin = sin.unsqueeze(1)?;
    }
    let orig_dtype = q.dtype();
    let q = if tof32 { &q.to_dtype(DType::F32)? } else { q };
    let k = if tof32 { &k.to_dtype(DType::F32)? } else { k };
    let cos = cos.to_dtype(q.dtype())?;
    let sin = sin.to_dtype(q.dtype())?;
    let q_embed = q
        .broadcast_mul(&cos)?
        .add(&roformer_rotate(q)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .add(&roformer_rotate(k)?.broadcast_mul(&sin)?)?
        .to_dtype(orig_dtype)?;
    Ok((q_embed, k_embed))
}

#[derive(Debug, Clone)]
pub struct Qwen2_5VLTextRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl Qwen2_5VLTextRotaryEmbedding {
    pub fn new(dim: usize, theta_base: f32) -> Self {
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        Self { inv_freq }
    }
    pub fn forward(
        &self,
        position_ids: &Tensor,
        dtype: DType,
        mrope_section: Vec<usize>,
    ) -> Result<(Tensor, Tensor)> {
        // position_ids shape: (3, bs, position) -> (3, bs, 1, position)
        let position_ids_expanded = position_ids
            .unsqueeze(D::Minus2)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        // inv_freq Vec<f32> -> Tensor(1, 1, head_dim / 2, 1) -> (3, bs, head_dim / 2, 1)
        let inv_freq_expanded = Tensor::from_vec(
            self.inv_freq.clone(),
            (1, 1, self.inv_freq.len(), 1),
            position_ids.device(),
        )?
        .broadcast_as((3, position_ids.dim(1)?, self.inv_freq.len(), 1))?
        .to_dtype(DType::F32)?
        .contiguous()?;

        // (3, bs, head_dim / 2, 1) matmul (3, bs, 1, position)
        //    -> (3, bs, head_dim / 2, seq_len) -> (3, bs, seq_len, head_dim / 2)
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded)?
            .transpose(2, 3)?;
        // let freqs = position_ids_expanded.matmul(&inv_freq_expanded)?;
        // (3, bs, seq_len, head_dim / 2) -> (3, bs, seq_len, head_dim)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        let mrope_section = mrope_section.repeat(2);
        let cos_select: Vec<Tensor> = cos
            .split(&mrope_section, D::Minus1)?
            .iter()
            .enumerate()
            .map(|(i, m)| m.i(i % 3).unwrap())
            .collect();
        // (bs, seq_len, head_dim) -> (bs, 1, seq_len, head_dim)
        let cos = Tensor::cat(&cos_select, D::Minus1)?
            .unsqueeze(1)?
            .contiguous()?;
        let sin_select: Vec<Tensor> = sin
            .split(&mrope_section, D::Minus1)?
            .iter()
            .enumerate()
            .map(|(i, m)| m.i(i % 3).unwrap())
            .collect();
        // (bs, seq_len, head_dim) -> (bs, 1, seq_len, head_dim)
        let sin = Tensor::cat(&sin_select, D::Minus1)?
            .unsqueeze(1)?
            .contiguous()?;
        Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
    }
}

#[derive(Debug, Clone)]
pub struct Qwen2_5VisionRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl Qwen2_5VisionRotaryEmbedding {
    pub fn new(dim: usize, theta_base: Option<f32>) -> Self {
        let theta_base = theta_base.unwrap_or(10000.0_f32);
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        Self { inv_freq }
    }

    pub fn forward(&self, seqlen: usize, device: &Device) -> Result<Tensor> {
        let seq = Tensor::arange(0.0_f32, seqlen as f32, device)?.reshape((seqlen, 1))?;
        let inv_freq = Tensor::from_vec(self.inv_freq.clone(), (1, self.inv_freq.len()), device)?;
        let freqs = seq.matmul(&inv_freq)?;
        Ok(freqs)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VLTextRotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl Qwen3VLTextRotaryEmbedding {
    pub fn new(dim: usize, theta_base: f32) -> Self {
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        Self { inv_freq }
    }

    pub fn apply_interleaved_mrope(
        &self,
        freqs: &Tensor,
        mrope_section: Vec<usize>,
    ) -> Result<Tensor> {
        let mut freqs_t = freqs.i(0)?.contiguous()?; //(3, bs, seq_len, head_dim //2) -> (bs, seq_len, head_dim //2)

        // for dim in 1..3 {
        for (dim, section) in mrope_section.iter().enumerate().skip(1) {
            // let length = mrope_section[dim] * 3;
            let length = section * 3;
            let idx = Tensor::arange_step(dim as u32, length as u32, 3, freqs.device())?;
            let src = freqs.i(dim)?.contiguous()?; // (bs, seq_len, head_dim //2)
            let src = src.index_select(&idx, D::Minus1)?.contiguous()?;
            let idx = idx
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as(src.shape())?
                .contiguous()?;
            freqs_t = freqs_t.scatter(&idx, &src, D::Minus1)?;
        }
        Ok(freqs_t)
    }

    pub fn apply_interleaved_mrope_asr(
        &self,
        freqs: &Tensor,
        mrope_section: Vec<usize>,
    ) -> Result<Tensor> {
        let mut freqs_t = freqs.i(0)?.contiguous()?; //(3, bs, seq_len, head_dim //2) -> (bs, seq_len, head_dim //2)

        // for dim in 1..3 {
        for (dim, offset) in (1..3).enumerate() {
            let dim = dim + 1;
            let length = mrope_section[dim];
            let idx = Tensor::arange_step(offset as u32, length as u32, 3, freqs.device())?;
            let src = freqs.i(dim)?.contiguous()?; // (bs, seq_len, head_dim //2)
            let src = src.index_select(&idx, D::Minus1)?.contiguous()?;
            let idx = idx
                .unsqueeze(0)?
                .unsqueeze(0)?
                .broadcast_as(src.shape())?
                .contiguous()?;
            freqs_t = freqs_t.scatter(&idx, &src, D::Minus1)?;
        }
        Ok(freqs_t)
    }

    pub fn forward_asr(
        &self,
        position_ids: &Tensor,
        dtype: DType,
        mrope_section: Vec<usize>,
    ) -> Result<(Tensor, Tensor)> {
        // position_ids shape: (3, bs, position) -> (3, bs, 1, position)
        let position_ids = if position_ids.rank() == 2 {
            let (bs, len) = position_ids.dims2()?;
            position_ids.unsqueeze(0)?.expand((3, bs, len))?
        } else {
            position_ids.clone()
        };
        let position_ids_expanded = position_ids
            .unsqueeze(D::Minus2)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        // inv_freq Vec<f32> -> Tensor(1, 1, head_dim / 2, 1) -> (3, bs, head_dim / 2, 1)
        let inv_freq_expanded = Tensor::from_vec(
            self.inv_freq.clone(),
            (1, 1, self.inv_freq.len(), 1),
            position_ids.device(),
        )?
        .broadcast_as((3, position_ids.dim(1)?, self.inv_freq.len(), 1))?
        .to_dtype(DType::F32)?
        .contiguous()?;

        // (3, bs, head_dim / 2, 1) matmul (3, bs, 1, position)
        //    -> (3, bs, head_dim / 2, seq_len) -> (3, bs, seq_len, head_dim / 2)
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded)?
            .transpose(2, 3)?;
        let freqs = self.apply_interleaved_mrope_asr(&freqs, mrope_section)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
    }

    pub fn forward(
        &self,
        position_ids: &Tensor,
        dtype: DType,
        mrope_section: Vec<usize>,
    ) -> Result<(Tensor, Tensor)> {
        // position_ids shape: (3, bs, position) -> (3, bs, 1, position)
        let position_ids = if position_ids.rank() == 2 {
            let (bs, len) = position_ids.dims2()?;
            position_ids.unsqueeze(0)?.expand((3, bs, len))?
        } else {
            position_ids.clone()
        };
        let position_ids_expanded = position_ids
            .unsqueeze(D::Minus2)?
            .to_dtype(DType::F32)?
            .contiguous()?;
        // inv_freq Vec<f32> -> Tensor(1, 1, head_dim / 2, 1) -> (3, bs, head_dim / 2, 1)
        let inv_freq_expanded = Tensor::from_vec(
            self.inv_freq.clone(),
            (1, 1, self.inv_freq.len(), 1),
            position_ids.device(),
        )?
        .broadcast_as((3, position_ids.dim(1)?, self.inv_freq.len(), 1))?
        .to_dtype(DType::F32)?
        .contiguous()?;

        // (3, bs, head_dim / 2, 1) matmul (3, bs, 1, position)
        //    -> (3, bs, head_dim / 2, seq_len) -> (3, bs, seq_len, head_dim / 2)
        let freqs = inv_freq_expanded
            .matmul(&position_ids_expanded)?
            .transpose(2, 3)?;
        let freqs = self.apply_interleaved_mrope(&freqs, mrope_section)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos.to_dtype(dtype)?, sin.to_dtype(dtype)?))
    }
}

pub struct RoPE {
    inv_freq: Tensor, // (1, dim / 2)
}

impl RoPE {
    pub fn new(dim: usize, theta_base: f32, device: &Device) -> Result<Self> {
        let inv_freq = compute_default_rope_parameters(dim, theta_base);
        let inv_freq = Tensor::from_slice(&inv_freq, (1, inv_freq.len()), device)?;

        Ok(Self { inv_freq })
    }
    pub fn forward(
        &self,
        seqlen_offset: usize,
        seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let positions = Tensor::arange(
            seqlen_offset as f32,
            (seqlen_offset + seq_len) as f32,
            device,
        )?
        .reshape((seq_len, 1))?; // (seq_len, 1)
        let freqs = positions.matmul(&self.inv_freq)?; // (seq_len, dim / 2)
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?.contiguous()?; // (seq_len, dim)
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok((cos, sin))
    }
}

pub fn get_xd_cos_sin(
    cos: &Tensor,
    sin: &Tensor,
    position_ids: &Tensor,
    xdrope_section: Vec<usize>,
) -> Result<(Tensor, Tensor)> {
    let x_dim = xdrope_section.len();
    // position_ids: (bs, 4, seq_len)
    let mut cos_vec = vec![];
    let mut sin_vec = vec![];
    let bs = position_ids.dim(0)?;
    for i in 0..bs {
        let pos_i = position_ids.i(i)?;
        let cos_i = index_select_2d(cos, &pos_i)?;
        let sin_i = index_select_2d(sin, &pos_i)?;
        cos_vec.push(cos_i);
        sin_vec.push(sin_i);
    }
    // (bs, 4, seq_len, dim) -> (bs, seq_len, 4, dim)
    let cos = Tensor::stack(&cos_vec, 0)?
        .permute((0, 2, 1, 3))?
        .contiguous()?;
    let sin = Tensor::stack(&sin_vec, 0)?
        .permute((0, 2, 1, 3))?
        .contiguous()?;
    let xdrope_section: Vec<usize> = xdrope_section.iter().map(|&i| i * 2).collect();
    let cos_select: Vec<Tensor> = split_tensor(&cos, &xdrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m)| m.i((.., .., i % x_dim)).unwrap())
        .collect();
    let sin_select: Vec<Tensor> = split_tensor(&sin, &xdrope_section, D::Minus1)?
        .iter()
        .enumerate()
        .map(|(i, m)| m.i((.., .., i % x_dim)).unwrap())
        .collect();

    let cos = Tensor::cat(&cos_select, D::Minus1)?;
    let sin = Tensor::cat(&sin_select, D::Minus1)?;
    Ok((cos, sin))
}
