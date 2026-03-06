use anyhow::{Result, anyhow};
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{
    Conv1d, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_b, linear_no_bias,
    ops::sigmoid, rms_norm,
};

use crate::{
    models::{
        common::{GateUpDownMLP, conv1d_depthwise, eager_attention_forward, get_conv1d, softplus},
        qwen3_5::config::{Qwen3_5Config, Qwen3_5TextConfig},
        qwen3vl::model::Qwen3VLVisionModel,
    },
    position_embed::rope::{Qwen3VLTextRotaryEmbedding, glm_asr_apply_rotary_pos_emb},
    utils::tensor_utils::{
        get_equal_mask, get_vision_next_indices, l2_normalize, masked_scatter_dim0, nonzero_index,
        prepare_causal_attention_mask, repeat_interleave, split_tensor, zero_index,
    },
};

pub struct Qwen3_5RMSNorm {
    eps: f64,
    weight: Tensor,
}

impl Qwen3_5RMSNorm {
    pub fn new(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let weight = weight.to_dtype(candle_core::DType::F32)?.affine(1.0, 1.0)?;
        Ok(Self { eps, weight })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = xs.to_dtype(candle_core::DType::F32)?;
        let norm_ = x
            .powf(2.0)?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.eps)?
            .sqrt()?;
        let norm = x.broadcast_div(&norm_)?;
        let norm = norm.broadcast_mul(&self.weight)?.to_dtype(xs.dtype())?;
        Ok(norm)
    }
}

pub struct Qwen3_5RMSNormGated {
    norm: RmsNorm,
}

impl Qwen3_5RMSNormGated {
    pub fn new(vb: VarBuilder, hidden_size: usize, eps: f64) -> Result<Self> {
        let norm = rms_norm(hidden_size, eps, vb)?;
        Ok(Self { norm })
    }

    pub fn forward(&self, xs: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.norm.forward(xs)?;
        if let Some(gate) = gate {
            xs = xs.broadcast_mul(&gate.silu()?)?;
        }
        Ok(xs)
    }
}

macro_rules! transmute_tensors {
    ($($tensor:expr),*) => {
        ($(
            $tensor.transpose(1, 2)?.contiguous()?.to_dtype(candle_core::DType::F32)?,
        )*)
    };
}

macro_rules! right_pad_zero_tensor {
    ($dim:expr, $pad_size:expr, $($tensor:expr),+) => {
        ($(
            $tensor.pad_with_zeros($dim, 0, $pad_size)?.contiguous()?,
        )+)
    };
}

macro_rules! reshape_chunk_tensor {
    ($chunk_size:expr, $($tensor:expr),*) => {
        ($(
            {
                let (bs, head, _, dim) = $tensor.dims4()?;
                $tensor.reshape((bs, head, (), $chunk_size, dim))?.contiguous()?
            },
        )*)
    };
}

pub struct Qwen3_5GatedDeltaNet {
    // hidden_size: usize,
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    // layer_idx: usize,
    // activation: Activation,
    // layer_norm_epsilon: f64,
    // QKV 投影
    // conv_dim: usize,
    conv1d: Conv1d,
    dt_bias: Tensor,
    a_log: Tensor,
    norm: Qwen3_5RMSNormGated,
    out_proj: Linear,

    // Z, B, A 投影
    in_proj_qkv: candle_nn::Linear,
    in_proj_z: candle_nn::Linear,
    in_proj_b: candle_nn::Linear,
    in_proj_a: candle_nn::Linear,
    conv_state_cache: Option<Tensor>,
    recurrent_state_cache: Option<Tensor>,
}

impl Qwen3_5GatedDeltaNet {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_v_heads = config.linear_num_value_heads;
        let num_k_heads = config.linear_num_key_heads;
        let head_k_dim = config.linear_key_head_dim;
        let head_v_dim = config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_kernel_size = config.linear_conv_kernel_dim;
        // let activation = config.hidden_act;
        let layer_norm_epsilon = config.rms_norm_eps;
        let conv_dim = key_dim * 2 + value_dim;
        let conv1d = get_conv1d(
            vb.pp("conv1d"),
            conv_dim,
            conv_dim,
            conv_kernel_size,
            conv_kernel_size - 1,
            1,
            1,
            conv_dim,
            false,
        )?;
        let dt_bias = vb.get(num_v_heads, "dt_bias")?;
        let a_log = vb.get(num_v_heads, "A_log")?;
        let norm = Qwen3_5RMSNormGated::new(vb.pp("norm"), head_v_dim, layer_norm_epsilon)?;

        let out_proj = linear_no_bias(value_dim, hidden_size, vb.pp("out_proj"))?;

        let in_proj_qkv = linear_no_bias(hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_a"))?;
        Ok(Self {
            // hidden_size,
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            // activation,
            // layer_norm_epsilon,
            // conv_dim,
            conv1d,
            dt_bias,
            a_log,
            norm,
            out_proj,
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv_state_cache: None,
            recurrent_state_cache: None,
        })
    }

    fn torch_causal_conv1d_update(&mut self, xs: &Tensor) -> Result<Tensor> {
        let conv_state = self.conv_state_cache.as_ref().unwrap();
        let seq_len = xs.dim(2)?;
        let state_len = conv_state.dim(D::Minus1)?;
        let conv_state_new = Tensor::cat(&[conv_state, xs], D::Minus1)?;
        let conv_update = conv_state_new.narrow(D::Minus1, seq_len, state_len)?;
        self.conv_state_cache = Some(conv_update);
        // too slow
        // let out = conv_state_new.conv1d(self.conv1d.weight(), 0, 1, 1, dim)?;
        let out = conv1d_depthwise(&conv_state_new, self.conv1d.weight(), self.conv1d.bias())?;
        let start = out.dim(D::Minus1)? - seq_len;
        let out = out.narrow(D::Minus1, start, seq_len)?.silu()?;
        Ok(out)
    }

    fn torch_chunk_gated_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        use_qk_l2norm_in_kernel: bool,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel {
            (l2_normalize(query, 3)?, l2_normalize(key, 3)?)
        } else {
            (query.clone(), key.clone())
        };
        let initial_dtype = query.dtype();
        let (query, key, value, beta, g) = transmute_tensors!(query, key, value, beta, g);
        let (batch_size, num_heads, sequence_length, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size;
        let (query, key, value, beta, g) =
            right_pad_zero_tensor!(2, pad_size, query, key, value, beta, g);
        let total_sequence_length = sequence_length + pad_size;
        let scale = 1.0 / (query.dim(D::Minus1)? as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        let v_beta = value.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let k_beta = key.broadcast_mul(&beta.unsqueeze(D::Minus1)?.contiguous()?)?;
        let (query, key, k_beta, v_beta) =
            reshape_chunk_tensor!(chunk_size, query, key, k_beta, v_beta);
        let g = g.reshape((g.dim(0)?, g.dim(1)?, (), chunk_size))?;
        let g = g.cumsum(D::Minus1)?;
        let decay_mask = g
            .unsqueeze(D::Minus1)?
            .broadcast_sub(&g.unsqueeze(D::Minus2)?)?
            .exp()?
            .to_dtype(candle_core::DType::F32)?;
        let tril_mask = Tensor::tril2(chunk_size, candle_core::DType::U32, query.device())?
            .broadcast_as(decay_mask.shape())?;
        let on_false = decay_mask.zeros_like()?;
        let decay_mask = tril_mask.where_cond(&decay_mask, &on_false)?.contiguous()?;
        // when rank = 5, matmul error, bs=1,squeeze(0)
        let mut attn = k_beta
            .squeeze(0)?
            .contiguous()?
            .matmul(
                &key.squeeze(0)?
                    .transpose(D::Minus1, D::Minus2)?
                    .contiguous()?,
            )?
            .unsqueeze(0)?
            .mul(&decay_mask)?
            .affine(-1.0, 0.0)?;
        // 包含对角线的上三角矩阵
        let mask = Tensor::triu2(chunk_size, candle_core::DType::U32, query.device())?
            .broadcast_as(decay_mask.shape())?;
        // attn的对角线为0,为1的取0, 为0的取attn
        attn = mask.where_cond(&on_false, &attn)?;
        let (d0, d1, d2, _, _) = attn.dims5()?;
        for i in 1..chunk_size {
            let row = attn.i((.., .., .., i, ..i))?.contiguous()?;
            let sub = attn.i((.., .., .., ..i, ..i))?.contiguous()?;
            let attn_i = row
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&sub)?
                .sum(D::Minus2)?
                .add(&row)?
                .unsqueeze(D::Minus2)?;
            attn = attn.slice_assign(&[(0..d0), (0..d1), (0..d2), (i..i + 1), (0..i)], &attn_i)?;
        }
        let attn = attn
            .broadcast_add(&Tensor::eye(chunk_size, attn.dtype(), attn.device())?)?
            .contiguous()?;
        // when rank = 5, matmul error, bs=1,squeeze(0)
        let value = attn.squeeze(0)?.matmul(&v_beta.squeeze(0)?)?.unsqueeze(0)?;
        let k_cumdecay = attn
            .squeeze(0)?
            .matmul(
                &k_beta
                    .broadcast_mul(&g.exp()?.unsqueeze(D::Minus1)?)?
                    .squeeze(0)?,
            )?
            .unsqueeze(0)?;
        let mut last_recurrent_state = if let Some(recurrent) = self.recurrent_state_cache.as_ref()
        {
            recurrent.clone()
        } else {
            Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                candle_core::DType::F32,
                value.device(),
            )?
        };

        let mut core_attn_out = value.zeros_like()?;
        let tril_mask = Tensor::tril2(chunk_size, candle_core::DType::U32, query.device())?
            .broadcast_as((batch_size, num_heads, chunk_size, chunk_size))?;
        let on_false = tril_mask.zeros_like()?.to_dtype(candle_core::DType::F32)?;
        let last_dim = core_attn_out.dim(D::Minus1)?;
        for i in 0..total_sequence_length / chunk_size {
            let q_i = query.i((.., .., i))?.contiguous()?;
            let k_i = key.i((.., .., i))?.contiguous()?;
            let v_i = value.i((.., .., i))?.contiguous()?;
            let g_i = g.i((.., .., i))?.contiguous()?;
            let attn = q_i
                .matmul(&k_i.transpose(D::Minus1, D::Minus2)?.contiguous()?)?
                .mul(&decay_mask.i((.., .., i))?)?;
            let attn = tril_mask.where_cond(&attn, &on_false)?.contiguous()?;
            let v_prime = k_cumdecay.i((.., .., i))?.matmul(&last_recurrent_state)?;
            let v_new = v_i.sub(&v_prime)?;
            let attn_inter = q_i
                .broadcast_mul(&g_i.unsqueeze(D::Minus1)?.exp()?)?
                .matmul(&last_recurrent_state)?;
            let out_i = attn_inter.add(&attn.matmul(&v_new)?)?.unsqueeze(2)?;
            core_attn_out = core_attn_out.slice_assign(
                &[
                    (0..batch_size),
                    (0..num_heads),
                    (i..i + 1),
                    (0..chunk_size),
                    (0..last_dim),
                ],
                &out_i,
            )?;
            let g_i_last_dim = g_i.dim(D::Minus1)?;
            last_recurrent_state = last_recurrent_state
                .broadcast_mul(
                    &g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?
                        .unsqueeze(D::Minus1)?
                        .exp()?,
                )?
                .add(
                    &k_i.broadcast_mul(
                        &g_i.narrow(D::Minus1, g_i_last_dim - 1, 1)?
                            .broadcast_sub(&g_i)?
                            .exp()?
                            .unsqueeze(D::Minus1)?,
                    )?
                    .transpose(D::Minus1, D::Minus2)?
                    .squeeze(0)?
                    .matmul(&v_new.squeeze(0)?)?
                    .unsqueeze(0)?,
                )?;
        }
        self.recurrent_state_cache = Some(last_recurrent_state);
        core_attn_out =
            core_attn_out.reshape((batch_size, num_heads, (), core_attn_out.dim(D::Minus1)?))?;
        core_attn_out = core_attn_out.narrow(2, 0, sequence_length)?;
        core_attn_out = core_attn_out
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;

        Ok(core_attn_out)
    }

    fn torch_recurrent_gated_delta_rule(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        use_qk_l2norm_in_kernel: bool,
    ) -> Result<Tensor> {
        let (query, key) = if use_qk_l2norm_in_kernel {
            (l2_normalize(query, 3)?, l2_normalize(key, 3)?)
        } else {
            (query.clone(), key.clone())
        };
        let initial_dtype = query.dtype();
        let (query, key, value, beta, g) = transmute_tensors!(query, key, value, beta, g);
        let (batch_size, num_heads, sequence_length, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(D::Minus1)?;
        let scale = 1.0 / (query.dim(D::Minus1)? as f64).sqrt();
        let query = query.affine(scale, 0.0)?;
        let mut last_recurrent_state = if let Some(recurrent) = self.recurrent_state_cache.as_ref()
        {
            recurrent.clone()
        } else {
            Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                candle_core::DType::F32,
                value.device(),
            )?
        };

        let mut core_attn_out = Tensor::zeros(
            (batch_size, num_heads, sequence_length, v_head_dim),
            candle_core::DType::F32,
            value.device(),
        )?;
        for i in 0..sequence_length {
            let q_i = query.i((.., .., i))?;
            let k_i = key.i((.., .., i))?;
            let v_i = value.i((.., .., i))?;
            let g_i = g
                .i((.., .., i))?
                .exp()?
                .unsqueeze(D::Minus1)?
                .unsqueeze(D::Minus1)?;
            let beta_i = beta.i((.., .., i))?.unsqueeze(D::Minus1)?;
            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_i)?;
            let kv_mem = last_recurrent_state
                .broadcast_mul(&k_i.unsqueeze(D::Minus1)?)?
                .sum(D::Minus2)?;
            let delta = v_i.broadcast_sub(&kv_mem)?.broadcast_mul(&beta_i)?;
            last_recurrent_state = last_recurrent_state.broadcast_add(
                &k_i.unsqueeze(D::Minus1)?
                    .broadcast_mul(&delta.unsqueeze(D::Minus2)?)?,
            )?;
            let out_i = last_recurrent_state
                .broadcast_mul(&q_i.unsqueeze(D::Minus1)?)?
                .sum_keepdim(D::Minus2)?;
            core_attn_out = core_attn_out.slice_assign(
                &[(0..batch_size), (0..num_heads), (i..i + 1), (0..v_head_dim)],
                &out_i,
            )?;
        }
        self.recurrent_state_cache = Some(last_recurrent_state);
        core_attn_out = core_attn_out
            .transpose(1, 2)?
            .contiguous()?
            .to_dtype(initial_dtype)?;

        Ok(core_attn_out)
    }

    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let xs = if let Some(mask) = attention_mask {
            xs.broadcast_mul(&mask.unsqueeze(D::Minus1)?)?
        } else {
            xs.clone()
        };
        let (bs, seq_len, _) = xs.dims3()?;
        let mut mixed_qkv = self.in_proj_qkv.forward(&xs)?.transpose(1, 2)?;
        let z = self
            .in_proj_z
            .forward(&xs)?
            .reshape((bs, seq_len, (), self.head_v_dim))?;
        let b = self.in_proj_b.forward(&xs)?;
        let a = self.in_proj_a.forward(&xs)?;
        let use_precomputed_states =
            self.conv_state_cache.is_some() && self.recurrent_state_cache.is_some() && seq_len == 1;
        if use_precomputed_states {
            mixed_qkv = self.torch_causal_conv1d_update(&mixed_qkv)?;
        } else {
            let pad = self.conv_kernel_size as isize - mixed_qkv.dim(D::Minus1)? as isize;
            let conv_state = if pad >= 0 {
                mixed_qkv.pad_with_zeros(D::Minus1, pad as usize, 0)?
            } else {
                mixed_qkv.narrow(D::Minus1, pad.unsigned_abs(), self.conv_kernel_size)?
            };
            self.conv_state_cache = Some(conv_state);
            mixed_qkv = mixed_qkv.pad_with_zeros(
                D::Minus1,
                self.conv_kernel_size - 1,
                self.conv_kernel_size - 1,
            )?;
            mixed_qkv = conv1d_depthwise(&mixed_qkv, self.conv1d.weight(), self.conv1d.bias())?;
            mixed_qkv = mixed_qkv.narrow(D::Minus1, 0, seq_len)?.silu()?;
            // too slowly
            // mixed_qkv = self
            //     .conv1d
            //     .forward(&mixed_qkv)?
            //     .narrow(D::Minus1, 0, seq_len)?
            //     .silu()?;
        }
        let mixed_qkv = mixed_qkv.transpose(1, 2)?;
        let qkv_split = split_tensor(
            &mixed_qkv,
            &[self.key_dim, self.key_dim, self.value_dim],
            D::Minus1,
        )?;

        let mut query = qkv_split[0].reshape((bs, seq_len, (), self.head_k_dim))?;
        let mut key = qkv_split[1].reshape((bs, seq_len, (), self.head_k_dim))?;
        let value = qkv_split[2].reshape((bs, seq_len, (), self.head_v_dim))?;
        let beta = sigmoid(&b)?;
        let a_plus_bias = softplus(
            &a.to_dtype(candle_core::DType::F32)?
                .broadcast_add(&self.dt_bias.to_dtype(candle_core::DType::F32)?)?,
        )?;
        let g = (-1.0 * self.a_log.to_dtype(candle_core::DType::F32)?.exp()?)?
            .broadcast_mul(&a_plus_bias)?;
        if self.num_v_heads / self.num_k_heads > 1 {
            query = repeat_interleave(&query, self.num_v_heads / self.num_k_heads, 2)?;
            key = repeat_interleave(&key, self.num_v_heads / self.num_k_heads, 2)?;
        }
        let core_attn_out = if !use_precomputed_states {
            self.torch_chunk_gated_delta_rule(&query, &key, &value, &g, &beta, true, 64)?
        } else {
            self.torch_recurrent_gated_delta_rule(&query, &key, &value, &g, &beta, true)?
        };
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = self.norm.forward(&core_attn_out, Some(&z))?;
        let core_attn_out = core_attn_out.reshape((bs, seq_len, ()))?;
        let output = self.out_proj.forward(&core_attn_out)?;

        Ok(output)
    }

    pub fn clear_cache(&mut self) {
        self.conv_state_cache = None;
        self.recurrent_state_cache = None;
    }
}

pub struct Qwen3_5Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Qwen3_5RMSNorm,
    k_norm: Qwen3_5RMSNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3_5Attention {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let q_proj = linear_b(
            hidden_size,
            num_attention_heads * head_dim * 2,
            config.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            config.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            config.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_attention_heads * head_dim,
            hidden_size,
            config.attention_bias,
            vb.pp("o_proj"),
        )?;
        let q_norm = Qwen3_5RMSNorm::new(vb.pp("q_norm"), head_dim, config.rms_norm_eps)?;
        let k_norm = Qwen3_5RMSNorm::new(vb.pp("k_norm"), head_dim, config.rms_norm_eps)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_chunk = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_attention_heads, self.head_dim * 2))?
            .chunk(2, D::Minus1)?;
        let query_states =
            query_chunk[0].reshape((b_sz, q_len, self.num_attention_heads, self.head_dim))?;
        let gate = query_chunk[1].reshape((b_sz, q_len, ()))?;

        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        let key_states = self.k_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            glm_asr_apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
        )?;
        let attn_output = attn_output
            .reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?
            .contiguous()?;
        let attn_output = attn_output.mul(&sigmoid(&gate)?)?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

pub struct Qwen3_5DecoderLayer {
    // hidden_size: usize,
    layer_type: String,
    linear_attn: Option<Qwen3_5GatedDeltaNet>,
    self_attn: Option<Qwen3_5Attention>,
    mlp: GateUpDownMLP,
    input_layernorm: Qwen3_5RMSNorm,
    post_attention_layernorm: Qwen3_5RMSNorm,
}

impl Qwen3_5DecoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig, layer_idx: usize) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let layer_type = config.layer_types[layer_idx].clone();
        let (linear_attn, self_attn) = if layer_type.eq("linear_attention") {
            let linear_attn = Qwen3_5GatedDeltaNet::new(vb.pp("linear_attn"), config)?;
            (Some(linear_attn), None)
        } else {
            let self_attn = Qwen3_5Attention::new(vb.pp("self_attn"), config)?;
            (None, Some(self_attn))
        };
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            hidden_size,
            config.intermediate_size,
            config.hidden_act,
            false,
            None,
            None,
            None,
        )?;
        let input_layernorm =
            Qwen3_5RMSNorm::new(vb.pp("input_layernorm"), hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = Qwen3_5RMSNorm::new(
            vb.pp("post_attention_layernorm"),
            hidden_size,
            config.rms_norm_eps,
        )?;
        Ok(Self {
            // hidden_size,
            layer_type,
            linear_attn,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = self.input_layernorm.forward(xs)?;
        if self.layer_type.eq("linear_attention")
            && let Some(linear_attn) = self.linear_attn.as_mut()
        {
            xs = linear_attn.forward(&xs, attention_mask)?;
        } else if let Some(self_attn) = self.self_attn.as_mut()
            && let Some(cos) = cos
            && let Some(sin) = sin
        {
            xs = self_attn.forward(&xs, cos, sin, attention_mask)?;
        }
        let residual = xs.add(&residual)?;
        xs = self.post_attention_layernorm.forward(&residual)?;
        xs = self.mlp.forward(&xs)?;
        xs = xs.add(&residual)?;
        Ok(xs)
    }

    pub fn clear_cache(&mut self) {
        if let Some(linear_attn) = self.linear_attn.as_mut() {
            linear_attn.clear_cache();
        }
        if let Some(self_attn) = self.self_attn.as_mut() {
            self_attn.clear_kv_cache();
        }
    }
}

pub struct Qwen3_5TextModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RMSNorm,
    rotary_emb: Qwen3VLTextRotaryEmbedding,
    mrope_section: Vec<usize>,
}

impl Qwen3_5TextModel {
    pub fn new(vb: VarBuilder, config: &Qwen3_5TextConfig) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            let layer = Qwen3_5DecoderLayer::new(vb_layers.pp(i), config, i)?;
            layers.push(layer);
        }
        let norm = Qwen3_5RMSNorm::new(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;
        let rope_dim =
            (config.head_dim as f32 * config.rope_parameters.partial_rotary_factor) as usize;
        let rotary_emb =
            Qwen3VLTextRotaryEmbedding::new(rope_dim, config.rope_parameters.rope_theta);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section: config.rope_parameters.mrope_section.clone(),
        })
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;

        let (cos, sin) = self.rotary_emb.forward(
            position_ids,
            inputs_embeds.dtype(),
            self.mrope_section.clone(),
        )?;
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    0,
                    inputs_embeds.device(),
                )?)
            }
        };
        for layer in self.layers.iter_mut() {
            let layer_mask =
                if layer.layer_type.ne("linear_attention") || (seq_len != 1 && b_size != 1) {
                    attention_mask.clone()
                } else {
                    None
                };
            xs = layer.forward(&xs, Some(&cos), Some(&sin), layer_mask.as_ref())?;
        }
        xs = self.norm.forward(&xs)?;
        Ok(xs)
    }

    pub fn clear_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_cache();
        }
    }
}

pub struct Qwen3_5Model {
    config: Qwen3_5Config,
    visual: Qwen3VLVisionModel,
    language_model: Qwen3_5TextModel,
    lm_head: Linear,
    rope_deltas: Option<Tensor>,
}

impl Qwen3_5Model {
    pub fn new(vb: VarBuilder, config: Qwen3_5Config) -> Result<Self> {
        let vb_m = vb.pp("model");
        let visual = Qwen3VLVisionModel::new(config.vision_config.clone(), vb_m.pp("visual"))?;
        let language_model = Qwen3_5TextModel::new(vb_m.pp("language_model"), &config.text_config)?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(language_model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };
        Ok(Self {
            config,
            visual,
            language_model,
            lm_head,
            rope_deltas: None,
        })
    }

    fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let video_grid_thw = match video_grid_thw {
            Some(thw) => {
                let grid_t = thw.i((.., 0))?.to_vec1::<u32>()?;
                let mut v_thw_vec = Vec::new();
                for (index, t) in grid_t.iter().enumerate() {
                    let mut thw_i = thw.i(index)?.to_vec1::<u32>()?;
                    // [12, 30, 50]
                    // [1, 30, 50]*t
                    thw_i[0] = 1;
                    v_thw_vec.push(
                        Tensor::new(thw_i, thw.device())?
                            .repeat(*t as usize)?
                            .reshape((*t as usize, ()))?,
                    );
                }
                Some(Tensor::cat(&v_thw_vec, 0)?)
            }
            None => None,
        };

        let spatial_merge_size = self.config.vision_config.spatial_merge_size;
        let image_token_id = self.config.image_token_id;
        let video_token_id = self.config.video_token_id;
        let vision_start_token_id = self.config.vision_start_token_id;
        let mut mrope_position_deltas = vec![];
        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            let total_input_ids = input_ids.clone();
            let mask_ = mask
                .cloned()
                .unwrap_or(Tensor::ones_like(&total_input_ids)?)
                .to_device(input_ids.device())?;
            let mut position_ids = Tensor::ones(
                (3, input_ids.dim(0)?, input_ids.dim(1)?),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            let mut image_index = 0;
            let mut video_index = 0;

            for i in 0..total_input_ids.dim(0)? {
                let mut input_ids_i = total_input_ids.i(i)?;
                let mask_i = mask_.i(i)?;
                // 推理时, attention_mask如果是全1向量,取非0索引的操作没必要
                if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                    let nonzero_idx = nonzero_index(&mask_i)?;
                    input_ids_i = input_ids_i.gather(&nonzero_idx, 0)?;
                }
                let mut text_start = 0;
                let mut text_end = 0;
                let mut thw = vec![];
                let mut llm_pos_ids_list: Vec<Tensor> = Vec::new();
                // vision start的下一个索引
                let vision_indices = get_vision_next_indices(&input_ids_i, vision_start_token_id);

                match vision_indices {
                    Ok(indeices) => {
                        let vision_tokens = input_ids_i.gather(&indeices, 0)?.to_vec1::<u32>()?;
                        let vision_indices_vec = indeices.to_vec1::<u32>()?;
                        for (j, &token) in vision_tokens.iter().enumerate() {
                            if token == image_token_id {
                                thw = image_grid_thw.unwrap().i(image_index)?.to_vec1::<u32>()?;
                                image_index += 1;
                                text_end = vision_indices_vec[j];
                            }
                            if token == video_token_id {
                                thw = video_grid_thw
                                    .as_ref()
                                    .unwrap()
                                    .i(video_index)?
                                    .to_vec1::<u32>()?;
                                text_end = vision_indices_vec[j];
                                video_index += 1;
                            }
                            let llm_grid_t = thw[0];
                            let llm_grid_h = thw[1] / spatial_merge_size as u32;
                            let llm_grid_w = thw[2] / spatial_merge_size as u32;
                            let text_len = text_end - text_start;
                            let start_idx = if !llm_pos_ids_list.is_empty() {
                                llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                                    .max_all()?
                                    .to_scalar::<u32>()?
                                    + 1
                            } else {
                                0
                            };
                            let pos_ids = Tensor::arange(
                                start_idx,
                                start_idx + text_len,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .broadcast_as((3usize, text_len as usize))?;
                            llm_pos_ids_list.push(pos_ids);

                            let t_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_t,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(D::Minus1)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize * llm_grid_w as usize,
                            ))?
                            .flatten_all()?;
                            let h_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_h,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(D::Minus1)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;
                            let w_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_w,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(0)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;

                            let thw_index = Tensor::stack(&[t_index, h_index, w_index], 0)?;
                            llm_pos_ids_list.push(thw_index);
                            text_start = text_end + llm_grid_t * llm_grid_h * llm_grid_w;
                        }
                    }
                    Err(e) => {
                        println!("get vision_indices err: {e}");
                    }
                };
                if text_start < input_ids_i.dim(0)? as u32 {
                    let start_idx = if !llm_pos_ids_list.is_empty() {
                        llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                            .max_all()?
                            .to_scalar::<u32>()?
                            + 1
                    } else {
                        0
                    };
                    let text_len = input_ids_i.dim(0)? as u32 - text_start;
                    let pos_ids =
                        Tensor::arange(start_idx, start_idx + text_len, input_ids_i.device())?
                            .unsqueeze(0)?
                            .broadcast_as((3usize, text_len as usize))?;
                    llm_pos_ids_list.push(pos_ids);
                }
                let llm_position = Tensor::cat(&llm_pos_ids_list, 1)?.reshape((3, 1, ()))?;
                position_ids = position_ids
                    .slice_assign(&[(0..3), (i..i + 1), (0..input_ids.dim(1)?)], &llm_position)?;
                let position_deltas = llm_position.max_all()?.to_scalar::<u32>()? as i64 + 1
                    - input_ids_i.dim(0)? as i64;
                mrope_position_deltas.push(position_deltas);
            }
            let mut mrope_position_deltas = Tensor::new(mrope_position_deltas, input_ids.device())?;
            if mrope_position_deltas.rank() == 1 {
                mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
            }
            Ok((position_ids.contiguous()?, mrope_position_deltas))
        } else if let Some(mask) = mask {
            let mut position_ids = mask
                .to_dtype(candle_core::DType::F64)?
                .cumsum(D::Minus1)?
                .to_dtype(candle_core::DType::U32)?
                .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;
            for i in 0..position_ids.dim(0)? {
                let mut position_ids_i = position_ids.i(i)?;
                let mask_i = mask.i(i)?;
                // 如果有pad, 将填充位置置为1
                // 当bs>1, 可能存在不同序列长度，需要添加pad使seq_len长度一致
                if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                    let zero_indices = zero_index(&mask_i)?;
                    let replace_1 = Tensor::ones(
                        zero_indices.dim(0)?,
                        candle_core::DType::U32,
                        input_ids.device(),
                    )?;
                    position_ids_i = position_ids_i
                        .scatter(&zero_indices, &replace_1, 0)?
                        .unsqueeze(0)?;
                    position_ids = position_ids
                        .slice_assign(&[(i..i + 1), (0..position_ids.dim(1)?)], &position_ids_i)?;
                }
            }
            position_ids = position_ids
                .unsqueeze(0)?
                .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?
                .contiguous()?;
            let mut mrope_position_deltas = position_ids
                .max(0)?
                .max(D::Minus1)?
                .broadcast_sub(&Tensor::new(
                    vec![mask.dim(D::Minus1)? as u32 - 1],
                    input_ids.device(),
                )?)?
                .contiguous()?;
            if mrope_position_deltas.rank() == 1 {
                mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
            }
            Ok((position_ids, mrope_position_deltas))
        } else {
            let position_ids =
                Tensor::arange(0_u32, input_ids.dim(D::Minus1)? as u32, input_ids.device())?
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(D::Minus1)?))?
                    .contiguous()?;
            let mrope_position_deltas = Tensor::zeros(
                (input_ids.dim(0)?, 1),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            Ok((position_ids, mrope_position_deltas))
        }
    }

    fn compute_3d_position_ids(
        &mut self,
        input_ids: &Tensor,
        inputs_embeds: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let position_ids = if self.rope_deltas.is_none() {
            let (position_ids, rope_deltas) =
                self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, None)?;
            self.rope_deltas = Some(rope_deltas);
            position_ids
        } else {
            let (bs, seq_len, _) = inputs_embeds.dims3()?;
            Tensor::arange(
                seqlen_offset as i64,
                (seqlen_offset + seq_len) as i64,
                input_ids.device(),
            )?
            .to_dtype(self.rope_deltas.as_ref().unwrap().dtype())?
            .unsqueeze(0)?
            .broadcast_as((bs, seq_len))?
            .broadcast_add(self.rope_deltas.as_ref().unwrap())?
            .unsqueeze(0)?
            .broadcast_as((3, bs, seq_len))?
            .contiguous()?
            .to_dtype(candle_core::DType::U32)?
        };
        Ok(position_ids)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        pixel_values_video: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        // cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.language_model.embed_tokens.forward(input_ids)?;
        if let Some(pixel_values) = pixel_values
            && let Some(image_grid_thw) = image_grid_thw
        {
            let (image_embeds, _) = self.visual.forward(pixel_values, image_grid_thw)?;
            let vision_mask = get_equal_mask(input_ids, self.config.image_token_id)?;
            let n_image_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;
            if n_image_tokens as usize != image_embeds.dim(0)? {
                return Err(anyhow!(format!(
                    "n_image_token num: {} not equal to image_embed len: {}",
                    n_image_tokens,
                    image_embeds.dim(0)?
                )));
            }
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embeds, &vision_mask)?;
        }
        if let Some(pixel_values_video) = pixel_values_video
            && let Some(video_grid_thw) = video_grid_thw
        {
            let (video_embeds, _) = self.visual.forward(pixel_values_video, video_grid_thw)?;
            let vision_mask = get_equal_mask(input_ids, self.config.video_token_id)?;
            let n_video_tokens = vision_mask.sum_all()?.to_scalar::<u32>()?;
            if n_video_tokens as usize != video_embeds.dim(0)? {
                return Err(anyhow!(format!(
                    "n_video_tokens num: {} not equal to video_embeds len: {}",
                    n_video_tokens,
                    video_embeds.dim(0)?
                )));
            }
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &video_embeds, &vision_mask)?;
        }

        let position_ids = self.compute_3d_position_ids(
            input_ids,
            &inputs_embeds,
            image_grid_thw,
            video_grid_thw,
            seqlen_offset,
        )?;
        let outputs = self.language_model.forward(&inputs_embeds, &position_ids)?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_cache(&mut self) {
        self.language_model.clear_cache();
    }
}
