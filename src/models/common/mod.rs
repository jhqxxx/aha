use anyhow::{Result, anyhow};
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{
    Activation, BatchNorm, BatchNormConfig, Conv1d, Conv1dConfig, Conv2d, Conv2dConfig,
    ConvTranspose1d, ConvTranspose1dConfig, Embedding, Init, LayerNorm, LayerNormConfig, Linear,
    Module, ModuleT, RmsNorm, VarBuilder, batch_norm, conv1d, conv1d_no_bias, conv2d,
    conv2d_no_bias, embedding, layer_norm, linear_b, linear_no_bias, ops::sigmoid, rms_norm,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    position_embed::rope::{RoPE, apply_rotary_pos_emb, apply_rotary_pos_emb_roformer},
    utils::tensor_utils::{prepare_causal_attention_mask, repeat_kv},
};

#[derive(Debug, Clone)]
pub struct GateUpDownMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GateUpDownMLP {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        act_fn: Activation,
        bias: bool,
        gate_pp_name: Option<&str>,
        up_pp_name: Option<&str>,
        down_pp_name: Option<&str>,
    ) -> Result<Self> {
        let gate_pp_name = gate_pp_name.unwrap_or("gate_proj");
        let up_pp_name = up_pp_name.unwrap_or("up_proj");
        let down_pp_name = down_pp_name.unwrap_or("down_proj");
        let gate_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(gate_pp_name))?;
        let up_proj = linear_b(hidden_size, intermediate_size, bias, vb.pp(up_pp_name))?;
        let down_proj = linear_b(intermediate_size, hidden_size, bias, vb.pp(down_pp_name))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

pub struct TwoLinearMLP {
    linear1: Linear,
    linear2: Linear,
    act: Activation,
}

impl TwoLinearMLP {
    pub fn new(
        vb: VarBuilder,
        // embedding_dim: usize,
        // mlp_dim: usize,
        in_dim: usize,
        middle_dim: usize,
        out_dim: usize,
        act: Activation,
        bias: bool,
        linear1_pp_name: &str,
        linear2_pp_name: &str,
    ) -> Result<Self> {
        let linear1 = linear_b(in_dim, middle_dim, bias, vb.pp(linear1_pp_name))?;
        let linear2 = linear_b(middle_dim, out_dim, bias, vb.pp(linear2_pp_name))?;

        Ok(Self {
            linear1,
            linear2,
            act,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs
            .apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
pub struct NaiveAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl NaiveAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: Option<usize>,
        bias: bool,
        q_proj_pp_name: Option<&str>,
        k_proj_pp_name: Option<&str>,
        v_proj_pp_name: Option<&str>,
        o_proj_pp_name: Option<&str>,
    ) -> Result<Self> {
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let head_dim = match head_dim {
            None => hidden_size / num_attention_heads,
            Some(dim) => dim,
        };
        let q_proj_pp_name = q_proj_pp_name.unwrap_or("q_proj");
        let k_proj_pp_name = k_proj_pp_name.unwrap_or("k_proj");
        let v_proj_pp_name = v_proj_pp_name.unwrap_or("v_proj");
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");
        let q_proj = linear_b(
            hidden_size,
            num_attention_heads * head_dim,
            bias,
            vb.pp(q_proj_pp_name),
        )?;
        let k_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            bias,
            vb.pp(k_proj_pp_name),
        )?;
        let v_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            bias,
            vb.pp(v_proj_pp_name),
        )?;
        let o_proj = linear_b(
            num_attention_heads * head_dim,
            hidden_size,
            bias,
            vb.pp(o_proj_pp_name),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_attention_heads,
            num_kv_heads: num_key_value_heads,
            num_kv_groups,
            head_dim,
            middle_size: num_attention_heads * head_dim,
            kv_cache: None,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) = if let Some(cos) = cos
            && let Some(sin) = sin
        {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
        } else {
            (query_states, key_states)
        };

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn forward_with_cache(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };

        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
pub struct QKVCatAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    middle_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl QKVCatAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        head_dim: Option<usize>,
        bias: bool,
        qkv_proj_pp_name: Option<&str>,
        o_proj_pp_name: Option<&str>,
    ) -> Result<Self> {
        let head_dim = match head_dim {
            None => hidden_size / num_attention_heads,
            Some(dim) => dim,
        };
        let qkv_proj_pp_name = qkv_proj_pp_name.unwrap_or("wqkv");
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");
        let qkv_proj = linear_b(
            hidden_size,
            3 * num_attention_heads * head_dim,
            bias,
            vb.pp(qkv_proj_pp_name),
        )?;
        let o_proj = linear_b(
            num_attention_heads * head_dim,
            hidden_size,
            bias,
            vb.pp(o_proj_pp_name),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            num_heads: num_attention_heads,
            head_dim,
            middle_size: num_attention_heads * head_dim,
            kv_cache: None,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
        use_roformer: bool,
    ) -> Result<Tensor> {
        let (b, q_len, _) = xs.dims3()?;
        // (3, B, n_head, seq_len, head_dim)
        let qkv = self
            .qkv_proj
            .forward(xs)?
            .reshape((b, q_len, 3, self.num_heads, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let query_states = qkv.i(0)?.contiguous()?;
        let key_states = qkv.i(1)?.contiguous()?;
        let value_states = qkv.i(2)?.contiguous()?;
        let (query_states, key_states) = if let Some(cos) = cos
            && let Some(sin) = sin
        {
            if use_roformer {
                apply_rotary_pos_emb_roformer(&query_states, &key_states, cos, sin, tof32)?
            } else {
                apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
            }
        } else {
            (query_states, key_states)
        };

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            None,
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn forward_with_cache(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        tof32: bool,
        use_roformer: bool,
    ) -> Result<Tensor> {
        let (b, q_len, _) = xs.dims3()?;
        let qkv = self
            .qkv_proj
            .forward(xs)?
            .reshape((b, q_len, 3, self.num_heads, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let query_states = qkv.i(0)?.contiguous()?;
        let key_states = qkv.i(1)?.contiguous()?;
        let value_states = qkv.i(2)?.contiguous()?;
        let (query_states, key_states) = if use_roformer {
            apply_rotary_pos_emb_roformer(&query_states, &key_states, cos, sin, tof32)?
        } else {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
        };
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };

        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            None,
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

pub struct NaiveAttnTwoLinearMLPBlock {
    self_attn: NaiveAttention,
    mlp: TwoLinearMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl NaiveAttnTwoLinearMLPBlock {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        linear1_pp_name: &str,
        linear2_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
    ) -> Result<Self> {
        let num_key_value_heads = match num_key_value_heads {
            Some(heads) => heads,
            None => num_attention_heads,
        };
        let self_attn = NaiveAttention::new(
            vb.pp(attn_pp_name),
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attn_bias,
            None,
            None,
            None,
            o_proj_pp_name,
        )?;
        let mlp = TwoLinearMLP::new(
            vb.pp(mlp_pp_name),
            hidden_size,
            intermediate_size,
            hidden_size,
            hidden_act,
            mlp_bias,
            linear1_pp_name,
            linear2_pp_name,
        )?;

        let input_layernorm =
            get_layer_norm(vb.pp(input_norm_pp_name), norm_eps, hidden_size, true)?;
        let post_attention_layernorm =
            get_layer_norm(vb.pp(post_norm_pp_name), norm_eps, hidden_size, true)?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, cos, sin, attention_mask, tof32)?;
        let residual = residual.add(&xs)?;
        let xs = self.post_attention_layernorm.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
}

pub struct NaiveAttnGateUpDownMLPBlock {
    self_attn: NaiveAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl NaiveAttnGateUpDownMLPBlock {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
    ) -> Result<Self> {
        let num_key_value_heads = match num_key_value_heads {
            Some(heads) => heads,
            None => num_attention_heads,
        };
        let self_attn = NaiveAttention::new(
            vb.pp(attn_pp_name),
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attn_bias,
            None,
            None,
            None,
            o_proj_pp_name,
        )?;
        let mlp = GateUpDownMLP::new(
            vb.pp(mlp_pp_name),
            hidden_size,
            intermediate_size,
            hidden_act,
            mlp_bias,
            None,
            None,
            None,
        )?;
        let input_layernorm = rms_norm(hidden_size, norm_eps, vb.pp(input_norm_pp_name))?;
        let post_attention_layernorm = rms_norm(hidden_size, norm_eps, vb.pp(post_norm_pp_name))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_with_cache(&xs, cos, sin, attention_mask, false)?;
        let residual = residual.add(&xs)?;
        let xs = self.post_attention_layernorm.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

pub fn eager_attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
) -> Result<Tensor> {
    // input q shape:(b, num_head, seq_len, dim)
    // input k/v shape:(b, num_kv_head, seq_len, dim)
    let key_states = match num_key_value_groups {
        Some(g) => repeat_kv(key_states.clone(), g)?.contiguous()?,
        None => key_states.clone(),
    };
    let value_states = match num_key_value_groups {
        Some(g) => repeat_kv(value_states.clone(), g)?.contiguous()?,
        None => value_states.clone(),
    };
    let query_states = query_states.contiguous()?;
    let key_states = key_states.contiguous()?;
    let value_states = value_states.contiguous()?;
    let attn_output = {
        #[cfg(not(feature = "flash-attn"))]
        {
            let attn_weights = query_states.matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?;
            let attn_weights = (attn_weights * scaling)?;
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(&mask.to_dtype(attn_weights.dtype())?)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        }
        #[cfg(feature = "flash-attn")]
        {
            // use flash-attn,
            // flash-attn shape: (bs, seq_len, num_head, head_dim)
            let query_states = query_states.transpose(1, 2)?;
            let key_states = key_states.transpose(1, 2)?;
            let value_states = value_states.transpose(1, 2)?;
            let attn_output = candle_flash_attn::flash_attn(
                &query_states,
                &key_states,
                &value_states,
                scaling as f32,
                attention_mask.is_some(),
            )?
            .transpose(1, 2)?;
            attn_output
        }
    };
    //(b, n_head, seq_len, dim) -> (b, seq_len, n_head, dim)
    let attn_output = attn_output.transpose(1, 2)?.contiguous()?;

    Ok(attn_output)
}

pub fn get_conv2d(
    vb: VarBuilder,
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding,
        stride,
        dilation,
        groups,
        cudnn_fwd_algo: None,
    };
    let conv2d = if bias {
        conv2d(in_c, out_c, kernel_size, cfg, vb)?
    } else {
        conv2d_no_bias(in_c, out_c, kernel_size, cfg, vb)?
    };
    Ok(conv2d)
}

pub fn get_conv1d(
    vb: VarBuilder,
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv1d> {
    let cfg = Conv1dConfig {
        padding,
        stride,
        dilation,
        groups,
        cudnn_fwd_algo: None,
    };
    let conv1d = if bias {
        conv1d(in_c, out_c, kernel_size, cfg, vb)?
    } else {
        conv1d_no_bias(in_c, out_c, kernel_size, cfg, vb)?
    };
    Ok(conv1d)
}

pub fn get_layer_norm(vb: VarBuilder, eps: f64, dim: usize, affine: bool) -> Result<LayerNorm> {
    let ln_config = LayerNormConfig {
        eps,
        remove_mean: true, // true for layernorm, false for RMSNorm
        affine,            // true for with bias, false for without bias
    };
    let norm = layer_norm(dim, ln_config, vb)?;
    Ok(norm)
}

pub fn get_layer_norm_without_weight(vb: VarBuilder, eps: f64, dim: usize) -> Result<LayerNorm> {
    let weight = Tensor::ones(dim, vb.dtype(), vb.device())?;
    let bias = Tensor::zeros(dim, vb.dtype(), vb.device())?;
    Ok(LayerNorm::new(weight, bias, eps))
}

pub fn get_batch_norm(vb: VarBuilder, eps: f64, dim: usize, affine: bool) -> Result<BatchNorm> {
    let bn_config = BatchNormConfig {
        eps,
        remove_mean: true,
        affine,
        momentum: 0.1,
    };
    let norm = batch_norm(dim, bn_config, vb)?;
    Ok(norm)
}

pub fn deform_conv2d_kernel(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    offset: &Tensor,
    mask: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    // 不考虑空洞卷积, bs = 1
    let (_, in_c, in_h, in_w) = input.dims4()?;
    let (out_channel, _, ker_h, ker_w) = weight.dims4()?;
    let out_h = ((in_h + 2 * padding - ker_h) / stride) + 1;
    let out_w = ((in_w + 2 * padding - ker_w) / stride) + 1;

    let num_kernels = in_c * out_h * out_w;
    let mask_vec = if let Some(mask) = mask {
        Some(mask.squeeze(0)?.to_vec3::<f32>()?)
    } else {
        None
    };
    let offset_vec = offset.squeeze(0)?.to_vec3::<f32>()?;
    let input_vec = input.squeeze(0)?.to_vec3::<f32>()?;
    let mut columns_vec = vec![vec![0.0f32; out_h * out_w]; in_c * ker_h * ker_w];
    for index in 0..num_kernels {
        let out_x = index % out_w;
        let out_y = (index / out_w) % out_h;
        let in_c = index / (out_w * out_h);
        let out_c = in_c * ker_h * ker_w;

        for i in 0..ker_h {
            for j in 0..ker_w {
                let mask_idx = i * ker_w + j;
                let offset_idx = 2 * mask_idx;
                let mask_value = if mask.is_some() {
                    mask_vec.as_ref().unwrap()[mask_idx][out_y][out_x]
                } else {
                    1.0
                };
                let offset_h = offset_vec[offset_idx][out_y][out_x];
                let offset_w = offset_vec[offset_idx + 1][out_y][out_x];
                let y = ((out_y * stride - padding) + i) as f32 + offset_h;
                let x = ((out_x * stride - padding) + j) as f32 + offset_w;
                let val = if y <= -1.0 || in_h as f32 <= y || x <= -1.0 || in_w as f32 <= x {
                    0.0
                } else {
                    let h_low = y.floor();
                    let w_low = x.floor();
                    let h_high = h_low + 1.0;
                    let w_high = w_low + 1.0;
                    let lh = y - h_low;
                    let lw = x - w_low;
                    let hh = 1.0 - lh;
                    let hw = 1.0 - lw;
                    let w1 = hh * hw;
                    let w2 = hh * lw;
                    let w3 = lh * hw;
                    let w4 = lh * lw;
                    let v1 = if h_low >= 0.0 && w_low >= 0.0 {
                        input_vec[in_c][h_low as usize][w_low as usize]
                    } else {
                        0.0
                    };
                    let v2 = if h_low >= 0.0 && w_high <= (in_w - 1) as f32 {
                        input_vec[in_c][h_low as usize][w_high as usize]
                    } else {
                        0.0
                    };
                    let v3 = if h_high <= (in_h - 1) as f32 && w_low >= 0.0 {
                        input_vec[in_c][h_high as usize][w_low as usize]
                    } else {
                        0.0
                    };
                    let v4 = if h_high <= (in_h - 1) as f32 && w_high <= (in_w - 1) as f32 {
                        input_vec[in_c][h_high as usize][w_high as usize]
                    } else {
                        0.0
                    };
                    w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                };
                columns_vec[out_c + i * ker_w + j][out_y * out_w + out_x] = mask_value * val;
            }
        }
    }

    let columns = Tensor::new(columns_vec, weight.device())?;
    let mut out =
        weight
            .flatten_from(1)?
            .matmul(&columns)?
            .reshape((1, out_channel, out_h, out_w))?;
    if let Some(bias) = bias {
        out = out.broadcast_add(bias)?;
    }
    Ok(out)
}

pub struct LlamaModel {
    pub embed_tokens: Embedding,
    layers: Vec<NaiveAttnGateUpDownMLPBlock>,
    norm: RmsNorm,
    rotary_emb: RoPE,
}

impl LlamaModel {
    pub fn new(
        vb: VarBuilder,
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
        rope_theta_base: f32,
    ) -> Result<Self> {
        let embed_tokens = embedding(vocab_size, hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..num_hidden_layers {
            let layers_i = NaiveAttnGateUpDownMLPBlock::new(
                vb_layers.pp(i),
                hidden_size,
                num_attention_heads,
                num_key_value_heads,
                head_dim,
                attn_bias,
                attn_pp_name,
                o_proj_pp_name,
                intermediate_size,
                hidden_act,
                mlp_bias,
                mlp_pp_name,
                norm_eps,
                input_norm_pp_name,
                post_norm_pp_name,
            )?;
            layers.push(layers_i);
        }
        let norm = rms_norm(hidden_size, norm_eps, vb.pp("norm"))?;
        let head_dim = head_dim.unwrap_or(hidden_size / num_attention_heads);
        let rotary_emb = RoPE::new(head_dim, rope_theta_base, vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
        })
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;

        let (cos, sin) = self
            .rotary_emb
            .forward(seqlen_offset, seq_len, inputs_embeds.device())?;
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    0,
                    xs.device(),
                )?)
            }
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
        }
        let xs = xs.apply(&self.norm)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

pub struct LlamaForCausalLM {
    pub model: LlamaModel,
    lm_head: Linear,
}

impl LlamaForCausalLM {
    pub fn new(
        vb: VarBuilder,
        vocab_size: usize,
        hidden_size: usize,
        num_hidden_layers: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
        rope_theta_base: f32,
    ) -> Result<Self> {
        let model = LlamaModel::new(
            vb.pp("model"),
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attn_bias,
            attn_pp_name,
            o_proj_pp_name,
            intermediate_size,
            hidden_act,
            mlp_bias,
            mlp_pp_name,
            norm_eps,
            input_norm_pp_name,
            post_norm_pp_name,
            rope_theta_base,
        )?;
        let lm_head = linear_no_bias(hidden_size, vocab_size, vb.pp("lm_head"))?;
        Ok(Self { model, lm_head })
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let outputs = self.model.forward(inputs_embeds, seqlen_offset)?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

pub fn conv1d_group_parallel(xs: &Tensor, conv1d: &Conv1d) -> Result<Tensor> {
    let groups = conv1d.config().groups;
    let xs = if groups == 1 {
        xs.conv1d_with_algo(
            conv1d.weight(),
            conv1d.config().padding,
            conv1d.config().stride,
            conv1d.config().dilation,
            groups,
            conv1d.config().cudnn_fwd_algo,
        )?
    } else {
        let blocks = xs.chunk(groups, 1)?;
        let kernel = conv1d.weight().chunk(groups, 0)?;
        let blocks = blocks
            // .iter()
            .par_iter()
            .zip(&kernel)
            .map(|(block, kernel)| {
                block
                    .conv1d_with_algo(
                        kernel,
                        conv1d.config().padding,
                        conv1d.config().stride,
                        conv1d.config().dilation,
                        1,
                        conv1d.config().cudnn_fwd_algo,
                    )
                    .map_err(|e| anyhow!(format!("tensor conv1d_with_algo error:{}", e)))
            })
            .collect::<Result<Vec<Tensor>>>()?;
        Tensor::cat(&blocks, 1)?
    };
    match conv1d.bias() {
        None => Ok(xs),
        Some(bias) => {
            let b = bias.dims1()?;
            let bias = bias.reshape((1, b, 1))?;
            Ok(xs.broadcast_add(&bias)?)
        }
    }
}

pub struct GLU {
    dim: usize,
}

impl GLU {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self { dim })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_ = xs.chunk(2, self.dim)?;
        let x_1 = sigmoid(x_[1].as_ref())?;
        let xs = x_1.mul(x_[0].as_ref())?;
        Ok(xs)
    }
}

pub struct GEGLU {
    dim: usize,
}

impl GEGLU {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self { dim })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_ = xs.chunk(2, self.dim)?;
        let x_1 = x_[1].as_ref().gelu()?;
        let xs = x_1.mul(x_[0].as_ref())?;
        Ok(xs)
    }
}

pub struct WNConv1d {
    conv: Conv1d,
}
impl WNConv1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        stride: usize,
        bias: bool,
    ) -> Result<Self> {
        let in_c = in_c / groups;
        let weight_g = vb.get((out_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((out_c, in_c, kernel_size), "weight_v")?;
        // let bias = vb.get(out_c, "bias").ok();
        let bias = if bias {
            vb.get(out_c, "bias").ok()
        } else {
            None
        };
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let cfg = Conv1dConfig {
            padding,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let conv = Conv1d::new(scaled_weight, bias, cfg);
        Ok(Self { conv })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        Ok(x)
    }
}

pub struct WNConvTranspose1d {
    conv_transpose: ConvTranspose1d,
}

impl WNConvTranspose1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        dilation: usize,
        kernel_size: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
        stride: usize,
    ) -> Result<Self> {
        let in_c = in_c / groups;
        let weight_g = vb.get((in_c, 1, 1), "weight_g")?;
        let weight_v = vb.get((in_c, out_c, kernel_size), "weight_v")?;
        let bias = vb.get(out_c, "bias").ok();
        let weight_norm = weight_v.sqr()?.sum_keepdim(1)?.sum_keepdim(2)?.sqrt()?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let config = ConvTranspose1dConfig {
            padding,
            output_padding,
            stride,
            dilation,
            groups,
        };
        let conv_transpose = ConvTranspose1d::new(scaled_weight, bias, config);
        Ok(Self { conv_transpose })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_transpose.forward(x)?;
        Ok(x)
    }
}

pub struct Conv2dWithBN {
    conv_0: Conv2d,
    bn_1: BatchNorm,
    bn_with_relu: bool,
}

impl Conv2dWithBN {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        ks: usize,
        padding: usize,
        stride: usize,
        bias: bool,
        bn_with_relu: bool,
    ) -> Result<Self> {
        let conv_0 = get_conv2d(vb.pp("0"), in_c, out_c, ks, padding, stride, 1, 1, bias)?;
        let bn_1 = get_batch_norm(vb.pp("1"), 1e-5, out_c, true)?;
        Ok(Self {
            conv_0,
            bn_1,
            bn_with_relu,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv_0.forward(x)?;
        let mut x = self.bn_1.forward_t(&x, false)?;
        if self.bn_with_relu {
            x = x.relu()?;
        }
        Ok(x)
    }
}

pub struct WNLinear {
    linear: Linear,
}
impl WNLinear {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize, bias: bool) -> Result<Self> {
        let weight_g = vb.get((out_dim, 1), "weight_g")?;
        let weight_v = vb.get((out_dim, in_dim), "weight_v")?;

        let bias = if bias {
            vb.get(out_dim, "bias").ok()
        } else {
            None
        };
        let weight_norm = weight_v.sqr()?.sum_keepdim(0)?.sqrt()?.affine(1.0, 1e-8)?;
        let normalized_weight = weight_v.broadcast_div(&weight_norm)?;
        let scaled_weight = normalized_weight.broadcast_mul(&weight_g)?;
        let linear = Linear::new(scaled_weight, bias);
        Ok(Self { linear })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear.forward(x)?;
        Ok(x)
    }
}

pub fn mish(xs: &Tensor) -> Result<Tensor> {
    let tanh = xs.exp()?.affine(1.0, 1.0)?.log()?.tanh()?;
    let xs = xs.mul(&tanh)?;
    Ok(xs)
}

pub struct GPT2Attention {
    num_heads: usize,
    head_dim: usize,
    c_attn: Linear,
    c_proj: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl GPT2Attention {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
        let c_attn_weight = vb
            .get_with_hints(
                (hidden_size, 3 * hidden_size),
                "c_attn.weight",
                Init::Const(1.0),
            )?
            .t()?;
        let c_attn_bias = vb.get_with_hints(3 * hidden_size, "c_attn.bias", Init::Const(0.0))?;
        let c_attn = Linear::new(c_attn_weight, Some(c_attn_bias));
        // let c_attn = linear_b(3 * hidden_size, hidden_size, true, vb.pp("c_attn"))?;
        let c_proj_weight = vb
            .get_with_hints(
                (hidden_size, hidden_size),
                "c_proj.weight",
                Init::Const(1.0),
            )?
            .t()?;
        let c_proj_bias = vb.get_with_hints(hidden_size, "c_proj.bias", Init::Const(0.0))?;
        let c_proj = Linear::new(c_proj_weight, Some(c_proj_bias));
        // let c_proj = linear_b(hidden_size, hidden_size, true, vb.pp("c_proj"))?;
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            c_attn,
            c_proj,
            kv_cache: None,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        let xs = self.c_attn.forward(xs)?;
        let xs_splits = xs.chunk(3, 2)?;
        let query_states = xs_splits[0]
            .as_ref()
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = xs_splits[1]
            .as_ref()
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = xs_splits[2]
            .as_ref()
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };

        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            None,
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b, seq_len, self.num_heads * self.head_dim))?;
        let attn_output = attn_output.apply(&self.c_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

pub struct GPT2MLP {
    linear1: Linear,
    linear2: Linear,
    act: Activation,
}

impl GPT2MLP {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        middle_dim: usize,
        out_dim: usize,
        act: Activation,
    ) -> Result<Self> {
        let c_fc_weight = vb
            .get_with_hints((in_dim, middle_dim), "c_fc.weight", Init::Const(1.0))?
            .t()?;
        let c_fc_bias = vb.get_with_hints(middle_dim, "c_fc.bias", Init::Const(0.0))?;
        let c_fc = Linear::new(c_fc_weight, Some(c_fc_bias));

        let c_proj_weight = vb
            .get_with_hints((middle_dim, out_dim), "c_proj.weight", Init::Const(1.0))?
            .t()?;
        let c_proj_bias = vb.get_with_hints(out_dim, "c_proj.bias", Init::Const(0.0))?;
        let c_proj = Linear::new(c_proj_weight, Some(c_proj_bias));

        Ok(Self {
            linear1: c_fc,
            linear2: c_proj,
            act,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs
            .apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)?;
        Ok(xs)
    }
}

pub struct GPT2Block {
    ln_1: LayerNorm,
    attn: GPT2Attention,
    ln_2: LayerNorm,
    mlp: GPT2MLP,
}

impl GPT2Block {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        inner_dim: Option<usize>,
    ) -> Result<Self> {
        let inner_dim = inner_dim.unwrap_or(4 * hidden_size);
        let ln_1 = get_layer_norm(vb.pp("ln_1"), 1e-5, hidden_size, true)?;
        let attn = GPT2Attention::new(vb.pp("attn"), hidden_size, num_heads)?;
        let ln_2 = get_layer_norm(vb.pp("ln_2"), 1e-5, hidden_size, true)?;
        let mlp = GPT2MLP::new(
            vb.pp("mlp"),
            hidden_size,
            inner_dim,
            hidden_size,
            Activation::NewGelu,
        )?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    pub fn forward(&mut self, xs: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.ln_1.forward(xs)?;
        let xs = self.attn.forward(&xs, attention_mask)?;
        let residual = xs.add(&residual)?;
        let xs = self.ln_2.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = xs.add(&residual)?;
        Ok(xs)
    }
    pub fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache()
    }
}

#[allow(unused)]
pub struct GPT2Model {
    wte: Embedding,
    // wpe: Embedding,
    h: Vec<GPT2Block>,
    ln_f: LayerNorm,
}

#[allow(unused)]
impl GPT2Model {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        num_hidden_layers: usize,
        wte_embeddings: &Tensor,
    ) -> Result<Self> {
        // let wte = embedding(vocab_size, hidden_size, vb.pp("wte"))?;
        let wte = Embedding::new(wte_embeddings.clone(), hidden_size);
        // let wpe = embedding(max_position_embeddings, hidden_size, vb.pp("wpe"))?;
        let vb_layers = vb.pp("h");
        let mut h = vec![];
        for i in 0..num_hidden_layers {
            let block = GPT2Block::new(vb_layers.pp(i), hidden_size, num_heads, None)?;
            h.push(block);
        }
        let ln_f = get_layer_norm(vb.pp("ln_f"), 1e-5, hidden_size, true)?;
        Ok(Self { wte, h, ln_f })
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    0,
                    xs.device(),
                )?)
            }
        };
        for block in &mut self.h {
            xs = block.forward(&xs, attention_mask.as_ref())?;
        }
        xs = self.ln_f.forward(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.h.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}
