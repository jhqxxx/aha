use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{
    Activation, Embedding, Init, LayerNorm, Linear, Module, VarBuilder, embedding, linear_b,
};

use crate::{
    models::common::modules::{TwoLinearMLP, eager_attention_forward, get_layer_norm},
    position_embed::rope::{RoPE, apply_rotary_pos_emb_interleave},
    utils::tensor_utils::prepare_causal_attention_mask,
};
pub mod config;

pub struct GPT2Attention {
    num_heads: usize,
    head_dim: usize,
    c_attn: Linear,
    c_proj: Linear,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl GPT2Attention {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
        let c_attn = linear_b(hidden_size, 3 * hidden_size, true, vb.pp("c_attn"))?;
        let c_proj = linear_b(hidden_size, hidden_size, true, vb.pp("c_proj"))?;
        let head_dim = hidden_size / num_heads;
        Ok(Self {
            num_heads,
            head_dim,
            c_attn,
            c_proj,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
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
        let (query_states, key_states) = if let Some(cos) = cos
            && let Some(sin) = sin
        {
            apply_rotary_pos_emb_interleave(&query_states, &key_states, cos, sin, false)?
        } else {
            (query_states, key_states)
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
        let scale: f64 = 1f64 / f64::sqrt(self.head_dim as f64);
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
    mlp: TwoLinearMLP,
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
        let mlp = TwoLinearMLP::new(
            vb.pp("mlp"),
            hidden_size,
            inner_dim,
            hidden_size,
            Activation::NewGelu,
            true,
            "fc_in",
            "fc_out",
        )?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
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
        let xs = self.ln_1.forward(xs)?;
        let xs = self.attn.forward(&xs, cos, sin, attention_mask)?;
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
    pub wte: Option<Embedding>,
    // wpe: Embedding, //rope not need
    h: Vec<GPT2Block>,
    ln_f: LayerNorm,
    rope: Option<RoPE>,
}

#[allow(unused)]
impl GPT2Model {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        num_hidden_layers: usize,
        vocab_size: usize,
        // n_positions: usize,
    ) -> Result<Self> {
        let wte = Some(embedding(vocab_size, hidden_size, vb.pp("wte"))?);
        let vb_layers = vb.pp("h");
        let mut h = vec![];
        for i in 0..num_hidden_layers {
            let block = GPT2Block::new(vb_layers.pp(i), hidden_size, num_heads, None)?;
            h.push(block);
        }
        let ln_f = get_layer_norm(vb.pp("ln_f"), 1e-5, hidden_size, true)?;
        let head_dim = hidden_size / num_heads;
        let rope = RoPE::new(head_dim, 10000.0, vb.device())?;
        Ok(Self {
            wte,
            h,
            ln_f,
            rope: Some(rope),
        })
    }

    pub fn new_without_wte(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        num_hidden_layers: usize,
        vocab_size: usize,
        // n_positions: usize,
    ) -> Result<Self> {
        let wte = None;
        let vb_layers = vb.pp("h");
        let mut h = vec![];
        for i in 0..num_hidden_layers {
            let block = GPT2Block::new(vb_layers.pp(i), hidden_size, num_heads, None)?;
            h.push(block);
        }
        let ln_f = get_layer_norm(vb.pp("ln_f"), 1e-5, hidden_size, true)?;
        let head_dim = hidden_size / num_heads;
        let rope = RoPE::new(head_dim, 10000.0, vb.device())?;
        Ok(Self {
            wte,
            h,
            ln_f,
            rope: Some(rope),
        })
    }

    pub fn new_with_wte(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        num_hidden_layers: usize,
        wte_embeddings: &Tensor,
    ) -> Result<Self> {
        let wte = Some(Embedding::new(wte_embeddings.clone(), hidden_size));
        let vb_layers = vb.pp("h");
        let mut h = vec![];
        for i in 0..num_hidden_layers {
            let block = GPT2Block::new(vb_layers.pp(i), hidden_size, num_heads, None)?;
            h.push(block);
        }
        let ln_f = get_layer_norm(vb.pp("ln_f"), 1e-5, hidden_size, true)?;
        Ok(Self {
            wte,
            h,
            ln_f,
            rope: None,
        })
    }

    pub fn forward(&mut self, inputs_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;
        let (cos, sin) = if let Some(rope) = &self.rope {
            let (cos, sin) =
                rope.forward_repeat_interleave(seqlen_offset, seq_len, inputs_embeds.device())?;
            (Some(cos), Some(sin))
        } else {
            (None, None)
        };

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
            xs = block.forward(&xs, cos.as_ref(), sin.as_ref(), attention_mask.as_ref())?;
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
