use crate::{
    models::common::{InferenceModel, modules::NaiveAttnGateUpDownMLPBlock},
    position_embed::rope::RoPE,
    utils::tensor_utils::prepare_causal_attention_mask,
};
use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{
    Activation, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};

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
    stop_token_ids: Vec<u32>,
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
        eos_ids: Vec<u32>,
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
        Ok(Self {
            model,
            lm_head,
            stop_token_ids: eos_ids,
        })
    }

    pub fn forward_embeds(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
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

impl InferenceModel for LlamaForCausalLM {
    fn forward_step(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let input_embeds = self.model.embed_tokens.forward(input_ids)?;
        self.forward_embeds(&input_embeds, seqlen_offset)
    }

    fn clear_cache(&mut self) {
        self.clear_kv_cache();
    }

    fn stop_token_ids(&self) -> Vec<u32> {
        self.stop_token_ids.clone()
    }
}
