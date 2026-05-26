use anyhow::Result;
use candle_core::{IndexOp, Tensor};
use candle_nn::{Conv1d, LayerNorm, Linear, Module, VarBuilder, linear, linear_no_bias};

use crate::{
    models::{
        common::{
            InferenceModel,
            modules::{TwoLinearMLP, eager_attention_forward, get_conv1d, get_layer_norm},
        },
        glm_asr_nano::config::{GlmAsrAudioConfig, GlmAsrNanoConfig},
        llama::LlamaForCausalLM,
    },
    position_embed::rope::{RoPE, glm_asr_apply_rotary_pos_emb},
    utils::tensor_utils::{get_equal_mask, masked_scatter_dim0},
};

#[derive(Debug, Clone)]
// pub struct AttentionNobias {
pub struct GlmAsrAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
}

impl GlmAsrAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: Option<usize>,
    ) -> Result<Self> {
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let head_dim = match head_dim {
            None => hidden_size / num_attention_heads,
            Some(dim) => dim,
        };
        let q_proj = linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear(num_attention_heads * head_dim, hidden_size, vb.pp("o_proj"))?;

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
            glm_asr_apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
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
}

pub struct GlmAsrEncoderLayer {
    self_attn: GlmAsrAttention,
    mlp: TwoLinearMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl GlmAsrEncoderLayer {
    pub fn new(vb: VarBuilder, audio_cfg: &GlmAsrAudioConfig) -> Result<Self> {
        let self_attn = GlmAsrAttention::new(
            vb.pp("self_attn"),
            audio_cfg.hidden_size,
            audio_cfg.num_attention_heads,
            audio_cfg.num_key_value_heads,
            Some(audio_cfg.head_dim),
        )?;
        let mlp = TwoLinearMLP::new(
            vb.pp("mlp"),
            audio_cfg.hidden_size,
            audio_cfg.intermediate_size,
            audio_cfg.hidden_size,
            audio_cfg.hidden_act,
            true,
            "fc1",
            "fc2",
        )?;
        let input_layernorm =
            get_layer_norm(vb.pp("input_layernorm"), 1e-5, audio_cfg.hidden_size, true)?;
        let post_attention_layernorm = get_layer_norm(
            vb.pp("post_attention_layernorm"),
            1e-5,
            audio_cfg.hidden_size,
            true,
        )?;
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

pub struct GlmAsrEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    layers: Vec<GlmAsrEncoderLayer>,
    norm: LayerNorm,
    rotary_emb: RoPE,
}

impl GlmAsrEncoder {
    pub fn new(vb: VarBuilder, audio_cfg: &GlmAsrAudioConfig) -> Result<Self> {
        let conv1 = get_conv1d(
            vb.pp("conv1"),
            audio_cfg.num_mel_bins,
            audio_cfg.hidden_size,
            3,
            1,
            1,
            1,
            1,
            true,
        )?;
        let conv2 = get_conv1d(
            vb.pp("conv2"),
            audio_cfg.hidden_size,
            audio_cfg.hidden_size,
            3,
            1,
            2,
            1,
            1,
            true,
        )?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..audio_cfg.num_hidden_layers {
            let layer_i = GlmAsrEncoderLayer::new(vb_layers.pp(i), audio_cfg)?;
            layers.push(layer_i);
        }
        let norm = get_layer_norm(vb.pp("norm"), 1e-5, audio_cfg.hidden_size, true)?;
        let dim = (audio_cfg.head_dim as f64 * audio_cfg.partial_rotary_factor) as usize;
        let rotary_emb = RoPE::new(dim, audio_cfg.rope_parameters.rope_theta, vb.device())?;
        Ok(Self {
            conv1,
            conv2,
            layers,
            norm,
            rotary_emb,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv1.forward(xs)?.gelu()?;
        let xs = self.conv2.forward(&xs)?.gelu()?;
        let mut xs = xs.transpose(1, 2)?;
        let (_, seq_len, _) = xs.dims3()?;
        let (cos, sin) = self.rotary_emb.forward(0, seq_len, xs.device())?;
        for encoder_layer in &self.layers {
            xs = encoder_layer.forward(&xs, Some(&cos), Some(&sin), None, false)?;
        }
        let xs = self.norm.forward(&xs)?;
        Ok(xs)
    }
}

pub struct GlmAsrNanoModel {
    config: GlmAsrNanoConfig,
    audio_tower: GlmAsrEncoder,
    multi_modal_projector: TwoLinearMLP,
    language_model: LlamaForCausalLM,
    stop_token_ids: Vec<u32>,
}

impl GlmAsrNanoModel {
    pub fn new(vb: VarBuilder, config: GlmAsrNanoConfig, eos_ids: Vec<u32>) -> Result<Self> {
        let audio_tower = GlmAsrEncoder::new(vb.pp("audio_tower"), &config.audio_config)?;
        let multi_modal_projector = TwoLinearMLP::new(
            vb.pp("multi_modal_projector"),
            config.audio_config.intermediate_size,
            config.text_config.hidden_size * 2,
            config.text_config.hidden_size,
            config.projector_hidden_act,
            true,
            "linear_1",
            "linear_2",
        )?;
        let language_model = LlamaForCausalLM::new(
            vb.pp("language_model"),
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            config.text_config.num_hidden_layers,
            config.text_config.num_attention_heads,
            Some(config.text_config.num_key_value_heads),
            Some(config.text_config.head_dim),
            config.text_config.attention_bias,
            "self_attn",
            Some("o_proj"),
            config.text_config.intermediate_size,
            config.text_config.hidden_act,
            config.text_config.mlp_bias,
            "mlp",
            config.text_config.rms_norm_eps,
            "input_layernorm",
            "post_attention_layernorm",
            config.text_config.rope_parameters.rope_theta,
            eos_ids.clone(),
        )?;
        Ok(Self {
            config,
            audio_tower,
            multi_modal_projector,
            language_model,
            stop_token_ids: eos_ids,
        })
    }

    pub fn get_audio_features(
        &self,
        input_features: &Tensor,
        audio_token_lengths: &[u32],
    ) -> Result<Tensor> {
        let audio_hidden_states = self.audio_tower.forward(input_features)?;
        let bs = audio_hidden_states.dim(0)?;
        let audio_hidden_states =
            audio_hidden_states.reshape((bs, (), self.config.audio_config.intermediate_size))?;
        let audio_embeds = self.multi_modal_projector.forward(&audio_hidden_states)?;
        let mut valid_audios = vec![];
        for (i, &len) in audio_token_lengths.iter().enumerate() {
            let len = len as usize;
            let audio_i = audio_embeds.i((i, 0..len, ..))?;
            valid_audios.push(audio_i);
        }
        let audio_embeds = Tensor::cat(&valid_audios, 0)?;

        Ok(audio_embeds)
    }

    pub fn forward(
        &mut self,
        input_features: Option<&Tensor>,
        audio_token_lengths: Option<&Tensor>,
        input_ids: &Tensor,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.language_model.model.embed_tokens.forward(input_ids)?;
        if let Some(input_features) = input_features
            && let Some(audio_token_len) = audio_token_lengths
        {
            let audio_token_len = audio_token_len.to_vec1::<u32>()?;
            let audio_token_mask = get_equal_mask(input_ids, self.config.audio_token_id)?;
            let audio_embeds = self.get_audio_features(input_features, &audio_token_len)?;
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &audio_embeds, &audio_token_mask)?;
        }
        let logits = self
            .language_model
            .forward_embeds(&inputs_embeds, seqlen_offset)?;
        Ok(logits)
    }
    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}

impl InferenceModel for GlmAsrNanoModel {
    fn forward_initial(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        data: crate::models::common::MultiModalData,
    ) -> Result<Tensor> {
        if data.data_vec.len() != 2 {
            return Err(anyhow::anyhow!(
                "GlmAsrNano process data error, must have input_features, audio_token_lengths"
            ));
        }
        let input_features = &data.data_vec[0];
        let audio_token_lengths = &data.data_vec[1];
        self.forward(
            input_features.as_ref(),
            audio_token_lengths.as_ref(),
            input_ids,
            seqlen_offset,
        )
    }

    fn forward_step(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        self.forward(None, None, input_ids, seqlen_offset)
    }

    fn clear_cache(&mut self) {
        self.clear_kv_cache();
    }

    fn stop_token_ids(&self) -> Vec<u32> {
        self.stop_token_ids.clone()
    }
}
