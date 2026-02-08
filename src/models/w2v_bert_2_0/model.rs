use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Activation, Conv1d, Embedding, Init, LayerNorm, Linear, Module, VarBuilder, embedding, linear,
    linear_b,
};

use crate::{
    models::{
        common::{GLU, TwoLinearMLP, eager_attention_forward, get_conv1d, get_layer_norm},
        w2v_bert_2_0::config::W2VBert2_0Config,
    },
    position_embed::rope::{RoPE, apply_rotary_pos_emb},
    utils::{find_type_files, tensor_utils::masked_fill_zeros},
};

pub struct Wav2Vec2BertFeatureProjection {
    layer_norm: LayerNorm,
    projection: Linear,
}

impl Wav2Vec2BertFeatureProjection {
    pub fn new(vb: VarBuilder, config: &W2VBert2_0Config) -> Result<Self> {
        let layer_norm = get_layer_norm(
            vb.pp("layer_norm"),
            config.layer_norm_eps,
            config.feature_projection_input_dim,
        )?;
        let projection = linear(
            config.feature_projection_input_dim,
            config.hidden_size,
            vb.pp("projection"),
        )?;
        Ok(Self {
            layer_norm,
            projection,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let norm_xs = self.layer_norm.forward(xs)?;
        let xs = self.projection.forward(&norm_xs)?;
        Ok((xs, norm_xs))
    }
}

pub struct Wav2Vec2BertSelfAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    head_dim: usize,
    num_heads: usize,
    position_embeddings_type: Option<String>,
    linear_pos: Option<Linear>,
    pos_bias_u: Option<Tensor>,
    pos_bias_v: Option<Tensor>,
    left_max_position_embeddings: usize,
    right_max_position_embeddings: usize,
    distance_embedding: Option<Embedding>,
}

impl Wav2Vec2BertSelfAttention {
    pub fn new(
        vb: VarBuilder,
        config: &W2VBert2_0Config,
        is_adapter_attention: bool,
    ) -> Result<Self> {
        let hidden_size = if is_adapter_attention {
            config.hidden_size
        } else {
            config.output_hidden_size
        };
        let head_dim = hidden_size / config.num_attention_heads;
        let num_heads = config.num_attention_heads;
        let left_max_position_embeddings = config.left_max_position_embeddings;
        let right_max_position_embeddings = config.right_max_position_embeddings;
        let position_embeddings_type = if !is_adapter_attention {
            Some(config.position_embeddings_type.clone())
        } else {
            None
        };
        let (linear_pos, pos_bias_u, pos_bias_v, distance_embedding) =
            if let Some(pos_type) = &position_embeddings_type {
                if pos_type.eq("relative") {
                    let linear_pos = Some(linear_b(
                        hidden_size,
                        hidden_size,
                        false,
                        vb.pp("linear_pos"),
                    )?);
                    let pos_bias_u = Some(vb.get_with_hints(
                        (config.num_attention_heads, head_dim),
                        "pos_bias_u",
                        Init::Const(0.),
                    )?);
                    let pos_bias_v = Some(vb.get_with_hints(
                        (config.num_attention_heads, head_dim),
                        "pos_bias_v",
                        Init::Const(0.),
                    )?);
                    (linear_pos, pos_bias_u, pos_bias_v, None)
                } else if pos_type.eq("relative_key") {
                    let num_positions =
                        left_max_position_embeddings + right_max_position_embeddings + 1;
                    let distance_embedding = Some(embedding(
                        num_positions,
                        head_dim,
                        vb.pp("distance_embedding"),
                    )?);
                    (None, None, None, distance_embedding)
                } else {
                    (None, None, None, None)
                }
            } else {
                (None, None, None, None)
            };
        let q_proj = linear_b(hidden_size, hidden_size, true, vb.pp("linear_q"))?;
        let k_proj = linear_b(hidden_size, hidden_size, true, vb.pp("linear_k"))?;
        let v_proj = linear_b(hidden_size, hidden_size, true, vb.pp("linear_v"))?;
        let o_proj = linear_b(hidden_size, hidden_size, true, vb.pp("linear_out"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            head_dim,
            num_heads,
            position_embeddings_type,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            left_max_position_embeddings,
            right_max_position_embeddings,
            distance_embedding,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        if let Some(pos_type) = &self.position_embeddings_type
            && pos_type.eq("rotary")
            && (cos.is_none() || sin.is_none())
        {
            return Err(anyhow::anyhow!(
                "rotary type position cos and sin can not be none"
            ));
        }

        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) = if let Some(cos) = cos
            && let Some(sin) = sin
        {
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?
        } else {
            (query_states, key_states)
        };
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attention_mask = if let Some(pos_type) = &self.position_embeddings_type
            && pos_type.eq("relative_key")
            && let Some(embed) = &self.distance_embedding
        {
            let query_length = query_states.dim(2)?;
            let key_length = key_states.dim(2)?;
            let position_ids_l =
                Tensor::arange(0i64, query_length as i64, xs.device())?.unsqueeze(D::Minus1)?;
            let position_ids_r =
                Tensor::arange(0i64, key_length as i64, xs.device())?.unsqueeze(0)?;
            let distance = position_ids_r.broadcast_sub(&position_ids_l)?;
            let distance = distance.clamp(
                -(self.left_max_position_embeddings as i64),
                self.right_max_position_embeddings as i64,
            )?;
            let distance = distance
                .affine(1.0, self.left_max_position_embeddings as f64)?
                .to_dtype(candle_core::DType::U32)?;
            let pos_emb = embed.forward(&distance)?.to_dtype(query_states.dtype())?; // (seq_q, seq_k, dim)
            let query_ = query_states.unsqueeze(D::Minus2)?; // (b, n_head, seq_q, 1, dim)
            let pos_emb = pos_emb.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, se_q, seq_k, dim)
            // torch.einsum("bhld,lrd->bhlr", query, positional_embedding)
            // (bs, n_head, seq_len, seq_len)
            let relative_position_attn_weights = query_
                .broadcast_mul(&pos_emb)?
                .sum(D::Minus1)?
                .affine(scale, 0.0)?;
            if let Some(mask) = attention_mask {
                // let mask = mask.unsqueeze(1)?.unsqueeze(D::Minus1)?;
                Some(relative_position_attn_weights.broadcast_add(&mask)?)
            } else {
                Some(relative_position_attn_weights)
            }
        } else {
            if let Some(mask) = attention_mask {
                // let mask = mask.unsqueeze(1)?.unsqueeze(D::Minus1)?;
                Some(mask.clone())
            } else {
                None
            }
        };

        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            None,
            attention_mask.as_ref(),
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }
}

pub struct Wav2Vec2BertConvolutionModule {
    layer_norm: LayerNorm,
    pointwise_conv1: Conv1d,
    glu: GLU,
    conv_depthwise_kernel_size: usize,
    depthwise_conv: Conv1d,
    depthwise_layer_norm: LayerNorm,
    act: Activation,
    pointwise_conv2: Conv1d,
}

impl Wav2Vec2BertConvolutionModule {
    pub fn new(vb: VarBuilder, config: &W2VBert2_0Config) -> Result<Self> {
        let layer_norm = get_layer_norm(
            vb.pp("layer_norm"),
            config.layer_norm_eps,
            config.hidden_size,
        )?;
        let pointwise_conv1 = get_conv1d(
            vb.pp("pointwise_conv1"),
            config.hidden_size,
            2 * config.hidden_size,
            1,
            0,
            1,
            1,
            1,
            false,
        )?;
        let glu = GLU::new(1)?;
        let conv_depthwise_kernel_size = config.conv_depthwise_kernel_size;
        let depthwise_conv = get_conv1d(
            vb.pp("depthwise_conv"),
            config.hidden_size,
            config.hidden_size,
            conv_depthwise_kernel_size,
            0,
            1,
            1,
            config.hidden_size,
            false,
        )?;
        let depthwise_layer_norm = get_layer_norm(
            vb.pp("depthwise_layer_norm"),
            config.layer_norm_eps,
            config.hidden_size,
        )?;
        let pointwise_conv2 = get_conv1d(
            vb.pp("pointwise_conv2"),
            config.hidden_size,
            config.hidden_size,
            1,
            0,
            1,
            1,
            1,
            false,
        )?;
        Ok(Self {
            layer_norm,
            pointwise_conv1,
            glu,
            conv_depthwise_kernel_size,
            depthwise_conv,
            depthwise_layer_norm,
            act: config.hidden_act,
            pointwise_conv2,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = self.layer_norm.forward(xs)?;
        if let Some(mask) = mask {
            xs = masked_fill_zeros(&xs, mask)?;
        }
        let xs = xs.transpose(1, 2)?;
        // (batch, 2*channel, dim)
        let xs = self.pointwise_conv1.forward(&xs)?;
        // (batch, channel, dim)
        let xs = self.glu.forward(&xs)?;
        let xs = xs.pad_with_zeros(D::Minus1, self.conv_depthwise_kernel_size - 1, 0)?;
        let xs = self.depthwise_conv.forward(&xs)?;
        let xs = self
            .depthwise_layer_norm
            .forward(&xs.transpose(1, 2)?)?
            .transpose(1, 2)?;
        let xs = xs.apply(&self.act)?;
        let xs = self.pointwise_conv2.forward(&xs)?;
        let xs = xs.transpose(1, 2)?;
        Ok(xs)
    }
}

pub struct Wav2Vec2BertEncoderLayer {
    ffn1_layer_norm: LayerNorm,
    ffn1: TwoLinearMLP,
    self_attn_layer_norm: LayerNorm,
    self_attn: Wav2Vec2BertSelfAttention,
    conv_module: Wav2Vec2BertConvolutionModule,
    ffn2_layer_norm: LayerNorm,
    ffn2: TwoLinearMLP,
    final_layer_norm: LayerNorm,
}

impl Wav2Vec2BertEncoderLayer {
    pub fn new(vb: VarBuilder, config: &W2VBert2_0Config) -> Result<Self> {
        let ffn1_layer_norm = get_layer_norm(
            vb.pp("ffn1_layer_norm"),
            config.layer_norm_eps,
            config.hidden_size,
        )?;
        let ffn1 = TwoLinearMLP::new(
            vb.pp("ffn1"),
            config.hidden_size,
            config.intermediate_size,
            config.hidden_size,
            config.hidden_act,
            true,
            "intermediate_dense",
            "output_dense",
        )?;
        let self_attn_layer_norm = get_layer_norm(
            vb.pp("self_attn_layer_norm"),
            config.layer_norm_eps,
            config.hidden_size,
        )?;
        let self_attn = Wav2Vec2BertSelfAttention::new(vb.pp("self_attn"), config, false)?;
        let conv_module = Wav2Vec2BertConvolutionModule::new(vb.pp("conv_module"), config)?;
        let ffn2_layer_norm = get_layer_norm(
            vb.pp("ffn2_layer_norm"),
            config.layer_norm_eps,
            config.hidden_size,
        )?;
        let ffn2 = TwoLinearMLP::new(
            vb.pp("ffn2"),
            config.hidden_size,
            config.intermediate_size,
            config.hidden_size,
            config.hidden_act,
            true,
            "intermediate_dense",
            "output_dense",
        )?;
        let final_layer_norm = get_layer_norm(
            vb.pp("final_layer_norm"),
            config.layer_norm_eps,
            config.hidden_size,
        )?;
        Ok(Self {
            ffn1_layer_norm,
            ffn1,
            self_attn_layer_norm,
            self_attn,
            conv_module,
            ffn2_layer_norm,
            ffn2,
            final_layer_norm,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        conv_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.ffn1_layer_norm.forward(xs)?;
        let xs = self.ffn1.forward(&xs)?;
        let residual = xs.affine(0.5, 0.0)?.add(&residual)?;
        let xs = self.self_attn_layer_norm.forward(&residual)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        let residual = xs.add(&residual)?;
        let xs = self.conv_module.forward(&residual, conv_attention_mask)?;
        let residual = xs.add(&residual)?;
        let xs = self.ffn2_layer_norm.forward(&residual)?;
        let xs = self.ffn2.forward(&xs)?;
        let xs = xs.affine(0.5, 0.0)?.add(&residual)?;
        let xs = self.final_layer_norm.forward(&xs)?;
        Ok(xs)
    }
}

pub struct ModelOutput {
    pub last_hidden_state: Tensor,
    pub specify_layer_id_hidden_state: Option<Tensor>,
    pub hidden_states: Option<Vec<Tensor>>,
}

pub struct Wav2Vec2BertEncoder {
    embed_positions: Option<RoPE>,
    layers: Vec<Wav2Vec2BertEncoderLayer>,
}

impl Wav2Vec2BertEncoder {
    pub fn new(vb: VarBuilder, config: &W2VBert2_0Config) -> Result<Self> {
        let embed_positions = if config.position_embeddings_type.eq("rotary") {
            let dim = config.hidden_size / config.num_attention_heads;
            let embed_positions = RoPE::new(dim, 10000.0, vb.device())?;
            Some(embed_positions)
        } else {
            None
        };
        let vb_layers = vb.pp("layers");
        let mut layers = vec![];
        for i in 0..config.num_hidden_layers {
            let layer = Wav2Vec2BertEncoderLayer::new(vb_layers.pp(i), config)?;
            layers.push(layer);
        }
        Ok(Self {
            embed_positions,
            layers,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        layer_id: Option<usize>,
        output_hidden_states: bool,
    ) -> Result<ModelOutput> {
        // xs: (bs, seq_len ,dim)
        // attention_mask: Some: (bs, seq_len)
        let (_, seq_len, _) = xs.dims3()?;
        let conv_attention_mask = attention_mask;
        let (mut xs, attention_mask) = if let Some(mask) = attention_mask {
            let xs = masked_fill_zeros(xs, mask)?;
            // (bs, 1, 1, seq_len)
            let attention_mask = mask.unsqueeze(1)?.unsqueeze(1)?;
            let neg_inf_t = attention_mask
                .zeros_like()?
                .to_dtype(xs.dtype())?
                .affine(1.0, f64::NEG_INFINITY)?;
            let attention_mask_f = attention_mask.to_dtype(xs.dtype())?;
            let attention_mask = attention_mask
                .where_cond(&attention_mask_f, &neg_inf_t)?
                .to_dtype(xs.dtype())?
                .affine(1.0, -1.0)?;
            (xs, Some(attention_mask))
        } else {
            (xs.clone(), None)
        };
        let (cos, sin) = if let Some(embed_posi) = &self.embed_positions {
            let (cos, sin) = embed_posi.forward(0, seq_len, xs.device())?;
            (Some(cos), Some(sin))
        } else {
            (None, None)
        };
        let mut hidden_states: Vec<Tensor> = vec![];
        let mut specify_layer_id_hidden_state = None;

        for (i, layer) in (&self.layers).iter().enumerate() {
            if output_hidden_states {
                hidden_states.push(xs.clone());
            }
            if let Some(id) = layer_id
                && id == i
            {
                specify_layer_id_hidden_state = Some(xs.clone());
            }
            xs = layer.forward(
                &xs,
                cos.as_ref(),
                sin.as_ref(),
                attention_mask.as_ref(),
                conv_attention_mask,
            )?;
        }
        let hidden_states = if hidden_states.len() > 0 {
            Some(hidden_states)
        } else {
            None
        };
        Ok(ModelOutput {
            last_hidden_state: xs,
            specify_layer_id_hidden_state,
            hidden_states,
        })
    }
}

pub struct W2VBert2_0Model {
    config: W2VBert2_0Config,
    feature_projection: Wav2Vec2BertFeatureProjection,
    masked_spec_embed: Option<Tensor>,
    encoder: Wav2Vec2BertEncoder,
    // config.add_adapter is false, adapter is None, Wav2Vec2BertAdapter not complish
    // adapter: Option<Wav2Vec2BertAdapter>,
    // config.use_intermediate_ffn_before_adapter is false, intermediate_ffn is None
    // intermediate_ffn: Option<Wav2Vec2BertFeedForward>,
}

impl W2VBert2_0Model {
    pub fn init(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let config_path = path.to_string() + "/config.json";
        let config: W2VBert2_0Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
        W2VBert2_0Model::new(vb, &config)
    }

    pub fn new(vb: VarBuilder, config: &W2VBert2_0Config) -> Result<Self> {
        let feature_projection =
            Wav2Vec2BertFeatureProjection::new(vb.pp("feature_projection"), config)?;
        let masked_spec_embed = if config.mask_time_prob > 0.0 || config.mask_time_prob > 0.0 {
            Some(
                vb.get_with_hints(config.hidden_size, "masked_spec_embed", Init::Uniform {
                    lo: 0.0,
                    up: 1.0,
                })?,
            )
        } else {
            None
        };
        let encoder = Wav2Vec2BertEncoder::new(vb.pp("encoder"), config)?;
        Ok(Self {
            config: config.clone(),
            feature_projection,
            masked_spec_embed,
            encoder,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        layer_id: Option<usize>,
        output_hidden_states: bool,
    ) -> Result<ModelOutput> {
        let (xs, _) = self.feature_projection.forward(xs)?;
        self.encoder
            .forward(&xs, attention_mask, layer_id, output_hidden_states)
    }
}
