use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{
    Activation, Conv2d, Embedding, LayerNorm, Linear, Module, RmsNorm, VarBuilder, embedding,
    linear, linear_no_bias, rms_norm,
};

use crate::{
    models::{
        common::{NaiveAttention, get_conv2d, get_layer_norm},
        qwen3::model::Qwen3DecoderLayer,
        qwen3_asr::{
            config::{
                Qwen3ASRAudioConfig, Qwen3ASRConfig, Qwen3ASRTextConfig, ThinkerConfig,
                qwen3asr_text_config2qwen3_config,
            },
            processor::get_feat_extract_output_lengths,
        },
    },
    position_embed::{
        rope::Qwen3VLTextRotaryEmbedding, sinusoidal_pe::SinusoidalPositionEncoderCat,
    },
    utils::tensor_utils::{
        get_equal_mask, masked_scatter_dim0, prepare_causal_attention_mask, split_tensor,
        split_tensor_with_size,
    },
};

pub struct Qwen3ASRAudioEncoderLayer {
    self_attn: NaiveAttention,
    self_attn_layer_norm: LayerNorm,
    activation_fn: Activation,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl Qwen3ASRAudioEncoderLayer {
    pub fn new(vb: VarBuilder, config: &Qwen3ASRAudioConfig) -> Result<Self> {
        let self_attn = NaiveAttention::new(
            vb.pp("self_attn"),
            config.d_model,
            config.encoder_attention_heads,
            config.encoder_attention_heads,
            None,
            true,
            Some("q_proj"),
            Some("k_proj"),
            Some("v_proj"),
            Some("out_proj"),
        )?;
        let self_attn_layer_norm =
            get_layer_norm(vb.pp("self_attn_layer_norm"), 1e-5, config.d_model, true)?;
        let activation_fn = config.activation_function;
        let fc1 = linear(config.d_model, config.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(config.encoder_ffn_dim, config.d_model, vb.pp("fc2"))?;
        let final_layer_norm =
            get_layer_norm(vb.pp("final_layer_norm"), 1e-5, config.d_model, true)?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            activation_fn,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.self_attn_layer_norm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, None, None, mask, false)?;
        let residual = xs.add(&residual)?;
        let xs = self.final_layer_norm.forward(&residual)?;
        let xs = self.fc1.forward(&xs)?.apply(&self.activation_fn)?;
        let xs = self.fc2.forward(&xs)?;
        let xs = xs.add(&residual)?;
        Ok(xs)
    }
}

pub struct Qwen3ASRAudioEncoder {
    n_window: usize,
    positional_embedding: SinusoidalPositionEncoderCat,
    layers: Vec<Qwen3ASRAudioEncoderLayer>,
    ln_post: LayerNorm,
    conv2d1: Conv2d,
    conv2d2: Conv2d,
    conv2d3: Conv2d,
    conv_out: Linear,
    proj1: Linear,
    act: Activation,
    proj2: Linear,
    // n_window_infer: usize,
    conv_chunksize: usize,
}

impl Qwen3ASRAudioEncoder {
    pub fn new(vb: VarBuilder, config: &Qwen3ASRAudioConfig) -> Result<Self> {
        let n_window = config.n_window;
        let positional_embedding =
            SinusoidalPositionEncoderCat::new(Some(config.d_model), true, vb.device())?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..config.encoder_layers {
            let layer = Qwen3ASRAudioEncoderLayer::new(vb_layers.pp(i), config)?;
            layers.push(layer);
        }
        let ln_post = get_layer_norm(vb.pp("ln_post"), 1e-5, config.d_model, true)?;
        let conv2d1 = get_conv2d(
            vb.pp("conv2d1"),
            1,
            config.downsample_hidden_size,
            3,
            1,
            2,
            1,
            1,
            true,
        )?;
        let conv2d2 = get_conv2d(
            vb.pp("conv2d2"),
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            1,
            2,
            1,
            1,
            true,
        )?;
        let conv2d3 = get_conv2d(
            vb.pp("conv2d3"),
            config.downsample_hidden_size,
            config.downsample_hidden_size,
            3,
            1,
            2,
            1,
            1,
            true,
        )?;
        let in_dim =
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) / 2 + 1) / 2 + 1) / 2);
        let conv_out = linear_no_bias(in_dim, config.d_model, vb.pp("conv_out"))?;
        let proj1 = linear(config.d_model, config.d_model, vb.pp("proj1"))?;
        let act = config.activation_function;
        let proj2 = linear(config.d_model, config.output_dim, vb.pp("proj2"))?;
        // let n_window_infer = config.n_window_infer;
        let conv_chunksize = config.conv_chunksize;
        Ok(Self {
            n_window,
            positional_embedding,
            layers,
            ln_post,
            conv2d1,
            conv2d2,
            conv2d3,
            conv_out,
            proj1,
            act,
            proj2,
            // n_window_infer,
            conv_chunksize,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: (feature_dim, feature_len)
        let feature_lens = xs.dim(1)?;
        // let aftercnn_lens = get_feat_extract_output_lengths(feature_lens);
        let chunk_num = feature_lens / (self.n_window * 2);
        let mut chunk_lengths = vec![self.n_window * 2; chunk_num];
        let last = feature_lens % (self.n_window * 2);
        if last > 0 {
            chunk_lengths.push(last);
        }
        let mut chunk_list = split_tensor(&xs.t()?, &chunk_lengths, 0)?;
        if last > 0 {
            let chunk_last = chunk_list
                .pop()
                .ok_or(anyhow::anyhow!(format!("chunk_list is empty")))?;
            let pad_size = self.n_window * 2 - last;
            let chunk_last = chunk_last.pad_with_zeros(0, 0, pad_size)?;
            chunk_list.push(chunk_last);
        }
        let padded_feature = Tensor::stack(&chunk_list, 0)?.transpose(1, 2)?;
        let feature_lens_after_cnn: Vec<usize> = chunk_lengths
            .iter()
            .map(|&i| get_feat_extract_output_lengths(i))
            .collect();
        let feature_len_after_cnn = feature_lens_after_cnn.iter().sum();
        let padded_feature = padded_feature.unsqueeze(1)?;
        let mut padded_embeds = vec![];
        let feature_splits = split_tensor_with_size(&padded_feature, self.conv_chunksize, 0)?;
        for chunk in feature_splits.iter() {
            let padded_embed = self.conv2d1.forward(chunk)?.gelu()?;
            let padded_embed = self.conv2d2.forward(&padded_embed)?.gelu()?;
            let padded_embed = self.conv2d3.forward(&padded_embed)?.gelu()?;
            padded_embeds.push(padded_embed);
        }
        let padded_embed = Tensor::cat(&padded_embeds, 0)?;
        let (b, c, f, t) = padded_embed.dims4()?;
        let padded_embed =
            padded_embed
                .permute((0, 3, 1, 2))?
                .contiguous()?
                .reshape((b, t, c * f))?;
        let padded_embed = self.conv_out.forward(&padded_embed)?;
        let padded_embed = self.positional_embedding.forward(&padded_embed, 0)?;
        let padded_embed = padded_embed.flatten(0, 1)?;
        let mut hidden_states = padded_embed
            .narrow(0, 0, feature_len_after_cnn)?
            .unsqueeze(0)?;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, None)?;
        }
        let hidden_states = hidden_states.squeeze(0)?;
        let hidden_states = self.ln_post.forward(&hidden_states)?;
        let hidden_states = self.proj1.forward(&hidden_states)?.apply(&self.act)?;
        let hidden_states = self.proj2.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

pub struct Qwen3ASRThinkerTextModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    rotary_emb: Qwen3VLTextRotaryEmbedding,
    mrope_section: Vec<usize>,
}

impl Qwen3ASRThinkerTextModel {
    pub fn new(vb: VarBuilder, cfg: &Qwen3ASRTextConfig) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        let qwen3cfg = qwen3asr_text_config2qwen3_config(cfg);
        for i in 0..cfg.num_hidden_layers {
            let layer = Qwen3DecoderLayer::new(&qwen3cfg, vb_layers.pp(i))?;
            layers.push(layer);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = Qwen3VLTextRotaryEmbedding::new(cfg.head_dim, cfg.rope_theta);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            mrope_section: cfg.rope_scaling.mrope_section.clone(),
        })
    }

    pub fn forward(
        &mut self,
        input_embeds: &Tensor,
        seqlen_offset: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = input_embeds.dims3()?;
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(
                seqlen_offset as u32,
                (seq_len + seqlen_offset) as u32,
                input_embeds.device(),
            )?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((3, b_size, seq_len))?,
        };
        let (cos, sin) = self.rotary_emb.forward_asr(
            &position_ids,
            input_embeds.dtype(),
            self.mrope_section.clone(),
        )?;
        let mut xs = input_embeds.clone();
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    0,
                    input_embeds.device(),
                )?)
            }
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
        }
        let xs = self.norm.forward(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

pub struct Qwen3ASRThinker {
    audio_tower: Qwen3ASRAudioEncoder,
    model: Qwen3ASRThinkerTextModel,
    audio_token_id: u32,
    lm_head: Linear,
}

impl Qwen3ASRThinker {
    pub fn new(vb: VarBuilder, config: &ThinkerConfig) -> Result<Self> {
        let audio_tower = Qwen3ASRAudioEncoder::new(vb.pp("audio_tower"), &config.audio_config)?;
        let model = Qwen3ASRThinkerTextModel::new(vb.pp("model"), &config.text_config)?;
        let lm_head = if config.text_config.tie_word_embeddings {
            Linear::new(model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(
                config.text_config.hidden_size,
                config.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };
        Ok(Self {
            audio_tower,
            model,
            audio_token_id: config.audio_token_id,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        input_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut input_embeds = self.model.embed_tokens.forward(input_ids)?;
        if let Some(input_features) = input_features {
            let audio_feature = self.audio_tower.forward(input_features)?;
            // println!("audio_feature: {}", audio_feature);
            let audio_mask = get_equal_mask(input_ids, self.audio_token_id)?;
            let n_audio_tokens = audio_mask.sum_all()?.to_scalar::<u32>()?;
            if n_audio_tokens as usize != audio_feature.dim(0)? {
                return Err(anyhow::anyhow!(format!(
                    "n_audio_tokens num: {} not equal to audio_feature len: {}",
                    n_audio_tokens,
                    audio_feature.dim(0)?
                )));
            }
            input_embeds = masked_scatter_dim0(&input_embeds, &audio_feature, &audio_mask)?;
        }
        let outputs = self.model.forward(&input_embeds, seqlen_offset, None)?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }
    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}

pub struct Qwen3ASRModel {
    thinker: Qwen3ASRThinker,
}

impl Qwen3ASRModel {
    pub fn new(vb: VarBuilder, config: &Qwen3ASRConfig) -> Result<Self> {
        let thinker = Qwen3ASRThinker::new(vb.pp("thinker"), &config.thinker_config)?;
        Ok(Self { thinker })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        input_features: Option<&Tensor>,
    ) -> Result<Tensor> {
        let logits = self
            .thinker
            .forward(input_ids, seqlen_offset, input_features)?;
        Ok(logits)
    }
    pub fn clear_kv_cache(&mut self) {
        self.thinker.clear_kv_cache();
    }
}
