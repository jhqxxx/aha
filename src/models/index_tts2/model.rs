use std::collections::HashMap;

use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Shape, Tensor, pickle::read_all_with_key};
use candle_nn::{
    Activation, Conv1d, Conv2d, Embedding, GroupNorm, Init, LayerNorm, Linear, Module, RmsNorm,
    VarBuilder, embedding, group_norm, linear, linear_b, ops::sigmoid, rms_norm,
};
use rand::Rng;

use crate::{
    models::{
        bigvgan::{BigVGAN, config::BigVGANConfig},
        campplus::CAMPPlus,
        common::{
            GEGLU, GLU, GPT2Model, GateUpDownMLP, QKVCatAttention, TwoLinearMLP, WNConv1d,
            WNLinear, eager_attention_forward, get_conv1d, get_conv2d, get_layer_norm,
            get_layer_norm_without_weight, mish,
        },
        feature_extractor::seamless_m4t_feature_extractor::SeamlessM4TFeatureExtractor,
        index_tts2::config::{
            DiTModelArgs, GptConfig, IndexTTS2Config, PreprocessParams, S2MelConfig,
        },
        mask_gct::model::RepCodec,
        w2v_bert_2_0::model::W2VBert2_0Model,
    },
    position_embed::rope::{RoPE, compute_default_rope_parameters},
    utils::{
        audio_utils::{
            create_hann_window, extract_audio_url, get_waveform_and_window_properties, kaldi_fbank,
            kaldi_get_mel_banks, load_audio, mel_filter_bank, resample_simple, torch_stft,
        },
        get_dtype, get_logit_processor, get_vb_model_path, load_tensor_from_pt,
        read_pth_tensor_info_cycle,
        tensor_utils::{
            cosine_similarity, interpolate_nearest_1d, l2_normalize, linspace, masked_fill_zeros,
            pad_reflect_last_dim, sequence_mask, split_tensor,
        },
    },
};
pub struct AdaptiveLayerNorm {
    project_layer: Linear,
    norm: RmsNorm,
    // d_model: usize,
}

impl AdaptiveLayerNorm {
    pub fn new(vb: VarBuilder, d_model: usize, eps: f64) -> Result<Self> {
        let project_layer = linear(d_model, d_model * 2, vb.pp("project_layer"))?;
        let norm = rms_norm(d_model, eps, vb.pp("norm"))?;
        Ok(Self {
            project_layer,
            norm,
            // d_model,
        })
    }

    pub fn forward(&self, xs: &Tensor, embedding: Option<&Tensor>) -> Result<Tensor> {
        if let Some(embedding) = embedding {
            let emb = self.project_layer.forward(embedding)?;
            // let emb_split = split_tensor_with_size(&emb, 2, D::Minus1)?;
            let emb_split = emb.chunk(2, D::Minus1)?;
            let weight = &emb_split[0];
            let bias = &emb_split[1];
            Ok(self
                .norm
                .forward(xs)?
                .broadcast_mul(weight)?
                .broadcast_add(bias)?)
        } else {
            Ok(self.norm.forward(xs)?)
        }
    }
}

pub struct DiTTransformerBlock {
    attention: QKVCatAttention,
    feed_forward: GateUpDownMLP,
    ffn_norm: AdaptiveLayerNorm,
    attention_norm: AdaptiveLayerNorm,
    skip_in_linear: Option<Linear>,
    uvit_skip_connection: bool,
    time_as_token: bool,
}

impl DiTTransformerBlock {
    pub fn new(vb: VarBuilder, config: &DiTModelArgs) -> Result<Self> {
        let attention = QKVCatAttention::new(
            vb.pp("attention"),
            config.dim,
            config.n_head,
            Some(config.head_dim),
            false,
            Some("wqkv"),
            Some("wo"),
        )?;

        let feed_forward = GateUpDownMLP::new(
            vb.pp("feed_forward"),
            config.dim,
            config.intermediate_size,
            candle_nn::Activation::Silu,
            false,
            Some("w1"),
            Some("w3"),
            Some("w2"),
        )?;
        let ffn_norm = AdaptiveLayerNorm::new(vb.pp("ffn_norm"), config.dim, config.norm_eps)?;
        let attention_norm =
            AdaptiveLayerNorm::new(vb.pp("attention_norm"), config.dim, config.norm_eps)?;
        let (skip_in_linear, uvit_skip_connection) = if config.uvit_skip_connection {
            let skip_in_linear = linear(config.dim * 2, config.dim, vb.pp("skip_in_linear"))?;
            (Some(skip_in_linear), config.uvit_skip_connection)
        } else {
            (None, false)
        };
        Ok(Self {
            attention,
            feed_forward,
            ffn_norm,
            attention_norm,
            skip_in_linear,
            uvit_skip_connection,
            time_as_token: config.time_as_token,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        c: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        mask: Option<&Tensor>,
        skip_in_x: Option<&Tensor>,
    ) -> Result<Tensor> {
        let c = if self.time_as_token { None } else { Some(c) };
        let mut xs = xs.clone();
        if self.uvit_skip_connection
            && let Some(skip_in_x) = skip_in_x
            && let Some(skip_in_linear) = &self.skip_in_linear
        {
            let cat = Tensor::cat(&[&xs, skip_in_x], D::Minus1)?;
            xs = skip_in_linear.forward(&cat)?;
        }
        let xs = self
            .attention
            .forward(
                &self.attention_norm.forward(&xs, c)?,
                cos,
                sin,
                mask,
                false,
                true,
            )?
            .add(&xs)?;
        let out = self
            .feed_forward
            .forward(&self.ffn_norm.forward(&xs, c)?)?
            .add(&xs)?;
        Ok(out)
    }
}

pub struct DiTTransformer {
    layers: Vec<DiTTransformerBlock>,
    norm: AdaptiveLayerNorm,
    rope: RoPE,
    uvit_skip_connection: bool,
    layers_emit_skip: Vec<usize>,
    layers_receive_skip: Vec<usize>,
}

impl DiTTransformer {
    pub fn new(vb: VarBuilder, config: &DiTModelArgs) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        let mut layers = vec![];
        for i in 0..config.n_layer {
            let layer = DiTTransformerBlock::new(vb_layers.pp(i), config)?;
            layers.push(layer);
        }
        let norm = AdaptiveLayerNorm::new(vb.pp("norm"), config.dim, config.norm_eps)?;
        let rope = RoPE::new(config.head_dim, 10000.0, vb.device())?;
        let mut layers_emit_skip: Vec<usize> = vec![];
        let mut layers_receive_skip: Vec<usize> = vec![];
        if config.uvit_skip_connection {
            layers_emit_skip = (0..config.n_layer)
                .filter(|&x| x < config.n_layer / 2)
                .collect();
            layers_receive_skip = (0..config.n_layer)
                .filter(|&x| x > config.n_layer / 2)
                .collect();
        }
        Ok(Self {
            layers,
            norm,
            rope,
            uvit_skip_connection: config.uvit_skip_connection,
            layers_emit_skip,
            layers_receive_skip,
        })
    }
    pub fn forward(&self, xs: &Tensor, c: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;
        let (cos, sin) = self.rope.forward(0, seq_len, xs.device())?;
        let mut skip_in_x_list = vec![];
        let mut xs = xs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let skip_in_x = if self.uvit_skip_connection && self.layers_receive_skip.contains(&i) {
                skip_in_x_list.pop()
            } else {
                None
            };
            xs = layer.forward(&xs, c, Some(&cos), Some(&sin), mask, skip_in_x.as_ref())?;
            if self.uvit_skip_connection && self.layers_emit_skip.contains(&i) {
                skip_in_x_list.push(xs.clone());
            }
        }
        xs = self.norm.forward(&xs, Some(c))?;
        Ok(xs)
    }
}

pub struct TimestepEmbedder {
    mlp: TwoLinearMLP,
    freqs: Tensor,
    scale: f64,
    frequency_embedding_size: usize,
}

impl TimestepEmbedder {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        frequency_embedding_size: usize,
    ) -> Result<Self> {
        let mlp = TwoLinearMLP::new(
            vb.pp("mlp"),
            frequency_embedding_size,
            hidden_size,
            hidden_size,
            candle_nn::Activation::Silu,
            true,
            "0",
            "2",
        )?;
        let scale = 1000.0;
        let half = frequency_embedding_size / 2;
        let freqs = Tensor::arange(0f32, half as f32, vb.device())?
            .affine(-(10000.0f64.ln()), 0.0)?
            .exp()?;
        Ok(Self {
            mlp,
            freqs,
            scale,
            frequency_embedding_size,
        })
    }
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let args = t
            .affine(self.scale, 0.0)?
            .unsqueeze(D::Minus1)?
            .broadcast_matmul(&self.freqs.unsqueeze(0)?)?;
        let mut embedding = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?;
        if !self.frequency_embedding_size.is_multiple_of(2) {
            embedding = embedding.pad_with_zeros(D::Minus1, 0, 1)?;
        }
        embedding = self.mlp.forward(&embedding)?;
        Ok(embedding)
    }
}

pub struct SConv1d {
    conv: WNConv1d,
    ks: usize,
    stride: usize,
    dilation: usize,
}

impl SConv1d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        ks: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv = WNConv1d::new(
            vb.pp("conv.conv"),
            in_c,
            out_c,
            ks,
            dilation,
            0,
            groups,
            stride,
            bias,
        )?;
        Ok(Self {
            conv,
            ks,
            stride,
            dilation,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let length = xs.dim(D::Minus1)?;
        let ks = (self.ks - 1) * self.dilation + 1;
        let padding_total = ks - self.stride;
        let n_frames = (length - ks + padding_total) as f32 / self.stride as f32 + 1.0;
        let idea_length = (n_frames.ceil() as usize - 1) * self.stride + (ks - padding_total);
        let extra_padding = idea_length - length;
        let padding_right = padding_total / 2;
        let padding_left = padding_total - padding_right;
        let xs = pad_reflect_last_dim(xs, (padding_left, padding_right + extra_padding))?;
        let xs = self.conv.forward(&xs)?;
        Ok(xs)
    }
}

pub struct Wavenet {
    cond_layer: Option<SConv1d>,
    in_layers: Vec<SConv1d>,
    res_skip_layers: Vec<SConv1d>,
    hidden_c: usize,
    n_layers: usize,
}

impl Wavenet {
    pub fn new(
        vb: VarBuilder,
        hidden_c: usize,
        ks: usize,
        dilation_rate: usize,
        n_layers: usize,
        gin_channels: usize,
    ) -> Result<Self> {
        let cond_layer = if gin_channels != 0 {
            Some(SConv1d::new(
                vb.pp("cond_layer"),
                gin_channels,
                2 * hidden_c * n_layers,
                1,
                1,
                1,
                1,
                true,
            )?)
        } else {
            None
        };
        let mut in_layers = vec![];
        let vb_layers = vb.pp("in_layers");
        let mut res_skip_layers = vec![];
        let vb_res_skip_layers = vb.pp("res_skip_layers");
        for i in 0..n_layers {
            let dilation = dilation_rate.pow(i as u32);
            let in_layer = SConv1d::new(
                vb_layers.pp(i),
                hidden_c,
                2 * hidden_c,
                ks,
                1,
                dilation,
                1,
                true,
            )?;
            in_layers.push(in_layer);
            let res_skip_c = if i < n_layers - 1 {
                2 * hidden_c
            } else {
                hidden_c
            };
            let res_skip_layer = SConv1d::new(
                vb_res_skip_layers.pp(i),
                hidden_c,
                res_skip_c,
                1,
                1,
                1,
                1,
                true,
            )?;
            res_skip_layers.push(res_skip_layer);
        }
        Ok(Self {
            cond_layer,
            in_layers,
            res_skip_layers,
            hidden_c,
            n_layers,
        })
    }

    pub fn fused_add_tanh_sigmoid_multiply(
        &self,
        input_a: &Tensor,
        input_b: &Tensor,
    ) -> Result<Tensor> {
        let in_act = input_a.broadcast_add(input_b)?;
        let parts = in_act.chunk(2, 1)?;
        let t_act = parts[0].tanh()?;
        let s_act = sigmoid(&parts[1])?;
        let acts = t_act.mul(&s_act)?;
        Ok(acts)
    }

    pub fn forward(&self, xs: &Tensor, x_mask: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let x_mask = x_mask.to_dtype(xs.dtype())?;
        let mut output = xs.zeros_like()?;
        let g = if let Some(g) = g
            && let Some(cond_layer) = &self.cond_layer
        {
            Some(cond_layer.forward(g)?)
        } else {
            None
        };
        let mut xs = xs.clone();
        for i in 0..self.n_layers {
            let xs_in = self.in_layers[i].forward(&xs)?;
            let g_l = if let Some(g) = &g {
                let cond_offset = i * 2 * self.hidden_c;
                g.narrow(1, cond_offset, 2 * self.hidden_c)?
            } else {
                xs_in.zeros_like()?
            };
            let acts = self.fused_add_tanh_sigmoid_multiply(&xs_in, &g_l)?;
            let res_skip_act = &self.res_skip_layers[i].forward(&acts)?;
            if i < self.n_layers - 1 {
                let res_acts = res_skip_act.narrow(1, 0, self.hidden_c)?;
                let out_acts = res_skip_act.narrow(1, self.hidden_c, self.hidden_c)?;
                xs = xs.add(&res_acts)?.broadcast_mul(&x_mask)?;
                output = output.add(&out_acts)?;
            } else {
                output = output.add(res_skip_act)?;
            }
        }
        output = output.broadcast_mul(&x_mask)?;
        Ok(output)
    }
}

pub struct FinalLayer {
    norm_final: LayerNorm,
    linear: WNLinear,
    ada_ln_modulation: Linear, // (silu+linear)
}

impl FinalLayer {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        patch_size: usize,
        out_c: usize,
    ) -> Result<Self> {
        let norm_final = get_layer_norm_without_weight(vb.pp("norm_final"), 1e-6, hidden_size)?;
        let linear = WNLinear::new(
            vb.pp("linear"),
            hidden_size,
            patch_size * patch_size * out_c,
            true,
        )?;
        let ada_ln_modulation = linear_b(
            hidden_size,
            2 * hidden_size,
            true,
            vb.pp("adaLN_modulation.1"),
        )?;
        Ok(Self {
            norm_final,
            linear,
            ada_ln_modulation,
        })
    }

    pub fn forward(&self, xs: &Tensor, c: &Tensor) -> Result<Tensor> {
        let linear_c = self.ada_ln_modulation.forward(c)?.chunk(2, 1)?;
        let xs = self.norm_final.forward(xs)?;
        let xs = linear_c[1]
            .unsqueeze(1)?
            .affine(1.0, 1.0)?
            .broadcast_mul(&xs)?
            .broadcast_add(&linear_c[0].unsqueeze(1)?)?;
        let xs = self.linear.forward(&xs)?;
        Ok(xs)
    }
}

#[allow(unused)]
pub struct DiT {
    transformer: DiTTransformer,
    x_embedder: WNLinear,
    cond_embedder: Embedding,
    cond_projection: Linear,
    t_embedder: TimestepEmbedder,
    input_pos: Tensor,
    t_embedder2: TimestepEmbedder,
    conv1: Linear,
    conv2: Conv1d,
    wavenet: Wavenet,
    final_layer: FinalLayer,
    res_projection: Linear,
    content_mask_embedder: Embedding,
    skip_linear: Linear,
    cond_x_merge_linear: Linear,
    style_in: Option<Linear>,
    time_as_token: bool,
    style_as_token: bool,
    uvit_skip_connection: bool,
    transformer_style_condition: bool,
    long_skip_connection: bool,
}

impl DiT {
    pub fn new(vb: VarBuilder, config: &S2MelConfig) -> Result<Self> {
        let time_as_token = config.di_t.time_as_token;
        let style_as_token = config.di_t.style_as_token;
        let uvit_skip_connection = config.di_t.uvit_skip_connection;
        let transformer_config = DiTModelArgs::new_from_dit_config(&config.di_t);
        let transformer = DiTTransformer::new(vb.pp("transformer"), &transformer_config)?;
        let x_embedder = WNLinear::new(
            vb.pp("x_embedder"),
            config.di_t.in_channels,
            config.di_t.hidden_dim,
            true,
        )?;
        let cond_embedder = embedding(
            config.di_t.content_codebook_size,
            config.di_t.hidden_dim,
            vb.pp("cond_embedder"),
        )?;
        let cond_projection = linear_b(
            config.di_t.content_dim,
            config.di_t.hidden_dim,
            true,
            vb.pp("cond_projection"),
        )?;
        let t_embedder = TimestepEmbedder::new(vb.pp("t_embedder"), config.di_t.hidden_dim, 256)?;
        let input_pos = Tensor::arange(0u32, 16384, vb.device())?;
        let t_embedder2 =
            TimestepEmbedder::new(vb.pp("t_embedder2"), config.wavenet.hidden_dim, 256)?;
        let conv1 = linear_b(
            config.di_t.hidden_dim,
            config.wavenet.hidden_dim,
            true,
            vb.pp("conv1"),
        )?;
        let conv2 = get_conv1d(
            vb.pp("conv2"),
            config.wavenet.hidden_dim,
            config.di_t.in_channels,
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let wavenet = Wavenet::new(
            vb.pp("wavenet"),
            config.wavenet.hidden_dim,
            config.wavenet.kernel_size,
            config.wavenet.dilation_rate,
            config.wavenet.num_layers,
            config.wavenet.hidden_dim,
        )?;
        let final_layer = FinalLayer::new(
            vb.pp("final_layer"),
            config.wavenet.hidden_dim,
            1,
            config.wavenet.hidden_dim,
        )?;
        let res_projection = linear(
            config.di_t.hidden_dim,
            config.wavenet.hidden_dim,
            vb.pp("res_projection"),
        )?;
        let content_mask_embedder =
            embedding(1, config.di_t.hidden_dim, vb.pp("content_mask_embedder"))?;
        let skip_linear = linear(
            config.di_t.hidden_dim + config.di_t.in_channels,
            config.di_t.hidden_dim,
            vb.pp("skip_linear"),
        )?;
        let in_dim = if config.di_t.style_condition && !config.di_t.style_as_token {
            config.di_t.hidden_dim + config.di_t.in_channels * 2 + config.style_encoder.dim
        } else {
            config.di_t.hidden_dim + config.di_t.in_channels * 2
        };
        let cond_x_merge_linear =
            linear(in_dim, config.di_t.hidden_dim, vb.pp("cond_x_merge_linear"))?;
        let style_in = if config.di_t.style_as_token {
            Some(linear(
                config.style_encoder.dim,
                config.di_t.hidden_dim,
                vb.pp("style_in"),
            )?)
        } else {
            None
        };
        Ok(Self {
            transformer,
            x_embedder,
            cond_embedder,
            cond_projection,
            t_embedder,
            input_pos,
            t_embedder2,
            conv1,
            conv2,
            wavenet,
            final_layer,
            res_projection,
            content_mask_embedder,
            skip_linear,
            cond_x_merge_linear,
            style_in,
            time_as_token,
            style_as_token,
            uvit_skip_connection,
            transformer_style_condition: config.di_t.style_condition,
            long_skip_connection: config.di_t.long_skip_connection,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        prompt_xs: &Tensor,
        x_lens: &Tensor,
        t: &Tensor,
        style: Option<&Tensor>,
        cond: &Tensor,
    ) -> Result<Tensor> {
        let (_, _, t_dim) = xs.dims3()?;
        let t1 = self.t_embedder.forward(t)?;
        let cond = self.cond_projection.forward(cond)?;
        let xs = xs.transpose(1, 2)?;
        let prompt_xs = prompt_xs.transpose(1, 2)?;
        let mut x_in = Tensor::cat(&[&xs, &prompt_xs, &cond], D::Minus1)?;
        if self.transformer_style_condition
            && !self.style_as_token
            && let Some(style) = style
        {
            let style = style.unsqueeze(1)?.repeat((1, t_dim, 1))?;
            x_in = Tensor::cat(&[&x_in, &style], D::Minus1)?.contiguous()?;
        }
        x_in = self.cond_x_merge_linear.forward(&x_in)?;
        if self.style_as_token
            && let Some(style_in) = self.style_in.as_ref()
            && let Some(style) = style
        {
            let style = style_in.forward(style)?.unsqueeze(1)?;
            x_in = Tensor::cat(&[&style, &x_in], 1)?.contiguous()?;
        }
        if self.time_as_token {
            let t1 = t1.unsqueeze(1)?;
            x_in = Tensor::cat(&[&t1, &x_in], 1)?.contiguous()?;
        }
        let mut x_lens = x_lens.clone();
        if self.style_as_token {
            x_lens = x_lens.affine(1.0, 1.0)?;
        }
        if self.time_as_token {
            x_lens = x_lens.affine(1.0, 1.0)?;
        }
        let x_mask = sequence_mask(&x_lens, Some(x_in.dim(1)? as u32))?
            .to_device(xs.device())?
            .unsqueeze(1)?;
        let mut x_res = self
            .transformer
            .forward(&x_in, &t1.unsqueeze(1)?, Some(&x_mask))?;
        if self.time_as_token {
            let last_dim = x_res.dim(D::Minus1)?;
            x_res = x_res.narrow(D::Minus1, 1, last_dim - 1)?.contiguous()?;
        }
        if self.style_as_token {
            let last_dim = x_res.dim(D::Minus1)?;
            x_res = x_res.narrow(D::Minus1, 1, last_dim - 1)?.contiguous()?;
        }
        if self.long_skip_connection {
            x_res = self
                .skip_linear
                .forward(&Tensor::cat(&[&x_res, &xs], D::Minus1)?.contiguous()?)?;
        }
        let xs = self.conv1.forward(&x_res)?;
        let xs = xs.transpose(1, 2)?;
        let t2 = self.t_embedder2.forward(t)?;
        let xs = self
            .wavenet
            .forward(&xs, &x_mask, Some(&t2.unsqueeze(2)?))?
            .transpose(1, 2)?
            .broadcast_add(&self.res_projection.forward(&x_res)?)?;
        let xs = self.final_layer.forward(&xs, &t1)?.transpose(1, 2)?;
        let xs = self.conv2.forward(&xs)?;
        Ok(xs)
    }
}

pub struct CFM {
    in_channels: usize,
    estimator: DiT,
    // criterion: l1Loss
    // sigma_min: f32,
}

impl CFM {
    pub fn new(vb: VarBuilder, config: &S2MelConfig) -> Result<Self> {
        let in_channels = config.di_t.in_channels;
        // let sigma_min = 1e-6;
        let estimator = DiT::new(vb.pp("estimator"), config)?;
        Ok(Self {
            in_channels,
            estimator,
            // sigma_min,
        })
    }

    pub fn inference(
        &self,
        mu: &Tensor,
        x_lens: &Tensor,
        prompt: &Tensor,
        style: &Tensor,
        n_timesteps: usize,
        inference_cfg_rate: f64,
    ) -> Result<Tensor> {
        let (b, t, _) = mu.dims3()?;
        let z = Tensor::randn(0.0f32, 1.0f32, (b, self.in_channels, t), mu.device())?;
        let t_span = linspace(0.0, 1.0, n_timesteps + 1, mu.device())?;
        let res = self.solve_euler(&z, x_lens, prompt, mu, style, &t_span, inference_cfg_rate)?;
        Ok(res)
    }

    pub fn solve_euler(
        &self,
        x: &Tensor,
        x_lens: &Tensor,
        prompt: &Tensor,
        mu: &Tensor,
        style: &Tensor,
        t_span: &Tensor,
        inference_cfg_rate: f64,
    ) -> Result<Tensor> {
        let mut x: Tensor = x.clone();
        let prompt_len = prompt.dim(D::Minus1)?;
        let prompt_x = Tensor::zeros_like(&x)?;
        prompt_x.slice_set(prompt, D::Minus1, 0)?;
        let x_0 = Tensor::zeros_like(&x)?
            .narrow(D::Minus1, 0, prompt_len)?
            .contiguous()?;
        x.slice_set(&x_0, D::Minus1, 0)?;
        let mut t = t_span.i(0)?;
        let t_len = t_span.dim(0)?;
        let mut res_x = x.clone();
        for step in 1..t_len {
            let dt = t_span.i(step)?.sub(&t_span.i(step - 1)?)?;
            let dphi_dt = if inference_cfg_rate > 0.0 {
                let stacked_prompt_x =
                    Tensor::cat(&[&prompt_x, &Tensor::zeros_like(&prompt_x)?], 0)?;
                let stacked_style = Tensor::cat(&[style, &Tensor::zeros_like(style)?], 0)?;
                let stacked_mu = Tensor::cat(&[mu, &Tensor::zeros_like(mu)?], 0)?;
                let stacked_x = Tensor::cat(&[&x, &x], 0)?;
                let stacked_t = Tensor::cat(&[&t.unsqueeze(0)?, &t.unsqueeze(0)?], 0)?;
                let stacked_dphi_dt = self.estimator.forward(
                    &stacked_x,
                    &stacked_prompt_x,
                    x_lens,
                    &stacked_t,
                    Some(&stacked_style),
                    &stacked_mu,
                )?;
                let dphi = stacked_dphi_dt.chunk(2, 0)?;
                dphi[0]
                    .affine(1.0 + inference_cfg_rate, 0.0)?
                    .sub(&dphi[1].affine(inference_cfg_rate, 0.0)?)?
            } else {
                self.estimator
                    .forward(&x, &prompt_x, x_lens, &t.unsqueeze(0)?, Some(style), mu)?
            };
            x = x.add(&dphi_dt.broadcast_mul(&dt)?)?;
            res_x = x.clone();
            t = t.add(&dt)?;
            x.slice_set(&x_0, D::Minus1, 0)?;
        }
        Ok(res_x)
    }
}

pub struct InterpolateModule {
    conv1d: Conv1d,
    norm: GroupNorm,
}

impl InterpolateModule {
    pub fn new(vb: &VarBuilder, index: usize, channels: usize, groups: usize) -> Result<Self> {
        let start_index = index * 3;
        let conv1d = get_conv1d(vb.pp(start_index), channels, channels, 3, 1, 1, 1, 1, true)?;
        let norm = group_norm(groups, channels, 1e-5, vb.pp(start_index + 1))?;
        Ok(Self { conv1d, norm })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv1d.forward(xs)?;
        let xs = self.norm.forward(&xs)?;
        let xs = mish(&xs)?;
        Ok(xs)
    }
}

#[allow(unused)]
pub struct InterpolateRegulator {
    sampling_ratios: Vec<usize>,
    out_channels: usize,
    model0_11: Vec<InterpolateModule>,
    model_12: Conv1d,
    embedding: Embedding,
    mask_token: Tensor,
    // quantizer_dropout: f32,
    content_in_proj: Linear,
    n_codebooks: usize,
    interpolate: bool,
}

#[allow(unused)]
impl InterpolateRegulator {
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        sampling_ratios: Vec<usize>,
        is_discrete: bool,
        in_channels: usize,
        vector_quantize: bool,
        codebook_size: usize,
        out_channels: Option<usize>,
        groups: usize,
        n_codebooks: usize,
        quantizer_dropout: f32,
        f0_condition: bool,
        n_f0_bins: usize,
    ) -> Result<Self> {
        let out_channels = out_channels.unwrap_or(channels);
        let vb_model = vb.pp("model");
        let interpolate = true;
        let mut model0_11 = vec![];
        for (index, _) in sampling_ratios.iter().enumerate() {
            let inter = InterpolateModule::new(&vb_model, index, channels, groups)?;
            model0_11.push(inter);
        }
        let model_12 = get_conv1d(
            vb_model.pp("12"),
            channels,
            out_channels,
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let embedding = embedding(codebook_size, channels, vb.pp("embedding"))?;
        let mask_token = vb.get_with_hints((1, channels), "mask_token", Init::Const(0.0))?;
        let content_in_proj = linear(in_channels, channels, vb.pp("content_in_proj"))?;
        Ok(Self {
            sampling_ratios,
            out_channels,
            model0_11,
            model_12,
            embedding,
            mask_token,
            // quantizer_dropout,
            content_in_proj,
            n_codebooks,
            interpolate,
        })
    }
    pub fn forward(&self, x: &Tensor, y_lens: &Tensor) -> Result<Tensor> {
        let mut xs = self.content_in_proj.forward(x)?;
        xs = xs.transpose(1, 2)?.contiguous()?;
        if self.interpolate {
            let size = y_lens.max_all()?.to_scalar::<u32>()? as usize;
            xs = interpolate_nearest_1d(&xs, size)?;
        }
        for model_i in self.model0_11.iter() {
            xs = model_i.forward(&xs)?;
        }
        xs = self.model_12.forward(&xs)?.transpose(1, 2)?.contiguous()?;
        Ok(xs)
    }
}

pub struct GptLayer {
    layer_0: Linear,
    layer_1: Linear,
    layer_2: Linear,
}

impl GptLayer {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let layer_0 = linear(1280, 256, vb.pp("0"))?;
        let layer_1 = linear(256, 128, vb.pp("1"))?;
        let layer_2 = linear(128, 1024, vb.pp("2"))?;
        Ok(Self {
            layer_0,
            layer_1,
            layer_2,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.layer_0.forward(xs)?;
        let xs = self.layer_1.forward(&xs)?;
        let xs = self.layer_2.forward(&xs)?;
        Ok(xs)
    }
}

pub struct MyModel {
    cfm: CFM,
    length_regulator: InterpolateRegulator,
    gpt_layer: GptLayer,
}

impl MyModel {
    pub fn new(
        model_path: &str,
        config: &S2MelConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let s2mel_path = model_path.to_string() + "/s2mel.pth";
        let length_regulator_dict =
            read_pth_tensor_info_cycle(s2mel_path.clone(), Some("net.length_regulator"))?;
        let length_regulator_vb = VarBuilder::from_tensors(length_regulator_dict, dtype, device);
        let length_regulator = InterpolateRegulator::new(
            length_regulator_vb,
            config.length_regulator.channels,
            config.length_regulator.sampling_ratios.clone(),
            config.length_regulator.is_discrete,
            config.length_regulator.in_channels,
            config.length_regulator.vector_quantize,
            config.length_regulator.content_codebook_size,
            None,
            1,
            config.length_regulator.n_codebooks,
            config.length_regulator.quantizer_dropout,
            config.length_regulator.f0_condition,
            config.length_regulator.n_f0_bins,
        )?;
        let cfm_dict = read_pth_tensor_info_cycle(s2mel_path.clone(), Some("net.cfm"))?;
        let cfm_vb = VarBuilder::from_tensors(cfm_dict, dtype, device);
        let cfm = CFM::new(cfm_vb, config)?;
        let gpt_layer_dict = read_pth_tensor_info_cycle(s2mel_path.clone(), Some("net.gpt_layer"))?;
        let gpt_layer_vb = VarBuilder::from_tensors(gpt_layer_dict, dtype, device);
        let gpt_layer = GptLayer::new(gpt_layer_vb)?;
        Ok(Self {
            cfm,
            length_regulator,
            gpt_layer,
        })
    }

    pub fn length_regulator_forward(
        &self,
        s_ori: &Tensor,
        target_lengths: &Tensor,
    ) -> Result<Tensor> {
        let xs = self.length_regulator.forward(s_ori, target_lengths)?;
        Ok(xs)
    }
}

pub struct RelPositionMultiHeadedAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl RelPositionMultiHeadedAttention {
    pub fn new(vb: VarBuilder, n_head: usize, n_feat: usize) -> Result<Self> {
        let head_dim = n_feat / n_head;
        let q_proj = linear_b(n_feat, n_feat, true, vb.pp("linear_q"))?;
        let k_proj = linear_b(n_feat, n_feat, true, vb.pp("linear_k"))?;
        let v_proj = linear_b(n_feat, n_feat, true, vb.pp("linear_v"))?;
        let o_proj = linear_b(n_feat, n_feat, true, vb.pp("linear_out"))?;
        let linear_pos = linear_b(n_feat, n_feat, false, vb.pp("linear_pos"))?;
        let pos_bias_u = vb
            .get_with_hints((n_head, head_dim), "pos_bias_u", Init::Const(0.0))?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        let pos_bias_v = vb
            .get_with_hints((n_head, head_dim), "pos_bias_v", Init::Const(0.0))?
            .unsqueeze(0)?
            .unsqueeze(0)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: n_head,
            head_dim,
            linear_pos,
            pos_bias_u,
            pos_bias_v,
            kv_cache: None,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        pos_emb: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let q_with_bias_u = query_states
            .broadcast_add(&self.pos_bias_u)?
            .transpose(1, 2)?
            .contiguous()?;
        let q_with_bias_v = query_states
            .broadcast_add(&self.pos_bias_v)?
            .transpose(1, 2)?
            .contiguous()?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let n_batch_pos = pos_emb.dim(0)?;
        let p = self
            .linear_pos
            .forward(pos_emb)?
            .reshape((n_batch_pos, (), self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let matrix_ac =
            q_with_bias_u.matmul(&key_states.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        let matrix_bd = q_with_bias_v.matmul(&p.transpose(D::Minus2, D::Minus1)?.contiguous()?)?;
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let scores = matrix_ac.add(&matrix_bd)?.affine(scale, 0.0)?;
        let attn_weights = match attention_mask {
            None => scores,
            Some(mask) => scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?;
        let attn_output = attn_output
            .reshape((b_sz, q_len, self.num_heads * self.head_dim))?
            .contiguous()?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn forward_with_cache(
        &mut self,
        xs: &Tensor,
        pos_emb: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states.reshape((b_sz, q_len, self.num_heads, self.head_dim))?;
        let q_with_bias_u = query_states
            .broadcast_add(&self.pos_bias_u)?
            .transpose(1, 2)?;
        let q_with_bias_v = query_states
            .broadcast_add(&self.pos_bias_v)?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
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
        let n_batch_pos = pos_emb.dim(0)?;
        let p = self.linear_pos.forward(pos_emb)?.reshape((
            n_batch_pos,
            (),
            self.num_heads,
            self.head_dim,
        ))?;
        let matrix_ac = q_with_bias_u.matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?;
        let matrix_bd = q_with_bias_v.matmul(&p.transpose(D::Minus2, D::Minus1)?)?;
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let scores = matrix_ac.add(&matrix_bd)?.affine(scale, 0.0)?;
        let attn_weights = match attention_mask {
            None => scores,
            Some(mask) => scores.broadcast_add(&mask.to_dtype(scores.dtype())?)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

pub struct ConvolutionModule {
    pointwise_conv1: Conv1d,
    glu: GLU,
    depthwise_conv: Conv1d,
    norm: LayerNorm,
    pointwise_conv2: Conv1d,
    activation: Activation,
}

impl ConvolutionModule {
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        activation: Activation,
        bias: bool,
    ) -> Result<Self> {
        let pointwise_conv1 = get_conv1d(
            vb.pp("pointwise_conv1"),
            channels,
            2 * channels,
            1,
            0,
            1,
            1,
            1,
            bias,
        )?;
        let padding = (kernel_size - 1) / 2;
        let depthwise_conv = get_conv1d(
            vb.pp("depthwise_conv"),
            channels,
            channels,
            kernel_size,
            padding,
            1,
            1,
            channels,
            bias,
        )?;
        let norm = get_layer_norm(vb.pp("norm"), 1e-5, channels, true)?;
        let pointwise_conv2 = get_conv1d(
            vb.pp("pointwise_conv2"),
            channels,
            channels,
            1,
            0,
            1,
            1,
            1,
            bias,
        )?;
        let glu = GLU::new(1)?;
        Ok(Self {
            pointwise_conv1,
            glu,
            depthwise_conv,
            norm,
            pointwise_conv2,
            activation,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask_pad: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.transpose(1, 2)?;
        if let Some(mask_pad) = mask_pad {
            xs = masked_fill_zeros(&xs, mask_pad)?;
        }
        xs = self.pointwise_conv1.forward(&xs)?;
        xs = self.glu.forward(&xs)?;
        xs = self.depthwise_conv.forward(&xs)?;
        xs = xs.transpose(1, 2)?;
        xs = self.norm.forward(&xs)?.apply(&self.activation)?;
        xs = xs.transpose(1, 2)?;
        xs = self.pointwise_conv2.forward(&xs)?;
        if let Some(mask_pad) = mask_pad {
            xs = masked_fill_zeros(&xs, mask_pad)?;
        }
        xs = xs.transpose(1, 2)?;
        Ok(xs)
    }
}

pub struct ConformerEncoderLayer {
    self_attn: RelPositionMultiHeadedAttention,
    feed_forward: TwoLinearMLP, // silu,
    // feed_forward_macaron: None,
    ff_scale: f32,
    conv_module: ConvolutionModule,
    norm_ff: LayerNorm,
    norm_mha: LayerNorm,
    norm_conv: LayerNorm,
    norm_final: LayerNorm,
    // concat_linear: None,
}

impl ConformerEncoderLayer {
    pub fn new(
        vb: VarBuilder,
        attention_heads: usize,
        output_size: usize,
        linear_units: usize,
    ) -> Result<Self> {
        let self_attn =
            RelPositionMultiHeadedAttention::new(vb.pp("self_attn"), attention_heads, output_size)?;
        let feed_forward = TwoLinearMLP::new(
            vb.pp("feed_forward"),
            output_size,
            linear_units,
            output_size,
            Activation::Silu,
            true,
            "w_1",
            "w_2",
        )?;
        let ff_scale = 1.0;
        let conv_module = ConvolutionModule::new(
            vb.pp("conv_module"),
            output_size,
            15,
            Activation::Silu,
            true,
        )?;
        let norm_ff = get_layer_norm(vb.pp("norm_ff"), 1e-5, output_size, true)?;
        let norm_mha = get_layer_norm(vb.pp("norm_mha"), 1e-5, output_size, true)?;
        let norm_conv = get_layer_norm(vb.pp("norm_conv"), 1e-5, output_size, true)?;
        let norm_final = get_layer_norm(vb.pp("norm_final"), 1e-5, output_size, true)?;
        Ok(Self {
            self_attn,
            feed_forward,
            ff_scale,
            conv_module,
            norm_ff,
            norm_mha,
            norm_conv,
            norm_final,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        pos_emb: &Tensor,
        mask_pad: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = self.norm_mha.forward(xs)?;
        xs = self.self_attn.forward(&xs, pos_emb, mask)?;
        xs = residual.add(&xs)?;
        let residual = xs.clone();
        xs = self.norm_conv.forward(&xs)?;
        xs = self.conv_module.forward(&xs, mask_pad)?;
        xs = residual.add(&xs)?;
        let residual = xs.clone();
        xs = self.norm_ff.forward(&xs)?;
        xs = self
            .feed_forward
            .forward(&xs)?
            .affine(self.ff_scale as f64, 1.0)?;
        xs = residual.add(&xs)?;
        xs = self.norm_final.forward(&xs)?;
        Ok(xs)
    }
}

pub struct RelPositionalEncoding {
    xscale: f64,
    inv_freq: Tensor,
}

impl RelPositionalEncoding {
    pub fn new(d_model: usize, device: &Device) -> Result<Self> {
        let xscale = (d_model as f64).sqrt();
        let inv_freq = compute_default_rope_parameters(d_model, 10000.0);
        let inv_freq = Tensor::from_slice(&inv_freq, (1, inv_freq.len()), device)?;

        Ok(Self { xscale, inv_freq })
    }
    pub fn forward(&self, xs: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let xs = xs.affine(self.xscale, 0.0)?;
        let seq_len = xs.dim(1)?;
        let positions =
            Tensor::arange(offset as f32, seq_len as f32, xs.device())?.reshape((seq_len, 1))?; // (max_len, 1)
        let freqs = positions.matmul(&self.inv_freq.to_device(xs.device())?)?; // (max_len, dim / 2)
        let sin = freqs.sin()?;
        let cos = freqs.cos()?;
        let pe = Tensor::stack(&[sin, cos], D::Minus1)?
            .flatten(1, 2)?
            .unsqueeze(0)?
            .to_dtype(xs.dtype())?;
        Ok((xs, pe))
    }
}

pub struct Conv2dSubsampling2 {
    conv_0: Conv2d, // conv+relu
    out_0: Linear,
    pos_enc: RelPositionalEncoding,
}

impl Conv2dSubsampling2 {
    pub fn new(vb: VarBuilder, in_dim: usize, out_dim: usize) -> Result<Self> {
        let conv_0 = get_conv2d(vb.pp("conv.0"), 1, out_dim, 3, 0, 2, 1, 1, true)?;
        let out_0 = linear_b(out_dim * ((in_dim - 1) / 2), out_dim, true, vb.pp("out.0"))?;
        let pos_enc = RelPositionalEncoding::new(out_dim, vb.device())?;
        Ok(Self {
            conv_0,
            out_0,
            pos_enc,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Option<Tensor>)> {
        let xs = xs.unsqueeze(1)?;
        let xs = self.conv_0.forward(&xs)?.relu()?;
        let (b, c, t, f) = xs.dims4()?;
        let xs = xs
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c * f))?
            .apply(&self.out_0)?;
        let (xs, pos_emb) = self.pos_enc.forward(&xs, offset)?;
        let mask = if let Some(mask) = mask {
            let t = mask.dim(2)?;
            let mask_index = Tensor::arange_step(2u32, t as u32, 2u32, xs.device())?;
            let mask = mask.index_select(&mask_index, 2)?;
            Some(mask)
        } else {
            None
        };

        Ok((xs, pos_emb, mask))
    }
}

pub struct ConformerEncoder {
    encoders: Vec<ConformerEncoderLayer>,
    embed: Conv2dSubsampling2,
    after_norm: LayerNorm,
}

impl ConformerEncoder {
    pub fn new(
        vb: VarBuilder,
        input_size: usize,
        attention_heads: usize,
        output_size: usize,
        linear_units: usize,
        num_blocks: usize,
    ) -> Result<Self> {
        let mut encoders = vec![];
        let vb_encoders = vb.pp("encoders");
        for i in 0..num_blocks {
            let layer = ConformerEncoderLayer::new(
                vb_encoders.pp(i),
                attention_heads,
                output_size,
                linear_units,
            )?;
            encoders.push(layer);
        }
        let embed = Conv2dSubsampling2::new(vb.pp("embed"), input_size, output_size)?;
        let after_norm = get_layer_norm(vb.pp("after_norm"), 1e-5, output_size, true)?;
        Ok(Self {
            encoders,
            embed,
            after_norm,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (mut xs, pos_emb, mask) = self.embed.forward(xs, None, 0)?;
        for layer in self.encoders.iter() {
            xs = layer.forward(&xs, mask.as_ref(), &pos_emb, mask.as_ref())?;
        }
        xs = self.after_norm.forward(&xs)?;
        Ok((xs, None))
    }
}

pub struct PerceiverResamplerAttention {
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
    n_head: usize,
    head_dim: usize,
    cross_attn_include_queries: bool,
}

impl PerceiverResamplerAttention {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        dim_context: Option<usize>,
        head_dim: usize,
        n_head: usize,
        cross_attn_include_queries: bool,
    ) -> Result<Self> {
        let dim_inner = head_dim * n_head;
        let dim_context = dim_context.unwrap_or(dim);
        let to_q = linear_b(dim, dim_inner, false, vb.pp("to_q"))?;
        let to_kv = linear_b(dim_context, dim_inner * 2, false, vb.pp("to_kv"))?;
        let to_out = linear_b(dim_inner, dim, false, vb.pp("to_out"))?;
        Ok(Self {
            to_q,
            to_kv,
            to_out,
            n_head,
            head_dim,
            cross_attn_include_queries,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        context: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let b_sz = xs.dim(0)?;
        let query_states = self.to_q.forward(xs)?;
        let context = if let Some(context) = context
            && self.cross_attn_include_queries
        {
            Tensor::cat(&[xs, context], D::Minus2)?
        } else {
            xs.clone()
        };
        let key_value = self.to_kv.forward(&context)?.chunk(2, D::Minus1)?;
        let query_states = query_states
            .reshape((b_sz, (), self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_value[0]
            .reshape((b_sz, (), self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = key_value[1]
            .reshape((b_sz, (), self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output =
            eager_attention_forward(&query_states, &key_states, &value_states, None, mask, scale)?;
        let attn_output = attn_output.reshape((b_sz, (), self.n_head * self.head_dim))?;
        let attn_output = attn_output.apply(&self.to_out)?;
        Ok(attn_output)
    }
}

pub struct PerceiverResamplerFeedForward {
    net_0: Linear,
    net_1: GEGLU,
    net_2: Linear,
}

impl PerceiverResamplerFeedForward {
    pub fn new(vb: VarBuilder, dim: usize, mult: usize) -> Result<Self> {
        let dim_inner = dim * mult * 2 / 3;
        let net_0 = linear(dim, dim_inner * 2, vb.pp("0"))?;
        let net_1 = GEGLU::new(2)?;
        let net_2 = linear(dim_inner, dim, vb.pp("2"))?;
        Ok(Self {
            net_0,
            net_1,
            net_2,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.net_0.forward(xs)?;
        let xs = self.net_1.forward(&xs)?;
        let xs = self.net_2.forward(&xs)?;
        Ok(xs)
    }
}

pub struct PerceiverResamplerLayer {
    layer_0: PerceiverResamplerAttention,
    layer_1: PerceiverResamplerFeedForward,
}

impl PerceiverResamplerLayer {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        head_dim: usize,
        n_head: usize,
        ff_mult: usize,
    ) -> Result<Self> {
        let layer_0 =
            PerceiverResamplerAttention::new(vb.pp("0"), dim, None, head_dim, n_head, true)?;
        let layer_1 = PerceiverResamplerFeedForward::new(vb.pp("1"), dim, ff_mult)?;
        Ok(Self { layer_0, layer_1 })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        context: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let latents = self.layer_0.forward(xs, context, mask)?.add(xs)?;
        let latents = self.layer_1.forward(&latents)?.add(&latents)?;
        Ok(latents)
    }
}

pub struct PerceiverRmsNorm {
    // dim: usize,
    scale: f64,
    gamma: Tensor,
}

impl PerceiverRmsNorm {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let scale = (dim as f64).sqrt();
        let gamma = vb.get_with_hints(dim, "gamma", Init::Const(1.0))?;
        Ok(Self { scale, gamma })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = l2_normalize(xs, xs.rank() - 1)?
            .affine(self.scale, 0.0)?
            .broadcast_mul(&self.gamma)?;
        Ok(out)
    }
}

#[allow(unused)]
pub struct PerceiverResampler {
    dim_context: usize,
    proj_context: Linear,
    latents: Tensor,
    layers: Vec<PerceiverResamplerLayer>,
    norm: PerceiverRmsNorm,
}

impl PerceiverResampler {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        depth: usize,
        dim_context: Option<usize>,
        num_latents: usize,
        head_dim: usize,
        n_head: usize,
        ff_mult: usize,
    ) -> Result<Self> {
        let dim_context = dim_context.unwrap_or(dim);
        let proj_context = linear(dim_context, dim, vb.pp("proj_context"))?;
        let latents = vb.get_with_hints((num_latents, dim), "latents", Init::Const(1.0))?;
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        for i in 0..depth {
            let layer =
                PerceiverResamplerLayer::new(vb_layers.pp(i), dim, head_dim, n_head, ff_mult)?;
            layers.push(layer);
        }
        let norm = PerceiverRmsNorm::new(vb.pp("norm"), dim)?;
        Ok(Self {
            dim_context,
            proj_context,
            latents,
            layers,
            norm,
        })
    }

    pub fn forward(&self, xs: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let b = xs.dim(0)?;
        let xs = self.proj_context.forward(xs)?;
        let mut latents = self.latents.unsqueeze(0)?.repeat((b, 1, 1))?;
        for layer in self.layers.iter() {
            latents = layer.forward(&latents, Some(&xs), mask)?;
        }
        let xs = self.norm.forward(&latents)?;
        Ok(xs)
    }
}

pub struct LearnedPositionEmbeddings {
    emb: Embedding,
}

impl LearnedPositionEmbeddings {
    pub fn new(vb: VarBuilder, seq_len: usize, model_dim: usize) -> Result<Self> {
        let emb = embedding(seq_len, model_dim, vb.pp("emb"))?;
        Ok(Self { emb })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let sl = match xs.rank() {
            1 => xs.dim(0)?,
            2 => xs.dim(1)?,
            3 => xs.dim(1)?,
            _ => return Err(anyhow!(" only support xs rank 1, 2, 3")),
        };
        let id = Tensor::arange(0, sl as u32, xs.device())?;
        let embed = self.emb.forward(&id)?.unsqueeze(0)?;
        Ok(embed)
    }
    pub fn get_fixed_embedding(&self, id: u32, device: &Device) -> Result<Tensor> {
        let id = Tensor::new(id, device)?;
        let embed = self.emb.forward(&id)?.unsqueeze(0)?;
        Ok(embed)
    }
}

#[allow(unused)]
pub struct UnifiedVoice {
    conditioning_encoder: ConformerEncoder,
    perceiver_encoder: PerceiverResampler,
    emo_conditioning_encoder: ConformerEncoder,
    emo_perceiver_encoder: PerceiverResampler,
    text_embedding: Embedding,
    emo_layer: Linear,
    emovec_layer: Linear,
    mel_embedding: Embedding,
    gpt: GPT2Model,
    mel_pos_embedding: LearnedPositionEmbeddings,
    text_pos_embedding: LearnedPositionEmbeddings,
    final_norm: LayerNorm,
    text_head: Linear,
    mel_head: Linear,
    speed_emb: Embedding,
    start_text_token: u32,
    stop_text_token: u32,
    start_mel_token: u32,
    stop_mel_token: u32,
    max_mel_tokens: usize,
    dtype: DType,
    device: Device,
}

impl UnifiedVoice {
    pub fn new(vb: VarBuilder, cfg: &GptConfig) -> Result<Self> {
        let conditioning_encoder = ConformerEncoder::new(
            vb.pp("conditioning_encoder"),
            1024,
            cfg.condition_module.attention_heads,
            cfg.condition_module.output_size,
            cfg.condition_module.linear_units,
            cfg.condition_module.num_blocks,
        )?;
        let perceiver_encoder = PerceiverResampler::new(
            vb.pp("perceiver_encoder"),
            cfg.model_dim,
            2,
            Some(cfg.condition_module.output_size),
            32,
            64,
            cfg.condition_module.attention_heads,
            cfg.condition_module.perceiver_mult,
        )?;
        let emo_conditioning_encoder = ConformerEncoder::new(
            vb.pp("emo_conditioning_encoder"),
            1024,
            cfg.emo_condition_module.attention_heads,
            cfg.emo_condition_module.output_size,
            cfg.emo_condition_module.linear_units,
            cfg.emo_condition_module.num_blocks,
        )?;
        let emo_perceiver_encoder = PerceiverResampler::new(
            vb.pp("emo_perceiver_encoder"),
            1024,
            2,
            Some(cfg.emo_condition_module.output_size),
            1,
            64,
            cfg.emo_condition_module.attention_heads,
            cfg.emo_condition_module.perceiver_mult,
        )?;

        let text_embedding = embedding(
            cfg.number_text_tokens + 1,
            cfg.model_dim,
            vb.pp("text_embedding"),
        )?;
        let emo_layer = linear(cfg.model_dim, cfg.model_dim, vb.pp("emo_layer"))?;
        let emovec_layer = linear(1024, cfg.model_dim, vb.pp("emovec_layer"))?;
        let mel_embedding = embedding(cfg.number_mel_codes, cfg.model_dim, vb.pp("mel_embedding"))?;

        let gpt = GPT2Model::new(
            vb.pp("gpt"),
            cfg.model_dim,
            cfg.heads,
            cfg.layers,
            mel_embedding.embeddings(),
        )?;
        let max_mel_seq_len = cfg.max_mel_tokens + 3;
        let max_text_seq_len = cfg.max_text_tokens + 2;
        let mel_pos_embedding = LearnedPositionEmbeddings::new(
            vb.pp("mel_pos_embedding"),
            max_mel_seq_len,
            cfg.model_dim,
        )?;
        let text_pos_embedding = LearnedPositionEmbeddings::new(
            vb.pp("text_pos_embedding"),
            max_text_seq_len,
            cfg.model_dim,
        )?;
        let final_norm = get_layer_norm(vb.pp("final_norm"), 1e-5, cfg.model_dim, true)?;
        let text_head = linear(
            cfg.model_dim,
            cfg.number_text_tokens + 1,
            vb.pp("text_head"),
        )?;
        let mel_head = linear(cfg.model_dim, cfg.number_mel_codes, vb.pp("mel_head"))?;
        let speed_emb = embedding(2, cfg.model_dim, vb.pp("speed_emb"))?;
        Ok(Self {
            conditioning_encoder,
            perceiver_encoder,
            emo_conditioning_encoder,
            emo_perceiver_encoder,
            text_embedding,
            emo_layer,
            emovec_layer,
            mel_embedding,
            gpt,
            mel_pos_embedding,
            text_pos_embedding,
            final_norm,
            text_head,
            mel_head,
            speed_emb,
            start_text_token: 0,
            stop_text_token: 1,
            start_mel_token: 8192,
            stop_mel_token: 8193,
            max_mel_tokens: cfg.max_mel_tokens,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    pub fn get_conditioning(&self, speech_conditioning_input: &Tensor) -> Result<Tensor> {
        let (speech_conditioning_input, mask) = self
            .conditioning_encoder
            .forward(speech_conditioning_input)?;
        let conds = self
            .perceiver_encoder
            .forward(&speech_conditioning_input, mask.as_ref())?;
        Ok(conds)
    }

    pub fn prepare_inputs(
        &self,
        conditional_latents: &Tensor,
        text_inputs: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let pad_left =
            Tensor::new(vec![self.start_text_token], text_inputs.device())?.unsqueeze(0)?;
        let pad_right =
            Tensor::new(vec![self.stop_text_token], text_inputs.device())?.unsqueeze(0)?;
        let text_input = Tensor::cat(&[&pad_left, text_inputs, &pad_right], D::Minus1)?;
        let seq_len = text_input.dim(D::Minus1)?;
        let text_input_pos =
            Tensor::arange(0u32, seq_len as u32, text_inputs.device())?.unsqueeze(0)?;
        let text_embedding = self.text_embedding.forward(&text_input)?;
        let text_pos_embedding = self.text_pos_embedding.forward(&text_input_pos)?;
        let text_emb = text_embedding.add(&text_pos_embedding)?;
        let conds_text_emb = Tensor::cat(&[conditional_latents, &text_emb], 1)?;
        let (bs, len, _) = conds_text_emb.dims3()?;
        let fake_inputs = Tensor::ones((bs, len), DType::U32, conds_text_emb.device())?;
        let start_mel_token =
            Tensor::new(vec![self.start_mel_token], conds_text_emb.device())?.unsqueeze(0)?;
        let fake_inputs = Tensor::cat(&[fake_inputs, start_mel_token], 1)?;
        Ok((fake_inputs, conds_text_emb))
    }

    pub fn gpt_forward(
        &mut self,
        input_ids: &Tensor,
        input_embed: Option<&Tensor>,
        offset: u32,
    ) -> Result<Tensor> {
        let emb = if let Some(input_embed) = input_embed {
            let mel_len = input_embed.dim(1)?;
            let text_inputs = input_ids.i((.., mel_len))?;
            let text_emb = self.mel_embedding.forward(&text_inputs)?.unsqueeze(1)?;
            let text_pos_emb = self.mel_pos_embedding.forward(&text_emb)?;
            let text_emb = text_emb.add(&text_pos_emb)?;
            Tensor::cat(&[input_embed, &text_emb], 1)?
        } else {
            let emb = self.mel_embedding.forward(input_ids)?;
            let emb_pos = self
                .mel_pos_embedding
                .get_fixed_embedding(offset, input_ids.device())?
                .unsqueeze(0)?;
            emb.add(&emb_pos)?
        };
        let outputs = self.gpt.forward(&emb)?;
        let outputs = self.final_norm.forward(&outputs)?;
        let logits = self.mel_head.forward(&outputs)?;
        let seq_len = logits.dim(1)?;
        let logits = logits.narrow(1, seq_len - 1, 1)?;

        Ok(logits)
    }

    pub fn generate(&mut self, input_ids: &Tensor, input_embed: &Tensor) -> Result<Tensor> {
        let mut logit_processor = get_logit_processor(Some(0.8), Some(0.8), Some(30), 35329);
        let mut generate_ids = vec![];
        let mut input_ids = input_ids.clone();
        let mut input_embed = Some(input_embed.clone());
        let mut offset = 0u32;
        let mut seq_len = input_ids.dim(1)? as u32;
        for _ in 0..self.max_mel_tokens {
            let logits = self.gpt_forward(&input_ids, input_embed.as_ref(), offset)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let next_token = logit_processor.sample(&logits)?;
            if next_token == self.start_mel_token || next_token == self.stop_mel_token {
                break;
            }
            generate_ids.push(next_token);
            input_ids = Tensor::from_vec(vec![next_token], (1, 1), &self.device)?;
            input_embed = None;
            offset += seq_len;
            seq_len = 1;
        }
        self.gpt.clear_kv_cache();
        let generate_ids = Tensor::new(generate_ids, &self.device)?;
        Ok(generate_ids)
    }

    pub fn inference_speech(
        &mut self,
        speech_condition: &Tensor,
        text_inputs: &Tensor,
        emo_vec: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let speech_conditioning_latent =
            self.get_conditioning(&speech_condition.to_dtype(self.dtype)?)?;
        let text_len = text_inputs.dim(0)?;
        let tmp = Tensor::zeros(text_len, DType::U32, speech_condition.device())?;
        let duration_emb = self.speed_emb.forward(&tmp)?.unsqueeze(1)?;
        let duration_emb_half = self
            .speed_emb
            .forward(&Tensor::ones_like(&tmp)?)?
            .unsqueeze(1)?;
        let speech_add_emovec = speech_conditioning_latent.broadcast_add(&emo_vec.unsqueeze(1)?)?;
        let conds_latent =
            Tensor::cat(&[&speech_add_emovec, &duration_emb_half, &duration_emb], 1)?;
        let (fake_inputs, inputs_embeds) = self.prepare_inputs(&conds_latent, text_inputs)?;
        let output = self.generate(&fake_inputs, &inputs_embeds)?;
        Ok((output, speech_conditioning_latent))
    }

    pub fn get_emo_conditioning(&self, speech_conditioning_latent: &Tensor) -> Result<Tensor> {
        let (speech_conditioning_input, mask) = self
            .emo_conditioning_encoder
            .forward(speech_conditioning_latent)?;
        let conds = self
            .emo_perceiver_encoder
            .forward(&speech_conditioning_input, mask.as_ref())?;
        let conds = conds.squeeze(1)?;
        Ok(conds)
    }

    pub fn get_emovec(&self, speech_conditioning_latent: &Tensor) -> Result<Tensor> {
        let emo_vec_syn_ori = self.get_emo_conditioning(speech_conditioning_latent)?;
        let emo_vec_syn = self.emovec_layer.forward(&emo_vec_syn_ori)?;
        let emo_vec = self.emo_layer.forward(&emo_vec_syn)?;
        Ok(emo_vec)
    }

    pub fn merge_emovec(
        &self,
        speech_conditioning_latent: &Tensor,
        emo_speech_conditioning_latent: &Tensor,
        alpha: f64,
    ) -> Result<Tensor> {
        let emo_vec = self.get_emovec(&emo_speech_conditioning_latent.to_dtype(self.dtype)?)?;
        let base_vec = self.get_emovec(&speech_conditioning_latent.to_dtype(self.dtype)?)?;
        let out = emo_vec.sub(&base_vec)?.affine(alpha, 0.0)?.add(&base_vec)?;
        Ok(out)
    }

    pub fn get_logits(
        &mut self,
        speech_conditioning_inputs: &Tensor,
        first_inputs: &Tensor,
        second_inputs: &Tensor,
    ) -> Result<Tensor> {
        let emb = Tensor::cat(
            &[speech_conditioning_inputs, first_inputs, second_inputs],
            1,
        )?;
        let gpt_out = self.gpt.forward(&emb)?;
        self.gpt.clear_kv_cache();
        let offset = speech_conditioning_inputs.dim(1)?;
        let enc = gpt_out.i((.., offset.., ..))?;
        let enc = self.final_norm.forward(&enc)?;
        let offset = first_inputs.dim(1)?;
        let mel_logits = enc.i((.., offset.., ..))?;
        Ok(mel_logits)
    }

    pub fn forward(
        &mut self,
        speech_conditioning_latent: &Tensor,
        text_inputs: &Tensor,
        mel_codes: &Tensor,
        emo_vec: &Tensor,
        use_speed: &Tensor,
    ) -> Result<Tensor> {
        let pad_left =
            Tensor::new(vec![self.start_text_token], text_inputs.device())?.unsqueeze(0)?;
        let pad_right =
            Tensor::new(vec![self.stop_text_token], text_inputs.device())?.unsqueeze(0)?;
        let text_inputs = Tensor::cat(&[&pad_left, text_inputs, &pad_right], D::Minus1)?;
        let pad_left =
            Tensor::new(vec![self.start_mel_token], text_inputs.device())?.unsqueeze(0)?;
        let pad_right =
            Tensor::new(vec![self.stop_mel_token], text_inputs.device())?.unsqueeze(0)?;
        let mel_codes = Tensor::cat(&[&pad_left, mel_codes, &pad_right], D::Minus1)?;
        let duration_emb = self
            .speed_emb
            .forward(&Tensor::zeros_like(use_speed)?)?
            .unsqueeze(1)?;
        let duration_emb_half = self
            .speed_emb
            .forward(&Tensor::ones_like(use_speed)?)?
            .unsqueeze(1)?;
        let speech_add_emovec = speech_conditioning_latent.broadcast_add(&emo_vec.unsqueeze(1)?)?;
        let conds = Tensor::cat(&[&speech_add_emovec, &duration_emb_half, &duration_emb], 1)?;
        let text_emb = self.text_embedding.forward(&text_inputs)?;
        let text_emb_pos = self.text_pos_embedding.forward(&text_inputs)?;
        let text_emb = text_emb.add(&text_emb_pos)?;
        let mel_emb = self.mel_embedding.forward(&mel_codes)?;
        let mel_emb_pos = self.mel_pos_embedding.forward(&mel_codes)?;
        let mel_emb = mel_emb.add(&mel_emb_pos)?;
        let mel_logits = self.get_logits(&conds, &text_emb, &mel_emb)?;
        let len = mel_logits.dim(1)? - 2;
        let mel_logits = mel_logits.i((.., 0..len, ..))?;
        Ok(mel_logits)
    }
}

pub struct IndexTTS2SpkCache {
    pub cache_spk_cond: Tensor,
    pub cache_s2mel_style: Tensor,
    pub cache_s2mel_prompt: Tensor,
    pub cache_mel: Tensor,
    pub cache_spk_audio_prompt: String,
}

pub struct IndexTTS2EmoCache {
    cache_emo_audio_prompt: String,
    emo_cond_emb: Tensor,
}

pub struct IndexTTS2Model {
    max_audio_length_seconds: usize,
    spk_cache: Option<IndexTTS2SpkCache>,
    emo_cache: Option<IndexTTS2EmoCache>,
    feature_extractor: SeamlessM4TFeatureExtractor,
    semantic_model: W2VBert2_0Model,
    semantic_mean: Tensor,
    semantic_std: Tensor,
    semantic_codec: RepCodec,
    s2mel_filters: Tensor,
    s2mel_windows: Tensor,
    s2mel_preprocess_params: PreprocessParams,
    window_shift: usize,
    window_size: usize,
    padded_window_size: usize,
    mel_energies: Tensor,
    campplus_model: CAMPPlus,
    s2mel: MyModel,
    emo_num: Vec<usize>,
    emo_matrix: Vec<Tensor>,
    spk_matrix: Vec<Tensor>,
    gpt: UnifiedVoice,
    bigvgan: BigVGAN,
    device: Device,
    dtype_f32: DType,
    // gpt_dtype: DType,
}

impl IndexTTS2Model {
    pub fn new(
        path: &str,
        save_dir: &str,
        config: &IndexTTS2Config,
        device: &Device,
    ) -> Result<Self> {
        let dtype_f32 = DType::F32;
        let gpt_dtype = get_dtype(None, "bfloat16");
        let feature_extractor = SeamlessM4TFeatureExtractor::new(
            // 80,
            80,
            crate::utils::tensor_utils::PaddingSide::Right,
            1.0,
            16000,
            2,
            device,
        )?;
        let w2vbert2_path = save_dir.to_string() + "/facebook/w2v-bert-2.0";
        let semantic_model = W2VBert2_0Model::init(&w2vbert2_path, device, dtype_f32)?;
        let semantic_mean_var_path = path.to_string() + "/" + &config.w2v_stat;
        let dict = read_all_with_key(semantic_mean_var_path, None)?;
        let mut semantic_mean = Tensor::new(0.0, device)?.to_dtype(dtype_f32)?;
        let mut semantic_std = Tensor::new(1.0, device)?.to_dtype(dtype_f32)?;
        for (k, v) in dict {
            if k.eq("mean") {
                semantic_mean = v.to_device(device)?.to_dtype(dtype_f32)?;
            } else if k.eq("var") {
                semantic_std = v.to_device(device)?.to_dtype(dtype_f32)?.sqrt()?;
            }
        }

        let semantic_codec_path =
            save_dir.to_string() + "/amphion/MaskGCT/semantic_codec/model.safetensors";
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[semantic_codec_path], dtype_f32, device)?
        };
        let semantic_codec = RepCodec::new(vb, &config.semantic_codec)?;
        let s2mel_filters = mel_filter_bank(
            config.s2mel.preprocess_params.spect_params.n_fft / 2 + 1,
            config.s2mel.preprocess_params.spect_params.n_mels,
            config.s2mel.preprocess_params.spect_params.fmin as f32,
            config
                .s2mel
                .preprocess_params
                .spect_params
                .fmax
                .unwrap_or(config.s2mel.preprocess_params.sr / 2) as f32,
            config.s2mel.preprocess_params.sr as f32,
            Some("slaney"),
            crate::utils::audio_utils::MelScale::Slaney,
            false,
            device,
        )?
        .t()?;
        let s2mel_windows = create_hann_window(
            config.s2mel.preprocess_params.spect_params.win_length,
            dtype_f32,
            device,
        )?;
        let (window_shift, window_size, padded_window_size) =
            get_waveform_and_window_properties(16000, 10.0, 25.0, true)?;
        let (mel_energies, _) =
            kaldi_get_mel_banks(80, padded_window_size, 16000_f32, 20.0, 0.0, device)?;
        let mel_energies = mel_energies.pad_with_zeros(D::Minus1, 0, 1)?.t()?;
        let campplus_model_path = save_dir.to_string()
            + "/iic/speech_campplus_sv_zh-cn_16k-common/campplus_cn_common.bin";
        let campplus_vb = get_vb_model_path(campplus_model_path, dtype_f32, device.clone(), None)?;
        let campplus_model = CAMPPlus::new(campplus_vb, 80, 192, 32, 4, 128)?;
        let s2mel = MyModel::new(path, &config.s2mel, dtype_f32, device)?;
        let emo_matrix_path = path.to_string() + "/" + &config.emo_matrix;
        let emo_num = config.emo_num.clone();
        let t_emo = load_tensor_from_pt(
            &emo_matrix_path,
            "feat2/data/0",
            Shape::from_dims(&[73, 1280]),
            device,
        )?;
        let emo_matrix = split_tensor(&t_emo, &emo_num, 0)?;
        let skp_matrix_path = path.to_string() + "/" + &config.spk_matrix;
        let t_spk = load_tensor_from_pt(
            &skp_matrix_path,
            "feat1/data/0",
            Shape::from_dims(&[73, 192]),
            device,
        )?;
        let spk_matrix = split_tensor(&t_spk, &emo_num, 0)?;
        let gpt_path = path.to_string() + "/" + &config.gpt_checkpoint;
        let gpt_dict = read_all_with_key(gpt_path, None)?;
        let mut dict_to_hashmap = HashMap::new();
        for (k, v) in gpt_dict {
            dict_to_hashmap.insert(k, v);
        }
        let gpt_vb = VarBuilder::from_tensors(dict_to_hashmap, gpt_dtype, device);
        let gpt = UnifiedVoice::new(gpt_vb, &config.gpt)?;
        let bigvgan_model_path = save_dir.to_string()
            + "/nv-community/bigvgan_v2_22khz_80band_256x/bigvgan_generator.pt";
        let bigvgan_vb = get_vb_model_path(
            bigvgan_model_path,
            dtype_f32,
            device.clone(),
            Some("generator"),
        )?;
        let bigvgan_config_path =
            save_dir.to_string() + "/nv-community/bigvgan_v2_22khz_80band_256x/config.json";
        let bigvgan_cfg: BigVGANConfig =
            serde_json::from_slice(&std::fs::read(bigvgan_config_path)?)?;
        let bigvgan = BigVGAN::new(bigvgan_vb, &bigvgan_cfg)?;
        Ok(Self {
            max_audio_length_seconds: 15,
            spk_cache: None,
            emo_cache: None,
            feature_extractor,
            semantic_model,
            semantic_mean,
            semantic_std,
            semantic_codec,
            s2mel_filters,
            s2mel_windows,
            s2mel_preprocess_params: config.s2mel.preprocess_params.clone(),
            window_shift,
            window_size,
            padded_window_size,
            mel_energies,
            campplus_model,
            s2mel,
            emo_num,
            emo_matrix,
            spk_matrix,
            gpt,
            device: device.clone(),
            dtype_f32,
            // gpt_dtype,
            bigvgan,
        })
    }

    pub fn get_emb(
        &self,
        input_features: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let output =
            self.semantic_model
                .forward(input_features, attention_mask, Some(17), false)?;

        let feature = &output.specify_layer_id_hidden_state.unwrap();
        let feature = feature
            .broadcast_sub(&self.semantic_mean)?
            .broadcast_div(&self.semantic_std)?;
        Ok(feature)
    }

    pub fn s2mel_spectrogram(&self, waveform: &Tensor) -> Result<Tensor> {
        let pad = (self.s2mel_preprocess_params.spect_params.n_fft
            - self.s2mel_preprocess_params.spect_params.hop_length)
            / 2;
        let pad_audio_22k = pad_reflect_last_dim(waveform, (pad, pad))?;
        let spec = torch_stft(
            &pad_audio_22k,
            self.s2mel_preprocess_params.spect_params.n_fft,
            self.s2mel_preprocess_params.spect_params.hop_length,
            &self.s2mel_windows,
        )?
        .transpose(1, 2)?;
        let spec = self.s2mel_filters.broadcast_matmul(&spec)?;
        let spec = spec.clamp(1e-5, f64::INFINITY)?.log()?;
        Ok(spec)
    }

    pub fn cut_audio(&self, audio: &Tensor, sr: usize) -> Result<(Tensor, usize)> {
        let max_audio_samples = self.max_audio_length_seconds * sr;
        let audio_lens = audio.dim(1)?;
        let audio = if audio_lens > max_audio_samples {
            audio.i((.., 0..max_audio_samples))?
        } else {
            audio.clone()
        };
        Ok((audio, sr))
    }

    pub fn process_spk_info(&self, audio_url: &str) -> Result<(Tensor, Tensor)> {
        let (audio, sr) = load_audio(audio_url, &self.device)?;
        let (audio, sr) = self.cut_audio(&audio, sr)?;
        let audio_22k = resample_simple(&audio, sr as i64, 22050)?;
        let audio_16k = resample_simple(&audio, sr as i64, 16000)?;

        Ok((audio_22k, audio_16k))
    }

    pub fn process_emo_info(&self, audio_url: &str) -> Result<Tensor> {
        let (audio, sr) = load_audio(audio_url, &self.device)?;
        let (audio, sr) = self.cut_audio(&audio, sr)?;
        let audio_16k = resample_simple(&audio, sr as i64, 16000)?;

        Ok(audio_16k)
    }

    fn is_use_spk_cache(&self, audio_url_vec: &[String]) -> bool {
        let mut flag = false;
        if audio_url_vec.is_empty() && self.spk_cache.is_some() {
            flag = true;
        }
        if !audio_url_vec.is_empty()
            && let Some(cache) = self.spk_cache.as_ref()
            && cache.cache_spk_audio_prompt.eq(&audio_url_vec[0])
        {
            flag = true;
        }
        flag
    }
    fn is_use_emo_cache(&self, audio_url_vec: &[String]) -> bool {
        let mut flag = false;
        if audio_url_vec.len() < 2 && self.emo_cache.is_some() {
            flag = true;
        }
        if audio_url_vec.len() >= 2
            && let Some(cache) = self.emo_cache.as_ref()
            && cache.cache_emo_audio_prompt.eq(&audio_url_vec[1])
        {
            flag = true;
        }
        if audio_url_vec.len() == 1
            && let Some(cache) = self.emo_cache.as_ref()
            && cache.cache_emo_audio_prompt.eq(&audio_url_vec[0])
        {
            flag = true;
        }
        flag
    }

    fn find_most_similar_cosine(&self, query_vector: &Tensor) -> Result<Vec<usize>> {
        let mut index = vec![];
        for temp in self.spk_matrix.iter() {
            let similarities = cosine_similarity(query_vector, temp)?.squeeze(0)?;
            let max_idx = similarities.argmax(0)?.to_scalar::<u32>()? as usize;
            index.push(max_idx);
        }
        Ok(index)
    }

    fn process_emo_vec(
        &self,
        emo_vec: Vec<f32>,
        use_random: bool,
        style: &Tensor,
        emo_weight: f64,
    ) -> Result<(Tensor, Tensor)> {
        let weight_vector = Tensor::new(emo_vec, &self.device)?.affine(emo_weight, 0.0)?;
        let mut rng = rand::rng();
        let random_index = if use_random {
            self.emo_num
                .iter()
                .map(|&x| rng.random_range(0..x.max(1)))
                .collect::<Vec<usize>>()
        } else {
            self.find_most_similar_cosine(style)?
        };
        let mut emo_matrix = vec![];
        for (i, &index) in random_index.iter().enumerate() {
            let tmp = self.emo_matrix[i].as_ref().i(index)?.unsqueeze(0)?;
            emo_matrix.push(tmp);
        }
        let emo_matrix = Tensor::cat(&emo_matrix, 0)?;
        let emo_vec_mat = weight_vector.unsqueeze(1)?.broadcast_mul(&emo_matrix)?;
        let emo_vec_mat = emo_vec_mat.sum(0)?;
        let emo_vec_mat = emo_vec_mat.unsqueeze(0)?;
        Ok((emo_vec_mat, weight_vector))
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        mes: &ChatCompletionParameters,
    ) -> Result<Tensor> {
        let audio_url_vec = extract_audio_url(mes);
        if audio_url_vec.is_empty() && self.spk_cache.is_none() && self.emo_cache.is_none() {
            return Err(anyhow!(
                "Missing audio input: please provide an audio prompt URL or initialize the speaker cache first"
            ));
        }
        let (spk_cond_emb, style, prompt_condition, ref_mel) =
            if self.is_use_spk_cache(&audio_url_vec) {
                let cache = self.spk_cache.as_ref().unwrap();
                (
                    cache.cache_spk_cond.clone(),
                    cache.cache_s2mel_style.clone(),
                    cache.cache_s2mel_prompt.clone(),
                    cache.cache_mel.clone(),
                )
            } else {
                let (audio_22k, audio_16k) = self.process_spk_info(&audio_url_vec[0])?;
                let (audio_16k_features, audio_16k_mask) =
                    self.feature_extractor.call(&audio_16k, 16000, true, true)?;
                let spk_cond_emb = self.get_emb(&audio_16k_features, audio_16k_mask.as_ref())?;
                let (_, s_ref) = self.semantic_codec.quantize(&spk_cond_emb)?;
                let ref_mel = self.s2mel_spectrogram(&audio_22k)?;
                let ref_target_lengths = Tensor::new(ref_mel.dim(2)? as u32, ref_mel.device())?;
                let feat = kaldi_fbank(
                    &audio_16k,
                    &self.mel_energies,
                    self.window_shift,
                    self.window_size,
                    self.padded_window_size,
                    0.0,
                )?
                .squeeze(0)?;
                let feat = feat.broadcast_sub(&feat.mean_keepdim(0)?)?.unsqueeze(0)?;
                let style = self.campplus_model.forward(&feat)?;
                let prompt_condition = self
                    .s2mel
                    .length_regulator
                    .forward(&s_ref, &ref_target_lengths)?;
                let cache = IndexTTS2SpkCache {
                    cache_spk_cond: spk_cond_emb.clone(),
                    cache_s2mel_style: style.clone(),
                    cache_s2mel_prompt: prompt_condition.clone(),
                    cache_mel: ref_mel.clone(),
                    cache_spk_audio_prompt: audio_url_vec[0].clone(),
                };
                self.spk_cache = Some(cache);
                (spk_cond_emb, style, prompt_condition, ref_mel)
            };

        let emo_vector = if let Some(map) = &mes.metadata
            && let Some(emo_vector_str) = map.get("emo_vector")
        {
            serde_json::from_str::<Vec<f32>>(emo_vector_str).ok()
            // match serde_json::from_str::<Vec<f32>>(emo_vector_str) {
            //     Ok(emo_vector) => Some(emo_vector),
            //     Err(_) => None,
            // }
        } else {
            None
        };

        let use_random = if let Some(map) = &mes.metadata
            && let Some(use_random) = map.get("use_random")
        {
            use_random.parse::<bool>().unwrap_or(false)
        } else {
            false
        };
        let emo_weight = if let Some(map) = &mes.metadata
            && let Some(emo_weight) = map.get("emo_weight")
        {
            emo_weight.parse::<f64>().unwrap_or(1.0)
        } else {
            1.0
        };
        let (emovec_mat, weight_vector) = if let Some(emo_vector) = emo_vector {
            let (emovec_mat, weight_vector) =
                self.process_emo_vec(emo_vector, use_random, &style, emo_weight)?;
            (Some(emovec_mat), Some(weight_vector))
        } else {
            (None, None)
        };

        let emo_cond_emb = if self.is_use_emo_cache(&audio_url_vec) {
            let cache = self.emo_cache.as_ref().unwrap();
            cache.emo_cond_emb.clone()
        } else {
            let emo_audio_prompt = if audio_url_vec.len() >= 2 {
                audio_url_vec[1].clone()
            } else {
                audio_url_vec[0].clone()
            };
            let emo_audio_16k = self.process_emo_info(&emo_audio_prompt)?;
            let (emo_audio_16k_features, emo_audio_16k_mask) =
                self.feature_extractor
                    .call(&emo_audio_16k, 16000, true, true)?;
            let emo_cond_emb =
                self.get_emb(&emo_audio_16k_features, emo_audio_16k_mask.as_ref())?;
            let cache = IndexTTS2EmoCache {
                cache_emo_audio_prompt: emo_audio_prompt,
                emo_cond_emb: emo_cond_emb.clone(),
            };
            self.emo_cache = Some(cache);
            emo_cond_emb
        };
        let mut emovec = self
            .gpt
            .merge_emovec(&spk_cond_emb, &emo_cond_emb, emo_weight)?;
        if let Some(weight_vector) = weight_vector
            && let Some(emovec_mat) = emovec_mat
        {
            let ratio = 1.0 - weight_vector.sum_all()?.to_scalar::<f32>()?;
            emovec = emovec
                .affine(ratio as f64, 0.0)?
                .add(&emovec_mat.to_dtype(emovec.dtype())?)?;
        }
        let (codes, speech_conditioning_latent) =
            self.gpt
                .inference_speech(&spk_cond_emb, input_ids, &emovec)?;
        let code_len = codes.dim(0)?;
        let codes = codes.unsqueeze(0)?;
        let use_speed = Tensor::zeros(spk_cond_emb.dim(0)?, DType::U32, &self.device)?;
        let latent = self.gpt.forward(
            &speech_conditioning_latent,
            input_ids,
            &codes,
            &emovec,
            &use_speed,
        )?;
        let latent = self
            .s2mel
            .gpt_layer
            .forward(&latent.to_dtype(self.dtype_f32)?)?;
        let s_infer = self.semantic_codec.quantizer.vq2emb(&codes)?;
        let s_infer = s_infer.transpose(1, 2)?;
        let s_infer = s_infer.add(&latent)?;
        let target_len = (code_len as f32 * 1.72) as u32;
        let target_len = Tensor::new(target_len, &self.device)?;
        let code = self.s2mel.length_regulator.forward(&s_infer, &target_len)?;
        let cat_condition = Tensor::cat(&[&prompt_condition, &code], 1)?;
        let x_lens = Tensor::new(vec![cat_condition.dim(1)? as u32], &self.device)?;
        let vc_target =
            self.s2mel
                .cfm
                .inference(&cat_condition, &x_lens, &ref_mel, &style, 25, 0.7)?;
        let ref_mel_len = ref_mel.dim(D::Minus1)?;
        let vc_target_len = vc_target.dim(D::Minus1)?;
        let vc_target = vc_target.narrow(D::Minus1, ref_mel_len, vc_target_len - ref_mel_len)?;
        let wav = self.bigvgan.forward(&vc_target)?.squeeze(0)?;
        Ok(wav)
    }
}
