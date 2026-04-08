use std::{cmp::max, collections::HashMap, f64};

use anyhow::{Ok, Result};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, linear_no_bias};
use candle_transformers::models::deepseek2::SplitOp;

use crate::{
    models::voxcpm::{
        audio_vae::AudioVAE,
        config::{CfmConfig, VoxCPMConfig, VoxMiniCPM4Config},
        minicpm4::MiniCPMModel,
        tokenizer::SingleChineseTokenizer,
    },
    utils::{audio_utils::load_audio_with_resample, tensor_utils::linspace},
};

pub struct ScalarQuantizationLayer {
    scale: usize,
    in_proj: Linear,
    out_proj: Linear,
}

impl ScalarQuantizationLayer {
    pub fn new(
        vb: VarBuilder,
        in_dim: usize,
        out_dim: usize,
        laten_dim: usize,
        scale: usize,
    ) -> Result<Self> {
        let in_proj = linear(in_dim, laten_dim, vb.pp("in_proj"))?;
        let out_proj = linear(laten_dim, out_dim, vb.pp("out_proj"))?;
        Ok(Self {
            scale,
            in_proj,
            out_proj,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.in_proj.forward(xs)?;
        let xs = xs.tanh()?;
        let xs = xs
            .affine(self.scale as f64, 0.0)?
            .round()?
            .affine(1.0 / self.scale as f64, 0.0)?;
        let xs = self.out_proj.forward(&xs)?;
        Ok(xs)
    }
}

pub struct SinusoidalPosEmb {
    dim: usize,
}

impl SinusoidalPosEmb {
    pub fn new(dim: usize) -> Result<Self> {
        assert_eq!(dim % 2, 0, "SinusoidalPosEmb requires dim to be even");
        Ok(Self { dim })
    }
    pub fn forward(&self, x: &Tensor, scale: usize) -> Result<Tensor> {
        let x = if x.rank() < 1 {
            x.unsqueeze(0)?
        } else {
            x.clone()
        };
        let half_dim = self.dim / 2;
        let dif = 10000.0_f64.ln() / (half_dim - 1) as f64;
        let emb = Tensor::arange(0.0, half_dim as f32, x.device())?
            .affine(-dif, 0.0)?
            .exp()?
            .to_dtype(x.dtype())?;
        let emb = x
            .unsqueeze(1)?
            .contiguous()?
            .affine(scale as f64, 0.0)?
            .matmul(&emb.unsqueeze(0)?.contiguous()?)?;
        let emb = Tensor::cat(&[emb.sin()?, emb.cos()?], D::Minus1)?;
        Ok(emb)
    }
}

pub struct TimestepEmbedding {
    linear_1: Linear,
    linear_2: Linear,
}

impl TimestepEmbedding {
    pub fn new(
        vb: VarBuilder,
        in_channels: usize,
        time_embed_dim: usize,
        out_dim: Option<usize>,
    ) -> Result<Self> {
        let linear_1 = linear(in_channels, time_embed_dim, vb.pp("linear_1"))?;
        let time_embed_dim_out = if let Some(out_dim) = out_dim {
            out_dim
        } else {
            time_embed_dim
        };
        let linear_2 = linear(time_embed_dim, time_embed_dim_out, vb.pp("linear_2"))?;
        Ok(Self { linear_1, linear_2 })
    }

    pub fn forward(&self, sample: &Tensor) -> Result<Tensor> {
        let sample = self.linear_1.forward(sample)?.silu()?;
        let sample = self.linear_2.forward(&sample)?;
        Ok(sample)
    }
}

pub struct VoxCPMLocDiT {
    in_proj: Linear,
    cond_proj: Linear,
    out_proj: Linear,
    time_embeddings: SinusoidalPosEmb,
    time_mlp: TimestepEmbedding,
    delta_time_mlp: TimestepEmbedding,
    decoder: MiniCPMModel,
    version: usize,
    // config: VoxMiniCPM4Config,
    // in_channels: usize,
}

impl VoxCPMLocDiT {
    pub fn new(vb: VarBuilder, config: VoxMiniCPM4Config, in_channels: usize) -> Result<Self> {
        let in_proj = linear(in_channels, config.hidden_size, vb.pp("in_proj"))?;
        let cond_proj = linear(in_channels, config.hidden_size, vb.pp("cond_proj"))?;
        let out_proj = linear(config.hidden_size, in_channels, vb.pp("out_proj"))?;
        let time_embeddings = SinusoidalPosEmb::new(config.hidden_size)?;
        let time_mlp = TimestepEmbedding::new(
            vb.pp("time_mlp"),
            config.hidden_size,
            config.hidden_size,
            None,
        )?;
        let delta_time_mlp = TimestepEmbedding::new(
            vb.pp("delta_time_mlp"),
            config.hidden_size,
            config.hidden_size,
            None,
        )?;
        assert_eq!(config.vocab_size, 0, "vocab_size must be 0 for local DiT");
        let decoder = MiniCPMModel::new(vb.pp("decoder"), config.clone())?;
        let version = if config.kv_channels.is_some() {
            2usize
        } else {
            1
        };
        Ok(Self {
            in_proj,
            cond_proj,
            out_proj,
            time_embeddings,
            time_mlp,
            delta_time_mlp,
            decoder,
            version,
            // config,
            // in_channels,
        })
    }

    pub fn forward(
        &mut self,
        x: &Tensor,
        mu: &Tensor,
        t: &Tensor,
        cond: &Tensor,
        dt: &Tensor,
    ) -> Result<Tensor> {
        let x = self.in_proj.forward(&x.transpose(1, 2)?.contiguous()?)?;
        let cond = self
            .cond_proj
            .forward(&cond.transpose(1, 2)?.contiguous()?)?;
        let prefix = cond.dim(1)?;
        let t = self.time_embeddings.forward(t, 1000)?.to_dtype(x.dtype())?;
        let t = self.time_mlp.forward(&t)?;
        let dt = self
            .time_embeddings
            .forward(dt, 1000)?
            .to_dtype(x.dtype())?;
        let dt = self.delta_time_mlp.forward(&dt)?;
        let t = t.add(&dt)?;
        let hidden = if self.version == 2 {
            let (b, _, dim) = x.dims3()?;
            let mu = mu.reshape((b, (), dim))?;
            let x = Tensor::cat(&[&mu, &t.unsqueeze(1)?, &cond, &x], 1)?;
            let hidden = self.decoder.forward(&x, 0, false)?;
            let select_len = hidden.dim(1)? - (prefix + mu.dim(1)? + 1);
            hidden.narrow(1, prefix + mu.dim(1)? + 1, select_len)?
        } else {
            let x = Tensor::cat(&[mu.add(&t)?.unsqueeze(1)?, cond, x], 1)?;
            let hidden = self.decoder.forward(&x, 0, false)?;
            let select_len = hidden.dim(1)? - (prefix + 1);
            hidden.narrow(1, prefix + 1, select_len)?
        };
        let hidden = self.out_proj.forward(&hidden)?;
        let hidden = hidden.transpose(1, 2)?.contiguous()?;
        Ok(hidden)
    }
}

pub struct UnifiedCFM {
    // solver: String,
    // sigma_min: f32,
    // t_scheduler: String,
    in_channels: usize,
    mean_mode: bool,
    estimator: VoxCPMLocDiT,
    // architecture: String,
}

impl UnifiedCFM {
    pub fn new(
        in_channels: usize,
        _cfm_params: CfmConfig,
        estimator: VoxCPMLocDiT,
        mean_mode: bool,
        // architecture: String,
    ) -> Result<Self> {
        // let solver = cfm_params.solver;
        // let sigma_min = cfm_params.sigma_min;
        // let t_scheduler = cfm_params.t_scheduler;
        Ok(Self {
            // solver,
            // sigma_min,
            // t_scheduler,
            in_channels,
            mean_mode,
            estimator,
            // architecture,
        })
    }

    pub fn forward(
        &mut self,
        mu: &Tensor,
        n_timesteps: usize,
        patch_size: usize,
        cond: &Tensor,
        temperature: f64,
        cfg_value: f64,
        sway_sampling_coef: f64,
        use_cfg_zero_star: bool,
    ) -> Result<Tensor> {
        let (b, _) = mu.dims2()?;
        let t = patch_size;
        let dtype = mu.dtype();
        let z = Tensor::randn(0.0f32, 1.0, (b, self.in_channels, t), mu.device())?
            .to_dtype(dtype)?
            .affine(temperature, 0.0)?;
        // let z = Tensor::ones((b, self.in_channels, t), dtype, mu.device())?;
        let t_span = linspace(1.0, 0.0, n_timesteps + 1, mu.device())?.to_dtype(dtype)?;
        let t_span = t_span
            .affine(f64::consts::PI / 2.0, 0.0)?
            .cos()?
            .affine(1.0, -1.0)?
            .add(&t_span)?
            .affine(sway_sampling_coef, 0.0)?
            .add(&t_span)?;
        let x = self.solve_euler(&z, &t_span, mu, cond, cfg_value, use_cfg_zero_star)?;
        Ok(x)
    }

    pub fn optimized_scale(
        &self,
        positive_flat: &Tensor,
        negative_flat: &Tensor,
    ) -> Result<Tensor> {
        let dot_product = positive_flat.mul(negative_flat)?.sum_keepdim(1)?;
        let squared_norm = negative_flat.powf(2.0)?.sum_keepdim(1)?.affine(1.0, 1e-8)?;
        let st_star = dot_product.div(&squared_norm)?;
        Ok(st_star)
    }

    pub fn solve_euler(
        &mut self,
        x: &Tensor,
        t_span: &Tensor,
        mu: &Tensor,
        cond: &Tensor,
        cfg_value: f64,
        use_cfg_zero_star: bool,
    ) -> Result<Tensor> {
        let mut t = t_span.i(0)?;
        let mut dt = t.sub(&t_span.i(1)?)?;
        let mut sol = Vec::new();
        let t_span_len = t_span.dim(0)?;
        let zero_init_steps = max(1, (t_span_len as f32 * 0.04) as usize);
        let mut dphi_dt;
        let mut x = x.clone();
        for step in 1..t_span_len {
            if use_cfg_zero_star && step <= zero_init_steps {
                // dphi_dt = Tensor::zeros(1, t_span.dtype(), t_span.device())?;
                dphi_dt = x.zeros_like()?;
            } else {
                let b = x.dim(0)?;
                // let x_in = Tensor::zeros((2*b, self.in_channels, x.dim(2)?), x.dtype(), x.device())?;
                let x_in = Tensor::cat(&[x.clone(), x.clone()], 0)?;
                let mu_in = Tensor::zeros((b, mu.dim(1)?), x.dtype(), x.device())?;
                let mu_in = Tensor::cat(&[mu.clone(), mu_in], 0)?;
                let t_in = t.broadcast_as(2 * b)?;
                let dt_in = if self.mean_mode {
                    dt.broadcast_as(2 * b)?
                } else {
                    Tensor::zeros(2 * b, x.dtype(), x.device())?
                };
                let cond_in = Tensor::cat(&[cond, cond], 0)?;
                dphi_dt = self
                    .estimator
                    .forward(&x_in, &mu_in, &t_in, &cond_in, &dt_in)?;
                let split = dphi_dt.split(&[b, b], 0)?;
                dphi_dt = split[0].clone();
                let cfg_dphi_dt = split[1].clone();
                let mut st_star = Tensor::ones(1, x.dtype(), x.device())?;
                if use_cfg_zero_star {
                    let positive_flat = dphi_dt.reshape((b, ()))?;
                    let negative_flat = cfg_dphi_dt.reshape((b, ()))?;
                    st_star = self.optimized_scale(&positive_flat, &negative_flat)?;
                    let mut vec_shape = vec![b];
                    let vec_shape1 = vec![1; dphi_dt.rank() - 1];
                    vec_shape.extend_from_slice(&vec_shape1);
                    st_star = st_star.reshape(vec_shape)?;
                }
                let cfg = cfg_dphi_dt.broadcast_mul(&st_star)?;
                dphi_dt = cfg.add(&dphi_dt.sub(&cfg)?.affine(cfg_value, 0.0)?)?; // step步的预测噪声
            }
            x = x.broadcast_sub(&dphi_dt.broadcast_mul(&dt)?)?; // 逐步去噪
            t = t.sub(&dt)?;
            sol.push(x.clone());
            if step < t_span_len - 1 {
                dt = t.sub(&t_span.i(step + 1)?)?;
            }
        }
        let ret = sol[sol.len() - 1].clone();
        Ok(ret)
    }
}

pub struct VoxCPMLocEnc {
    special_token: Tensor,
    in_proj: Linear,
    encoder: MiniCPMModel,
    hidden_size: usize,
}

impl VoxCPMLocEnc {
    pub fn new(vb: VarBuilder, config: VoxMiniCPM4Config, input_dim: usize) -> Result<Self> {
        let special_token = vb.get((1, 1, 1, config.hidden_size), "special_token")?;
        let in_proj = linear(input_dim, config.hidden_size, vb.pp("in_proj"))?;
        assert_eq!(
            config.vocab_size, 0,
            "vocab_size must be 0 for local encoder"
        );
        let hidden_size = config.hidden_size;
        let encoder = MiniCPMModel::new(vb.pp("encoder"), config)?;
        Ok(Self {
            special_token,
            in_proj,
            encoder,
            hidden_size,
        })
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _, _) = x.dims4()?;
        let x = self.in_proj.forward(x)?;
        let special_tokens = self.special_token.expand((b, t, 1, self.hidden_size))?;
        let x = Tensor::cat(&[special_tokens, x], 2)?;
        let (b, t, p, c) = x.dims4()?;
        let x = x.reshape((b * t, p, c))?;
        let outputs = self.encoder.forward(&x, 0, false)?;
        let cls_output = outputs.i((.., 0, ..))?;
        let cls_output = cls_output.reshape((b, t, c))?;
        Ok(cls_output)
    }
}

pub struct VoxCPMModel {
    config: VoxCPMConfig,
    patch_size: usize,
    audio_start_token: u32,
    // audio_end_token: u32,
    ref_audio_start_token: u32,
    ref_audio_end_token: u32,
    chunk_size: usize,
    sample_rate: usize,
    tokenizer: SingleChineseTokenizer,
    audio_vae: AudioVAE,
    base_lm: MiniCPMModel,
    residual_lm: MiniCPMModel,
    feat_encoder: VoxCPMLocEnc,
    feat_decoder: UnifiedCFM,
    fsq_layer: ScalarQuantizationLayer,
    enc_to_lm_proj: Linear,
    lm_to_dit_proj: Linear,
    res_to_dit_proj: Linear,
    fusion_concat_proj: Option<Linear>,
    stop_proj: Linear,
    stop_head: Linear,
    device: Device,
    dtype: DType,
}

impl VoxCPMModel {
    pub fn new(
        vb: VarBuilder,
        config: VoxCPMConfig,
        tokenizer: SingleChineseTokenizer,
        audio_vae: AudioVAE,
    ) -> Result<Self> {
        let base_lm = MiniCPMModel::new(vb.pp("base_lm"), config.lm_config.clone())?;
        let mut residual_lm_config = config.lm_config.clone();
        residual_lm_config.num_hidden_layers = config.residual_lm_num_layers;
        residual_lm_config.vocab_size = 0;
        residual_lm_config.no_rope = config.residual_lm_no_rope;
        let residual_lm = MiniCPMModel::new(vb.pp("residual_lm"), residual_lm_config)?;
        let mut encoder_config = config.lm_config.clone();
        encoder_config.hidden_size = config.encoder_config.hidden_dim;
        encoder_config.intermediate_size = config.encoder_config.ffn_dim;
        encoder_config.num_attention_heads = config.encoder_config.num_heads;
        encoder_config.num_hidden_layers = config.encoder_config.num_layers;
        encoder_config.kv_channels = config.encoder_config.kv_channels;
        encoder_config.vocab_size = 0;
        let feat_encoder =
            VoxCPMLocEnc::new(vb.pp("feat_encoder"), encoder_config, config.feat_dim)?;

        let mut decoder_config = config.lm_config.clone();
        decoder_config.hidden_size = config.dit_config.hidden_dim;
        decoder_config.intermediate_size = config.dit_config.ffn_dim;
        decoder_config.num_attention_heads = config.dit_config.num_heads;
        decoder_config.num_hidden_layers = config.dit_config.num_layers;
        decoder_config.kv_channels = config.dit_config.kv_channels;
        decoder_config.vocab_size = 0;
        let estimator = VoxCPMLocDiT::new(
            vb.pp("feat_decoder.estimator"),
            decoder_config,
            config.feat_dim,
        )?;
        let feat_decoder = UnifiedCFM::new(
            config.feat_dim,
            config.dit_config.cfm_config.clone(),
            estimator,
            false,
            // config.architecture.clone(),
        )?;
        let fsq_layer = ScalarQuantizationLayer::new(
            vb.pp("fsq_layer"),
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            config.scalar_quantization_latent_dim,
            config.scalar_quantization_scale,
        )?;
        let enc_to_lm_proj = linear(
            config.encoder_config.hidden_dim,
            config.lm_config.hidden_size,
            vb.pp("enc_to_lm_proj"),
        )?;
        let lm_to_dit_proj = linear(
            config.lm_config.hidden_size,
            config.dit_config.hidden_dim,
            vb.pp("lm_to_dit_proj"),
        )?;
        let res_to_dit_proj = linear(
            config.lm_config.hidden_size,
            config.dit_config.hidden_dim,
            vb.pp("res_to_dit_proj"),
        )?;

        let fusion_concat_proj = if config.architecture.to_lowercase().eq("voxcpm2") {
            Some(linear(
                config.lm_config.hidden_size * 2,
                config.lm_config.hidden_size,
                vb.pp("fusion_concat_proj"),
            )?)
        } else {
            None
        };

        let stop_proj = linear(
            config.lm_config.hidden_size,
            config.lm_config.hidden_size,
            vb.pp("stop_proj"),
        )?;
        let stop_head = linear_no_bias(config.lm_config.hidden_size, 2, vb.pp("stop_head"))?;

        let patch_size = config.patch_size;
        Ok(Self {
            config,
            patch_size,
            audio_start_token: 101,
            // audio_end_token: 102,
            ref_audio_start_token: 103,
            ref_audio_end_token: 104,
            chunk_size: audio_vae.chunk_size,
            sample_rate: audio_vae.sample_rate,
            tokenizer,
            audio_vae,
            base_lm,
            residual_lm,
            feat_encoder,
            feat_decoder,
            fsq_layer,
            enc_to_lm_proj,
            lm_to_dit_proj,
            res_to_dit_proj,
            fusion_concat_proj,
            stop_proj,
            stop_head,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn generate(
        &mut self,
        target_text: String,
        prompt_text: Option<String>,
        prompt_wav_path: Option<String>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        // retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        let (text_token, text_mask, audio_feat, audio_mask) = if let Some(prompt_text) = prompt_text
            && let Some(path) = prompt_wav_path
        {
            let text = prompt_text + &target_text;
            let text_token = self.tokenizer.encode(text)?;
            let text_token = Tensor::from_slice(&text_token, text_token.len(), &self.device)?;
            let audio_start = Tensor::new(vec![self.audio_start_token], &self.device)?;
            let text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
            let text_length = text_token.dim(0)?;
            let mut audio = load_audio_with_resample(&path, &self.device, Some(self.sample_rate))?;
            let patch_len = self.patch_size * self.chunk_size;
            if audio.dim(1)? % patch_len != 0 {
                audio =
                    audio.pad_with_zeros(D::Minus1, patch_len - audio.dim(1)? % patch_len, 0)?;
            }
            let audio_feat = self.audio_vae.encode(&audio, Some(self.sample_rate))?;
            let audio_feat = audio_feat
                .reshape((self.audio_vae.latent_dim, (), self.patch_size))?
                .permute((1, 2, 0))?;
            let audio_length = audio_feat.dim(0)?;
            let text_pad_token = Tensor::zeros(audio_length, DType::U32, &self.device)?;
            let text_token = Tensor::cat(&[text_token, text_pad_token], D::Minus1)?;
            let audio_pad_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                audio_feat.dtype(),
                &self.device,
            )?;
            let audio_feat = Tensor::cat(&[audio_pad_feat, audio_feat], 0)?;
            let text_mask = Tensor::cat(
                &[
                    Tensor::ones(text_length, self.dtype, &self.device)?,
                    Tensor::zeros(audio_length, self.dtype, &self.device)?,
                ],
                D::Minus1,
            )?;
            let audio_mask = Tensor::cat(
                &[
                    Tensor::zeros(text_length, self.dtype, &self.device)?,
                    Tensor::ones(audio_length, self.dtype, &self.device)?,
                ],
                D::Minus1,
            )?;
            (text_token, text_mask, audio_feat, audio_mask)
        } else if let Some(path) = prompt_wav_path {
            let text_token = self.tokenizer.encode(target_text.clone())?;
            let text_token = Tensor::from_slice(&text_token, text_token.len(), &self.device)?;
            let audio_start = Tensor::new(vec![self.audio_start_token], &self.device)?;
            let text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
            let text_length = text_token.dim(0)?;
            let mut audio = load_audio_with_resample(&path, &self.device, Some(self.sample_rate))?;
            let patch_len = self.patch_size * self.chunk_size;
            if audio.dim(1)? % patch_len != 0 {
                audio =
                    audio.pad_with_zeros(D::Minus1, 0, patch_len - audio.dim(1)? % patch_len)?;
            }
            let audio_feat = self.audio_vae.encode(&audio, Some(self.sample_rate))?;
            let audio_feat = audio_feat
                .reshape((self.audio_vae.latent_dim, (), self.patch_size))?
                .permute((1, 2, 0))?;
            let ref_len = audio_feat.dim(0)?;
            let z1 = Tensor::zeros(
                (1, self.patch_size, self.audio_vae.latent_dim),
                DType::F32,
                &self.device,
            )?;
            let ref_start = Tensor::new(vec![self.ref_audio_start_token], &self.device)?;
            let ref_end = Tensor::new(vec![self.ref_audio_end_token], &self.device)?;
            let ref_token = Tensor::zeros(ref_len, DType::U32, &self.device)?;
            let ref_tokens = Tensor::cat(&[&ref_start, &ref_token, &ref_end], 0)?;
            let feats = Tensor::cat(&[&z1, &audio_feat, &z1], 0)?;
            let t_mask = Tensor::cat(
                &[
                    Tensor::new(vec![1.0f32], &self.device)?.to_dtype(self.dtype)?,
                    Tensor::zeros(ref_len, self.dtype, &self.device)?,
                    Tensor::new(vec![1.0f32], &self.device)?.to_dtype(self.dtype)?,
                ],
                0,
            )?;
            let a_mask = Tensor::cat(
                &[
                    Tensor::new(vec![0.0f32], &self.device)?.to_dtype(self.dtype)?,
                    Tensor::ones(ref_len, self.dtype, &self.device)?,
                    Tensor::new(vec![0.0f32], &self.device)?.to_dtype(self.dtype)?,
                ],
                0,
            )?;
            let text_pad_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                DType::F32,
                &self.device,
            )?;
            let text_token = Tensor::cat(&[&ref_tokens, &text_token], 0)?;
            let audio_feat = Tensor::cat(&[&feats, &text_pad_feat], 0)?;
            let text_mask = Tensor::cat(
                &[
                    &t_mask,
                    &Tensor::ones(text_length, self.dtype, &self.device)?,
                ],
                0,
            )?;
            let audio_mask = Tensor::cat(
                &[
                    &a_mask,
                    &Tensor::zeros(text_length, self.dtype, &self.device)?,
                ],
                0,
            )?;
            (text_token, text_mask, audio_feat, audio_mask)
        } else {
            let text_token = self.tokenizer.encode(target_text.clone())?;
            let text_token = Tensor::from_slice(&text_token, text_token.len(), &self.device)?;
            let audio_start = Tensor::new(vec![self.audio_start_token], &self.device)?;
            let text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
            let text_length = text_token.dim(0)?;
            let audio_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                DType::F32,
                &self.device,
            )?;
            let text_mask = Tensor::ones(text_length, self.dtype, &self.device)?;
            let audio_mask = Tensor::zeros(text_length, self.dtype, &self.device)?;
            (text_token, text_mask, audio_feat, audio_mask)
        };
        let target_text_length = self.tokenizer.encode(target_text)?.len();
        // let max_len = if retry_badcase {
        //     (target_text_length as f64 * retry_badcase_ratio_threshold + 10.0) as usize
        // } else {
        //     max_len
        // };
        let max_len = max_len
            .min((target_text_length as f64 * retry_badcase_ratio_threshold + 10.0) as usize);
        let decode_audio = self._generate(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        )?;
        Ok(decode_audio)
    }

    fn _generate(
        &mut self,
        text_token: &Tensor,
        text_mask: &Tensor,
        audio_feat: &Tensor,
        audio_mask: &Tensor,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
    ) -> Result<Tensor> {
        let text_token = text_token.unsqueeze(0)?;
        let text_mask = text_mask.unsqueeze(0)?;
        let audio_feat = audio_feat.unsqueeze(0)?.to_dtype(self.dtype)?;
        let audio_mask = audio_mask.unsqueeze(0)?;

        let latent_pred = self.inference(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        )?;
        let decode_audio = self
            .audio_vae
            .decode(&latent_pred.to_dtype(DType::F32)?, None)?
            .squeeze(1)?;
        let decode_audio_len = decode_audio.dim(D::Minus1)? - 640 - 640;
        let decode_audio = decode_audio.narrow(D::Minus1, 640, decode_audio_len)?;
        Ok(decode_audio)
    }

    fn inference(
        &mut self,
        text: &Tensor,
        text_mask: &Tensor,
        feat: &Tensor,
        feat_mask: &Tensor,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
    ) -> Result<Tensor> {
        let (_, t, _, _) = feat.dims4()?;
        let feat_embed = self.feat_encoder.forward(feat)?; // [b, t, h_feat]
        let feat_embed = self.enc_to_lm_proj.forward(&feat_embed)?;
        let scale_emb = if self.config.lm_config.use_mup {
            self.config.lm_config.scale_emb
        } else {
            1.0
        };

        let text_embed = self
            .base_lm
            .embed_tokens
            .as_ref()
            .unwrap()
            .forward(text)?
            .affine(scale_emb as f64, 0.0)?;
        let combined_embed = text_mask
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&text_embed)?
            .add(&feat_mask.unsqueeze(D::Minus1)?.broadcast_mul(&feat_embed)?)?;
        let mut prefix_feat_cond = feat.i((.., t - 1, ..))?;
        let mut pred_feat_seq = Vec::new();
        // if feat_mask.i((1, t-1))?.to_scalar::<f32>()? == 0.0 {
        //     // TODO for stream
        // }
        let mut position_id = 0;
        let mut seq_len = t;
        let enc_outputs = self
            .base_lm
            .forward_with_cache(&combined_embed, position_id)?;
        let enc_outputs = self
            .fsq_layer
            .forward(&enc_outputs)?
            .broadcast_mul(&feat_mask.unsqueeze(D::Minus1)?)?
            .add(&enc_outputs.broadcast_mul(&text_mask.unsqueeze(D::Minus1)?)?)?;
        let mut lm_hidden = enc_outputs.i((.., t - 1, ..))?;
        let input_embeds = if let Some(fusion) = &self.fusion_concat_proj {
            let feat = feat_mask.unsqueeze(D::Minus1)?.broadcast_mul(&feat_embed)?;
            let concat = Tensor::cat(&[&enc_outputs, &feat], D::Minus1)?;
            fusion.forward(&concat)?
        } else {
            enc_outputs.add(&feat_mask.unsqueeze(D::Minus1)?.broadcast_mul(&feat_embed)?)?
        };
        let residual_enc_outputs = self
            .residual_lm
            .forward_with_cache(&input_embeds, position_id)?;
        let mut residual_hidden = residual_enc_outputs.i((.., t - 1, ..))?;
        for i in 0..max_len {
            let dit_hidden_1 = self.lm_to_dit_proj.forward(&lm_hidden)?; // [b, h_dit]
            let dit_hidden_2 = self.res_to_dit_proj.forward(&residual_hidden)?; // [b, h_dit]
            // let dit_hidden = dit_hidden_1.add(&dit_hidden_2)?;
            let dit_hidden = if self.fusion_concat_proj.is_some() {
                Tensor::cat(&[&dit_hidden_1, &dit_hidden_2], D::Minus1)?
            } else {
                dit_hidden_1.add(&dit_hidden_2)?
            };
            let cond = prefix_feat_cond.transpose(1, 2)?.contiguous()?;
            let pred_feat = self
                .feat_decoder
                .forward(
                    &dit_hidden,
                    inference_timesteps,
                    self.patch_size,
                    &cond,
                    1.0,
                    cfg_value,
                    1.0,
                    true,
                )?
                .transpose(1, 2)?; // [b, p, d]
            let curr_embed = self.feat_encoder.forward(&pred_feat.unsqueeze(1)?)?; // [b, 1, c]
            let curr_embed = self.enc_to_lm_proj.forward(&curr_embed)?;
            pred_feat_seq.push(pred_feat.unsqueeze(1)?);

            prefix_feat_cond = pred_feat;
            let stop_flag = self.stop_proj.forward(&lm_hidden)?.silu()?;
            let stop_flag = self
                .stop_head
                .forward(&stop_flag)?
                .argmax(D::Minus1)?
                .i(0)?
                .to_scalar::<u32>()?;
            if i > min_len && stop_flag == 1 {
                break;
            }
            position_id += seq_len;
            seq_len = 1;
            lm_hidden = self
                .base_lm
                .forward_with_cache(&curr_embed.i((.., 0, ..))?, position_id)?
                .squeeze(1)?;
            lm_hidden = self.fsq_layer.forward(&lm_hidden)?;
            let curr_residual_input = if let Some(fusion) = &self.fusion_concat_proj {
                let curr_embed = curr_embed.i((.., 0, ..))?;
                let concat = Tensor::cat(&[&lm_hidden, &curr_embed], D::Minus1)?;
                fusion.forward(&concat)?
            } else {
                lm_hidden.add(&curr_embed.i((.., 0, ..))?)?
            };
            residual_hidden = self
                .residual_lm
                .forward_with_cache(&curr_residual_input, position_id)?
                .squeeze(1)?;
        }
        let pred_seq = Tensor::cat(&pred_feat_seq, 1)?; // (b, t, p, d)
        let (b, _, _, d) = pred_seq.dims4()?;
        let feat_pred = pred_seq
            .permute((0, 3, 1, 2))?
            .reshape((b, d, ()))?
            .contiguous()?;
        self.clear_kv_cache();
        Ok(feat_pred)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_lm.clear_kv_cache();
        self.residual_lm.clear_kv_cache();
    }

    pub fn build_prompt_cache(
        &mut self,
        prompt_text: String,
        prompt_wav_path: String,
    ) -> Result<HashMap<String, Tensor>> {
        let text_token = self.tokenizer.encode(prompt_text)?;
        let text_token = Tensor::from_slice(&text_token, text_token.len(), &self.device)?;
        let mut audio =
            load_audio_with_resample(&prompt_wav_path, &self.device, Some(self.sample_rate))?;
        let patch_len = self.patch_size * self.chunk_size;
        if audio.dim(1)? % patch_len != 0 {
            audio = audio.pad_with_zeros(D::Minus1, 0, patch_len - audio.dim(1)? % patch_len)?;
        }
        let audio_feat = self.audio_vae.encode(&audio, Some(self.sample_rate))?;
        let audio_feat = audio_feat
            .reshape((self.audio_vae.latent_dim, (), self.patch_size))?
            .permute((1, 2, 0))?;
        let dim0 = audio_feat.dim(0)? - 1;
        let audio_feat = audio_feat.i(..dim0)?;
        let mut hashmap = HashMap::new();
        hashmap.insert("text_token".to_string(), text_token);
        hashmap.insert("audio_feat".to_string(), audio_feat);
        Ok(hashmap)
    }

    pub fn generate_with_prompt_cache(
        &mut self,
        target_text: String,
        prompt_cache: HashMap<String, Tensor>,
        min_len: usize,
        max_len: usize,
        inference_timesteps: usize,
        cfg_value: f64,
        retry_badcase: bool,
        retry_badcase_ratio_threshold: f64,
    ) -> Result<Tensor> {
        let target_text_token = self.tokenizer.encode(target_text.clone())?;
        let target_text_token =
            Tensor::from_slice(&target_text_token, target_text_token.len(), &self.device)?;
        let text_token = match prompt_cache.get("text_token") {
            Some(token) => Tensor::cat(&[token, &target_text_token], 0)?,
            None => target_text_token,
        };
        let audio_start = Tensor::new(vec![self.audio_start_token], &self.device)?;
        let text_token = Tensor::cat(&[text_token, audio_start], D::Minus1)?;
        let text_length = text_token.dim(0)?;
        let (audio_length, audio_feat) = match prompt_cache.get("audio_feat") {
            Some(feat) => (feat.dim(0)?, Some(feat.clone())),
            None => (0, None),
        };
        let (text_token, text_mask, audio_feat, audio_mask) = if audio_length > 0 {
            let audio_feat = audio_feat.unwrap();
            let audio_length = audio_feat.dim(0)?;
            let text_pad_token = Tensor::zeros(audio_length, DType::U32, &self.device)?;
            let text_token = Tensor::cat(&[text_token, text_pad_token], D::Minus1)?;
            let audio_pad_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                audio_feat.dtype(),
                &self.device,
            )?;
            let audio_feat = Tensor::cat(&[audio_pad_feat, audio_feat], 0)?;
            let text_mask = Tensor::cat(
                &[
                    Tensor::ones(text_length, self.dtype, &self.device)?,
                    Tensor::zeros(audio_length, self.dtype, &self.device)?,
                ],
                D::Minus1,
            )?;
            let audio_mask = Tensor::cat(
                &[
                    Tensor::zeros(text_length, self.dtype, &self.device)?,
                    Tensor::ones(audio_length, self.dtype, &self.device)?,
                ],
                D::Minus1,
            )?;
            (text_token, text_mask, audio_feat, audio_mask)
        } else {
            let audio_feat = Tensor::zeros(
                (text_length, self.patch_size, self.audio_vae.latent_dim),
                DType::F32,
                &self.device,
            )?;
            let text_mask = Tensor::ones(text_length, self.dtype, &self.device)?;
            let audio_mask = Tensor::zeros(text_length, self.dtype, &self.device)?;
            (text_token, text_mask, audio_feat, audio_mask)
        };
        let target_text_length = self.tokenizer.encode(target_text)?.len();
        let max_len = if retry_badcase {
            (target_text_length as f64 * retry_badcase_ratio_threshold + 10.0) as usize
        } else {
            max_len
        };
        let decode_audio = self._generate(
            &text_token,
            &text_mask,
            &audio_feat,
            &audio_mask,
            min_len,
            max_len,
            inference_timesteps,
            cfg_value,
        )?;
        Ok(decode_audio)
    }
}
