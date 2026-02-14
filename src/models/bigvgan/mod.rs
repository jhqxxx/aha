use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::{Init, VarBuilder};

use crate::{
    models::{
        bigvgan::config::BigVGANConfig,
        common::{WNConv1d, WNConvTranspose1d},
    },
    utils::tensor_utils::pad_replicate_last_dim,
};

pub mod config;

pub struct UpSample1d {
    // ratio: usize,
    stride: usize,
    pad: usize,
    pad_left: usize,
    pad_right: usize,
    filter: Tensor,
}

impl UpSample1d {
    pub fn new(vb: VarBuilder, ratio: usize, kernel_size: Option<usize>) -> Result<Self> {
        let stride = ratio;
        let kernel_size = kernel_size.unwrap_or(6 * ratio / 2 * 2);
        let pad = kernel_size / ratio - 1;
        let pad_left = pad * stride + (kernel_size - stride) / 2;
        let pad_right = pad * stride + (kernel_size - stride + 1) / 2;
        let filter = vb.get_with_hints((1, 1, kernel_size), "filter", Init::Const(0.0))?;

        Ok(Self {
            // ratio,
            stride,
            pad,
            pad_left,
            pad_right,
            filter,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let c = xs.dim(1)?;
        let xs = pad_replicate_last_dim(xs, (self.pad, self.pad))?;
        let xs = xs.conv_transpose1d(&self.filter.repeat((c, 1, 1))?, 0, 0, self.stride, 1, c)?;
        let xs_last_dim = xs.dim(D::Minus1)?;
        let xs_len = xs_last_dim - self.pad_left - self.pad_right;
        let xs = xs.narrow(D::Minus1, self.pad_left, xs_len)?;
        Ok(xs)
    }
}

pub struct DownSample1d {
    stride: usize,
    // kernel_size: usize,
    pad_left: usize,
    pad_right: usize,
    filter: Tensor,
}

impl DownSample1d {
    pub fn new(vb: VarBuilder, ratio: usize, kernel_size: Option<usize>) -> Result<Self> {
        let stride = ratio;
        let kernel_size = kernel_size.unwrap_or(6 * ratio / 2 * 2);
        let even = if kernel_size.is_multiple_of(2) { 1 } else { 0 };
        let pad_left = kernel_size / 2 - even;
        let pad_right = kernel_size / 2;
        let filter = vb.get_with_hints((1, 1, kernel_size), "lowpass.filter", Init::Const(0.0))?;
        Ok(Self {
            stride,
            // kernel_size,
            pad_left,
            pad_right,
            filter,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let c = xs.dim(1)?;
        let xs = pad_replicate_last_dim(xs, (self.pad_left, self.pad_right))?;
        let xs = xs.conv1d(&self.filter.repeat((c, 1, 1))?, 0, self.stride, 1, c)?;

        Ok(xs)
    }
}

pub struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
    no_div_by_zero: f64,
}

impl SnakeBeta {
    pub fn new(vb: VarBuilder, in_features: usize) -> Result<Self> {
        let no_div_by_zero = 0.000000001f64;
        let alpha = vb
            .get_with_hints(in_features, "alpha", Init::Const(0.0))?
            .unsqueeze(0)?
            .unsqueeze(D::Minus1)?
            .contiguous()?
            .exp()?;
        let beta = vb
            .get_with_hints(in_features, "beta", Init::Const(0.0))?
            .unsqueeze(0)?
            .unsqueeze(D::Minus1)?
            .contiguous()?
            .exp()?;
        Ok(Self {
            alpha,
            beta,
            no_div_by_zero,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let beta = (1.0 / self.beta.affine(1.0, self.no_div_by_zero)?)?;
        let xs = xs
            .broadcast_mul(&self.alpha)?
            .sin()?
            .powf(2.0)?
            .broadcast_mul(&beta)?
            .add(xs)?;
        Ok(xs)
    }
}

pub struct TorchActivation1d {
    upsample: UpSample1d,
    downsample: DownSample1d,
    act: SnakeBeta,
}

impl TorchActivation1d {
    pub fn new(
        vb: VarBuilder,
        up_ratio: usize,
        down_ratio: usize,
        up_kernel_size: usize,
        down_kernel_size: usize,
        channels: usize,
    ) -> Result<Self> {
        let upsample = UpSample1d::new(vb.pp("upsample"), up_ratio, Some(up_kernel_size))?;
        let downsample =
            DownSample1d::new(vb.pp("downsample"), down_ratio, Some(down_kernel_size))?;
        let act = SnakeBeta::new(vb.pp("act"), channels)?;
        Ok(Self {
            upsample,
            downsample,
            act,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.upsample.forward(xs)?;
        let xs = self.act.forward(&xs)?;
        let xs = self.downsample.forward(&xs)?;
        Ok(xs)
    }
}

pub struct AMPBlock1 {
    convs1: Vec<WNConv1d>,
    convs2: Vec<WNConv1d>,
    activations: Vec<TorchActivation1d>,
}

impl AMPBlock1 {
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        dilation: Vec<usize>,
    ) -> Result<Self> {
        let vb_convs1 = vb.pp("convs1");
        let mut convs1 = vec![];
        for (i, &d) in dilation.iter().enumerate() {
            let pad = ((kernel_size * d - d) as f32 / 2.0).round() as usize;
            let layer = WNConv1d::new(
                vb_convs1.pp(i),
                channels,
                channels,
                kernel_size,
                d,
                pad,
                1,
                1,
                true,
            )?;
            convs1.push(layer);
        }
        let vb_convs2 = vb.pp("convs2");
        let mut convs2 = vec![];
        for (i, _) in dilation.iter().enumerate() {
            let pad = ((kernel_size - 1) as f32 / 2.0).round() as usize;
            let layer = WNConv1d::new(
                vb_convs2.pp(i),
                channels,
                channels,
                kernel_size,
                1,
                pad,
                1,
                1,
                true,
            )?;
            convs2.push(layer);
        }

        let num_layer = convs1.len() + convs2.len();
        let act_vb = vb.pp("activations");
        let mut activations = vec![];
        for i in 0..num_layer {
            let layer = TorchActivation1d::new(act_vb.pp(i), 2, 2, 12, 12, channels)?;
            activations.push(layer);
        }

        Ok(Self {
            convs1,
            convs2,
            activations,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lens = self.convs1.len();
        let mut xs = xs.clone();
        for i in 0..lens {
            let xt = self.activations[i * 2].forward(&xs)?;
            let xt = self.convs1[i].forward(&xt)?;
            let xt = self.activations[i * 2 + 1].forward(&xt)?;
            let xt = self.convs2[i].forward(&xt)?;
            xs = xs.add(&xt)?;
        }
        Ok(xs)
    }
}

pub struct BigVGAN {
    num_kernels: usize,
    num_upsamples: usize,
    conv_pre: WNConv1d,
    ups: Vec<WNConvTranspose1d>,
    resblocks: Vec<AMPBlock1>,
    activation_post: TorchActivation1d,
    conv_post: WNConv1d,
    use_tanh_at_final: bool,
}

impl BigVGAN {
    pub fn new(vb: VarBuilder, cfg: &BigVGANConfig) -> Result<Self> {
        let num_kernels = cfg.resblock_kernel_sizes.len();
        let num_upsamples = cfg.upsample_rates.len();
        let conv_pre = WNConv1d::new(
            vb.pp("conv_pre"),
            cfg.num_mels,
            cfg.upsample_initial_channel,
            7,
            1,
            3,
            1,
            1,
            true,
        )?;

        let vb_ups = vb.pp("ups");
        let mut ups = vec![];
        for (i, (&u, &k)) in cfg
            .upsample_rates
            .iter()
            .zip(cfg.upsample_kernel_sizes.iter())
            .enumerate()
        {
            let in_c = cfg.upsample_initial_channel / (2_i32.pow(i as u32) as usize);
            let out_c = cfg.upsample_initial_channel / (2_i32.pow(i as u32 + 1) as usize);
            let pad = (k - u) / 2;
            let layer =
                WNConvTranspose1d::new(vb_ups.pp(i).pp("0"), in_c, out_c, 1, k, pad, 0, 1, u)?;
            ups.push(layer);
        }
        let vb_resblocks = vb.pp("resblocks");
        let mut resblocks = vec![];
        let ups_len = ups.len();
        let res_len = cfg.resblock_kernel_sizes.len();
        let mut ch = 0;
        for i in 0..ups_len {
            ch = cfg.upsample_initial_channel / (2_i32.pow(i as u32 + 1) as usize);
            for (j, (&k, d)) in cfg
                .resblock_kernel_sizes
                .iter()
                .zip(cfg.resblock_dilation_sizes.iter())
                .enumerate()
            {
                let layer = AMPBlock1::new(vb_resblocks.pp(i * res_len + j), ch, k, d.clone())?;
                resblocks.push(layer);
            }
        }
        let activation_post = TorchActivation1d::new(vb.pp("activation_post"), 2, 2, 12, 12, ch)?;
        let conv_post = WNConv1d::new(vb.pp("conv_post"), ch, 1, 7, 1, 3, 1, 1, false)?;

        Ok(Self {
            num_kernels,
            num_upsamples,
            conv_pre,
            ups,
            resblocks,
            activation_post,
            conv_post,
            use_tanh_at_final: cfg.use_tanh_at_final,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.conv_pre.forward(xs)?;
        for i in 0..self.num_upsamples {
            xs = self.ups[i].forward(&xs)?;
            let mut xs_j = xs.zeros_like()?;
            for j in 0..self.num_kernels {
                let j_ = self.resblocks[i * self.num_kernels + j].forward(&xs)?;
                xs_j = xs_j.add(&j_)?;
            }
            xs = xs_j.affine(1.0 / (self.num_kernels as f64), 0.0)?;
        }
        xs = self.activation_post.forward(&xs)?;
        xs = self.conv_post.forward(&xs)?;
        xs = if self.use_tanh_at_final {
            xs.tanh()?
        } else {
            xs.clamp(-1.0, 1.0)?
        };
        Ok(xs)
    }
}
