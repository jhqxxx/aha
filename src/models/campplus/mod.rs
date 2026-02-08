use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::{BatchNorm, Conv1d, Conv2d, Module, ModuleT, VarBuilder, ops::sigmoid};

use crate::{
    models::common::{get_batch_norm, get_conv1d, get_conv2d},
    utils::tensor_utils::{pool1d, statistics_pooling},
};

pub struct Shortcut {
    conv_0: Conv2d,
    bn_1: BatchNorm,
    stride: usize,
}

impl Shortcut {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        ks: usize,
        padding: usize,
        stride: usize,
        bias: bool,
    ) -> Result<Self> {
        let conv_0 = get_conv2d(vb.pp("0"), in_c, out_c, ks, padding, 1, 1, 1, bias)?;
        let bn_1 = get_batch_norm(vb.pp("1"), 1e-5, out_c, true)?;
        Ok(Self {
            conv_0,
            bn_1,
            stride,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_0.forward(x)?;
        if self.stride != 1 {
            let h_dim = x.dim(2)?;
            let half_h = h_dim / 2;
            let indices = Tensor::arange(0u32, half_h as u32, x.device())?.affine(2.0, 0.0)?;
            x = x.index_select(&indices, 2)?;
        }
        x = self.bn_1.forward_t(&x, false)?;
        Ok(x)
    }
}

pub struct BasicResBlock {
    stride: usize,
    conv1: Conv2d,
    bn1: BatchNorm,
    conv2: Conv2d,
    bn2: BatchNorm,
    shortcut: Option<Shortcut>,
}

impl BasicResBlock {
    pub fn new(
        vb: VarBuilder,
        in_planes: usize,
        planes: usize,
        stride: usize,
        expansion: usize,
    ) -> Result<Self> {
        let conv1 = get_conv2d(vb.pp("conv1"), in_planes, planes, 3, 1, 1, 1, 1, false)?;
        let bn1 = get_batch_norm(vb.pp("bn1"), 1e-5, planes, true)?;
        let conv2 = get_conv2d(vb.pp("conv2"), planes, planes, 3, 1, 1, 1, 1, false)?;
        let bn2 = get_batch_norm(vb.pp("bn2"), 1e-5, planes, true)?;
        let shortcut = if stride != 1 || in_planes != expansion * planes {
            Some(Shortcut::new(
                vb.pp("shortcut"),
                in_planes,
                expansion * planes,
                1,
                0,
                stride,
                false,
            )?)
        } else {
            None
        };
        Ok(Self {
            stride,
            conv1,
            bn1,
            conv2,
            bn2,
            shortcut,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let mut xs = self.conv1.forward(xs)?;
        // candle stride only surpport one size, h_stride = w_stride
        // infact model stride is (stride, 1)
        // but now setting stride all equal to 1
        // so h direction use indices select
        if self.stride != 1 {
            let h_dim = xs.dim(2)?;
            let half_h = h_dim / 2;
            let indices = Tensor::arange(0u32, half_h as u32, xs.device())?.affine(2.0, 0.0)?;
            xs = xs.index_select(&indices, 2)?;
        }
        let xs = self.bn1.forward_t(&xs, false)?.relu()?;
        let xs = self.conv2.forward(&xs)?;
        let mut xs = self.bn2.forward_t(&xs, false)?;
        if let Some(cut) = &self.shortcut {
            let shortcut = cut.forward(&residual)?;
            xs = xs.add(&shortcut)?;
        } else {
            xs = xs.add(&residual)?;
        }
        xs = xs.relu()?;
        Ok(xs)
    }
}

pub struct FCM {
    conv1: Conv2d,
    bn1: BatchNorm,
    layer1: Vec<BasicResBlock>,
    layer2: Vec<BasicResBlock>,
    conv2: Conv2d,
    bn2: BatchNorm,
    pub out_channels: usize,
}

impl FCM {
    pub fn new(
        vb: VarBuilder,
        num_blocks: &[usize],
        m_channels: usize,
        feat_dim: usize,
    ) -> Result<Self> {
        let conv1 = get_conv2d(vb.pp("conv1"), 1, m_channels, 3, 1, 1, 1, 1, false)?;
        let bn1 = get_batch_norm(vb.pp("bn1"), 1e-5, m_channels, true)?;
        let layer1_num_blocks = num_blocks[0] - 1;
        let strides: Vec<usize> = [2usize]
            .into_iter()
            .chain([1usize].into_iter().cycle().take(layer1_num_blocks))
            .collect();
        let mut layer1 = vec![];
        let vb_layer1 = vb.pp("layer1");
        for (i, stride) in strides.iter().enumerate() {
            let layer = BasicResBlock::new(vb_layer1.pp(i), m_channels, m_channels, *stride, 1)?;
            layer1.push(layer);
        }
        let layer2_num_blocks = num_blocks[1] - 1;
        let strides: Vec<usize> = [2usize]
            .into_iter()
            .chain([1usize].into_iter().cycle().take(layer2_num_blocks))
            .collect();
        let mut layer2 = vec![];
        let vb_layer2 = vb.pp("layer2");
        for (i, stride) in strides.iter().enumerate() {
            let layer = BasicResBlock::new(vb_layer2.pp(i), m_channels, m_channels, *stride, 1)?;
            layer2.push(layer);
        }
        let conv2 = get_conv2d(vb.pp("conv2"), m_channels, m_channels, 3, 1, 1, 1, 1, false)?;
        let bn2 = get_batch_norm(vb.pp("bn2"), 1e-5, m_channels, true)?;
        let out_channels = m_channels * (feat_dim / 8);
        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            conv2,
            bn2,
            out_channels,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.unsqueeze(1)?;
        let xs = self.conv1.forward(&xs)?;
        let mut xs = self.bn1.forward_t(&xs, false)?.relu()?;
        for layer in &self.layer1 {
            xs = layer.forward(&xs)?;
        }
        for layer in &self.layer2 {
            xs = layer.forward(&xs)?;
        }
        xs = self.conv2.forward(&xs)?;
        let h_dim = xs.dim(2)?;
        let half_h = h_dim / 2;
        let indices = Tensor::arange(0u32, half_h as u32, xs.device())?.affine(2.0, 0.0)?;
        xs = xs.index_select(&indices, 2)?;
        xs = self.bn2.forward_t(&xs, false)?.relu()?;
        let (bs, c, h, dim) = xs.dims4()?;
        xs = xs.reshape((bs, c * h, dim))?;
        Ok(xs)
    }
}

pub struct TDNNLayer {
    linear: Conv1d,
    nonlinear: BatchNorm,
}

impl TDNNLayer {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        ks: usize,
        stride: usize,
        dilation: usize,
        bias: bool,
    ) -> Result<Self> {
        let padding = (ks - 1) / 2 * dilation;
        let linear = get_conv1d(
            vb.pp("linear"),
            in_c,
            out_c,
            ks,
            padding,
            stride,
            dilation,
            1,
            bias,
        )?;
        let nonlinear = get_batch_norm(vb.pp("nonlinear.batchnorm"), 1e-5, out_c, true)?;
        Ok(Self { linear, nonlinear })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear.forward(xs)?;
        let xs = self.nonlinear.forward_t(&xs, false)?.relu()?;
        Ok(xs)
    }
}

pub struct CAMLayer {
    linear_local: Conv1d,
    linear1: Conv1d,
    linear2: Conv1d,
}

impl CAMLayer {
    pub fn new(
        vb: VarBuilder,
        bn_c: usize,
        out_c: usize,
        ks: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        bias: bool,
        reduction: usize,
    ) -> Result<Self> {
        let linear_local = get_conv1d(
            vb.pp("linear_local"),
            bn_c,
            out_c,
            ks,
            padding,
            stride,
            dilation,
            1,
            bias,
        )?;
        let linear1 = get_conv1d(
            vb.pp("linear1"),
            bn_c,
            bn_c / reduction,
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let linear2 = get_conv1d(
            vb.pp("linear2"),
            bn_c / reduction,
            out_c,
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        Ok(Self {
            linear_local,
            linear1,
            linear2,
        })
    }

    pub fn seg_pooling(&self, xs: &Tensor, seg_len: usize, stype: &str) -> Result<Tensor> {
        let x_dim = xs.dim(2)?;
        let seg = pool1d(xs, seg_len, true, stype)?;
        let (bs, c, dim) = seg.dims3()?;
        let seg = seg
            .unsqueeze(D::Minus1)?
            .expand((bs, c, dim, seg_len))?
            .reshape((bs, c, ()))?;
        let seg = seg.narrow(D::Minus1, 0, x_dim)?;
        Ok(seg)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let y = self.linear_local.forward(xs)?;
        let x_pool = self.seg_pooling(xs, 100, "avg")?;
        let context = xs.mean_keepdim(D::Minus1)?.broadcast_add(&x_pool)?;
        let context = self.linear1.forward(&context)?.relu()?;
        let m = sigmoid(&self.linear2.forward(&context)?)?;
        let res = y.mul(&m)?;
        Ok(res)
    }
}

pub struct CAMDenseTDNNLayer {
    nonlinear1: BatchNorm,
    linear1: Conv1d,
    nonlinear2: BatchNorm,
    cam_layer: CAMLayer,
}

impl CAMDenseTDNNLayer {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        bn_c: usize,
        ks: usize,
        stride: usize,
        dilation: usize,
        bias: bool,
    ) -> Result<Self> {
        let padding = (ks - 1) / 2 * dilation;
        let nonlinear1 = get_batch_norm(vb.pp("nonlinear1.batchnorm"), 1e-5, in_c, true)?;
        let linear1 = get_conv1d(vb.pp("linear1"), in_c, bn_c, 1, 0, 1, 1, 1, false)?;
        let nonlinear2 = get_batch_norm(vb.pp("nonlinear2.batchnorm"), 1e-5, bn_c, true)?;
        let cam_layer = CAMLayer::new(
            vb.pp("cam_layer"),
            bn_c,
            out_c,
            ks,
            stride,
            padding,
            dilation,
            bias,
            2,
        )?;
        Ok(Self {
            nonlinear1,
            linear1,
            nonlinear2,
            cam_layer,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.nonlinear1.forward_t(xs, false)?.relu()?;
        let xs = self.linear1.forward(&xs)?;
        let xs = self.nonlinear2.forward_t(&xs, false)?.relu()?;
        let xs = self.cam_layer.forward(&xs)?;
        Ok(xs)
    }
}

pub struct CAMDenseTDNNBlock {
    tdnns: Vec<CAMDenseTDNNLayer>,
}

impl CAMDenseTDNNBlock {
    pub fn new(
        vb: VarBuilder,
        num_layers: usize,
        in_c: usize,
        out_c: usize,
        bn_c: usize,
        ks: usize,
        stride: usize,
        dilation: usize,
        bias: bool,
    ) -> Result<Self> {
        let mut tdnns = vec![];
        for i in 0..num_layers {
            let layer = CAMDenseTDNNLayer::new(
                vb.pp(format!("tdnnd{}", i + 1)),
                in_c + i * out_c,
                out_c,
                bn_c,
                ks,
                stride,
                dilation,
                bias,
            )?;
            tdnns.push(layer);
        }
        Ok(Self { tdnns })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in &self.tdnns {
            let layer_out = layer.forward(&xs)?;
            xs = Tensor::cat(&[&xs, &layer_out], 1)?;
        }
        Ok(xs)
    }
}

pub struct TransitLayer {
    nonlinear: BatchNorm,
    linear: Conv1d,
}

impl TransitLayer {
    pub fn new(vb: VarBuilder, in_c: usize, out_c: usize, bias: bool) -> Result<Self> {
        let nonlinear = get_batch_norm(vb.pp("nonlinear.batchnorm"), 1e-5, in_c, true)?;
        let linear = get_conv1d(vb.pp("linear"), in_c, out_c, 1, 0, 1, 1, 1, bias)?;
        Ok(Self { nonlinear, linear })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.nonlinear.forward_t(xs, false)?.relu()?;
        let xs = self.linear.forward(&xs)?;
        Ok(xs)
    }
}

pub struct DenseLayer {
    linear: Conv1d,
    nonlinear: BatchNorm, // only batch norm, no relu
}

impl DenseLayer {
    pub fn new(vb: VarBuilder, in_c: usize, out_c: usize, bias: bool) -> Result<Self> {
        let linear = get_conv1d(vb.pp("linear"), in_c, out_c, 1, 0, 1, 1, 1, bias)?;
        let nonlinear = get_batch_norm(vb.pp("nonlinear.batchnorm"), 1e-5, out_c, false)?;
        Ok(Self { linear, nonlinear })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if xs.rank() == 2 {
            self.linear
                .forward(&xs.unsqueeze(D::Minus1)?)?
                .squeeze(D::Minus1)?
        } else {
            self.linear.forward(&xs)?
        };
        let xs = self.nonlinear.forward_t(&xs, false)?;
        Ok(xs)
    }
}

pub struct XVector {
    tdnn: TDNNLayer,
    blocks: Vec<CAMDenseTDNNBlock>,
    transits: Vec<TransitLayer>,
    out_nonlinear: BatchNorm,
    dense: DenseLayer,
}

impl XVector {
    pub fn new(
        vb: VarBuilder,
        channels: usize,
        init_channels: usize,
        growth_rate: usize,
        bn_size: usize,
        embedding_size: usize,
    ) -> Result<Self> {
        let tdnn = TDNNLayer::new(vb.pp("tdnn"), channels, init_channels, 5, 2, 1, false)?;
        let mut channels = init_channels;
        let mut blocks = vec![];
        let mut transits = vec![];
        let params = vec![(12, 3, 1), (24, 3, 2), (16, 3, 2)];
        for (i, (num_layers, ks, dilation)) in params.iter().enumerate() {
            let block = CAMDenseTDNNBlock::new(
                vb.pp(format!("block{}", i + 1)),
                *num_layers,
                channels,
                growth_rate,
                bn_size * growth_rate,
                *ks,
                1,
                *dilation,
                false,
            )?;
            blocks.push(block);
            channels = channels + num_layers * growth_rate;
            let transit = TransitLayer::new(
                vb.pp(format!("transit{}", i + 1)),
                channels,
                channels / 2,
                false,
            )?;
            transits.push(transit);
            channels /= 2;
        }
        let out_nonlinear = get_batch_norm(vb.pp("out_nonlinear.batchnorm"), 1e-5, channels, true)?;
        let dense = DenseLayer::new(vb.pp("dense"), channels * 2, embedding_size, false)?;
        Ok(Self {
            tdnn,
            blocks,
            transits,
            out_nonlinear,
            dense,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.tdnn.forward(xs)?;
        for i in 0..3 {
            let block = &self.blocks[i];
            xs = block.forward(&xs)?;
            let transit = &self.transits[i];
            xs = transit.forward(&xs)?;
        }
        xs = self.out_nonlinear.forward_t(&xs, false)?.relu()?;
        xs = statistics_pooling(&xs, D::Minus1, false)?;
        xs = self.dense.forward(&xs)?;
        Ok(xs)
    }
}

pub struct CAMPPlus {
    head: FCM,
    xvector: XVector,
}

impl CAMPPlus {
    pub fn new(
        vb: VarBuilder,
        feat_dim: usize,
        embedding_size: usize,
        growth_rate: usize,
        bn_size: usize,
        init_channels: usize,
    ) -> Result<Self> {
        let head = FCM::new(vb.pp("head"), &[2, 2], 32, feat_dim)?;
        let channels = head.out_channels;
        let xvector = XVector::new(
            vb.pp("xvector"),
            channels,
            init_channels,
            growth_rate,
            bn_size,
            embedding_size,
        )?;
        Ok(Self { head, xvector })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.permute((0, 2, 1))?;
        let xs = self.head.forward(&xs)?;
        let xs = self.xvector.forward(&xs)?;
        Ok(xs)
    }
}
