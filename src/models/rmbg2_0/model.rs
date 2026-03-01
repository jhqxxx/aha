use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Shape, Tensor};
use candle_nn::{
    Activation, BatchNorm, Conv2d, Init, LayerNorm, Linear, Module, ModuleT, VarBuilder, linear,
    linear_b, linear_no_bias, ops::sigmoid,
};

use crate::{
    models::common::{
        Conv2dWithBN, TwoLinearMLP, deform_conv2d_kernel, get_batch_norm, get_conv2d,
        get_layer_norm,
    },
    utils::interpolate::interpolate_bilinear,
    utils::tensor_utils::{get_equal_mask, index_select_2d, split_tensor_with_size},
};

struct PatchEmbed {
    proj: Conv2d,
    norm: Option<LayerNorm>,
    patch_size: usize,
    embed_dim: usize,
}

impl PatchEmbed {
    pub fn new(
        vb: VarBuilder,
        in_chans: usize,
        embed_dim: usize,
        patch_size: usize,
        patch_norm: bool,
    ) -> Result<Self> {
        let proj = get_conv2d(
            vb.pp("proj"),
            in_chans,
            embed_dim,
            patch_size,
            0,
            patch_size,
            1,
            1,
            true,
        )?;
        let norm = if patch_norm {
            Some(get_layer_norm(vb.pp("norm"), 1e-5, embed_dim, true)?)
        } else {
            None
        };
        Ok(Self {
            patch_size,
            proj,
            norm,
            embed_dim,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = xs.dims4()?;
        let mut xs = xs.clone();
        if w % self.patch_size != 0 {
            xs = xs.pad_with_zeros(3, 0, self.patch_size - w % self.patch_size)?;
        }
        if h % self.patch_size != 0 {
            xs = xs.pad_with_zeros(2, 0, self.patch_size - h % self.patch_size)?;
        }
        xs = self.proj.forward(&xs)?;
        if self.norm.is_some() {
            let (_, _, ph, pw) = xs.dims4()?;
            xs = xs.flatten_from(2)?.transpose(1, 2)?;
            xs = self.norm.as_ref().unwrap().forward(&xs)?;
            xs = xs.transpose(1, 2)?.reshape(((), self.embed_dim, ph, pw))?;
        }
        Ok(xs)
    }
}

pub struct WindowAttention {
    num_heads: usize,
    relative_position_bias: Tensor,
    qkv: Linear,
    proj: Linear,
    scaling: f64,
}

impl WindowAttention {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        window_size: (usize, usize),
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let qkv = linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;

        let proj = linear(dim, dim, vb.pp("proj"))?;
        let relative_position_bias_table = vb.get_with_hints(
            ((2 * window_size.0 - 1) * (2 * window_size.1 - 1), num_heads),
            "relative_position_bias_table",
            Init::Const(0.),
        )?; //2*Wh-1 * 2*Ww-1, nH

        let coords_h = Tensor::arange(0f32, window_size.0 as f32, vb.device())?
            .unsqueeze(1)?
            .broadcast_as(window_size)?;
        let coords_w = Tensor::arange(0f32, window_size.1 as f32, vb.device())?
            .unsqueeze(0)?
            .broadcast_as(window_size)?;

        let coords = Tensor::stack(&[coords_h, coords_w], 0)?.flatten_from(1)?; // (2, wh, ww)
        let coords1 = coords.unsqueeze(2)?;
        let coords2 = coords.unsqueeze(1)?;
        let relative_coords = coords1
            .broadcast_sub(&coords2)?
            .permute((1, 2, 0))?
            .contiguous()?; // (wh*ww, wh*ww, 2)
        let relative_coords_0 = relative_coords
            .i((.., .., 0))?
            .affine(1.0, window_size.0 as f64 - 1.0)?;
        let relative_coords_1 = relative_coords
            .i((.., .., 1))?
            .affine(1.0, window_size.1 as f64 - 1.0)?;
        let relative_coords_0 = relative_coords_0.affine(2.0 * window_size.1 as f64 - 1.0, 0.0)?;
        let relative_position_index = relative_coords_0
            .add(&relative_coords_1)?
            .to_dtype(candle_core::DType::U32)?;
        let relative_position_bias =
            index_select_2d(&relative_position_bias_table, &relative_position_index)?;
        Ok(Self {
            num_heads,
            relative_position_bias,
            qkv,
            proj,
            scaling,
        })
    }

    pub fn forward(&self, xs: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, _) = xs.dims3()?;
        // (3, B, n_head, h*w, head_dim)
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, seq_len, 3, self.num_heads, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let query_states = qkv.i(0)?.contiguous()?;
        let key_states = qkv.i(1)?.contiguous()?;
        let value_states = qkv.i(2)?.contiguous()?;
        let attn_bias = self
            .relative_position_bias
            .permute((2, 0, 1))?
            .contiguous()?
            .unsqueeze(0)?;
        let query_states = (query_states * self.scaling)?;

        let attn_weights = query_states.matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = attn_weights.broadcast_add(&attn_bias)?;
        let attn_weights = match attn_mask {
            None => attn_weights,
            Some(mask) => {
                let nw: usize = mask.dim(0)?;
                let attn_weights = attn_weights
                    .reshape((b / nw, nw, self.num_heads, seq_len, seq_len))?
                    .broadcast_add(
                        &mask
                            .unsqueeze(1)?
                            .unsqueeze(0)?
                            .to_dtype(attn_weights.dtype())?,
                    )?;

                attn_weights.reshape(((), self.num_heads, seq_len, seq_len))?
            }
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        //(b, n_head, seq_len, dim) -> (b, seq_len, n_head, dim)
        let xs = attn_output.transpose(1, 2)?.contiguous()?;
        // (b, h*w, n_head, dim)
        let xs = xs.reshape((b, seq_len, ()))?;
        let xs = self.proj.forward(&xs)?;
        Ok(xs)
    }
}

fn window_partition(x: &Tensor, window_size: usize) -> Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;

    let x = x.reshape((
        b,
        h / window_size,
        window_size,
        w / window_size,
        window_size,
        c,
    ))?;
    let windows =
        x.permute((0, 1, 3, 2, 4, 5))?
            .contiguous()?
            .reshape(((), window_size, window_size, c))?;
    Ok(windows)
}

fn window_reverse(windows: &Tensor, window_size: usize, pad_hw: (usize, usize)) -> Result<Tensor> {
    let (hp, wp) = pad_hw;
    let b = windows.dim(0)? / (hp * wp / window_size / window_size);
    let last_dim = windows.dim(D::Minus1)?;
    let x = windows.reshape(&[
        b,
        hp / window_size,
        wp / window_size,
        window_size,
        window_size,
        last_dim,
    ])?;
    let x = x
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b, hp, wp, ()))?;
    Ok(x)
}

struct SwinTransformerBlock {
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp: TwoLinearMLP,
    window_size: usize,
    shift_size: usize,
}

impl SwinTransformerBlock {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        act: Activation,
        window_size: usize,
        shift_size: usize,
    ) -> Result<Self> {
        let norm1 = get_layer_norm(vb.pp("norm1"), 1e-5, dim, true)?;

        let attn = WindowAttention::new(
            vb.pp("attn"),
            dim,
            num_heads,
            qkv_bias,
            (window_size, window_size),
        )?;
        let norm2 = get_layer_norm(vb.pp("norm2"), 1e-5, dim, true)?;
        let mlp_dim = (dim as f32 * mlp_ratio) as usize;
        let mlp = TwoLinearMLP::new(vb.pp("mlp"), dim, mlp_dim, dim, act, true, "fc1", "fc2")?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
            shift_size,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mask_matrix: Option<&Tensor>,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let (b, seq_len, c) = xs.dims3()?;
        assert_eq!(
            seq_len,
            h * w,
            "swin transformer block sq_len not equal to h*w"
        );
        let shortcut = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs = xs.reshape((b, h, w, c))?;
        let pad_h = (self.window_size - h % self.window_size) % self.window_size;
        let pad_w = (self.window_size - w % self.window_size) % self.window_size;
        let xs = xs.pad_with_zeros(1, 0, pad_h)?;
        let xs = xs.pad_with_zeros(2, 0, pad_w)?;
        let (_, hp, wp, _) = xs.dims4()?;

        let (shifted_x, attn_mask) = if self.shift_size > 0 {
            (
                xs.roll(-(self.shift_size as i32), 1)?
                    .roll(-(self.shift_size as i32), 2)?,
                mask_matrix,
            )
        } else {
            (xs, None)
        };
        let xs = window_partition(&shifted_x, self.window_size)?;
        let xs = xs.reshape(((), self.window_size * self.window_size, c))?;
        let xs = self.attn.forward(&xs, attn_mask)?;
        let xs = window_reverse(&xs, self.window_size, (hp, wp))?;
        let mut xs = if self.shift_size > 0 {
            xs.roll(self.shift_size as i32, 1)?
                .roll(self.shift_size as i32, 2)?
        } else {
            xs
        };
        if pad_h > 0 || pad_w > 0 {
            xs = xs.i((.., 0..h, 0..w, ..))?.contiguous()?;
        }
        let xs = xs.reshape((b, h * w, c))?;
        let x = shortcut.add(&xs)?;
        let x = x.add(&self.mlp.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

struct PatchMerging {
    reduction: Linear,
    norm: LayerNorm,
}

impl PatchMerging {
    pub fn new(vb: VarBuilder, dim: usize) -> Result<Self> {
        let reduction = linear_no_bias(4 * dim, 2 * dim, vb.pp("reduction"))?;
        let norm = get_layer_norm(vb.pp("norm"), 1e-5, 4 * dim, true)?;
        Ok(Self { reduction, norm })
    }

    pub fn forward(&self, xs: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (b, l, c) = xs.dims3()?;
        assert_eq!(l, h * w, "input feature has wrong size");
        let mut xs = xs.reshape((b, h, w, c))?;
        let pad_input = (h % 2 == 1) || (w % 2 == 1);
        if pad_input {
            xs = xs
                .pad_with_zeros(2, 0, w % 2)?
                .pad_with_zeros(1, 0, h % 2)?;
        }
        let shape = Shape::from_dims(&[b, h / 2, 2, w / 2, 2, c]);
        let xs = xs.reshape(shape)?;
        let x0 = xs.i((.., .., 0, .., 0, ..))?;
        let x1 = xs.i((.., .., 1, .., 0, ..))?;
        let x2 = xs.i((.., .., 0, .., 1, ..))?;
        let x3 = xs.i((.., .., 1, .., 1, ..))?;
        let xs = Tensor::cat(&[x0, x1, x2, x3], D::Minus1)?;
        let xs = xs.reshape((b, (), 4 * c))?;
        let xs = self.norm.forward(&xs)?;
        let xs = self.reduction.forward(&xs)?;
        Ok(xs)
    }
}

struct BasicLayer {
    window_size: usize,
    shift_size: usize,
    blocks: Vec<SwinTransformerBlock>,
    downsample: Option<PatchMerging>,
}

impl BasicLayer {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        depth: usize,
        num_heads: usize,
        window_size: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        downsample: bool,
    ) -> Result<Self> {
        let shift_size = window_size / 2;
        let mut blocks = vec![];
        let vb_blocks = vb.pp("blocks");
        for i in 0..depth {
            let block_shift_size = if i % 2 == 0 { 0usize } else { shift_size };

            let block = SwinTransformerBlock::new(
                vb_blocks.pp(i),
                dim,
                num_heads,
                mlp_ratio,
                qkv_bias,
                Activation::Gelu,
                window_size,
                block_shift_size,
            )?;
            blocks.push(block);
        }
        let downsample = if downsample {
            Some(PatchMerging::new(vb.pp("downsample"), dim)?)
        } else {
            None
        };
        Ok(Self {
            window_size,
            shift_size,
            blocks,
            downsample,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        h: usize,
        w: usize,
    ) -> Result<(Tensor, usize, usize, Tensor, usize, usize)> {
        let hp = (h as f32 / self.window_size as f32).ceil() as usize * self.window_size;
        let wp = (w as f32 / self.window_size as f32).ceil() as usize * self.window_size;
        let mut img_mask = Tensor::zeros((1, hp, wp, 1), xs.dtype(), xs.device())?;
        let h_slices = [
            (0usize, hp - self.window_size),
            (hp - self.window_size, hp - self.shift_size),
            (hp - self.shift_size, hp),
        ];
        let w_slices = [
            (0usize, wp - self.window_size),
            (wp - self.window_size, wp - self.shift_size),
            (wp - self.shift_size, wp),
        ];
        let mut cnt = 0f64;
        for (h_start, h_end) in h_slices {
            for (w_start, w_end) in w_slices {
                let mask_value = Tensor::zeros(
                    (1, h_end - h_start, w_end - w_start, 1),
                    xs.dtype(),
                    xs.device(),
                )?
                .affine(1.0, cnt)?;
                img_mask = img_mask.slice_assign(
                    &[(0..1), (h_start..h_end), (w_start..w_end), (0..1)],
                    &mask_value,
                )?;
                cnt += 1.0;
            }
        }

        let mask_windows = window_partition(&img_mask, self.window_size)?;
        let mask_windows = mask_windows.reshape(((), self.window_size * self.window_size))?;
        let attn_mask = mask_windows
            .unsqueeze(1)?
            .broadcast_sub(&mask_windows.unsqueeze(2)?)?;
        let equal_zero_mask = get_equal_mask(&attn_mask, 0)?;
        let attn_mask = equal_zero_mask.where_cond(
            &Tensor::new(0f32, xs.device())?.broadcast_as(equal_zero_mask.shape())?,
            &Tensor::new(-100f32, xs.device())?.broadcast_as(equal_zero_mask.shape())?,
        )?;
        let mut xs = xs.clone();
        for block in &self.blocks {
            xs = block.forward(&xs, Some(&attn_mask), h, w)?;
        }
        let (xs_down, wh, ww) = match self.downsample.as_ref() {
            Some(down) => {
                let xs_down = down.forward(&xs, h, w)?;
                // let wh = (h + 1) / 2;
                // let ww = (w + 1) / 2;
                let wh = h.div_ceil(2);
                let ww = w.div_ceil(2);
                (xs_down, wh, ww)
            }
            None => (xs.clone(), h, w),
        };
        Ok((xs, h, w, xs_down, wh, ww))
    }
}

pub struct SwinTransformer {
    patch_embed: PatchEmbed,
    num_layers: usize,
    // pos_drop: Dropout,
    layers: Vec<BasicLayer>,
    norms: Vec<LayerNorm>,
    out_indices: Vec<usize>,
    num_features: Vec<usize>,
}

impl SwinTransformer {
    pub fn new(
        vb: VarBuilder,
        patch_size: usize,
        in_channels: usize,
        embed_dim: usize,
        depths: Vec<usize>,
        num_heads: Vec<usize>,
        window_size: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        patch_norm: bool,
        out_indices: Vec<usize>,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            vb.pp("patch_embed"),
            in_channels,
            embed_dim,
            patch_size,
            patch_norm,
        )?;
        let num_layers = depths.len();
        let mut layers = vec![];
        let vb_layers = vb.pp("layers");
        let mut num_features = vec![];
        for i in 0..num_layers {
            let downsample = i < num_layers - 1;
            let dim_i = embed_dim * 2usize.pow(i as u32);
            num_features.push(dim_i);
            let layer_i = BasicLayer::new(
                vb_layers.pp(i),
                dim_i,
                depths[i],
                num_heads[i],
                window_size,
                mlp_ratio,
                qkv_bias,
                downsample,
            )?;
            layers.push(layer_i);
        }
        let mut norms = vec![];
        for i in out_indices.clone() {
            let layer_i = get_layer_norm(vb.pp(format!("norm{i}")), 1e-5, num_features[i], true)?;
            norms.push(layer_i);
        }
        Ok(Self {
            num_layers,
            patch_embed,
            layers,
            norms,
            out_indices,
            num_features,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Vec<Tensor>> {
        let xs = self.patch_embed.forward(xs)?;
        let (_, _, mut wh, mut ww) = xs.dims4()?;
        let mut outs = vec![];
        let mut xs = xs.flatten_from(2)?.transpose(1, 2)?;
        let mut norm_idx = 0;
        for i in 0..self.num_layers {
            let layer = &self.layers[i];
            let (x_out, h, w, xs_, wh_, ww_) = layer.forward(&xs, wh, ww)?;
            xs = xs_.clone();
            wh = wh_;
            ww = ww_;
            if self.out_indices.contains(&i) {
                let norm_layer = &self.norms[norm_idx];
                norm_idx += 1;
                let x_out = norm_layer.forward(&x_out)?;
                let out = x_out
                    .reshape(((), h, w, self.num_features[i]))?
                    .permute((0, 3, 1, 2))?
                    .contiguous()?;
                outs.push(out);
            }
        }
        Ok(outs)
    }
}

#[allow(unused)]
struct DeformableConv2d {
    offset_conv: Conv2d,
    modulator_conv: Conv2d,
    regular_conv: Conv2d,
    stride: usize,
    padding: usize,
    ks: usize,
}

#[allow(unused)]
impl DeformableConv2d {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Result<Self> {
        let offset_conv = get_conv2d(
            vb.pp("offset_conv"),
            in_c,
            2 * kernel_size * kernel_size,
            kernel_size,
            padding,
            stride,
            1,
            1,
            true,
        )?;

        let modulator_conv = get_conv2d(
            vb.pp("modulator_conv"),
            in_c,
            kernel_size * kernel_size,
            kernel_size,
            padding,
            stride,
            1,
            1,
            true,
        )?;

        let regular_conv = get_conv2d(
            vb.pp("regular_conv"),
            in_c,
            out_c,
            kernel_size,
            0,
            kernel_size,
            1,
            1,
            bias,
        )?;
        Ok(Self {
            offset_conv,
            modulator_conv,
            regular_conv,
            stride,
            padding,
            ks: kernel_size,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_use_kernel(xs)
        // if self.ks > 1 {
        //     self.forward_use_kernel(xs)
        // } else {
        //     self.forward_use_tensor(xs)
        // }
    }
    pub fn forward_use_kernel(&self, xs: &Tensor) -> Result<Tensor> {
        let offset = self.offset_conv.forward(xs)?; // (b, 2*k*k, out_h, out_w)

        let modulator = sigmoid(&self.modulator_conv.forward(xs)?)?
            .affine(2.0, 0.0)?
            .contiguous()?;
        let out = deform_conv2d_kernel(
            xs,
            self.regular_conv.weight(),
            self.regular_conv.bias(),
            &offset,
            Some(&modulator),
            self.stride,
            self.padding,
        )?;
        Ok(out)
    }

    pub fn forward_use_tensor(&self, xs: &Tensor) -> Result<Tensor> {
        let offset = self.offset_conv.forward(xs)?; // (b, 2*k*k, out_h, out_w)

        let modulator = sigmoid(&self.modulator_conv.forward(xs)?)?
            .affine(2.0, 0.0)?
            .contiguous()?;
        let n = offset.dim(1)? / 2;

        let xs = if self.padding > 0 {
            xs.pad_with_zeros(2, self.padding, self.padding)?
                .pad_with_zeros(3, self.padding, self.padding)?
        } else {
            xs.clone()
        };
        let offset = if self.ks > 3 {
            offset.to_device(&Device::Cpu)?
        } else {
            offset
        };
        let p = self.get_p(&offset)?;
        // drop(offset);
        // (b, h, w, 2n)
        let p = p.permute((0, 2, 3, 1))?.contiguous()?;
        let q_lt = p.floor()?;
        let q_rb = (&q_lt + 1.0)?;
        let (_, _, in_h, in_w) = xs.dims4()?;
        let in_h = in_h as f64;
        let in_w = in_w as f64;

        // 分开处理x和y坐标
        let p_x = p.narrow(3, 0, n)?.clamp(0.0, in_h - 1.0)?;
        let p_y = p.narrow(3, n, n)?.clamp(0.0, in_w - 1.0)?;
        // drop(p);
        let q_lt_x = q_lt.narrow(3, 0, n)?.clamp(0.0, in_h - 1.0)?;
        let q_lt_y = q_lt.narrow(3, n, n)?.clamp(0.0, in_w - 1.0)?;
        let q_rb_x = q_rb.narrow(3, 0, n)?.clamp(0.0, in_h - 1.0)?;
        let q_rb_y = q_rb.narrow(3, n, n)?.clamp(0.0, in_w - 1.0)?;
        // drop(q_lt);
        // drop(q_rb);
        // 转换为整数索引
        let q_lt_x_idx = q_lt_x.to_dtype(DType::U32)?;
        let q_lt_y_idx = q_lt_y.to_dtype(DType::U32)?;
        let q_rb_x_idx = q_rb_x.to_dtype(DType::U32)?;
        let q_rb_y_idx = q_rb_y.to_dtype(DType::U32)?;

        // 计算双线性权重
        let p_sub_lt_x = (&p_x - &q_lt_x)?;
        let one_sub_lt_x = (1.0 - &p_sub_lt_x)?;
        let p_sub_lt_y = (&p_y - &q_lt_y)?;
        let one_sub_lt_y = (1.0 - &p_sub_lt_y)?;
        // drop(q_lt_x);
        // drop(q_lt_y);
        // drop(q_rb_x);
        // drop(q_rb_y);
        let g_lt = (&one_sub_lt_x * &one_sub_lt_y)?;
        let g_rb = (&p_sub_lt_x * &p_sub_lt_y)?;
        let g_lb = (&one_sub_lt_x * &p_sub_lt_y)?;
        let g_rt = (&p_sub_lt_x * &one_sub_lt_y)?;
        // drop(p_sub_lt_x);
        // drop(one_sub_lt_x);
        // drop(p_sub_lt_y);
        // drop(one_sub_lt_y);

        let xs = if self.ks > 3 {
            xs.to_device(&Device::Cpu)?
        } else {
            xs
        };
        // 采样四个角点的特征
        let x_q_lt = self.get_x_q(&xs, &q_lt_x_idx, &q_lt_y_idx)?;
        let x_q_rb = self.get_x_q(&xs, &q_rb_x_idx, &q_rb_y_idx)?;
        let x_q_lb = self.get_x_q(&xs, &q_lt_x_idx, &q_rb_y_idx)?;
        let x_q_rt = self.get_x_q(&xs, &q_rb_x_idx, &q_lt_y_idx)?;
        // drop(q_lt_x_idx);
        // drop(q_lt_y_idx);
        // drop(q_rb_x_idx);
        // drop(q_rb_y_idx);
        // 双线性插值
        let x_offset = g_lt.unsqueeze(1)?.broadcast_mul(&x_q_lt)?;
        // drop(g_lt);
        // drop(x_q_lt);
        let x_offset = x_offset.add(&g_rb.unsqueeze(1)?.broadcast_mul(&x_q_rb)?)?;
        // drop(g_rb);
        // drop(x_q_rb);
        let x_offset = x_offset.add(&g_lb.unsqueeze(1)?.broadcast_mul(&x_q_lb)?)?;
        // drop(g_lb);
        // drop(x_q_lb);
        let x_offset = x_offset.add(&g_rt.unsqueeze(1)?.broadcast_mul(&x_q_rt)?)?;
        // drop(g_rt);
        // drop(x_q_rt);
        // (bs, n, h, w) -> (bs, h, w, n) -> (bs, 1, h, w, n)
        let m = modulator.permute((0, 2, 3, 1))?.unsqueeze(1)?;
        let x_offset = x_offset.to_device(m.device())?.broadcast_mul(&m)?;
        let x_offset = self.reshape_x_offset(&x_offset, self.ks)?;
        let xs = self.regular_conv.forward(&x_offset)?;
        Ok(xs)
    }
    fn reshape_x_offset(&self, xs: &Tensor, ks: usize) -> Result<Tensor> {
        let (b, c, h, w, _) = xs.dims5()?;

        let xs = xs.reshape((b, c, h, w, ks, ks))?;
        let xs = xs.permute((0, 1, 2, 4, 3, 5))?;
        let xs = xs.reshape((b, c, h * ks, w * ks))?;
        let x_offset = xs.contiguous()?;

        Ok(x_offset)
    }

    fn get_x_q(&self, xs: &Tensor, q_x: &Tensor, q_y: &Tensor) -> Result<Tensor> {
        let (b, h, w, n) = q_x.dims4()?;
        let padded_w = xs.dim(3)?;
        let c = xs.dim(1)?;

        // 展平输入: (b, c, H, W) -> (b, c, H*W)
        let xs_flat = xs.flatten_from(2)?;

        // 计算索引
        let index = q_x.affine(padded_w as f64, 0.0)?.add(q_y)?; // (b, h, w, n)

        // 扩展维度以匹配通道数
        let index = index
            .unsqueeze(1)?
            .expand((b, c, h, w, n))?
            .flatten_from(2)?; // (b, c, h*w*n)

        // 收集特征
        let xs = xs_flat.gather(&index, 2)?.reshape((b, c, h, w, n))?;
        Ok(xs)
    }

    fn get_p_n(&self, n: usize, dtype: DType, device: &Device) -> Result<Tensor> {
        let ks = self.ks as f32;
        let range = Tensor::arange_step(-(ks - 1.0) / 2.0, (ks - 1.0) / 2.0 + 1.0, 1.0, device)?;
        // 假设 ks=3
        // [(-1, -1), (-1, 0), (-1, 1)
        //  (0, -1), (0, 0), (0, 1)
        //  (1, -1), (1, 0), (1, 1)]
        // range: [-1, 0, 1]
        // unsqueeze(1) -> [[-1], [0], [1]]
        // broadcase_as(3, 3) -> [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
        // flatten_all -> [-1, -1, -1, 0, 0, 0, 1, 1, 1]
        let p_n_x = range
            .unsqueeze(1)?
            .broadcast_as((self.ks, self.ks))?
            .flatten_all()?;
        // range: [-1, 0, 1]
        // unsqueeze(0) -> [[-1, 0, 1]]
        // broadcase_as(3, 3) -> [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
        // flatten_all -> [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        let p_n_y = range
            .unsqueeze(0)?
            .broadcast_as((self.ks, self.ks))?
            .flatten_all()?;
        let p = Tensor::cat(&[p_n_x, p_n_y], 0)?
            .reshape((1, 2 * n, 1, 1))?
            .to_dtype(dtype)?
            .contiguous()?;
        Ok(p)
    }

    fn get_p_0(
        &self,
        h: usize,
        w: usize,
        n: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        // 假设 in featuremap h=w=5, padding=1, hp=wp=7
        // out featuremap h=w=5,
        // padding 后的 in featuremap
        // [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)
        //  (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)
        //  (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)
        //  (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)
        //  (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6)
        //  (5, 0), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6)
        //  (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)
        let start = self.padding as f32;
        // let start = 0.0f32;
        let p_0_x = Tensor::arange_step(
            start,
            start + h as f32 * self.stride as f32,
            self.stride as f32,
            device,
        )?
        .unsqueeze(1)?
        .broadcast_as((h, w))?
        .reshape((1, 1, h, w))?
        .repeat((1, n, 1, 1))?;
        let p_0_y = Tensor::arange_step(
            start,
            start + w as f32 * self.stride as f32,
            self.stride as f32,
            device,
        )?
        .unsqueeze(0)?
        .broadcast_as((h, w))?
        .reshape((1, 1, h, w))?
        .repeat((1, n, 1, 1))?;
        let p_0 = Tensor::cat(&[p_0_x, p_0_y], 1)?
            .to_dtype(dtype)?
            .contiguous()?;
        Ok(p_0)
    }
    fn get_p(&self, offset: &Tensor) -> Result<Tensor> {
        let (_, n, h, w) = offset.dims4()?;
        let n = n / 2;
        // (1, 2n, 1, 1)
        let p_n = self.get_p_n(n, offset.dtype(), offset.device())?;
        // (1, 2n, h, w)
        let p_0 = self.get_p_0(h, w, n, offset.dtype(), offset.device())?;
        let p = p_0
            .broadcast_add(&p_n)?
            .broadcast_add(offset)?
            .contiguous()?;
        Ok(p)
    }
}

struct _ASPPModuleDeformable {
    atrous_conv: DeformableConv2d,
    bn: BatchNorm,
}

impl _ASPPModuleDeformable {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        padding: usize,
    ) -> Result<Self> {
        let atrous_conv = DeformableConv2d::new(
            vb.pp("atrous_conv"),
            in_c,
            out_c,
            kernel_size,
            1,
            padding,
            false,
        )?;
        let bn = get_batch_norm(vb.pp("bn"), 1e-5, out_c, true)?;
        Ok(Self { atrous_conv, bn })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.atrous_conv.forward(xs)?;
        let xs = self.bn.forward_t(&xs, false)?.relu()?;
        Ok(xs)
    }
}

struct ASPPDeformable {
    aspp1: _ASPPModuleDeformable,
    // aspp_deforms: Vec<_ASPPModuleDeformable>,
    aspp_deforms_0: _ASPPModuleDeformable,
    aspp_deforms_1: _ASPPModuleDeformable,
    aspp_deforms_2: _ASPPModuleDeformable,
    // avgpool2d + conv2d + BatchNorm2d + relu
    global_avg_pool_1: Conv2d,
    global_avg_pool_2: BatchNorm,
    conv1: Conv2d,
    bn1: BatchNorm,
}

impl ASPPDeformable {
    pub fn new(
        vb: VarBuilder,
        in_c: usize,
        out_c: usize,
        parallel_block_sizes: Vec<usize>,
    ) -> Result<Self> {
        let in_channelster = 256;
        let aspp1 = _ASPPModuleDeformable::new(vb.pp("aspp1"), in_c, in_channelster, 1, 0)?;
        let vb_aspp_deforms = vb.pp("aspp_deforms");
        let aspp_deforms_0 = _ASPPModuleDeformable::new(
            vb_aspp_deforms.pp(0),
            in_c,
            in_channelster,
            parallel_block_sizes[0],
            parallel_block_sizes[0] / 2,
        )?;
        let aspp_deforms_1 = _ASPPModuleDeformable::new(
            vb_aspp_deforms.pp(1),
            in_c,
            in_channelster,
            parallel_block_sizes[1],
            parallel_block_sizes[1] / 2,
        )?;
        let aspp_deforms_2 = _ASPPModuleDeformable::new(
            vb_aspp_deforms.pp(2),
            in_c,
            in_channelster,
            parallel_block_sizes[2],
            parallel_block_sizes[2] / 2,
        )?;
        let global_avg_pool_1 = get_conv2d(
            vb.pp("global_avg_pool.1"),
            in_c,
            in_channelster,
            1,
            0,
            1,
            1,
            1,
            false,
        )?;
        let global_avg_pool_2 =
            get_batch_norm(vb.pp("global_avg_pool.2"), 1e-5, in_channelster, true)?;
        let conv1 = get_conv2d(
            vb.pp("conv1"),
            in_channelster * (2 + parallel_block_sizes.len()),
            out_c,
            1,
            0,
            1,
            1,
            1,
            false,
        )?;
        let bn1 = get_batch_norm(vb.pp("bn1"), 1e-5, out_c, true)?;
        Ok(Self {
            aspp1,
            aspp_deforms_0,
            aspp_deforms_1,
            aspp_deforms_2,
            global_avg_pool_1,
            global_avg_pool_2,
            conv1,
            bn1,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x1 = self.aspp1.forward(xs)?;
        let x_aspp_deforms_0 = self.aspp_deforms_0.forward(xs)?;
        let x_aspp_deforms_1 = self.aspp_deforms_1.forward(xs)?;
        let x_aspp_deforms_2 = self.aspp_deforms_2.forward(xs)?;

        let (_, _, h, w) = xs.dims4()?;
        assert_eq!(h, w, "avg_pool2d h, w mus be equal");
        let x5 = xs.avg_pool2d(h)?;
        let x5 = self.global_avg_pool_1.forward(&x5)?;
        let x5 = self.global_avg_pool_2.forward_t(&x5, false)?.relu()?;
        let (_, _, h, w) = x1.dims4()?;
        let x5 = interpolate_bilinear(&x5, (h, w), Some(true), None)?;
        let xs = Tensor::cat(
            &[x1, x_aspp_deforms_0, x_aspp_deforms_1, x_aspp_deforms_2, x5],
            1,
        )?;
        let xs = self.conv1.forward(&xs)?;
        let xs = self.bn1.forward_t(&xs, false)?.relu()?;
        Ok(xs)
    }
}

struct BasicDecBlk {
    conv_in: Conv2d,
    dec_att: ASPPDeformable,
    conv_out: Conv2d,
    bn_in: BatchNorm,
    bn_out: BatchNorm,
}

impl BasicDecBlk {
    pub fn new(vb: VarBuilder, in_c: usize, out_c: usize) -> Result<Self> {
        let inter_channels = 64;
        let conv_in = get_conv2d(vb.pp("conv_in"), in_c, inter_channels, 3, 1, 1, 1, 1, true)?;
        let dec_att = ASPPDeformable::new(vb.pp("dec_att"), inter_channels, inter_channels, vec![
            1, 3, 7,
        ])?;
        let conv_out = get_conv2d(
            vb.pp("conv_out"),
            inter_channels,
            out_c,
            3,
            1,
            1,
            1,
            1,
            true,
        )?;
        let bn_in = get_batch_norm(vb.pp("bn_in"), 1e-5, inter_channels, true)?;
        let bn_out = get_batch_norm(vb.pp("bn_out"), 1e-5, out_c, true)?;
        Ok(Self {
            conv_in,
            dec_att,
            conv_out,
            bn_in,
            bn_out,
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv_in.forward(xs)?;
        let xs = self.bn_in.forward_t(&xs, false)?;
        let xs = xs.relu()?;
        let xs = self.dec_att.forward(&xs)?;
        let xs = self.conv_out.forward(&xs)?;
        let xs = self.bn_out.forward_t(&xs, false)?;
        Ok(xs)
    }
}

struct SimpleConvs {
    conv1: Conv2d,
    conv_out: Conv2d,
}

impl SimpleConvs {
    pub fn new(vb: VarBuilder, in_c: usize, out_c: usize, inter_c: usize) -> Result<Self> {
        // inter_c = 64
        let conv1 = get_conv2d(vb.pp("conv1"), in_c, inter_c, 3, 1, 1, 1, 1, true)?;
        let conv_out = get_conv2d(vb.pp("conv_out"), inter_c, out_c, 3, 1, 1, 1, 1, true)?;
        Ok(Self { conv1, conv_out })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = self.conv_out.forward(&x)?;
        Ok(x)
    }
}

struct Decoder {
    ipt_blk5: SimpleConvs,
    ipt_blk4: SimpleConvs,
    ipt_blk3: SimpleConvs,
    ipt_blk2: SimpleConvs,
    ipt_blk1: SimpleConvs,
    decoder_block4: BasicDecBlk,
    decoder_block3: BasicDecBlk,
    decoder_block2: BasicDecBlk,
    decoder_block1: BasicDecBlk,
    conv_out1: Conv2d,
    // BasicLatBlk : conv, kernel_size=1
    lateral_block4: Conv2d,
    lateral_block3: Conv2d,
    lateral_block2: Conv2d,
    // conv_ms_spvn_4: Conv2d,
    // conv_ms_spvn_3: Conv2d,
    // conv_ms_spvn_2: Conv2d,
    gdt_convs_4: Conv2dWithBN,
    gdt_convs_3: Conv2dWithBN,
    gdt_convs_2: Conv2dWithBN,
    gdt_convs_attn_4: Conv2d,
    gdt_convs_attn_3: Conv2d,
    gdt_convs_attn_2: Conv2d,
}

impl Decoder {
    pub fn new(vb: VarBuilder, channels: Vec<usize>) -> Result<Self> {
        let ic = 64;
        let ipt_blk5 =
            SimpleConvs::new(vb.pp("ipt_blk5"), 2usize.pow(10) * 3, channels[0] / 8, ic)?;
        let ipt_blk4 = SimpleConvs::new(vb.pp("ipt_blk4"), 2usize.pow(8) * 3, channels[0] / 8, ic)?;
        let ipt_blk3 = SimpleConvs::new(vb.pp("ipt_blk3"), 2usize.pow(6) * 3, channels[1] / 8, ic)?;
        let ipt_blk2 = SimpleConvs::new(vb.pp("ipt_blk2"), 2usize.pow(4) * 3, channels[2] / 8, ic)?;
        let ipt_blk1 = SimpleConvs::new(vb.pp("ipt_blk1"), 3, channels[3] / 8, ic)?;

        let decoder_block4 = BasicDecBlk::new(
            vb.pp("decoder_block4"),
            channels[0] + channels[0] / 8,
            channels[1],
        )?;
        let decoder_block3 = BasicDecBlk::new(
            vb.pp("decoder_block3"),
            channels[1] + channels[0] / 8,
            channels[2],
        )?;
        let decoder_block2 = BasicDecBlk::new(
            vb.pp("decoder_block2"),
            channels[2] + channels[1] / 8,
            channels[3],
        )?;
        let decoder_block1 = BasicDecBlk::new(
            vb.pp("decoder_block1"),
            channels[3] + channels[2] / 8,
            channels[3] / 2,
        )?;

        let conv_out1 = get_conv2d(
            vb.pp("conv_out1.0"),
            channels[3] / 2 + channels[3] / 8,
            1,
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let lateral_block4 = get_conv2d(
            vb.pp("lateral_block4.conv"),
            channels[1],
            channels[1],
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let lateral_block3 = get_conv2d(
            vb.pp("lateral_block3.conv"),
            channels[2],
            channels[2],
            1,
            0,
            1,
            1,
            1,
            true,
        )?;
        let lateral_block2 = get_conv2d(
            vb.pp("lateral_block2.conv"),
            channels[3],
            channels[3],
            1,
            0,
            1,
            1,
            1,
            true,
        )?;

        // let conv_ms_spvn_4 =
        //     get_conv2d(vb.pp("conv_ms_spvn_4"), channels[1], 1, 1, 0, 1, 1, 1, true)?;
        // let conv_ms_spvn_3 =
        //     get_conv2d(vb.pp("conv_ms_spvn_3"), channels[2], 1, 1, 0, 1, 1, 1, true)?;
        // let conv_ms_spvn_2 =
        //     get_conv2d(vb.pp("conv_ms_spvn_2"), channels[3], 1, 1, 0, 1, 1, 1, true)?;
        let n = 16usize;
        let gdt_convs_4 =
            Conv2dWithBN::new(vb.pp("gdt_convs_4"), channels[1], n, 3, 1, 1, true, true)?;
        let gdt_convs_3 =
            Conv2dWithBN::new(vb.pp("gdt_convs_3"), channels[2], n, 3, 1, 1, true, true)?;
        let gdt_convs_2 =
            Conv2dWithBN::new(vb.pp("gdt_convs_2"), channels[3], n, 3, 1, 1, true, true)?;

        let gdt_convs_attn_4 = get_conv2d(vb.pp("gdt_convs_attn_4.0"), n, 1, 1, 0, 1, 1, 1, true)?;
        let gdt_convs_attn_3 = get_conv2d(vb.pp("gdt_convs_attn_3.0"), n, 1, 1, 0, 1, 1, 1, true)?;
        let gdt_convs_attn_2 = get_conv2d(vb.pp("gdt_convs_attn_2.0"), n, 1, 1, 0, 1, 1, 1, true)?;
        Ok(Self {
            ipt_blk5,
            ipt_blk4,
            ipt_blk3,
            ipt_blk2,
            ipt_blk1,
            decoder_block4,
            decoder_block3,
            decoder_block2,
            decoder_block1,
            conv_out1,
            lateral_block4,
            lateral_block3,
            lateral_block2,
            // conv_ms_spvn_4,
            // conv_ms_spvn_3,
            // conv_ms_spvn_2,
            gdt_convs_4,
            gdt_convs_3,
            gdt_convs_2,
            gdt_convs_attn_4,
            gdt_convs_attn_3,
            gdt_convs_attn_2,
        })
    }

    pub fn get_patches_batch(&self, x: &Tensor, p: &Tensor) -> Result<Tensor> {
        let (_, _, h, w) = p.dims4()?;
        let mut patches_batch = vec![];
        for idx in 0..x.dim(0)? {
            let x_i = x.i(idx)?.unsqueeze(0)?; // 保证bs的维度存在
            let columns_x = split_tensor_with_size(&x_i, w, D::Minus1)?;
            let mut patches_x = vec![];
            for col_x in columns_x {
                let pat_x = split_tensor_with_size(&col_x, h, D::Minus2)?;
                patches_x.extend_from_slice(&pat_x);
            }
            let patch_sample = Tensor::cat(&patches_x, 1)?;
            patches_batch.push(patch_sample);
        }
        let patch = Tensor::cat(&patches_batch, 0)?;
        Ok(patch)
    }

    pub fn forward(&self, features: Vec<&Tensor>) -> Result<Tensor> {
        let [x, x1, x2, x3, x4] = features[..] else {
            return Err(anyhow!(format!(
                "swintransformer output exactly 3 elements"
            )));
        };
        // let mut outs = vec![];
        let patches_batch = self.get_patches_batch(x, x4)?;
        let (_, _, x4_h, x4_w) = x4.dims4()?;
        let patches_batch = interpolate_bilinear(&patches_batch, (x4_h, x4_w), Some(true), None)?;
        let ipt_blk5_out = self.ipt_blk5.forward(&patches_batch)?;
        let x4 = Tensor::cat(&[x4, &ipt_blk5_out], 1)?;
        let p4 = self.decoder_block4.forward(&x4)?;
        // let m4 = self.conv_ms_spvn_4.forward(&p4)?;
        // outs.push(m4);
        let p4_gdt = self.gdt_convs_4.forward(&p4)?;
        let gdt_attn_4 = sigmoid(&self.gdt_convs_attn_4.forward(&p4_gdt)?)?;
        let p4 = p4.broadcast_mul(&gdt_attn_4)?;

        let (_, _, x3_h, x3_w) = x3.dims4()?;
        let p4_inter = interpolate_bilinear(&p4, (x3_h, x3_w), Some(true), None)?;
        let p3_ = self.lateral_block4.forward(x3)?;
        let p3_ = p4_inter.add(&p3_)?;
        let patches_batch = self.get_patches_batch(x, &p3_)?;
        let patches_batch = interpolate_bilinear(&patches_batch, (x3_h, x3_w), Some(true), None)?;
        let ipt_blk4_out = self.ipt_blk4.forward(&patches_batch)?;
        let p3_ = Tensor::cat(&[p3_, ipt_blk4_out], 1)?;
        let p3 = self.decoder_block3.forward(&p3_)?;
        // let m3 = self.conv_ms_spvn_3.forward(&p3)?;
        // outs.push(m3);
        let p3_gdt = self.gdt_convs_3.forward(&p3)?;
        let gdt_attn_3 = sigmoid(&self.gdt_convs_attn_3.forward(&p3_gdt)?)?;
        let p3 = p3.broadcast_mul(&gdt_attn_3)?;

        let (_, _, x2_h, x2_w) = x2.dims4()?;
        let p3_inter = interpolate_bilinear(&p3, (x2_h, x2_w), Some(true), None)?;
        let p2_ = self.lateral_block3.forward(x2)?;
        let p2_ = p3_inter.add(&p2_)?;
        let patches_batch = self.get_patches_batch(x, &p2_)?;
        let patches_batch = interpolate_bilinear(&patches_batch, (x2_h, x2_w), Some(true), None)?;
        let ipt_blk3_out = self.ipt_blk3.forward(&patches_batch)?;
        let p2_ = Tensor::cat(&[p2_, ipt_blk3_out], 1)?;
        let p2 = self.decoder_block2.forward(&p2_)?;
        // let m2 = self.conv_ms_spvn_2.forward(&p2)?;
        // outs.push(m2);
        let p2_gdt = self.gdt_convs_2.forward(&p2)?;
        let gdt_attn_2 = sigmoid(&self.gdt_convs_attn_2.forward(&p2_gdt)?)?;
        let p2 = p2.broadcast_mul(&gdt_attn_2)?;

        let (_, _, x1_h, x1_w) = x1.dims4()?;
        let p2_inter = interpolate_bilinear(&p2, (x1_h, x1_w), Some(true), None)?;
        let p1_ = self.lateral_block2.forward(x1)?;
        let p1_ = p2_inter.add(&p1_)?;
        let patches_batch = self.get_patches_batch(x, &p1_)?;
        let patches_batch = interpolate_bilinear(&patches_batch, (x1_h, x1_w), Some(true), None)?;
        let ipt_blk2_out = self.ipt_blk2.forward(&patches_batch)?;
        let p1_ = Tensor::cat(&[p1_, ipt_blk2_out], 1)?;
        let p1_ = self.decoder_block1.forward(&p1_)?;

        let (_, _, x_h, x_w) = x.dims4()?;
        let p1_ = interpolate_bilinear(&p1_, (x_h, x_w), Some(true), None)?;
        // let patches_batch = self.get_patches_batch(x, &p1_)?;
        // let patches_batch = interpolate_bilinear(&patches_batch, (x_h, x_w), Some(true))?;
        let ipt_blk1_out = self.ipt_blk1.forward(x)?;
        let p1_ = Tensor::cat(&[p1_, ipt_blk1_out], 1)?;
        let p1_out = self.conv_out1.forward(&p1_)?;
        let out = sigmoid(&p1_out)?;
        // outs.push(p1_out);
        Ok(out)
    }
}

pub struct BiRefNet {
    bb: SwinTransformer,
    squeeze_module_0: BasicDecBlk,
    decoder: Decoder,
}
impl BiRefNet {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let bb = SwinTransformer::new(
            vb.pp("bb"),
            4,
            3,
            192,
            vec![2, 2, 18, 2],
            vec![6, 12, 24, 48],
            12,
            4.0,
            true,
            true,
            vec![0, 1, 2, 3],
        )?;
        let channels = vec![3072, 1536, 768, 384];
        let in_c = channels.iter().sum();
        let squeeze_module_0 = BasicDecBlk::new(vb.pp("squeeze_module.0"), in_c, channels[0])?;
        let decoder = Decoder::new(vb.pp("decoder"), channels)?;
        Ok(Self {
            bb,
            squeeze_module_0,
            decoder,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let [ref x1, ref x2, ref x3, ref x4] = self.bb.forward(xs)?[..] else {
            return Err(anyhow!(format!(
                "swintransformer output exactly 3 elements"
            )));
        };

        let (_, _, h, w) = xs.dims4()?;
        let cat_xs = interpolate_bilinear(xs, (h / 2, w / 2), Some(true), None)?;
        let [ref x1_, ref x2_, ref x3_, ref x4_] = self.bb.forward(&cat_xs)?[..] else {
            return Err(anyhow!(format!(
                "swintransformer output exactly 3 elements"
            )));
        };
        let (_, _, x1_h, x1_w) = x1.dims4()?;
        let x1_ = interpolate_bilinear(x1_, (x1_h, x1_w), Some(true), None)?;
        let x1 = Tensor::cat(&[x1, &x1_], 1)?;
        let (_, _, x2_h, x2_w) = x2.dims4()?;
        let x2_ = interpolate_bilinear(x2_, (x2_h, x2_w), Some(true), None)?;
        let x2 = Tensor::cat(&[x2, &x2_], 1)?;
        let (_, _, x3_h, x3_w) = x3.dims4()?;
        let x3_ = interpolate_bilinear(x3_, (x3_h, x3_w), Some(true), None)?;
        let x3 = Tensor::cat(&[x3, &x3_], 1)?;
        let (_, _, x4_h, x4_w) = x4.dims4()?;
        let x4_ = interpolate_bilinear(x4_, (x4_h, x4_w), Some(true), None)?;
        let x4 = Tensor::cat(&[x4, &x4_], 1)?;
        let x1_resize = interpolate_bilinear(&x1, (x4_h, x4_w), Some(true), None)?;
        let x2_resize = interpolate_bilinear(&x2, (x4_h, x4_w), Some(true), None)?;
        let x3_resize = interpolate_bilinear(&x3, (x4_h, x4_w), Some(true), None)?;
        let x4 = Tensor::cat(&[x1_resize, x2_resize, x3_resize, x4], 1)?;

        let x4 = self.squeeze_module_0.forward(&x4)?;

        let features = vec![xs, &x1, &x2, &x3, &x4];
        let output = self.decoder.forward(features)?;
        Ok(output)
    }
}
