use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::{
    Conv1d, Embedding, Init, LayerNorm, Linear, Module, VarBuilder, embedding, linear,
};

use crate::{
    models::{
        common::{WNConv1d, get_conv1d, get_layer_norm},
        mask_gct::config::SemanticCodec,
    },
    utils::tensor_utils::{interpolate_nearest_1d, l2_normalize},
};

pub struct ConvNeXtBlock {
    dwconv: Conv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Option<Tensor>,
}

impl ConvNeXtBlock {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        intermediate_dim: usize,
        // layer_scale_init_value: f32,
    ) -> Result<Self> {
        let dwconv = get_conv1d(vb.pp("dwconv"), dim, dim, 7, 3, 1, 1, dim, true)?;
        let norm = get_layer_norm(vb.pp("norm"), 1e-6, dim)?;
        let pwconv1 = linear(dim, intermediate_dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(intermediate_dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get_with_hints(dim, "gamma", Init::Const(1.0))?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma: Some(gamma),
        })
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.dwconv.forward(xs)?;
        let xs = xs.transpose(1, 2)?;
        let xs = self.norm.forward(&xs)?;
        let xs = self.pwconv1.forward(&xs)?.gelu()?;
        let mut xs = self.pwconv2.forward(&xs)?;
        if let Some(gamma) = &self.gamma {
            xs = xs.broadcast_mul(&gamma)?;
        }
        let xs = xs.transpose(1, 2)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
}

pub struct VocosBackbone {
    embed: Conv1d,
    norm: LayerNorm,
    convnext: Vec<ConvNeXtBlock>,
    final_layer_norm: LayerNorm,
}

impl VocosBackbone {
    pub fn new(
        vb: VarBuilder,
        input_channels: usize,
        dim: usize,
        intermediate_dim: usize,
        num_layers: usize,
        // layer_scale_init_value: Option<f32>,
    ) -> Result<Self> {
        let embed = get_conv1d(vb.pp("embed"), input_channels, dim, 7, 3, 1, 1, 1, true)?;
        let norm = get_layer_norm(vb.pp("norm"), 1e-6, dim)?;
        let vb_convnext = vb.pp("convnext");
        let mut convnext = vec![];
        for i in 0..num_layers {
            let layer = ConvNeXtBlock::new(vb_convnext.pp(i), dim, intermediate_dim)?;
            convnext.push(layer);
        }
        let final_layer_norm = get_layer_norm(vb.pp("final_layer_norm"), 1e-6, dim)?;
        Ok(Self {
            embed,
            norm,
            convnext,
            final_layer_norm,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.embed.forward(xs)?;
        let mut xs = self.norm.forward(&xs.transpose(1, 2)?)?.transpose(1, 2)?;
        for layer in &self.convnext {
            xs = layer.forward(&xs)?;
        }
        xs = self.final_layer_norm.forward(&xs.transpose(1, 2)?)?;
        Ok(xs)
    }
}

pub struct FactorizedVectorQuantize {
    use_l2_normlize: bool,
    in_project: Option<WNConv1d>,
    out_project: Option<WNConv1d>,
    codebook: Embedding,
}

impl FactorizedVectorQuantize {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        codebook_size: usize,
        codebook_dim: usize,
        use_l2_normlize: bool,
    ) -> Result<Self> {
        let (in_project, out_project) = if input_dim != codebook_dim {
            let in_project = WNConv1d::new(
                vb.pp("in_project"),
                input_dim,
                codebook_dim,
                1,
                1,
                0,
                1,
                1,
                true,
            )?;
            let out_project = WNConv1d::new(
                vb.pp("out_project"),
                codebook_dim,
                input_dim,
                1,
                1,
                0,
                1,
                1,
                true,
            )?;
            (Some(in_project), Some(out_project))
        } else {
            (None, None)
        };
        let codebook = embedding(codebook_size, codebook_dim, vb.pp("codebook"))?;
        Ok(Self {
            use_l2_normlize,
            in_project,
            out_project,
            codebook,
        })
    }

    pub fn decode_latents(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, len, dim) = xs.dims3()?;
        let mut encodings = xs.transpose(1, 2)?.reshape((bs * dim, len))?;
        let mut codebook = self.codebook.embeddings().clone();
        if self.use_l2_normlize {
            encodings = l2_normalize(&encodings, 1)?;
            codebook = l2_normalize(&codebook, 1)?;
        }
        let dist1 = encodings.powf(2.0)?.sum_keepdim(1)?;
        let dist2 = encodings.affine(2.0, 0.0)?.matmul(&codebook.t()?)?;
        let dist3 = codebook.powf(2.0)?.sum_keepdim(1)?.t()?;
        let dist = dist1.broadcast_sub(&dist2)?.broadcast_add(&dist3)?;
        let indices = dist
            .affine(-1.0, 0.0)?
            .argmax(1)?
            .reshape((bs, ()))?
            .to_dtype(candle_core::DType::U32)?;
        let z_q = self.codebook.forward(&indices)?.transpose(1, 2)?;
        Ok((z_q, indices))
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut xs = xs.clone();
        if let Some(in_proj) = &self.in_project {
            xs = in_proj.forward(&xs)?;
        }
        let (z_q, indices) = self.decode_latents(&xs)?;
        let mut z_q = xs.add(&z_q.sub(&xs)?)?;
        if let Some(out_proj) = &self.out_project {
            z_q = out_proj.forward(&z_q)?;
        }
        Ok((z_q, indices))
    }
}

pub struct ResidualVQ {
    num_quantizers: usize,
    quantizers: Vec<FactorizedVectorQuantize>,
}

impl ResidualVQ {
    pub fn new(
        vb: VarBuilder,
        input_dim: usize,
        num_quantizers: usize,
        codebook_size: usize,
        codebook_dim: usize,
        // quantizer_type: &str,  // now only surpport "fvq"
    ) -> Result<Self> {
        let vb_quantizers = vb.pp("quantizers");
        let mut quantizers = vec![];
        for i in 0..num_quantizers {
            let quantizer = FactorizedVectorQuantize::new(
                vb_quantizers.pp(i),
                input_dim,
                codebook_size,
                codebook_dim,
                true,
            )?;
            quantizers.push(quantizer);
        }
        Ok(Self {
            num_quantizers,
            quantizers,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        n_quantizers: Option<usize>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let mut all_indices = vec![];
        let mut all_quantized = vec![];
        let n_quantizers = n_quantizers.unwrap_or(self.num_quantizers);
        let mut residual = xs.clone();
        let mut quantized_out = Tensor::new(0.0f32, xs.device())?.to_dtype(xs.dtype())?;
        for (i, quantizer) in (&self.quantizers).iter().enumerate() {
            if i >= n_quantizers {
                break;
            }
            let (z_q_i, indices_i) = quantizer.forward(&residual)?;
            quantized_out = quantized_out.broadcast_add(&z_q_i)?;
            residual = residual.sub(&z_q_i)?;
            all_indices.push(indices_i);
            all_quantized.push(z_q_i);
        }
        let all_indices = Tensor::stack(&all_indices, 0)?;
        let all_quantized = Tensor::stack(&all_quantized, 0)?;
        Ok((quantized_out, all_indices, all_quantized))
    }
}

pub struct RepCodec {
    downsample_scale: usize,
    down: Option<Conv1d>,
    up: Option<Conv1d>,
    encoder_0: VocosBackbone,
    encoder_1: Linear,
    decoder_0: VocosBackbone,
    decoder_1: Linear,
    quantizer: ResidualVQ,
}

impl RepCodec {
    pub fn new(vb: VarBuilder, config: &SemanticCodec) -> Result<Self> {
        let (down, up) = if config.downsample_scale > 1 {
            let down = get_conv1d(
                vb.pp("down"),
                config.hidden_size,
                config.hidden_size,
                3,
                1,
                2,
                1,
                1,
                true,
            )?;
            let up = get_conv1d(
                vb.pp("up"),
                config.hidden_size,
                config.hidden_size,
                3,
                1,
                1,
                1,
                1,
                true,
            )?;
            (Some(down), Some(up))
        } else {
            (None, None)
        };
        let encoder_0 = VocosBackbone::new(
            vb.pp("encoder.0"),
            config.hidden_size,
            config.vocos_dim,
            config.vocos_intermediate_dim,
            config.vocos_num_layers,
        )?;
        let encoder_1 = linear(config.vocos_dim, config.hidden_size, vb.pp("encoder.1"))?;
        let decoder_0 = VocosBackbone::new(
            vb.pp("decoder.0"),
            config.hidden_size,
            config.vocos_dim,
            config.vocos_intermediate_dim,
            config.vocos_num_layers,
        )?;
        let decoder_1 = linear(config.vocos_dim, config.hidden_size, vb.pp("decoder.1"))?;
        let quantizer = ResidualVQ::new(
            vb.pp("quantizer"),
            config.hidden_size,
            config.num_quantizers,
            config.codebook_size,
            config.codebook_dim,
        )?;
        Ok(Self {
            downsample_scale: config.downsample_scale,
            down,
            up,
            encoder_0,
            encoder_1,
            decoder_0,
            decoder_1,
            quantizer,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut xs = xs.clone();
        if let Some(down) = &self.down {
            xs = xs.transpose(1, 2)?;
            xs = down.forward(&xs)?.gelu()?;
            xs = xs.transpose(1, 2)?;
        }
        xs = self.encoder_0.forward(&xs.transpose(1, 2)?)?;
        xs = self.encoder_1.forward(&xs)?;
        xs = xs.transpose(1, 2)?;
        let (quantized_out, all_indices, _) = self.quantizer.forward(&xs, None)?;
        xs = self.decoder_0.forward(&quantized_out)?;
        if let Some(up) = &self.up {
            xs = xs.transpose(1, 2)?;
            let last_dim = xs.dim(D::Minus1)?;
            let target_size = last_dim * 2;
            xs = interpolate_nearest_1d(&xs, target_size)?;
            xs = up.forward(&xs)?.transpose(1, 2)?;
        }

        Ok((xs, all_indices))
    }

    pub fn quantize(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut xs = xs.clone();
        if let Some(down) = &self.down {
            xs = xs.transpose(1, 2)?;
            xs = down.forward(&xs)?.gelu()?;
            xs = xs.transpose(1, 2)?;
        }
        xs = self.encoder_0.forward(&xs.transpose(1, 2)?)?;
        xs = self.encoder_1.forward(&xs)?;
        xs = xs.transpose(1, 2)?;
        let (quantized_out, mut all_indices, _) = self.quantizer.forward(&xs, None)?;
        if all_indices.dim(0)? == 1 {
            all_indices = all_indices.squeeze(0)?;
        }
        let quantized_out = quantized_out.transpose(1, 2)?;
        Ok((all_indices, quantized_out))
    }
}
