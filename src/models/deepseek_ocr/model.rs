use anyhow::Result;
use candle_core::{D, IndexOp, Tensor};
use candle_nn::{
    Activation, Conv2d, Embedding, Init, LayerNorm, Linear, Module, RmsNorm, VarBuilder, embedding,
    linear, linear_b, linear_no_bias,
    ops::{sigmoid, softmax},
    rms_norm,
};
use candle_transformers::models::segment_anything::LayerNorm2d;

use crate::{
    models::{
        common::{
            GateUpDownMLP, NaiveAttention, TwoLinearMLP, eager_attention_forward, get_conv2d,
            get_layer_norm,
        },
        deepseek_ocr::config::{DeepseekOCRConfig, DeepseekV2Config},
    },
    position_embed::rope::RoPE,
    utils::tensor_utils::{
        index_select_2d, interpolate_bicubic, interpolate_linear_1d, masked_scatter_dim0, nonzero,
        onehot, prepare_causal_attention_mask, quick_gelu, topk,
    },
};

pub struct PatchEmbed {
    proj: Conv2d,
}

impl PatchEmbed {
    pub fn new(
        vb: VarBuilder,
        in_chans: usize,
        embed_dim: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let proj = get_conv2d(
            vb.pp("proj"),
            in_chans,
            embed_dim,
            kernel_size,
            padding,
            stride,
            1,
            1,
            true,
        )?;
        Ok(Self { proj })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.proj.forward(xs)?;
        let xs = xs.permute((0, 2, 3, 1))?;
        Ok(xs)
    }
}

pub struct Attention {
    num_heads: usize,
    // head_dim: usize,
    qkv: Linear,
    proj: Linear,
    scaling: f64,
    use_rel_pos: bool,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}

impl Attention {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        use_rel_pos: bool,
        input_size: Option<(usize, usize)>,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let qkv = linear_b(dim, dim * 3, qkv_bias, vb.pp("qkv"))?;

        let proj = linear(dim, dim, vb.pp("proj"))?;
        let mut rel_pos_h = None;
        let mut rel_pos_w = None;
        if use_rel_pos {
            if input_size.is_none() {
                return Err(anyhow::anyhow!(
                    "Input size must be provided if using relative positional encoding."
                ));
            }
            let input_size = input_size.unwrap();
            let h_len = 2 * input_size.0 - 1;
            let w_len = 2 * input_size.1 - 1;
            rel_pos_h = Some(vb.get_with_hints((h_len, head_dim), "rel_pos_h", Init::Const(0.))?);
            rel_pos_w = Some(vb.get_with_hints((w_len, head_dim), "rel_pos_w", Init::Const(0.))?);
        }

        Ok(Self {
            num_heads,
            // head_dim,
            qkv,
            proj,
            scaling,
            use_rel_pos,
            rel_pos_h,
            rel_pos_w,
        })
    }

    fn get_rel_pos(&self, q_size: usize, k_size: usize, rel_pos: &Tensor) -> Result<Tensor> {
        let max_rel_dist = 2 * std::cmp::max(q_size, k_size) - 1;
        let rel_pos_resized = if rel_pos.dim(0)? != max_rel_dist {
            let rel_pos_t = rel_pos
                .to_dtype(candle_core::DType::F32)?
                .t()?
                .unsqueeze(0)?
                .contiguous()?;
            let rel_pos_resized = interpolate_linear_1d(&rel_pos_t, max_rel_dist, None)?;
            rel_pos_resized
                .squeeze(0)?
                .t()?
                .contiguous()?
                .to_dtype(rel_pos.dtype())?
        } else {
            rel_pos.clone()
        };
        let q_coords = Tensor::arange(0 as f32, q_size as f32, rel_pos.device())?
            .unsqueeze(D::Minus1)?
            .affine((k_size as f64 / q_size as f64).max(1.0), 0.0)?;
        let k_coords = Tensor::arange(0 as f32, k_size as f32, rel_pos.device())?
            .unsqueeze(0)?
            .affine((q_size as f64 / k_size as f64).max(1.0), 0.0)?;
        let relative_coords = q_coords
            .broadcast_sub(&k_coords)?
            .affine(1.0, (k_size - 1) as f64)?
            .affine((q_size as f64 / k_size as f64).max(1.0), 0.0)?;
        let relative_coords = relative_coords
            .to_dtype(candle_core::DType::U32)?
            .contiguous()?;
        let rel_pos_resized = rel_pos_resized.contiguous()?;
        let res = index_select_2d(&rel_pos_resized, &relative_coords)?;
        Ok(res)
    }

    fn add_decomposed_rel_pos(
        &self,
        q: &Tensor,
        rel_pos_h: &Tensor,
        rel_pos_w: &Tensor,
        q_size: (usize, usize),
        k_size: (usize, usize),
    ) -> Result<(Tensor, Tensor)> {
        let (q_h, q_w) = q_size;
        let (k_h, k_w) = k_size;
        let rh = self.get_rel_pos(q_h, k_h, rel_pos_h)?; // (q_h, k_h, dim)
        let rw = self.get_rel_pos(q_w, k_w, rel_pos_w)?; // (q_w, k_w, dim)
        let (b, _, dim) = q.dims3()?;
        let r_q = q.reshape((b, q_h, q_w, dim))?.contiguous()?;
        let r_q_ = r_q.unsqueeze(D::Minus2)?; // (b, q_h, q_w, 1, dim)
        // rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        // rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        let rh_ = rh.unsqueeze(1)?.unsqueeze(0)?; // (1, h, 1, k, dim)
        let rel_h = r_q_.broadcast_mul(&rh_)?.sum(D::Minus1)?;
        let rw_ = rw.unsqueeze(0)?.unsqueeze(0)?; // (1, 1, w, k, dim)
        let rel_w = r_q_.broadcast_mul(&rw_)?.sum(D::Minus1)?;
        let rel_h = rel_h
            .unsqueeze(D::Minus1)?
            .reshape((b, q_h * q_w, k_h, 1))?;
        let rel_w = rel_w
            .unsqueeze(D::Minus2)?
            .reshape((b, q_h * q_w, 1, k_w))?;
        Ok((rel_h, rel_w))
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, h, w, _) = xs.dims4()?;
        // (3, B, n_head, h*w, head_dim)
        let qkv = self
            .qkv
            .forward(xs)?
            .reshape((b, h * w, 3, self.num_heads, ()))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?;
        let query_states = qkv.i(0)?.contiguous()?;
        let key_states = qkv.i(1)?.contiguous()?;
        let value_states = qkv.i(2)?.contiguous()?;
        let xs = if self.use_rel_pos {
            let q_reshape = query_states.reshape((b * self.num_heads, h * w, ()))?;
            let (rel_h, rel_w) = self.add_decomposed_rel_pos(
                &q_reshape,
                self.rel_pos_h.as_ref().unwrap(),
                self.rel_pos_w.as_ref().unwrap(),
                (h, w),
                (h, w),
            )?;
            let (_, rel_h_dim1, rel_h_dim2, rel_h_dim3) = rel_h.dims4()?;
            let rel_h = rel_h.reshape((b, self.num_heads, rel_h_dim1, rel_h_dim2, rel_h_dim3))?;
            let (_, rel_w_dim1, rel_w_dim2, rel_w_dim3) = rel_w.dims4()?;
            let rel_w = rel_w.reshape((b, self.num_heads, rel_w_dim1, rel_w_dim2, rel_w_dim3))?;
            let attn_bias = rel_h.broadcast_add(&rel_w)?.reshape((
                b,
                self.num_heads,
                rel_h_dim1,
                rel_h_dim2 * rel_w_dim3,
            ))?;
            eager_attention_forward(
                &query_states,
                &key_states,
                &value_states,
                None,
                Some(&attn_bias),
                self.scaling,
            )?
        } else {
            eager_attention_forward(
                &query_states,
                &key_states,
                &value_states,
                None,
                None,
                self.scaling,
            )?
        };
        // (b, h*w, n_head, dim)
        let xs = xs.reshape((b, h * w, ()))?.reshape((b, h, w, ()))?;
        let xs = self.proj.forward(&xs)?;
        Ok(xs)
    }
}

pub struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: TwoLinearMLP,
    window_size: usize,
}

impl Block {
    pub fn new(
        vb: VarBuilder,
        dim: usize,
        num_heads: usize,
        mlp_ratio: f32,
        qkv_bias: bool,
        eps: f64,
        act: Activation,
        use_rel_pos: bool,
        // rel_pos_zero_init: bool,
        window_size: usize,
        input_size: Option<(usize, usize)>,
    ) -> Result<Self> {
        let norm1 = get_layer_norm(vb.pp("norm1"), eps, dim)?;
        let input_size = if window_size == 0 {
            input_size
        } else {
            Some((window_size, window_size))
        };
        let attn = Attention::new(
            vb.pp("attn"),
            dim,
            num_heads,
            qkv_bias,
            use_rel_pos,
            input_size,
        )?;
        let norm2 = get_layer_norm(vb.pp("norm2"), eps, dim)?;
        let mlp_dim = (dim as f32 * mlp_ratio) as usize;
        let mlp = TwoLinearMLP::new(vb.pp("mlp"), dim, mlp_dim, dim, act, true, "lin1", "lin2")?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        })
    }

    pub fn window_partition(
        &self,
        x: &Tensor,
        window_size: usize,
    ) -> Result<(Tensor, (usize, usize))> {
        let (b, h, w, c) = x.dims4()?;
        let pad_h = (window_size - h % window_size) % window_size;
        let pad_w = (window_size - w % window_size) % window_size;
        let x = if pad_h > 0 || pad_w > 0 {
            let x = x.pad_with_zeros(1, 0, pad_h)?;
            x.pad_with_zeros(2, 0, pad_w)?
        } else {
            x.clone()
        };
        let hp = h + pad_h;
        let wp = w + pad_w;
        let x = x.reshape((
            b,
            hp / window_size,
            window_size,
            wp / window_size,
            window_size,
            c,
        ))?;
        let windows = x.permute((0, 1, 3, 2, 4, 5))?.contiguous()?.reshape((
            (),
            window_size,
            window_size,
            c,
        ))?;
        Ok((windows, (hp, wp)))
    }

    pub fn window_unpartition(
        &self,
        windows: &Tensor,
        window_size: usize,
        pad_hw: (usize, usize),
        hw: (usize, usize),
    ) -> Result<Tensor> {
        let (hp, wp) = pad_hw;
        let (h, w) = hw;
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
        let mut x = x
            .permute((0, 1, 3, 2, 4, 5))?
            .contiguous()?
            .reshape((b, hp, wp, ()))?;
        if hp > h || wp > w {
            x = x.i((.., 0..h, 0..w, ..))?
        }
        Ok(x)
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = xs.clone();
        let xs = self.norm1.forward(xs)?;
        let xs = if self.window_size > 0 {
            let h = xs.dim(1)?;
            let w = xs.dim(2)?;
            let (x, (hp, wp)) = self.window_partition(&xs, self.window_size)?;
            let x = self.attn.forward(&x)?;
            self.window_unpartition(&x, self.window_size, (hp, wp), (h, w))?
        } else {
            self.attn.forward(&xs)?
        };
        let x = shortcut.add(&xs)?;
        let x = x.add(&self.mlp.forward(&self.norm2.forward(&x)?)?)?;
        Ok(x)
    }
}

pub struct Neck {
    conv2d_0: Conv2d,
    layernorm_1: LayerNorm2d,
    conv2d_2: Conv2d,
    layernorm_3: LayerNorm2d,
}

impl Neck {
    pub fn new(vb: VarBuilder, embed_dim: usize, out_chans: usize) -> Result<Self> {
        let conv2d_0 = get_conv2d(vb.pp("0"), embed_dim, out_chans, 1, 0, 1, 1, 1, false)?;
        let layernorm_1 = LayerNorm2d::new(out_chans, 0.000001, vb.pp("1"))?;
        let conv2d_2 = get_conv2d(vb.pp("2"), out_chans, out_chans, 3, 1, 1, 1, 1, false)?;
        let layernorm_3 = LayerNorm2d::new(out_chans, 0.000001, vb.pp("3"))?;
        Ok(Self {
            conv2d_0,
            layernorm_1,
            conv2d_2,
            layernorm_3,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv2d_0.forward(xs)?;
        let xs = self.layernorm_1.forward(&xs)?;
        let xs = self.conv2d_2.forward(&xs)?;
        let xs = self.layernorm_3.forward(&xs)?;
        Ok(xs)
    }
}

pub struct ImageEncoderViT {
    // img_size: usize,
    patch_embed: PatchEmbed,
    pos_embed: Option<Tensor>,
    blocks: Vec<Block>,
    neck: Neck,
    net_2: Conv2d,
    net_3: Conv2d,
}

impl ImageEncoderViT {
    pub fn new(
        vb: VarBuilder,
        img_size: usize,
        patch_size: usize,
        in_chans: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        mlp_ratio: f32,
        out_chans: usize,
        qkv_bias: bool,
        act: Activation,
        use_abs_pos: bool,
        use_rel_pos: bool,
        // rel_pos_zero_init: bool,
        window_size: usize,
        global_attn_indexes: Vec<usize>,
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::new(
            vb.pp("patch_embed"),
            in_chans,
            embed_dim,
            patch_size,
            patch_size,
            0,
        )?;
        let pos_embed = if use_abs_pos {
            Some(vb.get_with_hints(
                (1, img_size / patch_size, img_size / patch_size, embed_dim),
                "pos_embed",
                Init::Const(0.),
            )?)
        } else {
            None
        };
        let mut blocks = Vec::new();
        let vb_blocks = vb.pp("blocks");
        for i in 0..depth {
            let window_size = if global_attn_indexes.contains(&i) {
                0
            } else {
                window_size
            };

            let block = Block::new(
                vb_blocks.pp(i),
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias,
                1e-6,
                act,
                use_rel_pos,
                // rel_pos_zero_init,
                window_size,
                Some((img_size / patch_size, img_size / patch_size)),
            )?;
            blocks.push(block);
        }

        let neck = Neck::new(vb.pp("neck"), embed_dim, out_chans)?;

        let net_2 = get_conv2d(vb.pp("net_2"), 256, 512, 3, 1, 2, 1, 1, false)?;
        let net_3 = get_conv2d(vb.pp("net_3"), 512, 1024, 3, 1, 2, 1, 1, false)?;
        Ok(Self {
            // img_size,
            patch_embed,
            pos_embed,
            blocks,
            neck,
            net_2,
            net_3,
        })
    }
    fn get_abs_pos_sam(&self, abs_pos: &Tensor, tgt_size: usize) -> Result<Tensor> {
        let src_size = abs_pos.dim(1)?;
        if src_size != tgt_size {
            let old_pos_embed = abs_pos.permute((0, 3, 1, 2))?;
            let new_pos_embed = interpolate_bicubic(
                &old_pos_embed,
                (tgt_size, tgt_size),
                Some(true),
                Some(false),
            )?;
            let new_pos_embed = new_pos_embed.permute((0, 2, 3, 1))?;
            Ok(new_pos_embed)
        } else {
            Ok(abs_pos.clone())
        }
    }
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = self.patch_embed.forward(xs)?;
        if self.pos_embed.is_some() {
            let dim1 = x.dim(1)?;
            let pos = self.get_abs_pos_sam(self.pos_embed.as_ref().unwrap(), dim1)?;
            x = x.broadcast_add(&pos)?;
        }
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        let x = x.permute((0, 3, 1, 2))?;
        let x = self.neck.forward(&x)?;
        let x = self.net_2.forward(&x)?;
        let x = self.net_3.forward(&x)?;
        Ok(x)
    }
}

pub struct CLIPVisionEmbeddings {
    class_embedding: Tensor,
    patch_embedding: Conv2d,
    // position_embedding: Embedding,
    // position_ids: Tensor,
    pos_embeds: Tensor,
    embed_dim: usize,
}

impl CLIPVisionEmbeddings {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        image_size: usize,
        patch_size: usize,
        num_channels: usize,
    ) -> Result<Self> {
        let class_embedding =
            vb.get_with_hints(hidden_size, "class_embedding", Init::Const(0.0))?;

        let patch_embedding = get_conv2d(
            vb.pp("patch_embedding"),
            num_channels,
            hidden_size,
            patch_size,
            0,
            patch_size,
            1,
            1,
            false,
        )?;

        let num_patches = (image_size / patch_size).pow(2);
        let num_positions = num_patches + 1;
        let position_embedding =
            embedding(num_positions, hidden_size, vb.pp("position_embedding"))?;
        let position_ids = Tensor::arange(0u32, num_positions as u32, vb.device())?;
        let pos_embeds = position_embedding.forward(&position_ids)?;
        Ok(Self {
            class_embedding,
            patch_embedding,
            // position_embedding,
            // position_ids,
            pos_embeds,
            embed_dim: hidden_size,
        })
    }

    fn get_abs_pos(&self, tgt_size: usize) -> Result<Tensor> {
        // println!("self.pos_embeds: {:?}", self.pos_embeds);
        let abs_pos_new = self.pos_embeds.clone();
        let (len, dim) = abs_pos_new.dims2()?;
        let src_size = ((len - 1) as f32).sqrt() as usize;
        let tgt_size = (tgt_size as f32).sqrt() as usize;
        let pos_embeds = if src_size != tgt_size {
            let cls_token = abs_pos_new.i(0)?.unsqueeze(0)?;
            let old_pos_embed = abs_pos_new.i(1..)?;
            let old_pos_embed = old_pos_embed
                .reshape((1, src_size, src_size, dim))?
                .permute((0, 3, 1, 2))?
                .contiguous()?;
            let new_pos_embed = interpolate_bicubic(
                &old_pos_embed,
                (tgt_size, tgt_size),
                Some(true),
                Some(false),
            )?;
            let new_pos_embed = new_pos_embed
                .permute((0, 2, 3, 1))?
                .reshape((tgt_size * tgt_size, dim))?;
            Tensor::cat(&[cls_token, new_pos_embed], 0)?.unsqueeze(0)?
        } else {
            self.pos_embeds.clone()
        };
        Ok(pos_embeds)
    }
    pub fn forward(&self, pixel_values: &Tensor, patch_embeds: Option<&Tensor>) -> Result<Tensor> {
        let bs = pixel_values.dim(0)?;
        let patch_embeds = match patch_embeds {
            Some(t) => t.clone(),
            None => self.patch_embedding.forward(pixel_values)?,
        };

        let patch_embeds = patch_embeds.flatten(2, D::Minus1)?.transpose(1, 2)?;
        let class_embeds = self.class_embedding.expand((bs, 1, self.embed_dim))?;
        let embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;
        let pos_embeds = self.get_abs_pos(embeddings.dim(1)?)?;
        let embeddings = embeddings.broadcast_add(&pos_embeds)?;
        Ok(embeddings)
    }
}

pub struct NoTPAttention {
    num_heads: usize,
    head_dim: usize,
    qkv_proj: Linear,
    out_proj: Linear,
    scaling: f64,
}

impl NoTPAttention {
    pub fn new(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
        let qkv_proj = linear(hidden_size, hidden_size * 3, vb.pp("qkv_proj"))?;
        let out_proj = linear(hidden_size, hidden_size, vb.pp("out_proj"))?;
        let head_dim = hidden_size / num_heads;
        let scaling = 1.0 / (head_dim as f64).sqrt();
        Ok(Self {
            num_heads,
            head_dim,
            qkv_proj,
            out_proj,
            scaling,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;
        let qkv = self.qkv_proj.forward(xs)?;
        let qkv = qkv
            .reshape((bs, seq_len, 3, self.num_heads, self.head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        let output = eager_attention_forward(&q, &k, &v, None, None, self.scaling)?;
        let output = output.reshape((bs, seq_len, ()))?;
        let output = self.out_proj.forward(&output)?;
        Ok(output)
    }
}

pub struct NoTPFeedForward {
    fc1: Linear,
    fc2: Linear,
}

impl NoTPFeedForward {
    pub fn new(vb: VarBuilder, dim: usize, hidden_dim: usize) -> Result<Self> {
        let fc1 = linear(dim, hidden_dim, vb.pp("fc1"))?;
        let fc2 = linear(hidden_dim, dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let output = self.fc1.forward(xs)?;
        let output = quick_gelu(&output)?;
        let output = self.fc2.forward(&output)?;
        Ok(output)
    }
}

pub struct NoTPTransformerBlock {
    self_attn: NoTPAttention,
    mlp: NoTPFeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}
impl NoTPTransformerBlock {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        ffn_hidden_size: usize,
        eps: f64,
    ) -> Result<Self> {
        let self_attn = NoTPAttention::new(vb.pp("self_attn"), hidden_size, num_heads)?;
        let mlp = NoTPFeedForward::new(vb.pp("mlp"), hidden_size, ffn_hidden_size)?;
        let layer_norm1 = get_layer_norm(vb.pp("layer_norm1"), eps, hidden_size)?;
        let layer_norm2 = get_layer_norm(vb.pp("layer_norm2"), eps, hidden_size)?;
        Ok(Self {
            self_attn,
            mlp,
            layer_norm1,
            layer_norm2,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.layer_norm1.forward(xs)?;
        let x = self.self_attn.forward(&x)?;
        let res = x.add(xs)?;
        let x = self.layer_norm2.forward(&res)?;
        let x = self.mlp.forward(&x)?;
        let out = x.add(&res)?;
        Ok(out)
    }
}

pub struct NoTPTransformer {
    layers: Vec<NoTPTransformerBlock>,
}
impl NoTPTransformer {
    pub fn new(
        vb: VarBuilder,
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        ffn_hidden_size: usize,
        eps: f64,
    ) -> Result<Self> {
        let mut layers = Vec::new();
        let vb_layers = vb.pp("layers");
        for i in 0..num_layers {
            let blocks = NoTPTransformerBlock::new(
                vb_layers.pp(i),
                hidden_size,
                num_heads,
                ffn_hidden_size,
                eps,
            )?;
            layers.push(blocks);
        }
        Ok(Self { layers })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

pub struct VitModel {
    embeddings: CLIPVisionEmbeddings,
    transformer: NoTPTransformer,
    pre_layrnorm: LayerNorm,
}

impl VitModel {
    pub fn new(
        vb: VarBuilder,
        image_size: usize,
        patch_size: usize,
        num_channels: usize,
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        ffn_hidden_size: usize,
        eps: f64,
    ) -> Result<Self> {
        let embeddings = CLIPVisionEmbeddings::new(
            vb.pp("embeddings"),
            hidden_size,
            image_size,
            patch_size,
            num_channels,
        )?;
        let transformer = NoTPTransformer::new(
            vb.pp("transformer"),
            num_layers,
            hidden_size,
            num_heads,
            ffn_hidden_size,
            eps,
        )?;
        let pre_layrnorm = get_layer_norm(vb.pp("pre_layrnorm"), eps, hidden_size)?;
        Ok(Self {
            embeddings,
            transformer,
            pre_layrnorm,
        })
    }

    pub fn forward(&self, xs: &Tensor, patch_embeds: Option<&Tensor>) -> Result<Tensor> {
        let x = self.embeddings.forward(xs, patch_embeds)?;
        let hidden_states = self.pre_layrnorm.forward(&x)?;
        let output = self.transformer.forward(&hidden_states)?;
        Ok(output)
    }
}

pub struct MoEGate {
    top_k: usize,
    // n_routed_experts: usize,
    routed_scaling_factor: f64,
    scoring_func: String,
    // alpha: f32,
    // seq_aux: bool,
    topk_method: String,
    // n_group: usize,
    // topk_group: usize,
    norm_topk_prob: bool,
    // gating_dim: usize,
    linear: Linear,
}

impl MoEGate {
    pub fn new(vb: VarBuilder, config: &DeepseekV2Config) -> Result<Self> {
        let linear = linear_no_bias(config.hidden_size, config.n_routed_experts, vb)?;
        Ok(Self {
            top_k: config.num_experts_per_tok,
            // n_routed_experts: config.n_routed_experts,
            routed_scaling_factor: config.routed_scaling_factor,
            scoring_func: config.scoring_func.clone(),
            // alpha: config.aux_loss_alpha,
            // seq_aux: config.seq_aux,
            topk_method: config.topk_method.clone(),
            // n_group: config.n_group,
            // topk_group: config.topk_group,
            norm_topk_prob: config.norm_topk_prob,
            // gating_dim: config.hidden_size,
            linear,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_, _, dim) = xs.dims3()?;
        let xs = xs.reshape(((), dim))?;
        let logits = self
            .linear
            .forward(&xs)?
            .to_dtype(candle_core::DType::F32)?;
        let scores = if self.scoring_func == "softmax" {
            softmax(&logits, D::Minus1)?
        } else if self.scoring_func == "sigmoid" {
            sigmoid(&logits)?
        } else {
            return Err(anyhow::anyhow!(format!(
                "insupportable scoring function for MoE gating: {}",
                self.scoring_func
            )));
        };
        let (topk_weight, topk_idx) = if self.topk_method == "greedy" {
            topk(&scores, self.top_k)?
        } else {
            return Err(anyhow::anyhow!(format!(
                "insupportable topk_method function for MoE gating: {}",
                self.topk_method
            )));
        };
        let topk_weight = if self.top_k > 1 && self.norm_topk_prob {
            topk_weight
                .broadcast_div(&topk_weight.sum_keepdim(D::Minus1)?.affine(1.0, 1e-20)?)?
                .affine(self.routed_scaling_factor, 0.0)?
        } else {
            topk_weight.affine(self.routed_scaling_factor, 0.0)?
        };
        let topk_weight = topk_weight.to_dtype(xs.dtype())?;
        Ok((topk_idx, topk_weight))
    }
}

pub struct DeepseekV2MoE {
    // num_experts_per_tok: usize,
    // ep_size: usize,
    // experts_per_rank: usize,
    // ep_rank: usize,
    experts: Vec<GateUpDownMLP>,
    gate: MoEGate,
    shared_experts: GateUpDownMLP,
}

impl DeepseekV2MoE {
    pub fn new(vb: VarBuilder, config: &DeepseekV2Config) -> Result<Self> {
        // let ep_size = 1;
        // let experts_per_rank = config.n_routed_experts;
        // let ep_rank = 0;
        let mut experts = Vec::new();
        let vb_experts = vb.pp("experts");
        for i in 0..config.n_routed_experts {
            let mlp = GateUpDownMLP::new(
                vb_experts.pp(i),
                config.hidden_size,
                config.moe_intermediate_size,
                Activation::Silu,
                false,
            )?;
            experts.push(mlp);
        }
        let gate = MoEGate::new(vb.pp("gate"), config)?;
        let shared_experts = GateUpDownMLP::new(
            vb.pp("shared_experts"),
            config.hidden_size,
            config.moe_intermediate_size * config.n_shared_experts,
            Activation::Silu,
            false,
        )?;
        Ok(Self {
            // num_experts_per_tok: config.num_experts_per_tok,
            // ep_size,
            // experts_per_rank,
            // ep_rank,
            experts,
            gate,
            shared_experts,
        })
    }

    fn moe_infer(&self, xs: &Tensor, topk_idx: &Tensor, topk_weight: &Tensor) -> Result<Tensor> {
        let expert_mask = onehot(topk_idx, self.experts.len())?
            .permute((2, 1, 0))?
            .to_dtype(candle_core::DType::U32)?;
        let expert_hit = expert_mask.sum((D::Minus1, D::Minus2))?;
        let expert_hit_vec = expert_hit.to_vec1::<u32>()?;
        let expert_hit_vec: Vec<usize> = expert_hit_vec
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val > 0 { Some(i) } else { None })
            .collect();
        let mut final_xs = xs.zeros_like()?;
        for i in expert_hit_vec {
            let expert = &self.experts[i];
            let tokens = expert_mask.i(i)?;
            let (topk_id, token_id) = nonzero(&tokens)?;
            let token_id_tensor = Tensor::new(token_id.as_slice(), xs.device())?;
            let select_tokens = xs.index_select(&token_id_tensor, 0)?;
            let select_xs = expert.forward(&select_tokens)?;
            let select_weight = topk_weight.index_select(&token_id_tensor, 0)?.gather(
                &Tensor::new(topk_id.as_slice(), xs.device())?.unsqueeze(D::Minus1)?,
                D::Minus1,
            )?;
            let select_xs = select_xs.broadcast_mul(&select_weight)?;
            final_xs = final_xs.index_add(&token_id_tensor, &select_xs, 0)?;
        }
        Ok(final_xs)
    }
}

impl Module for DeepseekV2MoE {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let identity = xs.clone();
        let (bs, seq_len, embedding_dim) = xs.dims3()?;
        let (topk_idx, topk_weight) = self
            .gate
            .forward(xs)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let xs = xs.reshape((bs * seq_len, embedding_dim))?;
        let xs = self
            .moe_infer(&xs, &topk_idx, &topk_weight)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))?;
        let xs = xs.reshape((bs, seq_len, embedding_dim))?;
        let xs_shared_experts = self.shared_experts.forward(&identity)?;
        let xs = xs.add(&xs_shared_experts)?;
        Ok(xs)
    }
}

pub enum DeepseekV2Proj {
    MOE(DeepseekV2MoE),
    MLP(GateUpDownMLP),
}

impl DeepseekV2Proj {
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            DeepseekV2Proj::MLP(model) => {
                let xs = model.forward(xs)?;
                Ok(xs)
            }
            DeepseekV2Proj::MOE(model) => {
                let xs = model.forward(xs)?;
                Ok(xs)
            }
        }
    }
}

pub struct DeepseekV2DecoderLayer {
    self_attn: NaiveAttention,
    mlp: DeepseekV2Proj,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DeepseekV2DecoderLayer {
    pub fn new(vb: VarBuilder, config: &DeepseekV2Config, layer_id: usize) -> Result<Self> {
        let self_attn = NaiveAttention::new(
            vb.pp("self_attn"),
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            None,
            false,
            None,
            None,
            None,
            None,
        )?;
        let mlp = if layer_id >= config.first_k_dense_replace
            && layer_id.is_multiple_of(config.moe_layer_freq)
        {
            DeepseekV2Proj::MOE(DeepseekV2MoE::new(vb.pp("mlp"), config)?)
        } else {
            DeepseekV2Proj::MLP(GateUpDownMLP::new(
                vb.pp("mlp"),
                config.hidden_size,
                config.intermediate_size,
                Activation::Silu,
                false,
            )?)
        };
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;

        let xs = self
            .self_attn
            .forward_with_cache(&xs, cos, sin, attention_mask, false)?;
        let residual = residual.add(&xs)?;
        let xs = self.post_attention_layernorm.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct DeepseekV2Model {
    embed_tokens: Embedding,
    layers: Vec<DeepseekV2DecoderLayer>,
    rope: RoPE,
    norm: RmsNorm,
}

impl DeepseekV2Model {
    pub fn new(vb: VarBuilder, config: DeepseekV2Config) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = Vec::new();
        let vb_layers = vb.pp("layers");
        for i in 0..config.num_hidden_layers {
            let layer = DeepseekV2DecoderLayer::new(vb_layers.pp(i), &config, i)?;
            layers.push(layer);
        }
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rope = RoPE::new(head_dim, 10000.0, vb.device())?;
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            rope,
            norm,
        })
    }
    pub fn forward(&mut self, xs: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;
        let (cos, sin) = self.rope.forward(seqlen_offset, seq_len, xs.device())?;

        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(prepare_causal_attention_mask(bs, seq_len, 0, xs.device())?)
            }
        };
        let mut xs = xs.clone();
        for layer in &mut self.layers {
            xs = layer.forward(&xs, &cos, &sin, attention_mask.as_ref())?;
        }
        let xs = self.norm.forward(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

pub struct DeepseekOCRModel {
    // config: DeepseekOCRConfig,
    sam_model: ImageEncoderViT,
    vision_model: VitModel,
    projector: Linear,
    language_model: DeepseekV2Model,
    image_newline: Tensor,
    view_seperator: Tensor,
    lm_head: Linear,
}

impl DeepseekOCRModel {
    pub fn new(vb: VarBuilder, config: DeepseekOCRConfig) -> Result<Self> {
        let vb_m = vb.pp("model");
        let sam_model = ImageEncoderViT::new(
            vb_m.pp("sam_model"),
            1024,
            16,
            3,
            768,
            12,
            12,
            4.0,
            256,
            true,
            Activation::Gelu,
            true,
            true,
            // true,
            14,
            config
                .vision_config
                .width
                .sam_vit_b
                .global_attn_indexes
                .clone(),
        )?;
        let vision_model = VitModel::new(
            vb_m.pp("vision_model"),
            224,
            14,
            3,
            24,
            1024,
            16,
            4096,
            1e-5,
        )?;
        let projector = linear(
            config.projector_config.input_dim,
            config.projector_config.n_embed,
            vb_m.pp("projector.layers"),
        )?;
        let image_newline = vb_m.get_with_hints(1280, "image_newline", Init::Const(0.))?;
        let view_seperator = vb_m.get_with_hints(1280, "view_seperator", Init::Const(0.))?;
        let language_model = DeepseekV2Model::new(vb_m, config.language_config.clone())?;
        let lm_head = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?;
        Ok(Self {
            // config,
            sam_model,
            vision_model,
            projector,
            language_model,
            image_newline,
            view_seperator,
            lm_head,
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        images_ori: Option<&Tensor>,
        image_crop: Option<&Tensor>,
        images_seq_mask: Option<&Tensor>,
        images_spatial_crop: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut input_embeds = self.language_model.embed_tokens.forward(input_ids)?;
        if input_ids.dim(1)? > 1
            && let Some(images_ori) = images_ori
            && let Some(image_crop) = image_crop
            && let Some(images_seq_mask) = images_seq_mask
            && let Some(images_spatial_crop) = images_spatial_crop
        {
            let image_num = images_ori.dim(0)?;
            let mut last_crop_num = 0;
            let mut images_in_this_batch = Vec::new();
            for i in 0..image_num {
                let image_ori_i = images_ori.i(i)?.unsqueeze(0)?;
                let global_local_features = if image_crop
                    .sum_all()?
                    .to_dtype(candle_core::DType::F32)?
                    .to_scalar::<f32>()?
                    != 0.0
                {
                    let images_spatial_crop_i = images_spatial_crop.i(i)?;
                    let width_crop_num = images_spatial_crop_i.i(0)?.to_scalar::<u32>()? as usize;
                    let height_crop_num = images_spatial_crop_i.i(1)?.to_scalar::<u32>()? as usize;
                    let crop_num = width_crop_num * height_crop_num;
                    let image_crop_i = image_crop.i(last_crop_num..last_crop_num + crop_num)?;
                    last_crop_num += crop_num;
                    let local_feature_1 = self.sam_model.forward(&image_crop_i)?;
                    let local_feature_2 = self
                        .vision_model
                        .forward(&image_crop_i, Some(&local_feature_1))?;
                    let local_feature_1 = local_feature_1.flatten(2, 3)?.permute((0, 2, 1))?;
                    let local_feature_2 = local_feature_2.i((.., 1..))?;
                    let local_features =
                        Tensor::cat(&[local_feature_2, local_feature_1], D::Minus1)?
                            .contiguous()?;
                    let local_features = self.projector.forward(&local_features)?;
                    let global_features_1 = self.sam_model.forward(&image_ori_i)?;
                    let global_features_2 = self
                        .vision_model
                        .forward(&image_ori_i, Some(&global_features_1))?;
                    let global_features_1 = global_features_1.flatten(2, 3)?.permute((0, 2, 1))?;
                    let global_features_2 = global_features_2.i((.., 1..))?;
                    let global_features =
                        Tensor::cat(&[global_features_2, global_features_1], D::Minus1)?;
                    let global_features = self.projector.forward(&global_features)?;
                    let (_, hw, n_dim) = global_features.dims3()?;
                    let h = (hw as f32).sqrt() as usize;
                    let w = h;
                    let (_, hw2, n_dim2) = local_features.dims3()?;
                    let h2 = (hw2 as f32).sqrt() as usize;
                    let w2 = h2;
                    let global_features = global_features.reshape((h, w, n_dim))?;
                    let image_newline = self.image_newline.unsqueeze(0)?.unsqueeze(0)?;
                    let global_cat = image_newline.expand((h, 1, n_dim))?;
                    let global_features = Tensor::cat(&[&global_features, &global_cat], 1)?;
                    let global_features = global_features.reshape(((), n_dim))?;
                    let local_features = local_features
                        .reshape((height_crop_num, width_crop_num, h2, w2, n_dim2))?
                        .permute((0, 2, 1, 3, 4))?
                        .reshape((height_crop_num * h2, width_crop_num * w2, n_dim2))?;
                    let local_cat = image_newline.expand((height_crop_num * h2, 1, n_dim2))?;
                    let local_features = Tensor::cat(&[&local_features, &local_cat], 1)?;
                    let local_features = local_features.reshape(((), n_dim2))?;
                    Tensor::cat(
                        &[
                            local_features,
                            global_features,
                            self.view_seperator.unsqueeze(0)?,
                        ],
                        0,
                    )?
                } else {
                    let global_features_1 = self.sam_model.forward(&image_ori_i)?;
                    let global_features_2 = self
                        .vision_model
                        .forward(&image_ori_i, Some(&global_features_1))?;
                    let global_features_1 = global_features_1.flatten(2, 3)?.permute((0, 2, 1))?;
                    let global_features_2 = global_features_2.i((.., 1..))?;
                    let global_features =
                        Tensor::cat(&[global_features_2, global_features_1], D::Minus1)?;
                    let global_features = self.projector.forward(&global_features)?;
                    let (_, hw, n_dim) = global_features.dims3()?;
                    let h = (hw as f32).sqrt() as usize;
                    let w = h;
                    let global_features = global_features.reshape((h, w, n_dim))?;
                    let image_newline = self.image_newline.unsqueeze(0)?.unsqueeze(0)?;
                    let global_cat = image_newline.expand((h, 1, n_dim))?;
                    let global_features = Tensor::cat(&[&global_features, &global_cat], 1)?;
                    let global_features = global_features.reshape(((), n_dim))?;
                    Tensor::cat(&[global_features, self.view_seperator.unsqueeze(0)?], 0)?
                };
                images_in_this_batch.push(global_local_features);
            }
            let images_in_this_batch = Tensor::cat(&images_in_this_batch, 0)?;
            input_embeds =
                masked_scatter_dim0(&input_embeds, &images_in_this_batch, images_seq_mask)?;
        }
        let outputs = self.language_model.forward(&input_embeds, seqlen_offset)?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.language_model.clear_kv_cache();
    }
}
