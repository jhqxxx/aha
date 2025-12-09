use anyhow::{Result, anyhow};
use candle_core::{D, IndexOp, Shape, Tensor};
use candle_nn::{
    Conv2d, Embedding, LayerNorm, Linear, Module, RmsNorm, VarBuilder, embedding, linear,
    linear_no_bias, rms_norm,
};
use num::integer::Roots;

use crate::{
    models::{
        common::{
            NaiveAttnGateUpDownMLPBlock, NaiveAttnTwoLinearMLPBlock, get_conv2d, get_layer_norm,
        },
        paddleocr_vl::config::{
            PaddleOCRVLConfig, PaddleOCRVLRopeScalingConfig, PaddleOCRVLVisionConfig,
        },
    },
    position_embed::rope::{Qwen2_5VLTextRotaryEmbedding, Qwen2_5VisionRotaryEmbedding},
    utils::tensor_utils::{
        get_vision_next_indices, interpolate_bilinear, masked_scatter_dim0, nonzero_index,
        prepare_causal_attention_mask, zero_index,
    },
};

pub struct Projector {
    merge_size: usize,
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
}

impl Projector {
    pub fn new(vb: VarBuilder, config: &PaddleOCRVLConfig) -> Result<Self> {
        let merge_size = config.vision_config.spatial_merge_size;
        let hidden_size = config.vision_config.hidden_size * merge_size * merge_size;
        let pre_norm = get_layer_norm(
            vb.pp("pre_norm"),
            config.rms_norm_eps,
            config.vision_config.hidden_size,
        )?;
        let linear_1 = linear(hidden_size, hidden_size, vb.pp("linear_1"))?;
        let linear_2 = linear(hidden_size, config.hidden_size, vb.pp("linear_2"))?;

        Ok(Self {
            merge_size,
            pre_norm,
            linear_1,
            linear_2,
        })
    }

    pub fn forward(&self, xs: &Tensor, image_grid_thw: &Tensor) -> Result<Tensor> {
        let img_num = image_grid_thw.dim(0)?;
        let mut processed_features = vec![];
        let start = 0usize;
        for i in 0..img_num {
            let [t, h, w] = image_grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            let end = start + (t * h * w) as usize;
            let xs_i = xs.i((start..end, ..))?;
            let xs_i = self.pre_norm.forward(&xs_i)?;
            let dim = xs_i.dim(1)?;
            let shape = Shape::from(vec![
                t as usize,
                h as usize / self.merge_size,
                self.merge_size,
                w as usize / self.merge_size,
                self.merge_size,
                dim,
            ]);
            let xs_i = xs_i
                .reshape((t as usize, h as usize, w as usize, dim))?
                .reshape(shape)?
                .permute((0, 1, 3, 2, 4, 5))?
                .reshape((
                    (t * h * w) as usize / self.merge_size / self.merge_size,
                    self.merge_size * self.merge_size * dim,
                ))?;
            let xs_i = self.linear_1.forward(&xs_i)?.gelu()?;
            let xs_i = self.linear_2.forward(&xs_i)?;
            processed_features.push(xs_i);
        }
        let xs = Tensor::cat(&processed_features, 0)?;
        Ok(xs)
    }
}

pub struct SiglipVisionEmbeddings {
    embed_dim: usize,
    patch_size: usize,
    patch_embedding: Conv2d,
    num_positions: usize,
    position_embedding: Embedding,
    packing_position_embedding: Embedding,
}

impl SiglipVisionEmbeddings {
    pub fn new(vb: VarBuilder, config: &PaddleOCRVLVisionConfig) -> Result<Self> {
        let embed_dim = config.hidden_size;
        let image_size = config.image_size;
        let patch_size = config.patch_size;
        let patch_embedding = get_conv2d(
            vb.pp("patch_embedding"),
            config.num_channels,
            embed_dim,
            patch_size,
            0,
            patch_size,
            1,
            1,
            true,
        )?;
        let num_positions = (image_size / patch_size).pow(2);
        let position_embedding = embedding(num_positions, embed_dim, vb.pp("position_embedding"))?;
        let packing_position_embedding =
            embedding(32768, embed_dim, vb.pp("packing_position_embedding"))?;
        Ok(Self {
            embed_dim,
            patch_size,
            patch_embedding,
            num_positions,
            position_embedding,
            packing_position_embedding,
        })
    }
    fn interpolate_pos_encoding(
        &self,
        h: usize,
        w: usize,
        is_after_patchify: bool,
    ) -> Result<Tensor> {
        let (new_height, new_width) = if is_after_patchify {
            (h, w)
        } else {
            (h / self.patch_size, w / self.patch_size)
        };
        let sqrt_num_positions = self.num_positions.sqrt();
        let patch_pos_embed = self
            .position_embedding
            .embeddings()
            .reshape((1, sqrt_num_positions, sqrt_num_positions, self.embed_dim))?
            .permute((0, 3, 1, 2))?;
        let patch_pos_embed =
            interpolate_bilinear(&patch_pos_embed, (new_height, new_width), Some(false))?;
        let patch_pos_embed =
            patch_pos_embed
                .permute((0, 2, 3, 1))?
                .reshape((1, (), self.embed_dim))?;
        Ok(patch_pos_embed)
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        position_ids: &Tensor,
        image_grid_thw: &Tensor,
        interpolate_pos_encoding: bool,
    ) -> Result<Tensor> {
        let (bs, seq_len, c, h, w) = pixel_values.dims5()?;
        let pixel_values = pixel_values.reshape((bs * seq_len, c, h, w))?;
        let patch_embeds = self.patch_embedding.forward(&pixel_values)?;
        // (bs*seq_len, c)
        let mut embeddings = patch_embeds.squeeze(D::Minus1)?.squeeze(D::Minus1)?;
        if interpolate_pos_encoding {
            let mut tmp_embeddings = vec![];
            let img_num = image_grid_thw.dim(0)?;
            let mut start = 0usize;
            for i in 0..img_num {
                let [t, h, w] = image_grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                    return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
                };
                let end = start + (t * h * w) as usize;
                let image_embeddings = embeddings.i(start..end)?;
                let position_embedding = self
                    .interpolate_pos_encoding(h as usize, w as usize, true)?
                    .squeeze(0)?
                    .repeat((t as usize, 1usize))?;
                let image_embeddings = image_embeddings.add(&position_embedding)?;
                tmp_embeddings.push(image_embeddings);
                start = end;
            }
            embeddings = Tensor::cat(&tmp_embeddings, 0)?.unsqueeze(0)?; // add bs dim
        } else {
            let packing_pos_embed = self.packing_position_embedding.forward(position_ids)?;
            embeddings = embeddings.add(&packing_pos_embed)?.unsqueeze(0)?;
        }
        Ok(embeddings)
    }
}

pub struct SiglipEncoder {
    layers: Vec<NaiveAttnTwoLinearMLPBlock>,
    rotary_pos_emb: Qwen2_5VisionRotaryEmbedding,
}

impl SiglipEncoder {
    pub fn new(vb: VarBuilder, config: &PaddleOCRVLVisionConfig) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        let mut layers = vec![];
        for i in 0..config.num_hidden_layers {
            let layer_i = NaiveAttnTwoLinearMLPBlock::new(
                vb_layers.pp(i),
                config.hidden_size,
                config.num_attention_heads,
                None,
                None,
                true,
                "self_attn",
                Some("out_proj"),
                config.intermediate_size,
                config.hidden_act,
                true,
                "mlp",
                "fc1",
                "fc2",
                config.layer_norm_eps,
                "layer_norm1",
                "layer_norm2",
            )?;
            layers.push(layer_i);
        }
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_pos_emb = Qwen2_5VisionRotaryEmbedding::new(head_dim / 2, Some(10000.0));
        Ok(Self {
            layers,
            rotary_pos_emb,
        })
    }

    pub fn forward(&self, xs: &Tensor, image_grid_thw: &Tensor) -> Result<Tensor> {
        let mut split_hids = vec![];
        let mut split_wids = vec![];
        for i in 0..image_grid_thw.dim(0)? {
            let [t, h, w] = image_grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
            };
            let pos_w: Vec<u32> = (0..h).flat_map(|_| 0u32..w).collect();
            let pos_w = pos_w.repeat(t as usize);
            let pos_w = Tensor::new(pos_w, xs.device())?;
            let pos_h: Vec<u32> = (0..h).flat_map(|h| vec![h; w as usize]).collect();
            let pos_h = pos_h.repeat(t as usize);
            let pos_h = Tensor::new(pos_h, xs.device())?;
            split_hids.push(pos_h);
            split_wids.push(pos_w);
        }
        let width_position_ids = Tensor::cat(&split_wids, 0)?;
        let height_position_ids = Tensor::cat(&split_hids, 0)?;
        let max_grid_size = image_grid_thw.i((.., 1..))?.max_all()?.to_scalar::<u32>()?;
        let rope_emb_max_grid = self
            .rotary_pos_emb
            .forward(max_grid_size as usize, xs.device())?;
        let rotary_pos_emb_h = rope_emb_max_grid.index_select(&height_position_ids, 0)?;
        let rotary_pos_emb_w = rope_emb_max_grid.index_select(&width_position_ids, 0)?;
        let rope_emb = Tensor::cat(&[rotary_pos_emb_h, rotary_pos_emb_w], 1)?.contiguous()?;
        let rope_emb = rope_emb.repeat((1, 2))?;
        let cos = rope_emb.cos()?;
        let sin = rope_emb.sin()?;
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, Some(&cos), Some(&sin), None, false)?;
        }
        Ok(xs)
    }
}

pub struct SiglipVisionModel {
    embeddings: SiglipVisionEmbeddings,
    encoder: SiglipEncoder,
    post_layernorm: LayerNorm,
}
impl SiglipVisionModel {
    pub fn new(vb: VarBuilder, config: &PaddleOCRVLVisionConfig) -> Result<Self> {
        let vb = vb.pp("vision_model");
        let embeddings = SiglipVisionEmbeddings::new(vb.pp("embeddings"), config)?;
        let encoder = SiglipEncoder::new(vb.pp("encoder"), config)?;
        let post_layernorm = get_layer_norm(
            vb.pp("post_layernorm"),
            config.layer_norm_eps,
            config.hidden_size,
        )?;
        Ok(Self {
            embeddings,
            encoder,
            post_layernorm,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        image_grid_thw: &Tensor,
        position_ids: &Tensor,
        interpolate_pos_encoding: bool,
    ) -> Result<Tensor> {
        let xs = self.embeddings.forward(
            pixel_values,
            position_ids,
            image_grid_thw,
            interpolate_pos_encoding,
        )?;
        let xs = self.encoder.forward(&xs, image_grid_thw)?;
        let xs = self.post_layernorm.forward(&xs)?;
        Ok(xs)
    }
}

pub struct Ernie4_5Model {
    embed_tokens: Embedding,
    layers: Vec<NaiveAttnGateUpDownMLPBlock>,
    norm: RmsNorm,
    rotary_emb: Qwen2_5VLTextRotaryEmbedding,
    rope_scaling: PaddleOCRVLRopeScalingConfig,
}

impl Ernie4_5Model {
    pub fn new(vb: VarBuilder, config: &PaddleOCRVLConfig) -> Result<Self> {
        let embed_tokens = embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let vb_layers = vb.pp("layers");
        let mut layers = vec![];
        for i in 0..config.num_hidden_layers {
            let layer_i = NaiveAttnGateUpDownMLPBlock::new(
                vb_layers.pp(i),
                config.hidden_size,
                config.num_attention_heads,
                Some(config.num_key_value_heads),
                Some(config.head_dim),
                config.use_bias,
                "self_attn",
                None,
                config.intermediate_size,
                config.hidden_act,
                config.use_bias,
                "mlp",
                config.rms_norm_eps,
                "input_layernorm",
                "post_attention_layernorm",
            )?;
            layers.push(layer_i);
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb =
            Qwen2_5VLTextRotaryEmbedding::new(config.head_dim, config.rope_theta as f32);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            rope_scaling: config.rope_scaling.clone(),
        })
    }

    pub fn forward(
        &mut self,
        inputs_embeds: &Tensor,
        seqlen_offset: usize,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len, _) = inputs_embeds.dims3()?;

        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(
                seqlen_offset as u32,
                (seq_len + seqlen_offset) as u32,
                inputs_embeds.device(),
            )?
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as((3, b_size, seq_len))?,
        };
        let (cos, sin) = self.rotary_emb.forward(
            &position_ids,
            inputs_embeds.dtype(),
            self.rope_scaling.mrope_section.clone(),
        )?;
        let mut xs = inputs_embeds.clone();
        let attention_mask: Option<&Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                Some(&prepare_causal_attention_mask(
                    b_size,
                    seq_len,
                    0,
                    xs.device(),
                )?)
            }
        };
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, &cos, &sin, attention_mask)?;
        }
        let xs = xs.apply(&self.norm)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }
}

pub struct PaddleOCRVLModel {
    mlp_ar: Projector,
    visual: SiglipVisionModel,
    model: Ernie4_5Model,
    pub cfg: PaddleOCRVLConfig,
    lm_head: Linear,
    rope_deltas: Option<Tensor>,
}

impl PaddleOCRVLModel {
    pub fn new(cfg: PaddleOCRVLConfig, vb: VarBuilder) -> Result<Self> {
        let mlp_ar = Projector::new(vb.pp("mlp_AR"), &cfg)?;
        let visual = SiglipVisionModel::new(vb.pp("visual"), &cfg.vision_config)?;
        let model = Ernie4_5Model::new(vb.pp("model"), &cfg)?;
        let vocab_size = cfg.vocab_size;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(model.embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            mlp_ar,
            visual,
            model,
            cfg,
            lm_head,
            rope_deltas: None,
        })
    }

    pub fn get_rope_index(
        &self,
        input_ids: &Tensor,
        image_grid_thw: Option<&Tensor>,
        video_grid_thw: Option<&Tensor>,
        mask: Option<&Tensor>,
        second_per_grid_ts: Option<Vec<f32>>,
    ) -> Result<(Tensor, Tensor)> {
        let spatial_merge_size = self.cfg.vision_config.spatial_merge_size;
        let mut mrope_position_deltas: Vec<i64> = Vec::new();
        if image_grid_thw.is_some() || video_grid_thw.is_some() {
            let total_input_ids = input_ids.clone();
            let mask_ = mask
                .cloned()
                .unwrap_or(Tensor::ones_like(&total_input_ids)?);
            let mut position_ids = Tensor::ones(
                (3, input_ids.dim(0)?, input_ids.dim(1)?),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            let mut image_index = 0;
            let mut video_index = 0;
            for i in 0..total_input_ids.dim(0)? {
                let mut input_ids_i = total_input_ids.i(i)?;
                let mask_i = mask_.i(i)?;
                // 推理时, attention_mask如果是全1向量,取非0索引的操作没必要
                if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                    let nonzero_idx = nonzero_index(&mask_i)?;
                    input_ids_i = input_ids_i.gather(&nonzero_idx, 0)?;
                }
                let mut text_start = 0;
                let mut text_end = 0;
                let mut thw = vec![];
                let mut second_per_grid_t = 0_f32;
                let mut llm_pos_ids_list: Vec<Tensor> = Vec::new();
                // vision start的下一个索引
                let vision_indices =
                    get_vision_next_indices(&input_ids_i, self.cfg.vision_start_token_id);
                match vision_indices {
                    Ok(indeices) => {
                        let vision_tokens = input_ids_i.gather(&indeices, 0)?.to_vec1::<u32>()?;
                        let vision_indices_vec = indeices.to_vec1::<u32>()?;
                        for (j, &token) in vision_tokens.iter().enumerate() {
                            if token == self.cfg.image_token_id {
                                thw = image_grid_thw.unwrap().i(image_index)?.to_vec1::<u32>()?;
                                image_index += 1;
                                text_end = vision_indices_vec[j];
                                second_per_grid_t = 0.0;
                            }
                            if token == self.cfg.video_token_id {
                                thw = video_grid_thw.unwrap().i(video_index)?.to_vec1::<u32>()?;
                                text_end = vision_indices_vec[j];
                                second_per_grid_t = match second_per_grid_ts {
                                    None => 1.0,
                                    Some(ref vec) => vec[video_index],
                                };
                                video_index += 1;
                            }
                            let llm_grid_t = thw[0];
                            let llm_grid_h = thw[1] / spatial_merge_size as u32;
                            let llm_grid_w = thw[2] / spatial_merge_size as u32;
                            let text_len = text_end - text_start;
                            let start_idx = if !llm_pos_ids_list.is_empty() {
                                llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                                    .max_all()?
                                    .to_scalar::<u32>()?
                                    + 1
                            } else {
                                0
                            };
                            let pos_ids = Tensor::arange(
                                start_idx,
                                start_idx + text_len,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .broadcast_as((3usize, text_len as usize))?;
                            llm_pos_ids_list.push(pos_ids);
                            let range_tensor = Tensor::arange(0, llm_grid_t, input_ids_i.device())?
                                .unsqueeze(D::Minus1)?;
                            let expanded_range = range_tensor.broadcast_as((
                                llm_grid_t as usize,
                                (llm_grid_h * llm_grid_w) as usize,
                            ))?;
                            let time_tensor = expanded_range
                                .broadcast_mul(&Tensor::new(
                                    vec![
                                        (second_per_grid_t
                                            * self.cfg.vision_config.tokens_per_second as f32)
                                            as u32,
                                    ],
                                    input_ids_i.device(),
                                )?)?
                                .broadcast_add(&Tensor::new(
                                    vec![start_idx + text_len],
                                    input_ids_i.device(),
                                )?)?;
                            let t_index = time_tensor.flatten_all()?;
                            let h_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_h,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(D::Minus1)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;
                            let w_index = Tensor::arange(
                                start_idx + text_len,
                                start_idx + text_len + llm_grid_w,
                                input_ids_i.device(),
                            )?
                            .unsqueeze(0)?
                            .unsqueeze(0)?
                            .broadcast_as((
                                llm_grid_t as usize,
                                llm_grid_h as usize,
                                llm_grid_w as usize,
                            ))?
                            .flatten_all()?;

                            let thw_index = Tensor::stack(&[t_index, h_index, w_index], 0)?;
                            llm_pos_ids_list.push(thw_index);
                            text_start = text_end + llm_grid_t * llm_grid_h * llm_grid_w;
                        }
                    }
                    Err(e) => {
                        println!("get vision_indices err: {}", e);
                    }
                };

                if text_start < input_ids_i.dim(0)? as u32 {
                    let start_idx = if !llm_pos_ids_list.is_empty() {
                        llm_pos_ids_list[llm_pos_ids_list.len() - 1]
                            .max_all()?
                            .to_scalar::<u32>()?
                            + 1
                    } else {
                        0
                    };
                    let text_len = input_ids_i.dim(0)? as u32 - text_start;
                    let pos_ids =
                        Tensor::arange(start_idx, start_idx + text_len, input_ids_i.device())?
                            .unsqueeze(0)?
                            .broadcast_as((3usize, text_len as usize))?;
                    llm_pos_ids_list.push(pos_ids);
                }
                let llm_position = Tensor::cat(&llm_pos_ids_list, 1)?.reshape((3, 1, ()))?;
                position_ids = position_ids
                    .slice_assign(&[(0..3), (i..i + 1), (0..input_ids.dim(1)?)], &llm_position)?;
                let position_deltas = llm_position.max_all()?.to_scalar::<u32>()? as i64 + 1
                    - input_ids_i.dim(0)? as i64;
                mrope_position_deltas.push(position_deltas);
            }

            let mut mrope_position_deltas = Tensor::new(mrope_position_deltas, input_ids.device())?;
            if mrope_position_deltas.rank() == 1 {
                mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
            }
            Ok((position_ids.contiguous()?, mrope_position_deltas))
        } else if let Some(mask) = mask {
            let mut position_ids = mask
                .to_dtype(candle_core::DType::F64)?
                .cumsum(D::Minus1)?
                .to_dtype(candle_core::DType::U32)?
                .broadcast_sub(&Tensor::new(vec![1_u32], input_ids.device())?)?;
            for i in 0..position_ids.dim(0)? {
                let mut position_ids_i = position_ids.i(i)?;
                let mask_i = mask.i(i)?;
                // 如果有pad, 将填充位置置为1
                // 当bs>1, 可能存在不同序列长度，需要添加pad使seq_len长度一致
                if mask_i.sum_all()?.to_scalar::<u32>()? != mask_i.dim(0)? as u32 {
                    let zero_indices = zero_index(&mask_i)?;
                    let replace_1 = Tensor::ones(
                        zero_indices.dim(0)?,
                        candle_core::DType::U32,
                        input_ids.device(),
                    )?;
                    position_ids_i = position_ids_i
                        .scatter(&zero_indices, &replace_1, 0)?
                        .unsqueeze(0)?;
                    position_ids = position_ids
                        .slice_assign(&[(i..i + 1), (0..position_ids.dim(1)?)], &position_ids_i)?;
                }
            }
            position_ids = position_ids
                .unsqueeze(0)?
                .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(1)?))?
                .contiguous()?;
            let mut mrope_position_deltas = position_ids
                .max(0)?
                .max(D::Minus1)?
                .broadcast_sub(&Tensor::new(
                    vec![mask.dim(D::Minus1)? as u32 - 1],
                    input_ids.device(),
                )?)?
                .contiguous()?;
            if mrope_position_deltas.rank() == 1 {
                mrope_position_deltas = mrope_position_deltas.unsqueeze(0)?;
            }
            Ok((position_ids, mrope_position_deltas))
        } else {
            let position_ids =
                Tensor::arange(0_u32, input_ids.dim(D::Minus1)? as u32, input_ids.device())?
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .broadcast_as((3, input_ids.dim(0)?, input_ids.dim(D::Minus1)?))?
                    .contiguous()?;
            let mrope_position_deltas = Tensor::zeros(
                (input_ids.dim(0)?, 1),
                input_ids.dtype(),
                input_ids.device(),
            )?;
            Ok((position_ids, mrope_position_deltas))
        }
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        image_grid_thw: Option<&Tensor>,
        image_mask: &Tensor,
        cache_position: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mut inputs_embeds = self.model.embed_tokens.forward(input_ids)?;
        if let Some(pixel_values) = pixel_values
            && let Some(image_grid_thw) = image_grid_thw
        {
            let pixel_values = pixel_values.unsqueeze(0)?;
            let mut siglip_position_ids = vec![];
            let mut sample_indices = vec![];
            let mut cu_seqlens = vec![0u32];
            let img_num = image_grid_thw.dim(0)?;
            for i in 0..img_num {
                let [t, h, w] = image_grid_thw.i(i)?.to_vec1::<u32>()?[..] else {
                    return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
                };
                let numel = h * w;
                let image_position_ids =
                    Tensor::arange(0, numel, pixel_values.device())?.repeat(t as usize)?;
                siglip_position_ids.push(image_position_ids);
                let indices =
                    Tensor::new(vec![i as u32; (numel * t) as usize], pixel_values.device())?;
                sample_indices.push(indices);
                cu_seqlens.push(cu_seqlens[cu_seqlens.len() - 1] + numel * t);
            }
            let siglip_position_ids = Tensor::cat(&siglip_position_ids, 0)?;

            let image_embed =
                self.visual
                    .forward(&pixel_values, image_grid_thw, &siglip_position_ids, true)?;
            let image_embed = image_embed.squeeze(0)?;
            let image_embed = self.mlp_ar.forward(&image_embed, image_grid_thw)?;
            inputs_embeds = masked_scatter_dim0(&inputs_embeds, &image_embed, image_mask)?;
        }
        let position_ids;
        let rope_deltas;
        if (cache_position.is_some() && cache_position.unwrap().i(0)?.to_scalar::<u32>()? == 0)
            || self.rope_deltas.is_none()
        {
            (position_ids, rope_deltas) =
                self.get_rope_index(input_ids, image_grid_thw, None, None, None)?;
            self.rope_deltas = Some(rope_deltas);
        } else {
            let (bs, seq_len, _) = inputs_embeds.dims3()?;
            let delta = if let Some(cache_position) = cache_position {
                cache_position
                    .i(0)?
                    .to_dtype(self.rope_deltas.as_ref().unwrap().dtype())?
                    .broadcast_add(self.rope_deltas.as_ref().unwrap())?
                    .contiguous()?
                    .to_dtype(candle_core::DType::U32)?
            } else {
                Tensor::zeros(1, inputs_embeds.dtype(), inputs_embeds.device())?
            };
            position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
                .unsqueeze(0)?
                .broadcast_as((bs, seq_len))?
                .broadcast_add(&delta)?
                .unsqueeze(0)?
                .broadcast_as((3, bs, seq_len))?
                .contiguous()?;
        }
        let outputs = self
            .model
            .forward(&inputs_embeds, seqlen_offset, Some(&position_ids))?;
        let seq_len = outputs.dim(1)?;
        let hidden_state = outputs.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
