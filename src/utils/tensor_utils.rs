use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, shape::Dim};
use candle_nn::ops::sigmoid;

pub enum PaddingSide {
    Left,
    Right,
}

pub fn masked_fill_zeros(hidden_states: &Tensor, mask: &Tensor) -> Result<Tensor> {
    // hidden_states: (bs, seq_len, hidden_dim)
    // mask: (bs, seq_len)
    let on_false = hidden_states.zeros_like()?;
    let mask = mask
        .unsqueeze(D::Minus1)?
        .broadcast_as(hidden_states.shape())?;
    let hidden_states = mask.where_cond(&hidden_states, &on_false)?;
    Ok(hidden_states)
}

pub fn attn_masked_fill(on_true: &Tensor, mask: &Tensor, on_false: f32) -> Result<Tensor> {
    let (mask_seq_len, _) = mask.dims2()?;
    let (_, _, seq_len, _) = on_true.dims4()?;
    assert!(
        mask_seq_len >= seq_len,
        "mask seq_len less than input data seq_len"
    );
    let mask = mask.i((..seq_len, ..seq_len))?;
    let mask = mask.broadcast_as(on_true.shape())?;
    let on_false = Tensor::new(on_false, on_true.device())?.broadcast_as(on_true.shape())?;
    let filled = mask.where_cond(on_true, &on_false)?;
    Ok(filled)
}

pub fn prepare_causal_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    device: &Device,
) -> Result<Tensor> {
    // Sliding window mask?
    // let mask: Vec<_> = (0..tgt_len)
    //     .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
    //     .collect();
    // let mask = Tensor::from_vec(mask, (tgt_len, tgt_len), device)?;
    let arange = Tensor::arange(0u32, tgt_len as u32, device)?;
    let arange = arange.unsqueeze(1)?.broadcast_as((tgt_len, tgt_len))?;
    let upper_triangle = arange.t()?.gt(&arange)?;
    let mask = upper_triangle.where_cond(
        &Tensor::new(f32::NEG_INFINITY, device)?.broadcast_as(arange.shape())?,
        &Tensor::new(0f32, device)?.broadcast_as(arange.shape())?,
    )?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    let mask = mask
        .expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(DType::F32)?;
    Ok(mask)
}

pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        let kv = Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((
            b_sz,
            n_kv_head * n_rep,
            seq_len,
            head_dim,
        ))?;
        Ok(kv)
    }
}

pub fn split_tensor<D: Dim>(t: &Tensor, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
    // 按给定长度切分tensor
    // 例： t:(25), splits: [5, 10, 5, 5] dim: 0,
    // 返回vec len=4, 其中tensor维度分别是:(5), (10), (5), (5)
    let dim = dim.to_index(t.shape(), "split")?;
    let mut split_res = Vec::new();
    let mut index = 0;
    for split in splits {
        split_res.push(t.narrow(dim, index, *split)?);
        index += *split;
    }
    Ok(split_res)
}

pub fn split_tensor_with_size<D: Dim>(
    t: &Tensor,
    splits_size: usize,
    dim: D,
) -> Result<Vec<Tensor>> {
    // 按给定size切分tensor
    // 例： t:(25), splits: 5 dim: 0,
    // 返回vec len=5, 其中tensor维度分别是:(5), (5), (5), (5), (5)
    let dim = dim.to_index(t.shape(), "split")?;
    let mut split_res = Vec::new();
    let dim_size = t.dim(dim)?;
    // assert_eq!(
    //     dim_size % splits_size,
    //     0,
    //     "input tensor dim size % splits_size must be equal to 0"
    // );
    for (i, split) in (0..dim_size).step_by(splits_size).enumerate() {
        let size = splits_size.min(dim_size - i * splits_size);
        split_res.push(t.narrow(dim, split, size)?);
    }
    Ok(split_res)
}

pub fn safe_arg_sort_last_dim(t: &Tensor, ascending: bool) -> Result<Tensor> {
    // tensor在GPU上时，维度超过1024， arg_sort_last_dim方法会报错
    // 所以维度大于1024时，放到CPU上处理
    let last_dim = t.dims()[t.rank() - 1];
    if last_dim <= 1024 {
        let t = t.arg_sort_last_dim(ascending)?;
        Ok(t)
    } else {
        let cpu_tensor = t.to_device(&Device::Cpu)?;
        let sorted_indices = cpu_tensor.arg_sort_last_dim(ascending)?;
        let t = sorted_indices.to_device(t.device())?;
        Ok(t)
    }
}

pub fn nonzero_index_vec(mask: &Tensor) -> Result<Vec<u32>> {
    // 根据mask矩阵选出其中不为0的元素所在索引, 返回vec
    // 只能处理1维数据
    let mut mask = mask.clone();
    if mask.dtype() != DType::U32 {
        mask = mask.to_dtype(DType::U32)?;
    }
    match mask.rank() {
        0 => Err(anyhow!(format!(
            "input rank must > 0, the input tensor rank: {}",
            mask.rank()
        ))),
        1 => {
            let mask_vector = mask.to_vec1::<u32>()?;
            let indices: Vec<u32> = mask_vector
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val != 0 { Some(idx as u32) } else { None })
                .collect();
            Ok(indices)
        }
        _ => Err(anyhow!(format!(
            "input rank not support, the input tensor rank: {}",
            mask.rank()
        ))),
    }
}

pub fn nonzero_index(mask: &Tensor) -> Result<Tensor> {
    // 根据mask矩阵选出其中不为1的元素所在索引, 返回Tensor
    let indices_tensor = match mask.rank() {
        0 => {
            return Err(anyhow!(format!(
                "input rank must > 0, the input tensor rank: {}",
                mask.rank()
            )));
        }
        1 => {
            let index_vec = nonzero_index_vec(mask)?;
            Tensor::from_slice(&index_vec, index_vec.len(), mask.device())?
        }
        _ => {
            return Err(anyhow!(format!(
                "input rank must == 1, the input tensor rank: {}",
                mask.rank()
            )));
        }
    };
    Ok(indices_tensor)
}

pub fn zero_index_vec(mask: &Tensor) -> Result<Vec<u32>> {
    // 根据mask矩阵选出其中为0的元素所在索引, 返回vec
    // 只能处理1维数据
    let mut mask = mask.clone();
    if mask.dtype() != DType::U32 {
        mask = mask.to_dtype(DType::U32)?;
    }
    match mask.rank() {
        0 => Err(anyhow!(format!(
            "input rank must > 0, the input tensor rank: {}",
            mask.rank()
        ))),
        1 => {
            let mask_vector = mask.to_vec1::<u32>()?;
            let indices: Vec<u32> = mask_vector
                .iter()
                .enumerate()
                .filter_map(|(idx, &val)| if val == 0 { Some(idx as u32) } else { None })
                .collect();
            Ok(indices)
        }
        _ => Err(anyhow!(format!(
            "input rank not support, the input tensor rank: {}",
            mask.rank()
        ))),
    }
}

pub fn zero_index(mask: &Tensor) -> Result<Tensor> {
    let index_vec = zero_index_vec(mask)?;
    let indices_tensor = Tensor::from_slice(&index_vec, index_vec.len(), mask.device())?;
    Ok(indices_tensor)
}

pub fn nonzero_slice(mask: &Tensor) -> Result<Vec<(usize, usize)>> {
    // 根据mask矩阵选出其中非0的元素所在索引
    // 根据索引获取连续索引间隔
    // 如不为零索引元素为[0, 3, 4, 5, 8, 9]
    // 间隔为: [(0, 1), (3, 6), (8, 10)]
    // 索引前闭后开
    let mut index_vec = nonzero_index_vec(mask)?;
    match index_vec.len() {
        0 => Ok(vec![]),
        1 => Ok(vec![(index_vec[0] as usize, (index_vec[0] + 1) as usize)]),
        _ => {
            let mut vec_slice = vec![];
            let mut start = index_vec.remove(0);
            let mut last = start;

            for i in index_vec {
                if i == (last + 1) {
                    last = i;
                    continue;
                } else {
                    vec_slice.push((start as usize, (last + 1) as usize));
                    start = i;
                    last = i;
                }
            }
            vec_slice.push((start as usize, (last + 1) as usize));
            Ok(vec_slice)
        }
    }
}

pub fn masked_scatter_dim0(original: &Tensor, replace: &Tensor, mask: &Tensor) -> Result<Tensor> {
    // 根据mask中非0元素所在索引,使用replace中的数据替换掉original中的数据
    // original: rank = 3: (bs, seq_len, hidden_dim)
    // replace: rank = 2: (seq_len, hidden_dim)
    // mask: rank = 2: (bs, seq_len)
    // 推理时bs=1,为了方便替换,将bs squeeze,替换后再unsqueeze
    // 按行替换
    if original.dim(0)? != 1 || mask.dim(0)? != 1 {
        return Err(anyhow!(format!(
            "masked_scatter_dim0 original bs: {} or mask bs :{} not equal to 1 ",
            original.dim(0)?,
            mask.dim(0)? != 1
        )));
    }
    let mut original = original.squeeze(0)?;
    let mask = mask.squeeze(0)?;
    let slices = nonzero_slice(&mask)?;
    let mut sub_start = 0usize;
    let mut sub_end;
    for (start, end) in slices {
        sub_end = sub_start + (end - start);
        let sub_replace = replace.i((sub_start..sub_end, ..))?;
        original = original.slice_assign(&[(start..end), (0..original.dim(1)?)], &sub_replace)?;
        sub_start = sub_end;
    }
    original = original.unsqueeze(0)?;
    Ok(original)
}

pub fn get_not_equal_mask(input_ids: &Tensor, token_ids: u32) -> Result<Tensor> {
    let image_token_id_tensor = Tensor::new(vec![token_ids], input_ids.device())?;
    let mask = input_ids
        .broadcast_ne(&image_token_id_tensor)?
        .to_dtype(candle_core::DType::U32)?;
    Ok(mask)
}

pub fn get_equal_mask(input_ids: &Tensor, token_ids: u32) -> Result<Tensor> {
    let image_token_id_tensor =
        Tensor::new(vec![token_ids], input_ids.device())?.to_dtype(input_ids.dtype())?;
    let mask = input_ids
        .broadcast_eq(&image_token_id_tensor)?
        .to_dtype(candle_core::DType::U32)?;
    Ok(mask)
}

pub fn get_eq_indices(input_ids: &Tensor, token_id: u32) -> Result<Tensor> {
    // input_ids -> shape: (seq_len)
    let mask = get_equal_mask(input_ids, token_id)?;
    let indices = nonzero_index(&mask)?;
    Ok(indices)
}

pub fn get_vision_next_indices(input_ids: &Tensor, token_id: u32) -> Result<Tensor> {
    // input_ids -> shape: (seq_len)
    let indices = get_eq_indices(input_ids, token_id)?;
    let indices = indices.broadcast_add(&Tensor::new(vec![1u32], input_ids.device())?)?;
    Ok(indices)
}

pub fn linspace(start: f32, end: f32, steps: usize, device: &Device) -> Result<Tensor> {
    assert!(steps > 0, "steps must be > 0");
    if steps == 1 {
        let t = Tensor::from_slice(&[start], 1, device)?;
        return Ok(t);
    }
    let step_size = (end - start) / (steps - 1) as f32;
    let data: Vec<f32> = (0..steps).map(|i| start + i as f32 * step_size).collect();

    let t = Tensor::from_slice(&data, steps, device)?;
    Ok(t)
}

pub fn bitor_tensor(mask1: &Tensor, mask2: &Tensor) -> Result<Tensor> {
    assert!(
        mask1.shape() == mask2.shape(),
        " bitor_tensor two tensor shape mask be equal"
    );
    let bitor = mask1.add(mask2)?.ne(&Tensor::zeros_like(mask1)?)?;
    Ok(bitor)
}

pub fn prod_tensor_last_dim(t: &Tensor) -> Result<Tensor> {
    let prod = match t.rank() {
        0 => t.clone(),
        1 => {
            let data_type = t.dtype();
            match data_type {
                DType::U8 => {
                    let t_vec = t.to_vec1::<u8>()?;
                    let prod = t_vec.iter().product::<u8>();
                    Tensor::from_slice(&[prod], 1, t.device())?
                }
                DType::U32 => {
                    let t_vec = t.to_vec1::<u32>()?;
                    let prod = t_vec.iter().product::<u32>();
                    Tensor::from_slice(&[prod], 1, t.device())?
                }
                DType::I64 => {
                    let t_vec = t.to_vec1::<i64>()?;
                    let prod = t_vec.iter().product::<i64>();
                    Tensor::from_slice(&[prod], 1, t.device())?
                }
                DType::F64 => {
                    let t_vec = t.to_vec1::<f64>()?;
                    let prod = t_vec.iter().product::<f64>();
                    Tensor::from_slice(&[prod], 1, t.device())?
                }
                _ => {
                    let t_vec = t.to_vec1::<f32>()?;
                    let prod = t_vec.iter().product::<f32>();
                    Tensor::from_slice(&[prod], 1, t.device())?
                }
            }
        }
        2 => {
            let data_type = t.dtype();
            match data_type {
                DType::U8 => {
                    let t_vec = t.to_vec2::<u8>()?;
                    let mut prod_vec = vec![];
                    for v in t_vec.iter() {
                        let prod = v.iter().product::<u8>();
                        prod_vec.push(prod);
                    }
                    Tensor::new(prod_vec, t.device())?
                }
                DType::U32 => {
                    let t_vec = t.to_vec2::<u32>()?;
                    let mut prod_vec = vec![];
                    for v in t_vec.iter() {
                        let prod = v.iter().product::<u32>();
                        prod_vec.push(prod);
                    }
                    Tensor::new(prod_vec, t.device())?
                }
                DType::I64 => {
                    let t_vec = t.to_vec2::<i64>()?;
                    let mut prod_vec = vec![];
                    for v in t_vec.iter() {
                        let prod = v.iter().product::<i64>();
                        prod_vec.push(prod);
                    }
                    Tensor::new(prod_vec, t.device())?
                }
                DType::F64 => {
                    let t_vec = t.to_vec2::<f64>()?;
                    let mut prod_vec = vec![];
                    for v in t_vec.iter() {
                        let prod = v.iter().product::<f64>();
                        prod_vec.push(prod);
                    }
                    Tensor::new(prod_vec, t.device())?
                }
                _ => {
                    let t_vec = t.to_vec2::<f32>()?;
                    let mut prod_vec = vec![];
                    for v in t_vec.iter() {
                        let prod = v.iter().product::<f32>();
                        prod_vec.push(prod);
                    }
                    Tensor::new(prod_vec, t.device())?
                }
            }
        }
        _ => {
            return Err(anyhow!(format!("can not action this dim")));
        }
    };
    Ok(prod)
}

pub fn mask_index_add(original: &Tensor, mask: &Tensor, add: &Tensor) -> Result<Tensor> {
    let visual_nonzero_index = nonzero_index(mask)?;
    let xs = original.index_add(&visual_nonzero_index, add, 0)?;
    Ok(xs)
}

pub fn compute_1d_coords(
    input_size: usize,
    output_size: usize,
    align_corner: Option<bool>,
) -> Result<Vec<f32>> {
    if input_size == 1 {
        Ok(vec![0f32; output_size])
    } else if let Some(align_) = align_corner
        && align_
    {
        Ok((0..output_size)
            .map(|i| i as f32 * (input_size - 1) as f32 / (output_size - 1) as f32)
            .collect())
    } else {
        Ok((0..output_size)
            .map(|i| {
                (i as f32 + 0.5) * (input_size as f32 / output_size as f32) - 0.5
                // coord.max(0.0).min((input_size - 1) as f32)
            })
            .collect())
    }
}

pub fn interpolate_linear_1d(
    t: &Tensor,
    target_size: usize,
    align_corner: Option<bool>,
) -> Result<Tensor> {
    // t: [b, channels, features]
    if t.rank() != 3 {
        return Err(anyhow::anyhow!(
            "Input rank must have equal to 3 dimensions"
        ));
    }
    let shape = t.dims();
    let orig_size = shape[shape.len() - 1];
    if orig_size == target_size {
        return Ok(t.clone());
    }
    let (bs, channels, _) = t.dims3()?;
    let mut output = Tensor::zeros((bs, channels, target_size), t.dtype(), t.device())?;
    let coords = compute_1d_coords(orig_size, target_size, align_corner)?;

    for b in 0..bs {
        for c in 0..channels {
            let input_slice = t.i((b, c))?;
            let mut out_i = Vec::new();
            // for x_out in 0..target_size {
            for &coord in coords.iter().take(target_size) {
                let coord = if coord < 0.0 { 0.0 } else { coord };
                let x0 = coord.floor() as usize;
                let x1 = std::cmp::min(x0 + 1, orig_size - 1);
                let weight = (coord - x0 as f32) as f64;
                let value0 = input_slice.get(x0)?;
                let value1 = input_slice.get(x1)?;
                let interpolated =
                    (value0.affine(1.0 - weight, 0.0)? + value1.affine(weight, 0.0)?)?;
                out_i.push(interpolated);
            }
            let out_i = Tensor::stack(&out_i, 0)?.unsqueeze(0)?.unsqueeze(0)?;
            output = output.slice_assign(&[(b..b + 1), (c..c + 1), (0..target_size)], &out_i)?;
        }
    }
    output = output.contiguous()?;
    Ok(output)
}

pub fn interpolate_nearest_1d(t: &Tensor, target_size: usize) -> Result<Tensor> {
    // t: [b, channels, features]
    if t.rank() != 3 {
        return Err(anyhow::anyhow!(
            "Input rank must have equal to 3 dimensions"
        ));
    }
    let shape = t.dims();
    let orig_size = shape[shape.len() - 1];
    if orig_size == target_size {
        return Ok(t.clone());
    }

    let (bs, channels, _) = t.dims3()?;
    let mut output = Tensor::zeros((bs, channels, target_size), t.dtype(), t.device())?;
    let coords = compute_1d_coords(orig_size, target_size, None)?;

    for b in 0..bs {
        for c in 0..channels {
            let input_slice = t.i((b, c))?;
            let mut out_i = Vec::new();

            for &coord in coords.iter().take(target_size) {
                // Nearest neighbor: round to nearest integer coordinate
                let nearest_idx = coord.floor() as usize;
                let clamped_idx = nearest_idx.min(orig_size - 1);

                let value = input_slice.get(clamped_idx)?;
                out_i.push(value);
            }
            let out_i = Tensor::stack(&out_i, 0)?.unsqueeze(0)?.unsqueeze(0)?;
            output = output.slice_assign(&[(b..b + 1), (c..c + 1), (0..target_size)], &out_i)?;
        }
    }
    output = output.contiguous()?;
    Ok(output)
}

pub fn interpolate_bilinear(
    input: &Tensor,
    target_size: (usize, usize),
    align_corner: Option<bool>,
) -> Result<Tensor> {
    // input: [b, channels, height, width]
    if input.rank() != 4 {
        return Err(anyhow::anyhow!(
            "Input rank must have equal to 4 dimensions [b, c, h, w]"
        ));
    }

    let (bs, channels, input_height, input_width) = input.dims4()?;
    let (target_height, target_width) = target_size;

    // If size is the same, return clone
    if input_height == target_height && input_width == target_width {
        return Ok(input.clone());
    }

    let align_corners = align_corner.unwrap_or(false);

    // Compute scaling factors
    let height_scale = if align_corners && target_height > 1 {
        (input_height - 1) as f64 / (target_height - 1) as f64
    } else {
        input_height as f64 / target_height as f64
    };

    let width_scale = if align_corners && target_width > 1 {
        (input_width - 1) as f64 / (target_width - 1) as f64
    } else {
        input_width as f64 / target_width as f64
    };
    let dim0 = bs * channels;
    let input_3dim = input.reshape((dim0, input_height, input_width))?;
    let input_data = input_3dim.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; target_width]; target_height]; dim0];

    for c in 0..dim0 {
        for out_y in 0..target_height {
            let src_y = if align_corners {
                out_y as f64 * height_scale
            } else {
                (out_y as f64 + 0.5) * height_scale - 0.5
            };
            let src_y = src_y.max(0.0).min((input_height - 1) as f64);
            let y0 = src_y.floor() as usize;
            let y1 = (y0 + 1).min(input_height - 1);
            let dy = (src_y - y0 as f64) as f32;
            for out_x in 0..target_width {
                let src_x = if align_corners {
                    out_x as f64 * width_scale
                } else {
                    (out_x as f64 + 0.5) * width_scale - 0.5
                };
                let src_x = src_x.max(0.0).min((input_width - 1) as f64);
                let x0 = src_x.floor() as usize;
                let x1 = (x0 + 1).min(input_width - 1);
                let q00 = input_data[c][y0][x0];
                let q01 = input_data[c][y0][x1];
                let q10 = input_data[c][y1][x0];
                let q11 = input_data[c][y1][x1];
                let dx = (src_x - x0 as f64) as f32;
                let interpolated = q00 * (1.0 - dx) * (1.0 - dy)
                    + q01 * dx * (1.0 - dy)
                    + q10 * (1.0 - dx) * dy
                    + q11 * dx * dy;
                output_data[c][out_y][out_x] = interpolated;
            }
        }
    }
    let output = Tensor::new(output_data, input.device())?
        .reshape((bs, channels, target_height, target_width))?
        .to_dtype(input.dtype())?;
    Ok(output.contiguous()?)
}

fn compute_scale(input_size: usize, output_size: usize, align_corners: bool) -> f64 {
    if align_corners && output_size > 1 {
        (input_size - 1) as f64 / (output_size - 1) as f64
    } else {
        input_size as f64 / output_size as f64
    }
}

fn bicubic_filter(x: f64) -> f64 {
    let a = -0.75;
    let x = x.abs();
    if x < 1.0 {
        ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * a
    } else {
        0.0
    }
}

pub fn interpolate_bicubic_antialias(
    input: &Tensor,
    batch_size: usize,
    channels: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    height_scale: f64,
    width_scale: f64,
    align_corners: bool,
) -> Result<Tensor> {
    // tensor没有to_vec4, 所以把bs和channels先合在一起
    let dim0 = batch_size * channels;
    let input_3dim = input.reshape((dim0, input_height, input_width))?;
    let input_data = input_3dim.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; output_width]; output_height]; dim0];
    let support = 2.0 * height_scale.max(width_scale);
    for c in 0..dim0 {
        for out_y in 0..output_height {
            let center_y = if align_corners {
                out_y as f64 * height_scale
            } else {
                (out_y as f64 + 0.5) * height_scale - 0.5
            };
            let start_y = (center_y - support).ceil() as isize;
            let end_y = (center_y + support).floor() as isize;
            for out_x in 0..output_width {
                let center_x = if align_corners {
                    out_x as f64 * width_scale
                } else {
                    (out_x as f64 + 0.5) * width_scale - 0.5
                };
                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                let start_x = (center_x - support).ceil() as isize;
                let end_x = (center_x + support).floor() as isize;
                for iy in start_y..end_y {
                    for ix in start_x..end_x {
                        if iy >= 0
                            && iy < input_height as isize
                            && ix >= 0
                            && ix < input_width as isize
                        {
                            let dx = (ix as f64 - center_x).abs();
                            let dy = (iy as f64 - center_y).abs();
                            let wx = bicubic_filter(dx / width_scale.max(1.0));
                            let wy = bicubic_filter(dy / height_scale.max(1.0));
                            let weight = (wx * wy) as f32;
                            sum += input_data[c][iy as usize][ix as usize] * weight;
                            weight_sum += weight;
                        }
                    }
                }
                if weight_sum > 0.0 {
                    output_data[c][out_y][out_x] = sum / weight_sum;
                } else {
                    output_data[c][out_y][out_x] = 0.0;
                }
            }
        }
    }
    let output = Tensor::new(output_data, input.device())?
        .reshape((batch_size, channels, output_height, output_width))?
        .to_dtype(input.dtype())?;
    Ok(output)
}

fn get_cubic_coefficients(t: f64) -> [f64; 4] {
    let a = -0.75;

    let x1 = t;
    let coeff0 = cubic_convolution2(x1 + 1.0, a);
    let coeff1 = cubic_convolution1(x1, a);

    let x2 = 1.0 - t;
    let coeff2 = cubic_convolution1(x2, a);
    let coeff3 = cubic_convolution2(x2 + 1.0, a);

    [coeff0, coeff1, coeff2, coeff3]
}

// 三次卷积函数1
fn cubic_convolution1(x: f64, a: f64) -> f64 {
    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
}

// 三次卷积函数2
fn cubic_convolution2(x: f64, a: f64) -> f64 {
    ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
}

fn cubic_interp1d(x0: f32, x1: f32, x2: f32, x3: f32, t: f64) -> f32 {
    let coeffs = get_cubic_coefficients(t);
    x0 * coeffs[0] as f32 + x1 * coeffs[1] as f32 + x2 * coeffs[2] as f32 + x3 * coeffs[3] as f32
}

pub fn interpolate_bicubic_standard(
    input: &Tensor,
    batch_size: usize,
    channels: usize,
    input_height: usize,
    input_width: usize,
    output_height: usize,
    output_width: usize,
    height_scale: f64,
    width_scale: f64,
    align_corners: bool,
) -> Result<Tensor> {
    // tensor没有to_vec4, 所以把bs和channels先合在一起
    let dim0 = batch_size * channels;
    let input_3dim = input.reshape((dim0, input_height, input_width))?;
    let input_data = input_3dim.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; output_width]; output_height]; dim0];
    for c in 0..dim0 {
        for out_y in 0..output_height {
            let center_y = if align_corners {
                out_y as f64 * height_scale
            } else {
                (out_y as f64 + 0.5) * height_scale - 0.5
            };
            let in_y = center_y.floor() as isize;
            let t_y = center_y - in_y as f64;
            for out_x in 0..output_width {
                let center_x = if align_corners {
                    out_x as f64 * width_scale
                } else {
                    (out_x as f64 + 0.5) * width_scale - 0.5
                };
                let in_x = center_x.floor() as isize;
                let t_x = center_x - in_x as f64;
                let mut coefficients = [0.0; 4];
                // for k in 0..4 {
                for (k, coefficients_k) in coefficients.iter_mut().enumerate() {
                    let row = (in_y - 1 + k as isize)
                        .max(0)
                        .min(input_height as isize - 1) as usize;
                    let x_minus_1 = input_data[c][row]
                        [(in_x - 1).max(0).min(input_width as isize - 1) as usize];
                    let x_plus_0 =
                        input_data[c][row][in_x.max(0).min(input_width as isize - 1) as usize];
                    let x_plus_1 = input_data[c][row]
                        [(in_x + 1).max(0).min(input_width as isize - 1) as usize];
                    let x_plus_2 = input_data[c][row]
                        [(in_x + 2).max(0).min(input_width as isize - 1) as usize];

                    // coefficients[k] = cubic_interp1d(x_minus_1, x_plus_0, x_plus_1, x_plus_2, t_x);
                    *coefficients_k = cubic_interp1d(x_minus_1, x_plus_0, x_plus_1, x_plus_2, t_x);
                }
                output_data[c][out_y][out_x] = cubic_interp1d(
                    coefficients[0],
                    coefficients[1],
                    coefficients[2],
                    coefficients[3],
                    t_y,
                );
            }
        }
    }
    let output = Tensor::new(output_data, input.device())?
        .reshape((batch_size, channels, output_height, output_width))?
        .to_dtype(input.dtype())?;
    Ok(output)
}

pub fn interpolate_bicubic(
    input: &Tensor,
    target_size: (usize, usize),
    antialias: Option<bool>,
    align_corner: Option<bool>,
) -> Result<Tensor> {
    if input.rank() != 4 {
        return Err(anyhow::anyhow!(
            "Input rank must have at least 3 dimensions"
        ));
    }
    // if input.dim(0)? != 1 {
    //     return Err(anyhow::anyhow!("Input batch_size must be 1"));
    // }
    let (batch_size, channels, input_height, input_width) = input.dims4()?;
    let (output_height, output_width) = target_size;
    if output_height == input_height && output_width == input_width {
        return Ok(input.clone());
    }
    let align_corners = match align_corner {
        Some(true) => true,
        Some(false) => false,
        None => false,
    };
    let height_scale = compute_scale(input_height, output_height, align_corners);
    let width_scale = compute_scale(input_width, output_width, align_corners);
    // let input_squeeze = input.squeeze(0)?;
    let output = if let Some(antialias_) = antialias
        && antialias_
        && (input_height > output_height || input_width > output_width)
    {
        interpolate_bicubic_antialias(
            input,
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            height_scale,
            width_scale,
            align_corners,
        )?
    } else {
        interpolate_bicubic_standard(
            input,
            batch_size,
            channels,
            input_height,
            input_width,
            output_height,
            output_width,
            height_scale,
            width_scale,
            align_corners,
        )?
    };
    let output = output.to_dtype(input.dtype())?.to_device(input.device())?;
    Ok(output)
}

pub fn index_select_2d(t: &Tensor, index: &Tensor) -> Result<Tensor> {
    if t.rank() != 2 && index.rank() != 2 {
        return Err(anyhow::anyhow!("t and index rank must be equal to 2"));
    }
    let mut res_vec = Vec::new();
    let index_dim0 = index.dim(0)?;
    for i in 0..index_dim0 {
        let index_i = index.i(i)?;
        let rel_i = t.index_select(&index_i, 0)?;
        res_vec.push(rel_i);
    }
    let res = Tensor::stack(&res_vec, 0)?;
    Ok(res)
}

pub fn quick_gelu(xs: &Tensor) -> Result<Tensor> {
    let x = xs.affine(1.702, 0.0)?;
    let x = sigmoid(&x)?;
    Ok(xs.mul(&x)?)
}

pub fn topk(weight: &Tensor, topk: usize) -> Result<(Tensor, Tensor)> {
    let topk_idx = weight
        .arg_sort_last_dim(false)?
        .narrow(D::Minus1, 0, topk)?
        .contiguous()?;
    let topk_weight = weight.gather(&topk_idx, D::Minus1)?;
    Ok((topk_weight, topk_idx))
}

pub fn onehot(input: &Tensor, len: usize) -> Result<Tensor> {
    let mut shape = input.dims().to_vec();
    shape.push(len);
    let expand_input = input.unsqueeze(D::Minus1)?.broadcast_as(shape)?;
    let range =
        Tensor::arange(0u32, len as u32, input.device())?.broadcast_as(expand_input.dims())?;
    let onehot = expand_input.eq(&range)?;
    Ok(onehot)
}

pub fn nonzero(input: &Tensor) -> Result<(Vec<u32>, Vec<u32>)> {
    assert!(input.rank() == 2, "input rank must be 2!");
    let mut topk_ids = Vec::new();
    let mut token_ids_all = Vec::new();
    let topk = input.dim(0)?;
    let input_vec = input.to_vec2::<u32>()?;
    for (i, vec) in input_vec.iter().enumerate().take(topk) {
        let token_ids: Vec<u32> = vec
            .iter()
            .enumerate()
            .filter_map(|(idx, &val)| if val > 0 { Some(idx as u32) } else { None })
            .collect();
        let token_len = token_ids.len();
        topk_ids.extend_from_slice(&vec![i as u32; token_len]);
        token_ids_all.extend_from_slice(&token_ids);
    }
    Ok((topk_ids, token_ids_all))
}

pub fn pad_reflect_last_dim(t: &Tensor, pad: (usize, usize)) -> Result<Tensor> {
    let (pad_l, pad_r) = pad;
    let last_dim = t.dim(D::Minus1)?;
    if pad_l >= last_dim || pad_r >= last_dim {
        return Err(anyhow!(format!(
            "input pad_l {}, pad_r {} must less than t last_dim: {}",
            pad_l, pad_r, last_dim
        )));
    }
    let mut pad_tensor = t.clone();
    if pad_l > 0 {
        let left = pad_tensor.narrow(D::Minus1, 1, pad_l)?.contiguous()?;
        let last_dim_id = left.rank() - 1;
        let left_flip = left.flip(&[last_dim_id])?;
        pad_tensor = Tensor::cat(&[&left_flip, &pad_tensor], D::Minus1)?;
    }
    if pad_r > 0 {
        let start_i = last_dim - pad_r;
        let right = pad_tensor.narrow(D::Minus1, start_i, pad_r)?.contiguous()?;
        let last_dim_id = right.rank() - 1;
        let right_flip = right.flip(&[last_dim_id])?;
        pad_tensor = Tensor::cat(&[&pad_tensor, &right_flip], D::Minus1)?;
    }
    Ok(pad_tensor)
}

pub fn pad_replicate_last_dim(t: &Tensor, pad: (usize, usize)) -> Result<Tensor> {
    let (pad_l, pad_r) = pad;
    let last_dim = t.dim(D::Minus1)?;

    let mut pad_tensor = t.clone();
    if pad_l > 0 {
        let left = pad_tensor.narrow(D::Minus1, 0, 1)?.contiguous()?;
        let rank = left.rank();
        let mut shape = vec![1usize; rank - 1];
        shape.push(pad_l);
        let left_pad = left.repeat(shape)?;
        pad_tensor = Tensor::cat(&[&left_pad, &pad_tensor], D::Minus1)?;
    }
    if pad_r > 0 {
        let start_i = last_dim - 1;
        let right = pad_tensor.narrow(D::Minus1, start_i, 1)?.contiguous()?;
        let rank = right.rank();
        let mut shape = vec![1usize; rank - 1];
        shape.push(pad_r);
        let right_pad = right.repeat(shape)?;
        pad_tensor = Tensor::cat(&[&pad_tensor, &right_pad], D::Minus1)?;
    }
    Ok(pad_tensor)
}

pub fn log10(t: &Tensor) -> Result<Tensor> {
    Ok(t.log()?.affine(1.0 / 10.0_f64.ln(), 0.0)?)
}

pub fn z_score_normalize(t: &Tensor, dim: usize) -> Result<Tensor> {
    let rank = t.rank();
    if dim >= rank {
        return Err(anyhow!(format!("input dim {} must < rank {}", dim, rank)));
    }
    Ok(t.broadcast_sub(&t.mean_keepdim(dim)?)?
        .broadcast_div(&t.var_keepdim(dim)?.sqrt()?)?)
}

pub fn l2_normalize(t: &Tensor, dim: usize) -> Result<Tensor> {
    let rank = t.rank();
    if dim >= rank {
        return Err(anyhow!(format!("input dim {} must < rank {}", dim, rank)));
    }
    let l2_norm = t.sqr()?.sum_keepdim(dim)?.sqrt()?;
    Ok(t.broadcast_div(&l2_norm)?)
}

pub fn l1_normalize(t: &Tensor, dim: usize) -> Result<Tensor> {
    let rank = t.rank();
    if dim >= rank {
        return Err(anyhow!(format!("input dim {} must < rank {}", dim, rank)));
    }
    let l1_norm = t.abs()?.sum_keepdim(dim)?;
    Ok(t.broadcast_div(&l1_norm)?)
}

pub fn pool1d(xs: &Tensor, pool_size: usize, ceil_mode: bool, stype: &str) -> Result<Tensor> {
    // xs: (bs, c, dim)
    // ceil_mode: 是否保留不完整窗口，为true时通过pad实现
    if pool_size == 0 {
        return Err(anyhow!("pool_size must be greater than 0"));
    }
    let (bs, c, dim) = xs.dims3()?;
    let xs_reshape = if ceil_mode {
        let remain = dim % pool_size;
        if remain > 0 {
            let pad = pool_size - remain;
            let xs_pad = pad_replicate_last_dim(xs, (0, pad))?;
            xs_pad.reshape((bs, c, (), pool_size))?
        } else {
            xs.reshape((bs, c, (), pool_size))?
        }
    } else {
        let remain = dim % pool_size;
        if remain > 0 {
            let xs_del = xs.narrow(D::Minus1, 0, dim - remain)?;
            xs_del.reshape((bs, c, (), pool_size))?
        } else {
            xs.reshape((bs, c, (), pool_size))?
        }
    };
    let xs_pool = match stype {
        "avg" => xs_reshape.mean(D::Minus1)?,
        "max" => xs_reshape.max(D::Minus1)?,
        "min" => xs_reshape.min(D::Minus1)?,
        _ => {
            return Err(anyhow!(
                "unsupported pool type: {}, supported types are: avg, max, min",
                stype
            ));
        }
    };
    Ok(xs_pool)
}

pub fn statistics_pooling(xs: &Tensor, dim: D, keepdim: bool) -> Result<Tensor> {
    let mean = xs.mean(dim)?;
    let std = xs.var(dim)?.sqrt()?;
    let mut stats = Tensor::cat(&[mean, std], D::Minus1)?;
    if keepdim {
        stats = stats.unsqueeze(dim)?;
    }
    Ok(stats)
}
pub fn float_range_normalize(t: &Tensor) -> Result<Tensor> {
    let peak = t
        .to_dtype(DType::F32)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    if peak == 0.0 {
        return Ok(t.clone());
    }
    let mut t = t.clone();
    if peak > 1.0 {
        t = t.affine(1.0 / peak as f64, 0.0)?;
    }
    t = t.clamp(-1.0, 1.0)?;
    Ok(t)
}
