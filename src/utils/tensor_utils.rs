use std::f32;

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, IndexOp, Tensor, shape::Dim};

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
    let hidden_states = mask.where_cond(hidden_states, &on_false)?;
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

pub fn prepare_mask(mask: &Tensor) -> Result<Tensor> {
    //(bs, seq_len)
    // [[1, 1, 1, 1, 0, 0]]
    // ->
    // [[1, 1, 1, 1, 0, 0],
    //  [1, 1, 1, 1, 0, 0],
    //  [1, 1, 1, 1, 0, 0],
    //  [1, 1, 1, 1, 0, 0],
    //  [1, 1, 1, 1, 0, 0],
    //  [1, 1, 1, 1, 0, 0],]
    // (bs, 1, 1, seq_len)
    let seq_len = mask.dim(1)?;
    let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
    let mask = mask.repeat((1, 1, seq_len, 1))?;
    let on_true = mask.zeros_like()?.to_dtype(DType::F32)?;
    let on_false = Tensor::new(f32::NEG_INFINITY, mask.device())?.broadcast_as(mask.shape())?;
    let mask = mask.where_cond(&on_true, &on_false)?;
    Ok(mask)
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

pub fn sequence_mask(length: &Tensor, max_length: Option<u32>) -> Result<Tensor> {
    let max_length = max_length.unwrap_or(length.max_all()?.to_scalar::<u32>()?);
    let x = Tensor::arange(0, max_length, length.device())?.unsqueeze(0)?;
    let length = length.unsqueeze(1)?;
    let mask = x.broadcast_lt(&length)?;
    Ok(mask)
}

pub fn repeat_interleave(t: &Tensor, repeats: usize, dim: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(t.clone());
    }
    let rank = t.rank();
    if dim >= rank {
        return Err(anyhow!(
            "Dimension {} is out of range for tensor with {} dimensions",
            dim,
            rank
        ));
    }

    let dims = t.dims();
    let mut indices = Vec::with_capacity(dims[dim] * repeats);
    for i in 0..dims[dim] {
        for _ in 0..repeats {
            indices.push(i as u32);
        }
    }

    let indices_tensor = Tensor::from_vec(indices, (dims[dim] * repeats,), t.device())?;
    let t = t.index_select(&indices_tensor, dim)?;
    Ok(t)
}
