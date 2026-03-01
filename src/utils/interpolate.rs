use anyhow::{Result, anyhow};
use candle_core::{DType, Tensor};

fn compute_scale(input_size: usize, output_size: usize, align_corners: bool) -> f32 {
    if align_corners && output_size > 1 {
        (input_size - 1) as f32 / (output_size - 1) as f32
    } else {
        input_size as f32 / output_size as f32
    }
}
pub fn compute_1d_coords(
    input_size: usize,
    output_size: usize,
    align_corner: Option<bool>,
) -> Result<Vec<f32>> {
    if input_size == 0 {
        return Err(anyhow!("input_size must be > 0"));
    }
    if output_size == 0 {
        return Err(anyhow!("output_size must be > 0"));
    }
    if input_size == 1 {
        return Ok(vec![0f32; output_size]);
    }
    let align_corners = align_corner.unwrap_or(false);
    let scale = compute_scale(input_size, output_size, align_corners);
    if align_corners {
        Ok((0..output_size).map(|i| i as f32 * scale).collect())
    } else {
        Ok((0..output_size)
            .map(|i| {
                let coord = (i as f32 + 0.5) * scale - 0.5;
                coord.clamp(0.0, (input_size - 1) as f32)
            })
            .collect())
    }
}

pub fn interpolate_nearest_1d(t: &Tensor, target_size: usize) -> Result<Tensor> {
    // t: [b, channels, features]
    if t.rank() != 3 {
        return Err(anyhow::anyhow!(
            "Input rank must have equal to 3 dimensions"
        ));
    }

    let (bs, channels, orig_size) = t.dims3()?;
    if orig_size == target_size {
        return Ok(t.clone());
    }
    let input_data = t.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; target_size]; channels]; bs];
    for b in 0..bs {
        for c in 0..channels {
            for i in 0..target_size {
                // Nearest neighbor: round to nearest integer coordinate
                let coord = if target_size == 1 {
                    (orig_size - 1) as f32 / 2.0
                } else {
                    (i as f32 + 0.5) * (orig_size as f32 / target_size as f32) - 0.5
                };
                let nearest_idx = coord.round() as usize;
                let clamped_idx = nearest_idx.clamp(0, orig_size - 1);

                output_data[b][c][i] = input_data[b][c][clamped_idx];
            }
        }
    }
    let output = Tensor::new(output_data, t.device())?.to_dtype(t.dtype())?;
    Ok(output)
}

pub fn interpolate_nearest_2d(input: &Tensor, target_size: (usize, usize)) -> Result<Tensor> {
    // t: [batch, channels, height, width]
    if input.rank() != 4 {
        return Err(anyhow::anyhow!(
            "Input tensor must have 4 dimensions [N, C, H, W], got rank {}",
            input.rank()
        ));
    }

    let (bs, channels, orig_h, orig_w) = input.dims4()?;
    let (target_h, target_w) = target_size;

    // 如果尺寸相同，直接返回克隆
    if orig_h == target_h && orig_w == target_w {
        return Ok(input.clone());
    }

    // 将输入数据转为 Vec 以便索引
    let dim0 = bs * channels;
    let input_3dim = input.reshape((dim0, orig_h, orig_w))?;
    let input_data = input_3dim.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; target_w]; target_h]; dim0];

    for c in 0..dim0 {
        for i in 0..target_h {
            // 计算高度方向的最近邻索引
            let coord_h = if target_h == 1 {
                (orig_h - 1) as f32 / 2.0
            } else {
                (i as f32 + 0.5) * (orig_h as f32 / target_h as f32) - 0.5
            };
            let nearest_h = coord_h.round() as usize;
            let clamped_h = nearest_h.clamp(0, orig_h - 1);

            for j in 0..target_w {
                // 计算宽度方向的最近邻索引
                let coord_w = if target_w == 1 {
                    (orig_w - 1) as f32 / 2.0
                } else {
                    (j as f32 + 0.5) * (orig_w as f32 / target_w as f32) - 0.5
                };
                let nearest_w = coord_w.round() as usize;
                let clamped_w = nearest_w.clamp(0, orig_w - 1);

                output_data[c][i][j] = input_data[c][clamped_h][clamped_w];
            }
        }
    }
    let output = Tensor::new(output_data, input.device())?
        .reshape((bs, channels, target_h, target_w))?
        .to_dtype(input.dtype())?
        .contiguous()?;
    Ok(output)
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
    let (bs, channels, orig_size) = t.dims3()?;
    if orig_size == target_size {
        return Ok(t.clone());
    }
    let coords = compute_1d_coords(orig_size, target_size, align_corner)?;
    let input_data = t.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; target_size]; channels]; bs];

    for b in 0..bs {
        for c in 0..channels {
            for (i, &coord) in coords.iter().enumerate() {
                let coord = coord.clamp(0.0, (orig_size - 1) as f32);
                let x0 = coord.floor() as usize;
                let x1 = (x0 + 1).min(orig_size - 1);
                let weight = coord - x0 as f32;
                let value0 = input_data[b][c][x0];
                let value1 = input_data[b][c][x1];

                output_data[b][c][i] = value0 * (1.0 - weight) + value1 * weight;
            }
        }
    }
    let output = Tensor::new(output_data, t.device())?.to_dtype(t.dtype())?;
    Ok(output)
}

fn antialias_filter(x: f32) -> f32 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

pub fn interpolate_bilinear(
    input: &Tensor,
    target_size: (usize, usize),
    align_corner: Option<bool>,
    antialias: Option<bool>,
) -> Result<Tensor> {
    if input.rank() != 4 {
        return Err(anyhow::anyhow!(
            "Input rank must have equal to 4 dimensions [b, c, h, w]"
        ));
    }

    let (_, _, input_height, input_width) = input.dims4()?;
    let (target_height, target_width) = target_size;

    if input_height == target_height && input_width == target_width {
        return Ok(input.clone());
    }

    let output = if antialias.unwrap_or(false)
        && (target_height < input_height || target_width < input_width)
    {
        interpolate_bilinear_antialias(input, target_size)?
    } else {
        interpolate_bilinear_standard(input, target_size, align_corner)?
    };
    let output = output.to_dtype(input.dtype())?.to_device(input.device())?;
    Ok(output)
}

pub fn interpolate_bilinear_standard(
    input: &Tensor,
    target_size: (usize, usize),
    align_corner: Option<bool>,
) -> Result<Tensor> {
    let (bs, channels, input_height, input_width) = input.dims4()?;
    let (target_height, target_width) = target_size;

    // 计算两个维度的采样坐标
    let coords_h = compute_1d_coords(input_height, target_height, align_corner)?;
    let coords_w = compute_1d_coords(input_width, target_width, align_corner)?;

    let dim0 = bs * channels;
    let input_3dim = input.reshape((dim0, input_height, input_width))?;
    let input_data = input_3dim.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; target_width]; target_height]; dim0];

    for c in 0..dim0 {
        for (i, &coord_h) in coords_h.iter().enumerate() {
            let coord_h = coord_h.clamp(0.0, (input_height - 1) as f32);
            let y0 = coord_h.floor() as usize;
            let y1 = (y0 + 1).min(input_height - 1);
            let dy = coord_h - y0 as f32;
            for (j, &coord_w) in coords_w.iter().enumerate() {
                let coord_w = coord_w.clamp(0.0, (input_width - 1) as f32);
                let x0 = coord_w.floor() as usize;
                let x1 = (x0 + 1).min(input_width - 1);
                let dx = coord_w - x0 as f32;

                let q00 = input_data[c][y0][x0];
                let q01 = input_data[c][y0][x1];
                let q10 = input_data[c][y1][x0];
                let q11 = input_data[c][y1][x1];
                output_data[c][i][j] = q00 * (1.0 - dx) * (1.0 - dy)
                    + q01 * dx * (1.0 - dy)
                    + q10 * (1.0 - dx) * dy
                    + q11 * dx * dy;
            }
        }
    }
    let output = Tensor::new(output_data, input.device())?
        .reshape((bs, channels, target_height, target_width))?
        .to_dtype(input.dtype())?
        .contiguous()?;
    Ok(output)
}

pub fn interpolate_bilinear_antialias(
    input: &Tensor,
    target_size: (usize, usize),
) -> Result<Tensor> {
    let (bs, channels, input_height, input_width) = input.dims4()?;
    let (target_height, target_width) = target_size;

    let scale_h = input_height as f32 / target_height as f32;
    let scale_w = input_width as f32 / target_width as f32;

    let dim0 = bs * channels;
    let input_3dim = input.reshape((dim0, input_height, input_width))?;
    let input_data = input_3dim.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; target_width]; target_height]; dim0];

    let support_size = scale_h.max(scale_w);
    for c in 0..dim0 {
        for out_y in 0..target_height {
            let center_y = (out_y as f32 + 0.5) * scale_h - 0.5;
            let start_y = (center_y - support_size).max(0.0) as usize;
            let end_y = (center_y + support_size).min(input_height as f32 - 1.0) as usize;
            for out_x in 0..target_width {
                let center_x = (out_x as f32 + 0.5) * scale_w - 0.5;
                let start_x = (center_x - support_size).max(0.0) as usize;
                let end_x = (center_x + support_size).min(input_width as f32 - 1.0) as usize;
                let mut total_weight = 0.0;
                let mut weighted_sum = 0.0;

                for src_y in start_y..end_y + 1 {
                    for src_x in start_x..end_x + 1 {
                        let dist_x = (src_x as f32 - center_x).abs();
                        let dist_y = (src_y as f32 - center_y).abs();
                        let weight_x = antialias_filter(dist_x / scale_w);
                        let weight_y = antialias_filter(dist_y / scale_h);
                        let weight = weight_x * weight_y;
                        weighted_sum += input_data[c][src_y][src_x] * weight;
                        total_weight += weight;
                    }
                }
                let result = if total_weight > 0.0 {
                    weighted_sum / total_weight
                } else {
                    let y = center_y.round().clamp(0.0, (input_height - 1) as f32) as usize;
                    let x = center_x.round().clamp(0.0, (input_width - 1) as f32) as usize;
                    input_data[c][y][x]
                };
                output_data[c][out_y][out_x] = result;
            }
        }
    }
    let output = Tensor::new(output_data, input.device())?
        .reshape((bs, channels, target_height, target_width))?
        .to_dtype(input.dtype())?
        .contiguous()?;
    Ok(output)
}

fn bicubic_filter(x: f32, a: f32) -> f32 {
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
    target_size: (usize, usize),
) -> Result<Tensor> {
    let (bs, channels, input_height, input_width) = input.dims4()?;
    let (target_height, target_width) = target_size;

    let scale_h = input_height as f32 / target_height as f32;
    let scale_w = input_width as f32 / target_width as f32;

    // tensor没有to_vec4, 所以把bs和channels先合在一起
    let dim0 = bs * channels;
    let input_3dim = input.reshape((dim0, input_height, input_width))?;
    let input_data = input_3dim.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; target_width]; target_height]; dim0];
    let scale = scale_h.max(scale_w);
    let support_size = if scale >= 1.0 {
        (2.0 * scale).ceil()
    } else {
        2.0
    };
    for c in 0..dim0 {
        for out_y in 0..target_height {
            let center_y = (out_y as f32 + 0.5) * scale_h - 0.5;
            let start_y = (center_y - support_size).ceil() as isize;
            let end_y = (center_y + support_size).floor() as isize;
            for out_x in 0..target_width {
                let center_x = (out_x as f32 + 0.5) * scale_w - 0.5;
                let start_x = (center_x - support_size).ceil() as isize;
                let end_x = (center_x + support_size).floor() as isize;
                let mut sum = 0.0;
                let mut weight_sum = 0.0;
                for iy in start_y..end_y + 1 {
                    for ix in start_x..end_x + 1 {
                        if iy >= 0
                            && iy < input_height as isize
                            && ix >= 0
                            && ix < input_width as isize
                        {
                            let dx = (ix as f32 - center_x).abs();
                            let dy = (iy as f32 - center_y).abs();
                            let wx = bicubic_filter(dx / scale_w.max(1.0), -0.5);
                            let wy = bicubic_filter(dy / scale_h.max(1.0), -0.5);
                            let weight = wx * wy;
                            sum += input_data[c][iy as usize][ix as usize] * weight;
                            weight_sum += weight;
                        }
                    }
                }
                if weight_sum > 0.0 {
                    output_data[c][out_y][out_x] = sum / weight_sum;
                } else {
                    let y = center_y.round().clamp(0.0, (input_height - 1) as f32) as usize;
                    let x = center_x.round().clamp(0.0, (input_width - 1) as f32) as usize;
                    output_data[c][out_y][out_x] = input_data[c][y][x];
                }
            }
        }
    }
    let output = Tensor::new(output_data, input.device())?
        .reshape((bs, channels, target_height, target_width))?
        .to_dtype(input.dtype())?
        .contiguous()?;
    Ok(output)
}

// 三次卷积函数1
fn cubic_convolution1(x: f64, a: f64) -> f64 {
    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
}

// 三次卷积函数2
fn cubic_convolution2(x: f64, a: f64) -> f64 {
    (((x - 5.0) * x + 8.0) * x - 4.0) * a
}

fn get_cubic_coefficients(t: f64, a: f64) -> [f64; 4] {
    let coeff0 = cubic_convolution2(t + 1.0, a);
    let coeff1 = cubic_convolution1(t, a);
    let coeff2 = cubic_convolution1(1.0 - t, a);
    let coeff3 = cubic_convolution2(1.0 - t + 1.0, a);

    [coeff0, coeff1, coeff2, coeff3]
}

fn cubic_interp1d(x0: f32, x1: f32, x2: f32, x3: f32, t: f64, a: f64) -> f32 {
    let coeffs = get_cubic_coefficients(t, a);
    x0 * coeffs[0] as f32 + x1 * coeffs[1] as f32 + x2 * coeffs[2] as f32 + x3 * coeffs[3] as f32
}

pub fn interpolate_bicubic_standard(
    input: &Tensor,
    target_size: (usize, usize),
    align_corner: Option<bool>,
) -> Result<Tensor> {
    let (bs, channels, input_height, input_width) = input.dims4()?;
    let (target_height, target_width) = target_size;
    let align_corners = align_corner.unwrap_or(false);
    let scale_h = compute_scale(input_height, target_height, align_corners);
    let scale_w = compute_scale(input_width, target_width, align_corners);

    let dim0 = bs * channels;
    let input_3dim = input.reshape((dim0, input_height, input_width))?;
    let input_data = input_3dim.to_dtype(DType::F32)?.to_vec3::<f32>()?;
    let mut output_data = vec![vec![vec![0.0f32; target_width]; target_height]; dim0];
    for c in 0..dim0 {
        for out_y in 0..target_height {
            let center_y = if align_corners {
                out_y as f32 * scale_h
            } else {
                (out_y as f32 + 0.5) * scale_h - 0.5
            }
            .clamp(0.0, (input_height - 1) as f32);
            let in_y = center_y.floor() as isize;
            let t_y = center_y - in_y as f32;
            for out_x in 0..target_width {
                let center_x = if align_corners {
                    out_x as f32 * scale_w
                } else {
                    (out_x as f32 + 0.5) * scale_w - 0.5
                }
                .clamp(0.0, (input_width - 1) as f32);
                let in_x: isize = center_x.floor() as isize;
                let t_x = center_x - in_x as f32;
                let mut coefficients = [0.0; 4];
                for k in 0..4 {
                    let row = (in_y - 1 + k as isize).clamp(0, input_height as isize - 1) as usize;
                    let x_minus_1 =
                        input_data[c][row][(in_x - 1).clamp(0, input_width as isize - 1) as usize];
                    let x_plus_0 =
                        input_data[c][row][in_x.clamp(0, input_width as isize - 1) as usize];
                    let x_plus_1 =
                        input_data[c][row][(in_x + 1).clamp(0, input_width as isize - 1) as usize];
                    let x_plus_2 =
                        input_data[c][row][(in_x + 2).clamp(0, input_width as isize - 1) as usize];

                    coefficients[k] =
                        cubic_interp1d(x_minus_1, x_plus_0, x_plus_1, x_plus_2, t_x as f64, -0.75);
                }
                output_data[c][out_y][out_x] = cubic_interp1d(
                    coefficients[0],
                    coefficients[1],
                    coefficients[2],
                    coefficients[3],
                    t_y as f64,
                    -0.75,
                );
            }
        }
    }
    let output = Tensor::new(output_data, input.device())?
        .reshape((bs, channels, target_height, target_width))?
        .to_dtype(input.dtype())?
        .contiguous()?;
    Ok(output)
}

pub fn interpolate_bicubic(
    input: &Tensor,
    target_size: (usize, usize),
    align_corner: Option<bool>,
    antialias: Option<bool>,
) -> Result<Tensor> {
    if input.rank() != 4 {
        return Err(anyhow::anyhow!(
            "Input rank must have at least 3 dimensions"
        ));
    }
    let (_, _, input_height, input_width) = input.dims4()?;
    let (output_height, output_width) = target_size;
    if output_height == input_height && output_width == input_width {
        return Ok(input.clone());
    }
    // let input_squeeze = input.squeeze(0)?;
    let output = if antialias.unwrap_or(false)
        && (input_height > output_height || input_width > output_width)
    {
        interpolate_bicubic_antialias(input, target_size)?
    } else {
        interpolate_bicubic_standard(input, target_size, align_corner)?
    };
    let output = output.to_dtype(input.dtype())?.to_device(input.device())?;
    Ok(output)
}
