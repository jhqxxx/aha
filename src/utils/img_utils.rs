use std::io::Cursor;
use std::{collections::HashSet, path::PathBuf};

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionParameters, ChatMessage, ChatMessageContent, ChatMessageContentPart,
};
use anyhow::{Result, anyhow};
use base64::{Engine, engine::general_purpose};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, ImageBuffer, ImageReader, Rgb, RgbImage, imageops};
use rayon::prelude::*;

use crate::utils::{ceil_by_factor, floor_by_factor, round_by_factor};

pub fn load_image_from_url(url: &str) -> Result<DynamicImage> {
    tokio::task::block_in_place(|| {
        let response = reqwest::blocking::get(url)
            .map_err(|e| anyhow!(format!("Failed to fetch image from url: {}", e)))?;
        let bytes = response
            .bytes()
            .map_err(|e| anyhow!(format!("Failed to get image bytes: {}", e)))?;

        let cursor = Cursor::new(bytes);
        let img = ImageReader::new(cursor)
            .with_guessed_format()
            .map_err(|e| anyhow!(format!("Failed to read image format: {}", e)))?
            .decode()
            .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?;
        Ok(img)
    })
}

pub fn load_image_from_base64(base64_data: &str) -> Result<DynamicImage> {
    let image_data = general_purpose::STANDARD
        .decode(base64_data)
        .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?;
    let cursor = Cursor::new(image_data);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| anyhow!(format!("Failed to read image format: {}", e)))?
        .decode()
        .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?;
    Ok(img)
}

pub fn get_image(file: &str) -> Result<DynamicImage> {
    let mut img = None;
    if file.starts_with("http://") || file.starts_with("https://") {
        img = Some(load_image_from_url(file)?);
    }
    if file.starts_with("file://") {
        // let mut path = file.to_owned();
        // path = path.split_off(7);
        let path = url::Url::parse(file)?;
        let path = path.to_file_path();
        let path = match path {
            Ok(path) => path,
            Err(_) => {
                let mut path = file.to_owned();
                path = path.split_off(7);
                PathBuf::from(path)
            }
        };
        img = Some(
            ImageReader::open(path)
                .map_err(|e| anyhow!(format!("Failed to open file: {}", e)))?
                .decode()
                .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?,
        );
    }
    if file.starts_with("data:image") && file.contains("base64,") {
        let data: Vec<&str> = file.split("base64,").collect();
        let data = data[1];
        img = Some(load_image_from_base64(data)?);
    }
    if let Some(img) = img {
        return Ok(img);
    }
    Err(anyhow!("get image from message failed".to_string()))
}

pub fn extract_image_url(mes: &ChatCompletionParameters) -> Vec<&String> {
    let mut img_vec = Vec::new();
    for chat_mes in &mes.messages {
        if let ChatMessage::User { content, .. } = chat_mes
            && let ChatMessageContent::ContentPart(part_vec) = content
        {
            for part in part_vec {
                if let ChatMessageContentPart::Image(img_part) = part {
                    img_vec.push(&img_part.image_url.url);
                }
            }
        }
    }
    img_vec
}

pub fn extract_images(mes: &ChatCompletionParameters) -> Result<Vec<DynamicImage>> {
    let img_url_vec = extract_image_url(mes);
    // 并行下载图片
    img_url_vec
        .par_iter()
        .map(|url| get_image(url))
        .collect()
}

pub fn generate_target_ratios_sorted(min_num: u32, max_num: u32) -> Vec<(u32, u32)> {
    let mut target_ratios = HashSet::new();

    for n in min_num..=max_num {
        for i in 1..=n {
            for j in 1..=n {
                let product = i * j;
                if product <= max_num && product >= min_num {
                    target_ratios.insert((i, j));
                }
            }
        }
    }
    // Convert to vector and sort by the product of elements (i*j)
    let mut sorted_ratios: Vec<(u32, u32)> = target_ratios.into_iter().collect();
    sorted_ratios.sort_by_key(|&(i, j)| i * j);

    sorted_ratios
}

pub fn find_closest_aspect_ratio(
    aspect_ratio: f64,
    target_ratios: &[(u32, u32)],
    width: u32,
    height: u32,
    image_size: u32,
) -> (u32, u32) {
    let mut best_ratio_diff = f64::INFINITY;
    let mut best_ratio = (1, 1);
    let area = width * height;

    for &ratio in target_ratios {
        let target_aspect_ratio = ratio.0 as f64 / ratio.1 as f64;
        let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();

        if ratio_diff < best_ratio_diff {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff - best_ratio_diff).abs() < 1e-10 {
            // 当多个候选比例具有相同的宽高比差异时，根据图像的实际面积来选择最优比例。
            let target_area = 0.5 * (image_size as f64).powi(2) * (ratio.0 * ratio.1) as f64;
            if area as f64 > target_area {
                best_ratio = ratio;
            }
        }
    }

    best_ratio
}

pub fn dynamic_preprocess(
    image: &DynamicImage,
    image_size: u32,
    use_thumbnail: bool,
) -> Result<(Vec<DynamicImage>, (u32, u32))> {
    let orig_width = image.width();
    let orig_height = image.height();
    let aspect_ratio = orig_width as f64 / orig_height as f64;
    // 控制分块数量在2-9之间
    let target_ratios = generate_target_ratios_sorted(2, 9);
    let target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        &target_ratios,
        orig_width,
        orig_height,
        image_size,
    );
    let target_width = image_size * target_aspect_ratio.0;
    let target_height = image_size * target_aspect_ratio.1;
    let blocks = target_aspect_ratio.0 * target_aspect_ratio.1;
    let mut resized_img = image.resize_exact(
        target_width,
        target_height,
        image::imageops::FilterType::CatmullRom,
    );
    let mut processed_images = Vec::new();
    let grid_width = target_width / image_size;
    for i in 0..blocks {
        // Calculate box coordinates
        let x1 = (i % grid_width) * image_size;
        let y1 = (i / grid_width) * image_size;

        // Crop the image
        let split_img = resized_img.crop(x1, y1, image_size, image_size);
        processed_images.push(split_img);
    }
    assert_eq!(processed_images.len() as u32, blocks);

    if use_thumbnail && processed_images.len() != 1 {
        let thumbnail_img = image.resize_exact(
            image_size,
            image_size,
            image::imageops::FilterType::CatmullRom,
        );
        processed_images.push(thumbnail_img);
    }
    Ok((processed_images, target_aspect_ratio))
}

pub fn resize_with_edge_padding(
    img: &DynamicImage,
    width: u32,
    height: u32,
    color: [u8; 3],
) -> DynamicImage {
    // 按图像原比例resize,可能不是输入的宽高
    let mut img = img.resize(width, height, image::imageops::FilterType::CatmullRom);
    // 使用输入像素颜色填充为输入宽高
    if img.height() != height || img.width() != width {
        let (img_h, img_w) = (img.height(), img.width());
        let img_buffer = img.to_rgb8();
        let mut canvas: ImageBuffer<Rgb<u8>, Vec<u8>> =
            RgbImage::from_pixel(width, height, Rgb(color));
        let x_offset = (width - img_w) / 2;
        let y_offset = (height - img_h) / 2;
        imageops::overlay(&mut canvas, &img_buffer, x_offset as i64, y_offset as i64);
        img = DynamicImage::ImageRgb8(canvas);
    }
    img
}

pub fn img_transform(
    img: &DynamicImage,
    mean: &Tensor,
    std: &Tensor,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img_h = img.height();
    let img_w = img.width();
    let img_vec = img.to_rgb8().into_raw();
    // (h, w, c) => (c, h, w)
    let img_tensor = Tensor::from_slice(&img_vec, (img_h as usize, img_w as usize, 3), device)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?;
    // 0-255 rescale to 0-1
    let img_tensor = img_tensor.affine(1.0 / 255.0, 0.)?;
    // normalize
    let img_tensor = img_tensor
        .broadcast_sub(&mean.to_dtype(DType::F32)?)?
        .broadcast_div(&std.to_dtype(DType::F32)?)?
        .to_dtype(dtype)?;
    Ok(img_tensor)
}

pub fn img_smart_resize(
    img_h: u32,
    img_w: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32)> {
    if std::cmp::max(img_h, img_w) / std::cmp::min(img_h, img_w) > 200 {
        return Err(anyhow!(format!(
            "absolute aspect ratio mush be smaller than {}, got {}",
            200,
            std::cmp::max(img_h, img_w) / std::cmp::min(img_h, img_w)
        )));
    }
    let image_factor = factor;
    let mut h_bar = std::cmp::max(image_factor, round_by_factor(img_h, image_factor));
    let mut w_bar = std::cmp::max(image_factor, round_by_factor(img_w, image_factor));

    if h_bar * w_bar > max_pixels {
        let beta = ((img_h * img_w) as f32 / max_pixels as f32).sqrt();
        h_bar = std::cmp::max(
            image_factor,
            floor_by_factor(img_h as f32 / beta, image_factor),
        );
        w_bar = std::cmp::max(
            image_factor,
            floor_by_factor(img_w as f32 / beta, image_factor),
        );
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (img_h * img_w) as f32).sqrt();
        h_bar = ceil_by_factor(img_h as f32 * beta, image_factor);
        w_bar = ceil_by_factor(img_w as f32 * beta, image_factor);
    }
    Ok((h_bar, w_bar))
}

pub fn img_transform_with_resize(
    img: &DynamicImage,
    h: u32,
    w: u32,
    mean: &Tensor,
    std: &Tensor,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img_resize = img.resize_exact(w, h, imageops::FilterType::CatmullRom);
    let img_tensor = img_transform(&img_resize, mean, std, device, dtype)?;
    Ok(img_tensor)
}

pub fn float_tensor_to_dynamic_image(tensor: &Tensor) -> Result<DynamicImage> {
    let tensor = tensor.affine(255.0, 0.0)?.clamp(0.0, 255.0)?;
    let tensor_u8 = tensor.to_dtype(DType::U8)?.to_device(&Device::Cpu)?;
    let (c, h, w) = tensor_u8.dims3()?;
    match c {
        1 => {
            let tensor_u8 = tensor_u8.reshape((h, w))?;
            let data: Vec<u8> = tensor_u8.flatten_all()?.to_vec1()?;
            let img = ImageBuffer::from_raw(w as u32, h as u32, data)
                .ok_or_else(|| anyhow!("Failed to create image buffer"))?;
            Ok(DynamicImage::ImageLuma8(img))
        }
        3 => {
            let tensor_u8 = tensor_u8.permute((1, 2, 0))?;
            let data: Vec<u8> = tensor_u8.flatten_all()?.to_vec1()?;
            let img = ImageBuffer::from_raw(w as u32, h as u32, data)
                .ok_or_else(|| anyhow!("Failed to create image buffer"))?;
            Ok(DynamicImage::ImageRgb8(img))
        }
        _ => Err(anyhow!(format!("Unsupported number of channels: {}", c))),
    }
}
