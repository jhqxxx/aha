use std::time::Instant;

use anyhow::Result;
use image::{ImageReader, Rgba, RgbaImage};
use rayon::prelude::*;

/// 测试像素组合性能对比
#[test]
fn test_pixel_combine_performance() -> Result<()> {
    // cargo test test_pixel_combine_performance -r -- --nocapture
    let img_path = "./assets/img/gougou.jpg";
    let img = ImageReader::open(img_path)?.decode()?;
    // 缩小图片以加快测试
    let img = img.resize(1024, 1024, image::imageops::FilterType::Nearest);
    let width = img.width();
    let height = img.height();
    let rgb_img = img.to_rgb8();

    // 模拟 alpha 通道
    let alpha_gray = image::GrayImage::from_fn(width, height, |x, y| {
        image::Luma([((x + y) % 256) as u8])
    });

    let iterations = 10;

    println!("=== 像素组合性能测试 ===");
    println!("图片尺寸: {}x{}", width, height);
    println!();

    // 旧方法：逐像素操作
    let start = Instant::now();
    for _ in 0..iterations {
        let mut rgba_img = RgbaImage::new(width, height);
        for (x, y, pixel) in rgb_img.enumerate_pixels() {
            let alpha_value = alpha_gray.get_pixel(x, y).0[0];
            let rgba_pixel = Rgba([pixel.0[0], pixel.0[1], pixel.0[2], alpha_value]);
            rgba_img.put_pixel(x, y, rgba_pixel);
        }
        std::hint::black_box(&rgba_img);
    }
    let old_duration = start.elapsed();
    println!(
        "旧方法（逐像素）: {:?}, 平均: {:?}",
        old_duration,
        old_duration / iterations
    );

    // 新方法：串行索引赋值
    let start = Instant::now();
    for _ in 0..iterations {
        let rgb_raw = rgb_img.as_raw();
        let alpha_raw = alpha_gray.as_raw();
        let pixel_count = (width * height) as usize;
        let mut rgba_raw = vec![0u8; pixel_count * 4];

        for i in 0..pixel_count {
            let dst = i * 4;
            let src = i * 3;
            rgba_raw[dst] = rgb_raw[src];
            rgba_raw[dst + 1] = rgb_raw[src + 1];
            rgba_raw[dst + 2] = rgb_raw[src + 2];
            rgba_raw[dst + 3] = alpha_raw[i];
        }

        let rgba_img = RgbaImage::from_raw(width, height, rgba_raw).unwrap();
        std::hint::black_box(&rgba_img);
    }
    let serial_duration = start.elapsed();
    println!(
        "新方法（串行索引）: {:?}, 平均: {:?}",
        serial_duration,
        serial_duration / iterations
    );

    // 新方法：并行分块写入
    let start = Instant::now();
    for _ in 0..iterations {
        let rgb_raw = rgb_img.as_raw();
        let alpha_raw = alpha_gray.as_raw();
        let pixel_count = (width * height) as usize;
        let mut rgba_raw = vec![0u8; pixel_count * 4];

        rgba_raw
            .par_chunks_mut(4)
            .enumerate()
            .for_each(|(i, chunk)| {
                let src = i * 3;
                chunk[0] = rgb_raw[src];
                chunk[1] = rgb_raw[src + 1];
                chunk[2] = rgb_raw[src + 2];
                chunk[3] = alpha_raw[i];
            });

        let rgba_img = RgbaImage::from_raw(width, height, rgba_raw).unwrap();
        std::hint::black_box(&rgba_img);
    }
    let parallel_duration = start.elapsed();
    println!(
        "新方法（并行）: {:?}, 平均: {:?}",
        parallel_duration,
        parallel_duration / iterations
    );

    let speedup_serial = old_duration.as_secs_f64() / serial_duration.as_secs_f64();
    let speedup_parallel = old_duration.as_secs_f64() / parallel_duration.as_secs_f64();
    println!();
    println!("串行索引 vs 逐像素: {:.2}x", speedup_serial);
    println!("并行 vs 逐像素: {:.2}x", speedup_parallel);

    Ok(())
}

/// 测试图片 resize 性能对比（串行 vs 并行）
#[test]
fn test_image_resize_parallel_vs_serial() -> Result<()> {
    // cargo test test_image_resize_parallel_vs_serial -r -- --nocapture
    let img_path = "./assets/img/gougou.jpg";
    let img = ImageReader::open(img_path)?.decode()?;
    // 缩小原图以加快测试
    let img = img.resize(2048, 2048, image::imageops::FilterType::Nearest);

    let num_images = 4;
    let imgs: Vec<_> = (0..num_images).map(|_| img.clone()).collect();
    let target_h = 1024u32;
    let target_w = 1024u32;

    let iterations = 10;

    println!("=== 图片 Resize 性能测试 ===");
    println!("图片数量: {}", num_images);
    println!("原始尺寸: {}x{}", img.width(), img.height());
    println!("目标尺寸: {}x{}", target_w, target_h);
    println!();

    // 串行 resize
    let start = Instant::now();
    for _ in 0..iterations {
        let mut results = Vec::with_capacity(num_images);
        for img in &imgs {
            let resized =
                img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);
            results.push(resized);
        }
        std::hint::black_box(&results);
    }
    let serial_duration = start.elapsed();
    println!(
        "串行 resize: {:?}, 平均: {:?}",
        serial_duration,
        serial_duration / iterations
    );

    // 并行 resize
    let start = Instant::now();
    for _ in 0..iterations {
        let results: Vec<_> = imgs
            .par_iter()
            .map(|img| {
                img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom)
            })
            .collect();
        std::hint::black_box(&results);
    }
    let parallel_duration = start.elapsed();
    println!(
        "并行 resize: {:?}, 平均: {:?}",
        parallel_duration,
        parallel_duration / iterations
    );

    let speedup = serial_duration.as_secs_f64() / parallel_duration.as_secs_f64();
    println!();
    println!("并行 vs 串行: {:.2}x", speedup);

    Ok(())
}

/// 测试后处理阶段并行 vs 串行性能（纯图像操作）
#[test]
fn test_postprocess_parallel_vs_serial() -> Result<()> {
    // cargo test test_postprocess_parallel_vs_serial -r -- --nocapture
    let img_path = "./assets/img/gougou.jpg";
    let img = ImageReader::open(img_path)?.decode()?;
    // 缩小图片以加快测试
    let img = img.resize(1024, 1024, image::imageops::FilterType::Nearest);
    let width = img.width();
    let height = img.height();

    // 模拟多张图片的后处理数据
    let num_images = 4;
    let rgb_imgs: Vec<_> = (0..num_images).map(|_| img.to_rgb8()).collect();
    let alpha_grays: Vec<_> = (0..num_images)
        .map(|_| {
            image::GrayImage::from_fn(width, height, |x, y| image::Luma([((x + y) % 256) as u8]))
        })
        .collect();

    let iterations = 10;

    println!("=== 后处理阶段性能测试（纯图像操作）===");
    println!("图片数量: {}", num_images);
    println!("图片尺寸: {}x{}", width, height);
    println!();

    // 串行后处理（for-in 循环）
    let start = Instant::now();
    for _ in 0..iterations {
        let mut results = Vec::with_capacity(num_images);
        for i in 0..num_images {
            let rgb_raw = rgb_imgs[i].as_raw();
            let alpha_raw = alpha_grays[i].as_raw();
            let pixel_count = (width * height) as usize;
            let mut rgba_raw = vec![0u8; pixel_count * 4];

            for j in 0..pixel_count {
                let dst = j * 4;
                let src = j * 3;
                rgba_raw[dst] = rgb_raw[src];
                rgba_raw[dst + 1] = rgb_raw[src + 1];
                rgba_raw[dst + 2] = rgb_raw[src + 2];
                rgba_raw[dst + 3] = alpha_raw[j];
            }

            let rgba_img = RgbaImage::from_raw(width, height, rgba_raw).unwrap();
            results.push(rgba_img);
        }
        std::hint::black_box(&results);
    }
    let serial_duration = start.elapsed();
    println!(
        "串行后处理: {:?}, 平均: {:?}",
        serial_duration,
        serial_duration / iterations
    );

    // 并行后处理（外层并行 + 内层并行）
    let start = Instant::now();
    for _ in 0..iterations {
        let results: Vec<_> = (0..num_images)
            .into_par_iter()
            .map(|i| {
                let rgb_raw = rgb_imgs[i].as_raw();
                let alpha_raw = alpha_grays[i].as_raw();
                let pixel_count = (width * height) as usize;
                let mut rgba_raw = vec![0u8; pixel_count * 4];

                rgba_raw
                    .par_chunks_mut(4)
                    .enumerate()
                    .for_each(|(j, chunk)| {
                        let src = j * 3;
                        chunk[0] = rgb_raw[src];
                        chunk[1] = rgb_raw[src + 1];
                        chunk[2] = rgb_raw[src + 2];
                        chunk[3] = alpha_raw[j];
                    });

                RgbaImage::from_raw(width, height, rgba_raw).unwrap()
            })
            .collect();
        std::hint::black_box(&results);
    }
    let parallel_duration = start.elapsed();
    println!(
        "并行后处理: {:?}, 平均: {:?}",
        parallel_duration,
        parallel_duration / iterations
    );

    let speedup = serial_duration.as_secs_f64() / parallel_duration.as_secs_f64();
    println!();
    println!("后处理性能提升: {:.2}x", speedup);

    Ok(())
}

/// 测试完整图像处理流程（resize + 像素合并）串行 vs 并行
#[test]
fn test_full_image_pipeline_parallel_vs_serial() -> Result<()> {
    // cargo test test_full_image_pipeline_parallel_vs_serial -r -- --nocapture
    let img_path = "./assets/img/gougou.jpg";
    let img = ImageReader::open(img_path)?.decode()?;
    // 缩小图片以加快测试
    let img = img.resize(2048, 2048, image::imageops::FilterType::Nearest);
    let orig_width = img.width();
    let orig_height = img.height();

    let num_images = 4;
    let imgs: Vec<_> = (0..num_images).map(|_| img.clone()).collect();
    let target_h = 1024u32;
    let target_w = 1024u32;

    // 模拟 alpha 蒙版
    let alpha_grays: Vec<_> = (0..num_images)
        .map(|_| {
            image::GrayImage::from_fn(orig_width, orig_height, |x, y| {
                image::Luma([((x + y) % 256) as u8])
            })
        })
        .collect();

    let iterations = 10;

    println!("=== 完整图像处理流程性能测试 ===");
    println!("图片数量: {}", num_images);
    println!("原始尺寸: {}x{}", orig_width, orig_height);
    println!("处理尺寸: {}x{}", target_w, target_h);
    println!();

    // 串行处理流程
    let start = Instant::now();
    for _ in 0..iterations {
        let mut results = Vec::with_capacity(num_images);
        for i in 0..num_images {
            // 预处理：resize 到模型输入尺寸
            let _resized =
                imgs[i].resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom);

            // 后处理：合并 RGB 和 alpha
            let rgb_img = imgs[i].to_rgb8();
            let rgb_raw = rgb_img.as_raw();
            let alpha_raw = alpha_grays[i].as_raw();
            let pixel_count = (orig_width * orig_height) as usize;
            let mut rgba_raw = vec![0u8; pixel_count * 4];

            for j in 0..pixel_count {
                let dst = j * 4;
                let src = j * 3;
                rgba_raw[dst] = rgb_raw[src];
                rgba_raw[dst + 1] = rgb_raw[src + 1];
                rgba_raw[dst + 2] = rgb_raw[src + 2];
                rgba_raw[dst + 3] = alpha_raw[j];
            }

            let rgba_img = RgbaImage::from_raw(orig_width, orig_height, rgba_raw).unwrap();
            results.push(rgba_img);
        }
        std::hint::black_box(&results);
    }
    let serial_duration = start.elapsed();
    println!(
        "串行流程: {:?}, 平均: {:?}",
        serial_duration,
        serial_duration / iterations
    );

    // 并行处理流程
    let start = Instant::now();
    for _ in 0..iterations {
        // 并行预处理
        let _resized: Vec<_> = imgs
            .par_iter()
            .map(|img| {
                img.resize_exact(target_w, target_h, image::imageops::FilterType::CatmullRom)
            })
            .collect();

        // 并行后处理
        let results: Vec<_> = (0..num_images)
            .into_par_iter()
            .map(|i| {
                let rgb_img = imgs[i].to_rgb8();
                let rgb_raw = rgb_img.as_raw();
                let alpha_raw = alpha_grays[i].as_raw();
                let pixel_count = (orig_width * orig_height) as usize;
                let mut rgba_raw = vec![0u8; pixel_count * 4];

                rgba_raw
                    .par_chunks_mut(4)
                    .enumerate()
                    .for_each(|(j, chunk)| {
                        let src = j * 3;
                        chunk[0] = rgb_raw[src];
                        chunk[1] = rgb_raw[src + 1];
                        chunk[2] = rgb_raw[src + 2];
                        chunk[3] = alpha_raw[j];
                    });

                RgbaImage::from_raw(orig_width, orig_height, rgba_raw).unwrap()
            })
            .collect();
        std::hint::black_box(&results);
    }
    let parallel_duration = start.elapsed();
    println!(
        "并行流程: {:?}, 平均: {:?}",
        parallel_duration,
        parallel_duration / iterations
    );

    let speedup = serial_duration.as_secs_f64() / parallel_duration.as_secs_f64();
    println!();
    println!("完整流程性能提升: {:.2}x", speedup);

    Ok(())
}
