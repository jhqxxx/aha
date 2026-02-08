// use std::io::Cursor;

use std::time::Instant;

use aha::utils::tensor_utils::interpolate_nearest_1d;
use anyhow::Result;
use candle_core::Tensor;
// use symphonia::core::io::MediaSourceStream;

#[test]
fn messy_test() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda,ffmpeg messy_test -r -- --nocapture
    let device = &candle_core::Device::Cpu;
    let t = Tensor::arange(0.0f32, 40.0, device)?.broadcast_as((1, 40, 40))?;
    println!("t: {}", t);
    let i_start = Instant::now();
    let t_inter = interpolate_nearest_1d(&t, 20)?;
    let i_duration = i_start.elapsed();
    println!(
        "Time elapsed in interpolate_nearest_1d is: {:?}",
        i_duration
    );
    println!("t_inter: {}", t_inter);
    // let url = "https://sis-sample-audio.obs.cn-north-1.myhuaweicloud.com/16k16bit.mp3";
    // let client = reqwest::blocking::Client::new();
    // let response = client.get(url).send()?;
    // let vec_u8 = response.bytes()?.to_vec();
    // let mut content = Cursor::new(vec_u8);
    // let mss = MediaSourceStream::new(Box::new(content), Default::default());
    // let window = create_hann_window(400, DType::F32, device)?;
    // println!("window: {}", window);
    // let audio_path = "file:///home/jhq/Videos/voice_01.wav";
    // let audio_path = "/home/jhq/Videos/zh.mp3";
    // let audio_path = "/home/jhq/Videos/zh.mp3";
    // // let audio_tensor = load_and_resample_audio_rubato(audio_path, 16000, device)?;
    // // let audio_tensor = load_audio_with_resample(audio_path, device, Some(16000))?;
    // // println!("audio_tensor: {}", audio_tensor);
    // #[cfg(feature = "ffmpeg")]
    // {
    //     use aha::utils::audio_utils::load_and_resample_audio_ffmpeg;

    //     let audio_tensor = load_and_resample_audio_ffmpeg(audio_path, Some(16000), device)?;
    //     println!("audio_tensor: {}", audio_tensor);
    // }

    // // let path = get_default_save_dir();
    // // let x = Tensor::new(array, device)
    // let x = Tensor::arange(0.0, 9.0, device)?;
    // println!("x: {}", x);
    // let x = x
    //     .unsqueeze(0)?
    //     .unsqueeze(0)?
    //     .broadcast_as((5, 5, 9))?
    //     .reshape((5, 5, 3, 3))?;
    // println!("x: {}", x);
    // let x = x.permute((0, 2, 1, 3))?;
    // println!("x: {}", x);
    // let x = x.reshape((15, 15))?;
    // println!("x: {}", x);
    // let xs = Tensor::rand(0.0, 5.0, (1, 1, 3, 3), device)?;
    // println!("xs: {}", xs);
    // let xs = xs.pad_with_zeros(3, 2, 2)?
    //             .pad_with_zeros(2, 2, 2)?;
    // println!("xs: {}", xs);
    // let xs = Tensor::arange(0.0, 25.0, device)?;
    // println!("xs: {}", xs);
    // let splits = split_tensor_with_size(&xs, 5, 0)?;
    // for v in splits {
    //     println!("v: {}", v);
    // }
    // let xs = Tensor::arange(0.0, 25.0, device)?.broadcast_as((1, 1, 5, 5))?;
    // println!("xs: {}", xs);
    // let xs = xs.avg_pool2d(5)?;
    // println!("xs: {}", xs);
    // let xs = Tensor::rand(0.0, 1.0, (1, 4, 4, 2), device)?;
    // println!("xs: {}", xs);
    // let shape = Shape::from_dims(&[1, 2, 2, 2, 2, 2]);
    // let xs = xs.reshape(shape)?;
    // println!("xs: {}", xs);
    // let x0 = xs.i((.., .., 0, .., 0, ..))?;
    // let x1 = xs.i((.., .., 1, .., 0, ..))?;
    // let x2 = xs.i((.., .., 0, .., 1, ..))?;
    // let x3 = xs.i((.., .., 1, .., 1, ..))?;
    // let xs = Tensor::cat(&[x0, x1, x2, x3], D::Minus1)?;
    // println!("xs: {}", xs);
    // let xs = xs.reshape((1, (), 4 * 2))?;
    // println!("xs: {}", xs);
    // let path_str = "file://./assets/img/ocr_test1.png";
    // let path = url::Url::from_str(path_str)?;
    // let path = path.to_file_path();
    // let path = match path {
    //     Ok(path) => path,
    //     Err(_) => {
    //         let mut path = path_str.to_owned();
    //         path = path.split_off(7);
    //         PathBuf::from(path)
    //     }
    // };
    // println!("to file path: {:?}", path);

    // let device = &candle_core::Device::Cpu;
    // let t = Tensor::arange(0.0f32, 40.0, device)?.broadcast_as((1, 1, 40, 40))?;
    // println!("t: {}", t);
    // let i_start = Instant::now();
    // let t_inter = interpolate_bilinear(&t, (20, 20), Some(false))?;
    // let i_duration = i_start.elapsed();
    // println!("Time elapsed in interpolate_bilinear is: {:?}", i_duration);
    // println!("t_inter: {}", t_inter);
    // let x: Vec<u32> = (0..5).flat_map(|_| 0u32..10).collect();
    // let id: Vec<u32> = (0..5).flat_map(|h| vec![h; 10]).collect();
    // println!("x: {:?}", id);
    // let t = Tensor::randn(0.0f32, 1.0, (1, 768, 64, 64), device)?;
    // let t = Tensor::arange(0u32, 10, device)?.broadcast_as((1, 10))?;
    // let eq = t.broadcast_eq(&Tensor::new(5u32, device)?)?;
    // println!("eq: {}", eq);
    // let t = Tensor::arange(0.0f32, 10.0, device)?.broadcast_as((1, 1, 10, 10))?;
    // println!("t: {}", t);
    // let t_resized = interpolate_bicubic(&t, (5, 5), Some(true), Some(false))?;
    // println!("t_resized: {}", t_resized);
    // let t1 = Tensor::rand(0.0, 1.0, (1, 5, 5, 10), device)?;
    // let t2 = Tensor::rand(0.0, 1.0, (5, 8, 10), device)?;
    // let t2 = t2.t()?;
    // println!("t2: {:?}", t2);
    // let re = t1.broadcast_matmul(&t2)?;
    // println!("re: {:?}", re);
    // let index = Tensor::arange(0u32, 10u32, device)?;
    // let index_2d_vec = vec![index;5];
    // let index_2d = Tensor::stack(&index_2d_vec, 0)?;
    // println!("index_2d: {}", index_2d);
    // let t = Tensor::rand(0.0, 1.0, (20, 8), device)?;
    // println!("t: {}", t);
    // let res = index_select_2d(&t, &index_2d)?;
    // println!("res: {}", res);
    // let t = Tensor::arange(0.0, 10.0, device)?
    //     .unsqueeze(0)?
    //     .unsqueeze(0)?;
    // println!("t: {}", t);
    // let t_resized = interpolate_linear(&t, 20, None)?;
    // println!("t_resized: {}", t_resized);

    // let grid_thw = Tensor::new(vec![vec![3u32, 12, 20], vec![5, 30, 25]], device)?;
    // let cu_seqlens = grid_thw.i((.., 1))?.mul(&grid_thw.i((.., 2))?)?;
    // let grid_t = grid_thw.i((.., 0))?.to_vec1::<u32>()?;
    // println!("cu_seqlens: {}", cu_seqlens);
    // println!("cu_seqlens rank: {}", cu_seqlens.rank());
    // println!("grid_t: {:?}", grid_t);
    // let image_mask = Tensor::new(vec![0u32, 0, 0, 1, 0, 1], device)?;
    // let video_mask = Tensor::new(vec![0u32, 1, 0, 1, 0, 1], device)?;
    // let visual_mask = bitor_tensor(&image_mask, &video_mask)?;
    // println!("visual_mask: {}", visual_mask);
    // let x = Tensor::arange_step(0.0_f32, 5., 0.5, &device)?;
    // let x_int = x.to_dtype(candle_core::DType::U32)?;
    // println!("x: {}", x);
    // println!("x_int: {}", x_int);
    // let x_affine = x_int.affine(1.0, 1.0)?;
    // println!("x_affine: {}", x_affine);
    // let x_clamp = x_affine.clamp(0u32, 3u32)?;
    // println!("x_clamp: {}", x_clamp);
    // let wav_path = "./assets/audio/voice_01.wav";
    // let audio_tensor = load_audio_with_resample(wav_path, device, Some(16000))?;
    // println!("audio_tensor: {}", audio_tensor);
    // let string = "你好啊".to_string();
    // let vec_str: Vec<String>= string.chars().map(|c| c.to_string()).collect();
    // println!("vec_str: {:?}", vec_str);
    // let t = Tensor::rand(-1.0, 1.0, (2, 2), &device)?;
    // println!("t: {}", t);
    // let re_t = t.recip()?;
    // println!("re_t: {}", re_t);
    Ok(())
}
