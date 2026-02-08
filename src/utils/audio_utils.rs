use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread;
use std::{f64::consts::PI, io::Cursor};

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionParameters, ChatCompletionResponse, ChatMessage, ChatMessageContent,
    ChatMessageContentPart,
};
use anyhow::{Result, anyhow};
// use audioadapter_buffers::direct::InterleavedSlice;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Module};
#[cfg(feature = "ffmpeg")]
use ffmpeg_next as ffmpeg;
use hound::{SampleFormat, WavReader};
use num::integer::gcd;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use realfft::RealFftPlanner;
// use rubato::{
//     Async, FixedAsync, Indexing, Resampler, SincInterpolationParameters, SincInterpolationType,
//     WindowFunction,
// };
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::utils::get_default_save_dir;
use crate::utils::tensor_utils::{
    linspace, log10, pad_reflect_last_dim, pad_replicate_last_dim, split_tensor,
};

// 重采样方法枚举
#[derive(Debug, Clone, Copy)]
pub enum ResamplingMethod {
    SincInterpHann,
    SincInterpKaiser,
}

// 零阶修正贝塞尔函数 I0
fn i0(x: f32) -> f32 {
    let mut result = 1.0;
    let mut term = 1.0;
    let half_x_sq = x * x / 4.0;

    for k in 1..50 {
        term = term * half_x_sq / (k * k) as f32;
        result += term;

        if term < 1e-12 {
            break;
        }
    }

    result
}

// 获取sinc重采样核
pub fn get_sinc_resample_kernel(
    orig_freq: i64,
    new_freq: i64,
    gcd_val: i64,
    lowpass_filter_width: i64,
    rolloff: f64,
    resampling_method: ResamplingMethod,
    beta: Option<f32>,
    device: &Device,
) -> Result<(Tensor, i64)> {
    if orig_freq <= 0 || new_freq <= 0 {
        return Err(anyhow!("Frequencies must be positive".to_string()));
    }

    if lowpass_filter_width <= 0 {
        return Err(anyhow!(
            "Low pass filter width should be positive".to_string()
        ));
    }
    // 24k
    // 16k
    // 8k
    // 3 / 2
    // conv: 6
    // out_channel: 2
    // in_channel: 1
    // ks= 23 / stride = 3
    // [2, 1, 23]

    let orig_freq = orig_freq / gcd_val;
    let new_freq = new_freq / gcd_val;

    let base_freq = (orig_freq.min(new_freq) as f64) * rolloff;

    let width_f = (lowpass_filter_width as f64) * (orig_freq as f64) / base_freq;
    let width = width_f.ceil() as i64;
    // 创建索引数组 [1, 1, 2*width + orig_freq]
    let idx = Tensor::arange(-width as f32, (width + orig_freq) as f32, device)?
        .affine(1.0 / orig_freq as f64, 0.0)?
        .unsqueeze(0)?
        .unsqueeze(0)?;
    // 创建时间数组 t [new_freq, 1, idx_len]
    let t = Tensor::arange_step(0.0, -new_freq as f32, -1.0, device)?
        .affine(1.0 / new_freq as f64, 0.0)?
        .unsqueeze(D::Minus1)?
        .unsqueeze(D::Minus1)?
        .broadcast_add(&idx)?
        .affine(base_freq, 0.0)?;
    let t = t.clamp(-lowpass_filter_width as f32, lowpass_filter_width as f32)?;
    // 计算窗口函数
    let window = match resampling_method {
        ResamplingMethod::SincInterpHann => {
            let window_arg = t.affine(PI / (lowpass_filter_width as f64) / 2.0, 0.0)?;
            window_arg.cos()?.sqr()?
        }
        ResamplingMethod::SincInterpKaiser => {
            let beta_val = beta.unwrap_or(14.769_656_f32);
            let i0_beta = i0(beta_val);

            let normalized_t = t.affine(1.0 / lowpass_filter_width as f64, 0.0)?;
            let arg = (1.0 - normalized_t.sqr()?)?;
            // 处理arg为负数的情况
            let sqrt_arg = arg.relu()?.sqrt()?;
            let sqrt_dims = sqrt_arg.dims();
            let sqrt_arg_vec = sqrt_arg.flatten_all()?.to_vec1::<f32>()?;

            let window_val: Vec<f32> = sqrt_arg_vec
                .iter()
                .map(|x| i0(beta_val * x) / i0_beta)
                .collect();
            Tensor::new(window_val, device)?.reshape(sqrt_dims)?
        }
    };

    // 计算sinc核
    let scale = base_freq / (orig_freq as f64);
    let t_scaled = t.affine(PI, 0.0)?;

    let t_zeros = Tensor::zeros_like(&t_scaled)?;
    let t_ones = Tensor::ones_like(&t_scaled)?;
    let mask = t_scaled.eq(&t_zeros)?;
    let sinc = mask.where_cond(&t_ones, &t_scaled.sin()?.div(&t_scaled)?)?;
    let kernels = sinc.mul(&window)?.affine(scale, 0.0)?;

    Ok((kernels, width))
}

// 应用sinc重采样核
pub fn apply_sinc_resample_kernel(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    gcd_val: i64,
    kernel: &Tensor,
    width: i64,
) -> Result<Tensor> {
    let orig_freq = orig_freq / gcd_val;
    let new_freq = new_freq / gcd_val;

    // 获取波形形状
    let dims = waveform.dims();
    let waveform_flat = waveform.reshape(((), dims[dims.len() - 1]))?;

    let (num_wavs, length) = waveform_flat.dims2()?;
    let padded_waveform =
        waveform.pad_with_zeros(D::Minus1, width as usize, (width + orig_freq) as usize)?;

    // 添加通道维度 [batch_size, 1, padded_length]
    let waveform_3d = padded_waveform.unsqueeze(1)?;
    let config = Conv1dConfig {
        padding: 0,
        stride: orig_freq as usize,
        dilation: 1,
        groups: 1,
        cudnn_fwd_algo: None,
    };
    // (len -k +padding ) / stride +1
    // out: 2
    let conv1d = Conv1d::new(kernel.clone(), None, config);
    // 执行卷积
    // kernel形状: [new_freq_reduced, 1, kernel_len]
    // 输出形状: [batch_size, new_freq_reduced, output_length]
    let conv_output = conv1d.forward(&waveform_3d)?;

    // 转置并重塑 [batch_size, output_length * new_freq_reduced]
    let conv_transposed = conv_output.transpose(1, 2)?.reshape((num_wavs, ()))?;

    // 计算目标长度
    let target_length = ((new_freq as f64 * length as f64) / orig_freq as f64).ceil() as usize;

    // 截取目标长度
    let resampled_flat =
        conv_transposed.narrow(1, 0, target_length.min(conv_transposed.dim(1)?))?;
    let mut new_dims = dims.to_vec();
    let last_dim = new_dims.len() - 1;
    new_dims[last_dim] = resampled_flat.dim(1)?;
    // 恢复原始批次形状

    let resampled = resampled_flat.reshape(new_dims)?;

    Ok(resampled)
}

// 主要的重采样函数
pub fn resample(
    waveform: &Tensor,
    orig_freq: i64,
    new_freq: i64,
    lowpass_filter_width: i64,
    rolloff: f64,
    resampling_method: ResamplingMethod,
    beta: Option<f32>,
) -> Result<Tensor> {
    if orig_freq <= 0 || new_freq <= 0 {
        return Err(anyhow!("Frequencies must be positive".to_string(),));
    }

    if orig_freq == new_freq {
        return Ok(waveform.clone());
    }

    let gcd_val = gcd(orig_freq, new_freq);
    let device = waveform.device();

    let (kernel, width) = get_sinc_resample_kernel(
        orig_freq,
        new_freq,
        gcd_val,
        lowpass_filter_width,
        rolloff,
        resampling_method,
        beta,
        device,
    )?;
    let t = apply_sinc_resample_kernel(waveform, orig_freq, new_freq, gcd_val, &kernel, width)?;
    Ok(t)
}

// 为方便使用提供的简化版本
pub fn resample_simple(waveform: &Tensor, orig_freq: i64, new_freq: i64) -> Result<Tensor> {
    resample(
        waveform,
        orig_freq,
        new_freq,
        6,
        0.99,
        ResamplingMethod::SincInterpHann,
        None,
    )
}

pub fn load_audio_from_url(url: &str) -> Result<PathBuf> {
    tokio::task::block_in_place(|| {
        let client = reqwest::blocking::Client::new();
        let response = client.get(url).send()?;
        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "Failed to download file: {}",
                response.status()
            ));
        }
        let temp_dir = get_default_save_dir().expect("Failed to get home directory");
        let temp_dir = PathBuf::from(temp_dir);
        let temp_path = if url.contains("wav") {
            temp_dir.join("temp_audio.wav")
        } else if url.contains("mp3") {
            temp_dir.join("temp_audio.mp3")
        } else {
            return Err(anyhow::anyhow!("load audio only surpport wav/mp3 format"));
        };

        let mut file = std::fs::File::create(&temp_path)?;
        let mut content = Cursor::new(response.bytes()?);
        std::io::copy(&mut content, &mut file)?;

        // Return the temp directory to keep it alive until the function ends
        Ok(temp_path)
    })
}

// pub fn _load_audio_bytes_from_url(url: &str) -> Result<Vec<u8>> {
//     tokio::task::block_in_place(|| {
//         let client = reqwest::blocking::Client::new();
//         let response = client.get(url).send()?;
//         if !response.status().is_success() {
//             return Err(anyhow::anyhow!(
//                 "Failed to download file: {}",
//                 response.status()
//             ));
//         }
//         let bytes = response.bytes()?.to_vec();
//         Ok(bytes)
//     })
// }

pub fn load_audio_bytes_from_url(url: &str) -> Result<Vec<u8>> {
    let url = url.to_string();
    thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let response = reqwest::get(&url).await?;
            if !response.status().is_success() {
                return Err(anyhow::anyhow!(
                    "Failed to download file: {}",
                    response.status()
                ));
            }
            let bytes = response.bytes().await?.to_vec();
            Ok(bytes)
        })
    })
    .join()
    .unwrap()
}

pub fn get_audio_path(path_str: &str) -> Result<PathBuf> {
    if path_str.starts_with("http://") || path_str.starts_with("https://") {
        // Download file from network
        load_audio_from_url(path_str)
    } else if path_str.starts_with("file://") {
        // Convert file:// URL to local path
        let path = url::Url::parse(path_str)?;
        let path = path.to_file_path();
        let path = match path {
            Ok(path) => path,
            Err(_) => {
                let mut path = path_str.to_owned();
                path = path.split_off(7);
                PathBuf::from(path)
            }
        };
        Ok(path)
    } else if path_str.starts_with("data:audio") && path_str.contains("base64,") {
        let data: Vec<&str> = path_str.split("base64,").collect();
        let file_mes = data[0];
        let data = data[1];
        let temp_dir = get_default_save_dir().expect("Failed to get home directory");
        let temp_dir = PathBuf::from(temp_dir);
        let temp_path = if file_mes.contains("wav") {
            temp_dir.join("temp_audio.wav")
        } else if file_mes.contains("mpeg") {
            temp_dir.join("temp_audio.mp3")
        } else {
            return Err(anyhow::anyhow!(
                "base64 audio only surpport wav/mpeg(mp3) format"
            ));
        };
        save_audio_from_base64(data, &temp_path)?;
        Ok(temp_path)
    } else {
        Err(anyhow::anyhow!("get audio path error {}", path_str))
    }
}

pub fn get_audio_bytes_vec(path_str: &str) -> Result<Vec<u8>> {
    if path_str.starts_with("http://") || path_str.starts_with("https://") {
        // Download file from network
        load_audio_bytes_from_url(path_str)
    } else if path_str.starts_with("file://") {
        // Convert file:// URL to local path
        let path = url::Url::parse(path_str)?;
        let path = path.to_file_path();
        let path = match path {
            Ok(path) => path,
            Err(_) => {
                let mut path = path_str.to_owned();
                path = path.split_off(7);
                PathBuf::from(path)
            }
        };
        let bytes = std::fs::read(path)?;
        Ok(bytes)
    } else if path_str.starts_with("data:audio") && path_str.contains("base64,") {
        let data: Vec<&str> = path_str.split("base64,").collect();
        let data = data[1];
        let data = BASE64_STANDARD.decode(data)?;
        Ok(data)
    } else {
        Err(anyhow::anyhow!("get audio path error {}", path_str))
    }
}

pub fn load_audio_use_hound(audio_path: PathBuf, device: &Device) -> Result<(Tensor, usize)> {
    let mut reader = WavReader::open(audio_path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        SampleFormat::Int => {
            // 将整数样本转换为浮点数 [-1.0, 1.0]
            // println!("spec.bits_per_sample: {}", spec.bits_per_sample);
            match spec.bits_per_sample {
                8 => reader
                    .samples::<i8>()
                    .map(|s| s.map(|sample| sample as f32 / i8::MAX as f32))
                    .collect::<Result<Vec<_>, _>>()?,
                16 => reader
                    .samples::<i16>()
                    .map(|s| s.map(|sample| sample as f32 / i16::MAX as f32))
                    .collect::<Result<Vec<_>, _>>()?,
                24 => reader
                    .samples::<i32>()
                    .map(|s| s.map(|sample| sample as f32 / 8388607.0))
                    .collect::<Result<Vec<_>, _>>()?,
                _ => {
                    return Err(anyhow::anyhow!(
                        "Unsupported bit depth: {}",
                        spec.bits_per_sample
                    ));
                }
            }
        }
        SampleFormat::Float => {
            // 直接读取浮点数样本
            reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?
        }
    };
    let sample_rate = spec.sample_rate;
    let mut audio_tensor = Tensor::from_slice(
        &samples,
        (
            samples.len() / spec.channels as usize,
            spec.channels as usize,
        ),
        device,
    )?
    .t()?;
    // println!("audio channels: {}", spec.channels);
    if spec.channels > 1 {
        // 对channel通道求平均， channel维度变为1
        audio_tensor = audio_tensor.mean_keepdim(0)?;
    }
    Ok((audio_tensor, sample_rate as usize))
}

pub fn get_audio_format_from_bytes(bytes: &[u8]) -> Result<String> {
    if bytes.len() < 12 {
        return Err(anyhow::anyhow!("bytes too short: {}", bytes.len()));
    }
    // Check for different audio formats based on their magic bytes
    if bytes.starts_with(&[0x52, 0x49, 0x46, 0x46]) && bytes.len() >= 12 {
        // RIFF header - typically WAV files
        if bytes.len() >= 8 && bytes[8..12] == [0x57, 0x41, 0x56, 0x45] {
            Ok("wav".to_string())
        } else {
            Ok("riff".to_string())
        }
    } else if bytes.starts_with(&[0xFF, 0xFB])
        || bytes.starts_with(&[0xFF, 0xF3])
        || bytes.starts_with(&[0xFF, 0xF2])
    {
        // MP3 header with different bitrates and options
        Ok("mp3".to_string())
    } else if bytes.len() >= 3 && bytes[0..3] == [0x49, 0x44, 0x33] {
        // ID3 tag - typically MP3 files
        Ok("mp3".to_string())
    } else if bytes.len() >= 4 && bytes[0..4] == [0x46, 0x4F, 0x52, 0x4D] {
        // FORM header - AIFF files
        Ok("aiff".to_string())
    } else if bytes.len() >= 8 && bytes[0..4] == [0x4F, 0x67, 0x67, 0x53] {
        // OggS header - OGG files
        Ok("ogg".to_string())
    } else if bytes.len() >= 4 && bytes[0..4] == [0x66, 0x4C, 0x61, 0x43] {
        // fLaC header - FLAC files
        Ok("flac".to_string())
    } else if bytes.len() >= 8 && bytes[4..8] == [0x6D, 0x70, 0x34, 0x20] {
        // M4A header
        Ok("m4a".to_string())
    } else if bytes.len() >= 8 && bytes[4..8] == [0x6D, 0x70, 0x34, 0x61] {
        // MP4A header
        Ok("mp4".to_string())
    } else {
        Err(anyhow::anyhow!("Unknown format "))
    }
}

pub fn load_audio_use_symphonia(audio_vec: Vec<u8>, device: &Device) -> Result<(Tensor, usize)> {
    let extension = get_audio_format_from_bytes(&audio_vec)?;
    let content = Cursor::new(audio_vec);
    let mss = MediaSourceStream::new(Box::new(content), Default::default());

    let mut hint = Hint::new();

    hint.with_extension(&extension);

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;

    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or("No default track found")
        .map_err(|e| anyhow!("symphonia read err: {}", e))?;
    let mut channels = 1;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    // 创建解码器
    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    // 用于存储所有音频样本的缓冲区
    let mut all_samples: Vec<Vec<f32>> = Vec::new();

    // 循环读取数据包并解码
    while let Ok(packet) = format.next_packet() {
        match decoder.decode(&packet) {
            Ok(decoded) => {
                match decoded {
                    AudioBufferRef::F32(buf) => {
                        channels = buf.spec().channels.count();
                        // 对于浮点格式
                        for channel in 0..channels {
                            if all_samples.len() <= channel {
                                all_samples.push(Vec::new());
                            }
                            let channel_data = buf.chan(channel);
                            all_samples[channel].extend_from_slice(channel_data);
                        }
                    }
                    AudioBufferRef::S16(buf) => {
                        channels = buf.spec().channels.count();
                        // 对于16位整数格式，转换为f32
                        for channel in 0..channels {
                            if all_samples.len() <= channel {
                                all_samples.push(Vec::new());
                            }
                            let channel_data = buf.chan(channel);
                            let float_samples: Vec<f32> = channel_data
                                .iter()
                                .map(|&s| s as f32 / 32768.0) // 转换为[-1, 1]
                                .collect();
                            all_samples[channel].extend(float_samples);
                        }
                    }
                    AudioBufferRef::S24(buf) => {
                        channels = buf.spec().channels.count();
                        // 处理24位音频
                        for channel in 0..channels {
                            if all_samples.len() <= channel {
                                all_samples.push(Vec::new());
                            }
                            let channel_data = buf.chan(channel);
                            let float_samples: Vec<f32> = channel_data
                                .iter()
                                .map(|&s| s.inner() as f32 / 8388608.0) // 转换为[-1, 1]
                                .collect();
                            all_samples[channel].extend(float_samples);
                        }
                    }
                    _ => {
                        println!("不支持的音频格式");
                    }
                }
            }
            Err(e) => {
                eprintln!("解码错误: {}", e);
                break;
            }
        }
    }
    let mut audio_tensor = Tensor::new(all_samples, device)?;
    if channels > 1 {
        // 对channel通道求平均， channel维度变为1
        audio_tensor = audio_tensor.mean_keepdim(0)?;
    }
    Ok((audio_tensor, sample_rate as usize))
}

pub fn load_audio(path: &str, device: &Device) -> Result<(Tensor, usize)> {
    let audio_vec = get_audio_bytes_vec(path)?;
    load_audio_use_symphonia(audio_vec, device)
}

pub fn load_audio_with_resample(
    path: &str,
    device: &Device,
    target_sample_rate: Option<usize>,
) -> Result<Tensor> {
    // hound 只支持wav文件
    // let audio_path = get_audio_path(path)?;
    // let (mut audio, sr) = load_audio_use_hound(audio_path, device)?;

    let audio_vec = get_audio_bytes_vec(path)?;
    let (mut audio, sr) = load_audio_use_symphonia(audio_vec, device)?;
    if let Some(target_sample_rate) = target_sample_rate
        && target_sample_rate != sr
    {
        audio = resample_simple(&audio, sr as i64, target_sample_rate as i64)?;
    }
    Ok(audio)
}

pub fn save_wav(audio: &Tensor, save_path: &str, sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    assert_eq!(audio.dim(0)?, 1, "audio channel must be 1");
    let max = audio.abs()?.max_all()?;
    let max = max.to_scalar::<f32>()?;
    let ratio = if max > 1.0 { 32767.0 / max } else { 32767.0 };
    let audio = audio.squeeze(0)?;
    let audio_vec = audio.to_vec1::<f32>()?;
    let mut writer = hound::WavWriter::create(save_path, spec).unwrap();
    for i in audio_vec {
        let sample_i16 = (i * ratio).round() as i16;
        writer.write_sample(sample_i16).unwrap();
    }
    writer.finalize().unwrap();
    Ok(())
}

pub fn get_audio_wav_u8(audio: &Tensor, sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    assert_eq!(audio.dim(0)?, 1, "audio channel must be 1");
    let max = audio.abs()?.max_all()?;
    let max = max.to_scalar::<f32>()?;
    let ratio = if max > 1.0 { 32767.0 / max } else { 32767.0 };
    let audio = audio.squeeze(0)?;
    let audio_vec = audio.to_vec1::<f32>()?;
    let mut cursor = Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
    for i in audio_vec {
        let sample_i16 = (i * ratio).round() as i16;
        writer.write_sample(sample_i16)?;
    }
    writer.finalize()?;
    let wav_buffer = cursor.into_inner();
    Ok(wav_buffer)
}

pub fn extract_audio_url(mes: &ChatCompletionParameters) -> Vec<String> {
    let mut audio_vec = Vec::new();
    for chat_mes in mes.messages.clone() {
        if let ChatMessage::User { content, .. } = chat_mes.clone()
            && let ChatMessageContent::ContentPart(part_vec) = content
        {
            for part in part_vec {
                if let ChatMessageContentPart::Audio(audio_part) = part {
                    let audio_url = audio_part.audio_url;
                    audio_vec.push(audio_url.url);
                }
            }
        }
    }
    audio_vec
}

pub fn extract_audios(
    mes: &ChatCompletionParameters,
    device: &Device,
    target_sample_rate: Option<usize>,
) -> Result<Vec<Tensor>> {
    let audio_url_vec = extract_audio_url(mes);
    // 并行加载音频
    audio_url_vec
        .par_iter()
        .map(|url| load_audio_with_resample(url, device, target_sample_rate))
        .collect()
    // #[cfg(not(feature = "ffmpeg"))]
    // {
    //     audio_url_vec
    //         .par_iter()
    //         .map(|url| load_audio_with_resample(url, device, target_sample_rate))
    //         .collect()
    // }
    // #[cfg(feature = "ffmpeg")]
    // {
    //     // 该方法wav文件解析有问题
    //     use crate::utils::audio_utils::load_and_resample_audio_ffmpeg;
    //     audio_url_vec
    //         .par_iter()
    //         .map(|url| load_and_resample_audio_ffmpeg(url, target_sample_rate, device))
    //         .collect()
    // }

    // 使用rubato重采样
    // audio_url_vec
    //     .par_iter()
    //     .map(|url| load_and_resample_audio_rubato(url, device, target_sample_rate))
    //     .collect()
}

// 从 ChatCompletionResponse 中提取音频数据
pub fn extract_audio_base64_from_response(
    response: &ChatCompletionResponse,
) -> Result<Vec<String>> {
    let mut audio_data_list = Vec::new();

    for choice in &response.choices {
        if let ChatMessage::Assistant {
            content: Some(ChatMessageContent::ContentPart(parts)),
            ..
        } = &choice.message
        {
            for part in parts.clone() {
                if let ChatMessageContentPart::Audio(audio_part) = part {
                    // if let Some(audio_data) = &audio_part.audio_url {
                    //     audio_data_list.push(audio_data.data.clone());
                    // }
                    let audio_url = audio_part.audio_url;
                    audio_data_list.push(audio_url.url);
                }
            }
        }
    }

    Ok(audio_data_list)
}

// 将 base64 音频数据解码并保存到文件
pub fn save_audio_from_base64<P: AsRef<Path>>(base64_data: &str, file_path: P) -> Result<()> {
    // 解码 base64 数据
    let data: Vec<&str> = base64_data.split("base64,").collect();
    let data = data[1];
    let decoded_data = BASE64_STANDARD.decode(data)?;

    // 创建文件并写入数据
    let mut file = File::create(file_path)?;
    file.write_all(&decoded_data)?;

    Ok(())
}

// 组合函数：从响应中提取音频并保存到文件
pub fn extract_and_save_audio_from_response(
    response: &ChatCompletionResponse,
    directory: &str,
) -> Result<Vec<String>> {
    let audio_data_list = extract_audio_base64_from_response(response)?;
    let mut saved_files = Vec::new();

    for (index, audio_data) in audio_data_list.iter().enumerate() {
        let file_path = format!("{}/audio_{}.wav", directory, index);
        save_audio_from_base64(audio_data, &file_path)?;
        saved_files.push(file_path);
    }

    Ok(saved_files)
}

#[cfg(feature = "ffmpeg")]
pub fn load_and_resample_audio_ffmpeg(
    file_path: &str,
    target_sample_rate: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    // 方法只支持mp3
    // wav文件会报错：
    // [SWR @ 0x745ff0037840] Input channel layout "" is invalid or unsupported.
    // Error: Invalid argument
    // 未解决
    ffmpeg::init().map_err(|e| anyhow!(format!("Failed to initialize ffmpeg: {}", e)))?;

    // 打开文件
    let mut ictx = ffmpeg::format::input(&Path::new(file_path))
        .map_err(|e| anyhow!(format!("Failed to open audio file: {}", e)))?;

    // 找到音频流
    let stream = ictx
        .streams()
        .best(ffmpeg::media::Type::Audio)
        .ok_or_else(|| anyhow!(format!("No audio stream found")))?;
    let stream_index = stream.index();

    // 获取解码器
    let codec_params = stream.parameters();
    let mut decoder = ffmpeg::codec::context::Context::from_parameters(codec_params)
        .map_err(|e| anyhow!(format!("无法创建解码器上下文: {}", e)))?
        .decoder()
        .audio()
        .map_err(|e| anyhow!(format!("不是音频解码器: {}", e)))?;

    // // 直接更改输入的channel_layout也会报错：Error: Input changed
    // let src_channels = decoder.channels();
    // let layout = decoder.channel_layout();
    // if layout.is_empty() || layout.channels() == 0 {
    //     // 如果没有有效的 channel layout，使用基于通道数的默认布局
    //     let layout = ffmpeg::channel_layout::ChannelLayout::default(src_channels as i32);
    //     decoder.set_channel_layout(layout);
    // }
    let original_sample_rate = decoder.rate() as usize;
    let needs_resampling = match target_sample_rate {
        None => false,
        Some(target_sr) => target_sr != original_sample_rate,
    };
    // 存储音频数据
    let mut audio_buffer = vec![];
    if !needs_resampling {
        // 不需要重采样，直接解码音频
        for (stream, packet) in ictx.packets() {
            if stream.index() == stream_index {
                decoder.send_packet(&packet)?;
                let mut decoded = ffmpeg::util::frame::Audio::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    let planes = decoded.planes();
                    if planes == 1 {
                        let data_slice = decoded.plane::<f32>(0);
                        audio_buffer.extend_from_slice(data_slice);
                    } else {
                        let mut channel_data: Vec<&[f32]> = vec![];
                        for plane_idx in 0..planes {
                            let plane_data = decoded.plane::<f32>(plane_idx);
                            channel_data.push(plane_data);
                        }
                        let channel_len = channel_data[0].len();
                        for sample_idx in 0..channel_len {
                            let mut sum = 0.0f32;
                            for channel in &channel_data {
                                sum += channel[sample_idx];
                            }
                            let avg = sum / planes as f32;
                            audio_buffer.push(avg);
                        }
                    }
                }
            }
        }
    } else {
        let target_sample_rate = target_sample_rate.unwrap_or(16000);
        // 创建重采样器， 通道为1
        let mut resampler = ffmpeg::software::resampling::context::Context::get(
            decoder.format(),
            decoder.channel_layout(),
            decoder.rate() as u32,
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
            ffmpeg::channel_layout::ChannelLayout::default(1),
            target_sample_rate as u32,
        )
        .map_err(|e| anyhow!(format!("无法创建重采样器: {}", e)))?;

        // let mut resampler = decoder.resampler(
        //     ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
        //     ffmpeg::channel_layout::ChannelLayout::default(target_channels as i32),
        //     target_sample_rate,
        // )?;

        // 处理所有包
        for (stream, packet) in ictx.packets() {
            if stream.index() == stream_index {
                // 解码
                decoder.send_packet(&packet)?;

                let mut decoded = ffmpeg::util::frame::Audio::empty();
                while decoder.receive_frame(&mut decoded).is_ok() {
                    // 重采样
                    let mut resampled = ffmpeg::util::frame::Audio::empty();
                    resampler.run(&decoded, &mut resampled)?;

                    // 提取数据,Planar格式
                    let data_slice = resampled.plane::<f32>(0);
                    audio_buffer.extend_from_slice(data_slice);
                }
            }
        }

        // 处理剩余数据
        decoder.send_eof()?;

        let mut decoded = ffmpeg::util::frame::Audio::empty();
        while decoder.receive_frame(&mut decoded).is_ok() {
            let mut resampled = ffmpeg::util::frame::Audio::empty();
            resampler.run(&decoded, &mut resampled)?;

            // 提取数据,Planar格式
            let data_slice = resampled.plane::<f32>(0);
            audio_buffer.extend_from_slice(data_slice);
        }
    }

    let audio_tensor = Tensor::new(audio_buffer, device)?;

    Ok(audio_tensor)
}

// // 使用rubato库做重采样
// pub fn load_and_resample_audio_rubato(
//     path: &str,
//     device: &Device,
//     target_sample_rate: Option<usize>,
// ) -> Result<Tensor> {
//     let audio_vec = get_audio_bytes_vec(path)?;
//     let (mut audio, sr) = load_audio_use_symphonia(audio_vec, device)?;
//     let mono_audio = audio.squeeze(0)?.to_vec1::<f32>()?;
//     if let Some(target_sample_rate) = target_sample_rate
//         && target_sample_rate != sr
//     {
//         let params = SincInterpolationParameters {
//             sinc_len: 256,
//             f_cutoff: 0.99,
//             interpolation: SincInterpolationType::Cubic,
//             oversampling_factor: 256,
//             window: WindowFunction::BlackmanHarris2,
//         };
//         let input_len = mono_audio.len();
//         let mut resampler = Async::<f64>::new_sinc(
//             target_sample_rate as f64 / sr as f64, // 重采样比例
//             1.0,                                   // 输出/输入采样率比
//             &params,
//             input_len,
//             1, // 单通道
//             FixedAsync::Input,
//         )
//         .map_err(|e| anyhow!(format!("无法创建重采样器: {}", e)))?;

//         let audio_f64: Vec<f64> = mono_audio.iter().map(|x| *x as f64).collect();
//         let input_adapter = InterleavedSlice::new(&audio_f64, 1, input_len)?;

//         let mut outdata = vec![0.0f64; input_len * 2];
//         let mut output_adapter = InterleavedSlice::new_mut(&mut outdata, 1, input_len * 2)?;
//         // Preparations
//         let mut indexing = Indexing {
//             input_offset: 0,
//             output_offset: 0,
//             active_channels_mask: None,
//             partial_len: None,
//         };
//         let mut input_frames_left = input_len;
//         let mut input_frames_next = resampler.input_frames_max();
//         while input_frames_left >= input_frames_next {
//             let (frames_read, frames_written) = resampler.process_into_buffer(
//                 &input_adapter,
//                 &mut output_adapter,
//                 Some(&indexing),
//             )?;
//             indexing.input_offset += frames_read;
//             indexing.output_offset += frames_written;
//             input_frames_left -= frames_read;
//             input_frames_next = resampler.input_frames_next();
//         }
//         indexing.partial_len = Some(input_frames_left);
//         let (_nbr_in, _nbr_out) = resampler
//             .process_into_buffer(&input_adapter, &mut output_adapter, Some(&indexing))
//             .unwrap();
//         let output_len = input_len * target_sample_rate / sr;
//         audio = Tensor::new(&outdata[0..output_len], device)?
//             .to_dtype(candle_core::DType::F32)?
//             .unsqueeze(0)?;
//     }
//     Ok(audio)
// }

// pub fn create_hann_window(window_size: usize, dtype: DType, device: &Device) -> Result<Tensor> {
//     let n = window_size as f64;
//     let window: Vec<f32> = (0..window_size)
//         .map(|i| {
//             let i_f64 = i as f64;
//             let val = 0.5 * (1.0 - (2.0 * PI * i_f64 / n).cos());
//             val as f32
//         })
//         .collect();
//     Ok(Tensor::from_vec(window, window_size, device)?.to_dtype(dtype)?)
// }

pub fn create_hann_window(window_size: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    if window_size < 1 {
        return Err(anyhow::anyhow!("window_size must bigger than 0"));
    }
    if window_size == 1 {
        return Ok(Tensor::new(1.0f32, device)?.to_dtype(dtype)?);
    }
    let n = window_size as f64 - 1.0;
    let start = 1_i64 - window_size as i64;
    let end = window_size as i64;
    let window: Vec<f32> = (start..end)
        .step_by(2)
        .map(|i| {
            let i_f64 = i as f64;
            let val = 0.5 + 0.5 * (PI * i_f64 / n).cos();
            val as f32
        })
        .collect();
    Ok(Tensor::from_vec(window, window_size, device)?.to_dtype(dtype)?)
}

pub fn create_povey_window(window_size: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let window = create_hann_window(window_size, dtype, device)?;
    Ok(window.powf(0.85)?)
}

pub fn crate_hamming_window(
    window_size: usize,
    periodic: bool,
    alpha: f64,
    beta: f64,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let denominator = if periodic {
        window_size as f64
    } else {
        (window_size - 1) as f64
    };

    let window: Vec<f32> = (0..window_size)
        .map(|i| {
            let i_f64 = i as f64;
            let val = alpha - beta * (2.0 * std::f64::consts::PI * i_f64 / denominator).cos();
            val as f32
        })
        .collect();

    Ok(Tensor::from_vec(window, window_size, device)?.to_dtype(dtype)?)
}

/// 梅尔频率刻度类型
#[derive(Debug, Clone, Copy)]
pub enum MelScale {
    Htk,
    Kaldi,
    Slaney,
}

/// 将赫兹转换为梅尔频率
pub fn hertz_to_mel(freq: f32, mel_scale: MelScale) -> f32 {
    match mel_scale {
        MelScale::Htk => 2595.0 * ((1.0 + freq / 700.0).log10()),
        MelScale::Kaldi => 1127.0 * ((1.0 + freq / 700.0).ln()),
        MelScale::Slaney => {
            let min_log_hertz = 1000.0;
            let min_log_mel = 15.0;
            let logstep = 27.0 / 6.4_f32.ln();
            let mut mels = 3.0 * freq / 200.0;

            if freq >= min_log_hertz {
                mels = min_log_mel + (freq / min_log_hertz).ln() * logstep;
            }
            mels
        }
    }
}

/// 将梅尔频率转换为赫兹
pub fn mel_to_hertz(mels: f32, mel_scale: MelScale) -> f32 {
    match mel_scale {
        MelScale::Htk => 700.0 * (10.0_f32.powf(mels / 2595.0) - 1.0),
        MelScale::Kaldi => 700.0 * (f32::exp(mels / 1127.0) - 1.0),
        MelScale::Slaney => {
            let min_log_hertz = 1000.0;
            let min_log_mel = 15.0;
            let logstep = 6.4_f32.ln() / 27.0;
            let mut freq = 200.0 * mels / 3.0;

            if mels >= min_log_mel {
                freq = min_log_hertz * f32::exp(logstep * (mels - min_log_mel));
            }
            freq
        }
    }
}

pub fn create_triangular_filter_bank(fft_freqs: &Tensor, filter_freqs: &Tensor) -> Result<Tensor> {
    // fft_freqs/filter_freqs -> 1d
    let len = filter_freqs.dim(0)?;
    let filter_diff = filter_freqs
        .narrow(0, 1, len - 1)?
        .sub(&filter_freqs.narrow(0, 0, len - 1)?)?;
    let slopes = filter_freqs
        .unsqueeze(0)?
        .broadcast_sub(&fft_freqs.unsqueeze(1)?)?;
    let down_slopes = slopes
        .narrow(D::Minus1, 0, len - 2)?
        .affine(-1.0, 0.0)?
        .broadcast_div(&filter_diff.narrow(0, 0, len - 2)?)?;
    let up_slopes = slopes
        .narrow(D::Minus1, 2, len - 2)?
        .broadcast_div(&filter_diff.narrow(0, 1, len - 2)?)?;
    let res = down_slopes
        .minimum(&up_slopes)?
        .maximum(&Tensor::zeros_like(&down_slopes)?)?;
    Ok(res)
}

/// 创建梅尔滤波器组
pub fn mel_filter_bank(
    num_frequency_bins: usize,
    num_mel_filters: usize,
    min_frequency: f32,
    max_frequency: f32,
    sampling_rate: f32,
    norm: Option<&str>,
    mel_scale: MelScale,
    triangularize_in_mel_space: bool,
    device: &Device,
) -> Result<Tensor> {
    // 参数验证
    if let Some(n) = norm
        && n != "slaney"
    {
        return Err(anyhow::anyhow!("norm must be one of None or 'slaney'"));
    }
    if num_frequency_bins < 2 {
        return Err(anyhow::anyhow!(
            "Require num_frequency_bins: {} >= 2",
            num_frequency_bins
        ));
    }
    if min_frequency > max_frequency {
        return Err(anyhow::anyhow!(
            "Require min_frequency: {} <= max_frequency: {}",
            min_frequency,
            max_frequency
        ));
    }
    // 计算梅尔频率范围
    let mel_min = hertz_to_mel(min_frequency, mel_scale);
    let mel_max = hertz_to_mel(max_frequency, mel_scale);

    // 在梅尔刻度上均匀分布频率点（包括边界点）
    let mel_freqs = linspace(mel_min, mel_max, num_mel_filters + 2, device)?;

    // 将梅尔频率转换回赫兹频率
    let filter_freqs: Vec<f32> = mel_freqs
        .to_vec1::<f32>()?
        .iter()
        .map(|&m| mel_to_hertz(m, mel_scale))
        .collect();
    let mut filter_freqs = Tensor::new(filter_freqs, device)?;

    let fft_freqs = if triangularize_in_mel_space {
        // 在梅尔空间中应用三角滤波器
        let fft_bin_width = sampling_rate / ((num_frequency_bins as f32 - 1.0) * 2.0);
        let fft_vec: Vec<f32> = (0..num_frequency_bins)
            .map(|i| hertz_to_mel(fft_bin_width * i as f32, mel_scale))
            .collect();
        filter_freqs = mel_freqs;
        Tensor::new(fft_vec, device)?
    } else {
        // 在赫兹频率上
        linspace(0.0, sampling_rate / 2.0, num_frequency_bins, device)?
    };

    // 创建三角滤波器组
    let mut mel_filters = create_triangular_filter_bank(&fft_freqs, &filter_freqs)?;

    // 如果需要，进行归一化
    if let Some(n) = norm
        && n == "slaney"
    {
        // Slaney风格的归一化
        let enorm = (2.0
            / filter_freqs
                .i(2..num_mel_filters + 2)?
                .sub(&filter_freqs.i(0..num_mel_filters)?)?)?
        .unsqueeze(0)?;
        mel_filters = mel_filters.broadcast_mul(&enorm)?;
    }

    // // 检查是否有零值滤波器
    // let mel_max = mel_filters.max(0)?;
    // let mel_max_eq_zero = mel_max.eq(&Tensor::zeros_like(&mel_max)?)?;
    // let eq_zero_index = zero_index_vec(&mel_max_eq_zero)?;
    // if eq_zero_index.len() > 0 {
    //     println!("At least one mel filter has all zero values.");
    // }

    Ok(mel_filters)
}

pub fn stft_audio(n_fft: usize, frame_wave: &[f32]) -> Result<Vec<f32>> {
    let mut real_planner = RealFftPlanner::<f32>::new();
    let r2c = real_planner.plan_fft_forward(n_fft);
    let mut spectrum = r2c.make_output_vec();
    let mut frame_wave = frame_wave.to_owned();
    r2c.process(&mut frame_wave, &mut spectrum)?;
    let output: Vec<f32> = spectrum.iter().map(|complex| complex.norm_sqr()).collect();
    Ok(output)
}

pub fn apply_stft(waveform: &Tensor) -> Result<Tensor> {
    // waveform: (bs, n_frames, window_size)
    let mut wave_fft = vec![];
    let (batch_size, _, window_size) = waveform.dims3()?;
    for bs in 0..batch_size {
        let wave_i = waveform.i(bs)?;
        let wave_i_vec = wave_i.to_vec2::<f32>()?;
        let wave_i_fft_vec: Result<Vec<Vec<f32>>> = wave_i_vec
            .par_iter()
            .map(|frame_wave| stft_audio(window_size, frame_wave))
            .collect();
        let wave_i_fft_vec = wave_i_fft_vec?;

        let wave_i_fft = Tensor::new(wave_i_fft_vec, waveform.device())?.unsqueeze(0)?;
        wave_fft.push(wave_i_fft);
    }
    let magnitudes = Tensor::cat(&wave_fft, 0)?;
    Ok(magnitudes)
}

pub fn torch_stft(
    waveform: &Tensor,
    n_fft: usize,
    hop_length: usize,
    window: &Tensor,
) -> Result<Tensor> {
    // waveform: already padding
    // (bs, n_frames, n_fft)
    let frames = extract_frames(&waveform, n_fft, hop_length)?;
    // 应用汉明窗口
    let result = frames.broadcast_mul(window)?;
    // 傅立叶变换
    let magnitudes = apply_stft(&result)?;
    Ok(magnitudes)
}

pub fn kaldi_fbank(
    waveform: &Tensor,
    mel_energies: &Tensor,
    window_shift: usize,
    window_size: usize,
    padded_window_size: usize,
    dither: f32,
    // energy_floor: f32,
    // window_type: &str,
    // sample_frequency: usize,
    // snip_edges: bool,
) -> Result<Tensor> {
    let (strided_input, _) = get_window(
        waveform,
        padded_window_size,
        window_size,
        window_shift,
        dither,
        true,
        true,
        0.97,
    )?;

    let spectrum = apply_stft(&strided_input)?;
    let mel_energies = spectrum.broadcast_matmul(mel_energies)?;
    let epsilon =
        Tensor::new(1.192_092_9e-7_f32, waveform.device())?.broadcast_as(mel_energies.shape())?;
    let mel_energies = mel_energies.maximum(&epsilon)?.log()?;

    Ok(mel_energies)
}

pub fn apply_lfr(inputs: &Tensor, lfr_m: usize, lfr_n: usize) -> Result<Tensor> {
    let (t, feat_dim) = inputs.dims2()?;
    let t_lfr = (t as f32 / lfr_n as f32).ceil() as usize;
    let left_padding_size = (lfr_m - 1) / 2;
    let left_padding = inputs.narrow(0, 0, 1)?.repeat((left_padding_size, 1))?;
    let mut inputs = Tensor::cat(&[&left_padding, inputs], 0)?;
    let t = t + left_padding_size;
    let last_idx = (t - lfr_m) / lfr_n + 1;
    let num_padding = lfr_m - (t - last_idx * lfr_n);
    if num_padding > 0 {
        let num_padding =
            (2 * lfr_m - 2 * t + (t_lfr - 1 + last_idx) * lfr_n) / 2 * (t_lfr - last_idx);
        let right_padding = inputs.narrow(0, t - 1, 1)?.repeat((num_padding, 1))?;
        inputs = Tensor::cat(&[&inputs, &right_padding], 0)?;
    }
    let mut outputs = vec![];
    for i in 0..t_lfr {
        let start = i * lfr_n;
        let frame = inputs
            .narrow(0, start, lfr_m)?
            .reshape((1, lfr_m * feat_dim))?;
        outputs.push(frame);
    }
    let lfr_outputs = Tensor::cat(&outputs, 0)?;
    Ok(lfr_outputs)
}

pub fn get_waveform_and_window_properties(
    sample_frequency: usize,
    frame_shift: f32,
    frame_length: f32,
    round_to_power_of_two: bool,
) -> Result<(usize, usize, usize)> {
    let window_shift = (sample_frequency as f32 * frame_shift * 0.001) as usize;
    let window_size = (sample_frequency as f32 * frame_length * 0.001) as usize;
    let padded_window_size = if round_to_power_of_two {
        (window_size - 1).next_power_of_two()
    } else {
        window_size
    };
    Ok((window_shift, window_size, padded_window_size))
}

pub fn get_window(
    waveform: &Tensor,
    padded_window_size: usize,
    window_size: usize,
    window_shift: usize,
    dither: f32,
    remove_dc_offset: bool,
    raw_energy: bool,
    preemphasis_coefficient: f32,
) -> Result<(Tensor, Tensor)> {
    let mut strided_input = extract_frames(waveform, window_size, window_shift)?;
    // (ba, m, window_size)
    if dither != 0.0 {
        let rand_gauss = strided_input
            .randn_like(0.0, 1.0)?
            .affine(dither as f64, 0.0)?;
        strided_input = strided_input.add(&rand_gauss)?;
    }
    if remove_dc_offset {
        let row_means = strided_input.mean_keepdim(D::Minus1)?;
        strided_input = strided_input.broadcast_sub(&row_means)?;
    }
    let signal_log_energy = if raw_energy {
        let energy = strided_input.powf(2.0)?.sum(1)?.log()?;
        Some(energy)
    } else {
        None
    };

    if preemphasis_coefficient != 0.0 {
        let offset_strided_input = pad_replicate_last_dim(&strided_input, (1, 0))?
            .affine(preemphasis_coefficient as f64, 0.0)?;
        strided_input =
            strided_input.sub(&offset_strided_input.narrow(D::Minus1, 0, window_size)?)?;
    }

    let windows = crate_hamming_window(
        window_size,
        false,
        0.54,
        0.46,
        waveform.dtype(),
        waveform.device(),
    )?
    .unsqueeze(0)?
    .unsqueeze(0)?;

    strided_input = strided_input.broadcast_mul(&windows)?;

    if padded_window_size != window_size {
        let padding_right = padded_window_size - window_size;
        strided_input = strided_input.pad_with_zeros(D::Minus1, 0, padding_right)?;
    }

    let signal_log_energy = signal_log_energy.unwrap_or(strided_input.powf(2.0)?.sum(1)?.log()?);
    Ok((strided_input, signal_log_energy))
}

/// 提取音频帧
pub fn extract_frames(
    waveform: &Tensor,
    window_size: usize,
    window_shift: usize,
) -> Result<Tensor> {
    // waveform ->(1, audio_len)
    let waveform_len = waveform.dim(1)?;
    let n_frames = 1 + (waveform_len - window_size) / window_shift;
    let mut frames = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        let start = i * window_shift;
        let frame = waveform.narrow(D::Minus1, start, window_size)?;
        frames.push(frame);
    }

    let result = Tensor::cat(&frames, D::Minus1)?;
    let bs = result.dim(0)?;
    let reshaped = result.reshape((bs, n_frames, window_size))?;
    Ok(reshaped)
}

pub fn inverse_mel_scale(mel_freq: &Tensor) -> Result<Tensor> {
    Ok(mel_freq
        .affine(1.0 / 1127.0, 0.0)?
        .exp()?
        .affine(1.0, -1.0)?
        .affine(700.0, 0.0)?)
}

pub fn mel_scale(freq: &Tensor) -> Result<Tensor> {
    Ok(freq.affine(1.0 / 700.0, 1.0)?.log()?.affine(1127.0, 0.0)?)
}

pub fn kaldi_get_mel_banks(
    num_bins: usize,
    window_length_padded: usize,
    sample_freq: f32,
    low_freq: f32,
    high_freq: f32,
    // vtln_low: f32,
    // vtln_high: f32,
    // vtln_warp_factor: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    assert!(num_bins > 3, "Must have at least 3 mel bins");
    assert!(
        window_length_padded.is_multiple_of(2),
        "window_length_padded must be even"
    );

    let num_fft_bins = window_length_padded as f32 / 2.0;
    let nyquist = 0.5 * sample_freq;

    let mut high_freq = high_freq;
    if high_freq <= 0.0 {
        high_freq += nyquist;
    }

    assert!(
        (0.0 <= low_freq && low_freq < nyquist)
            && (0.0 < high_freq && high_freq <= nyquist)
            && (low_freq < high_freq),
        "Bad values in options: low-freq {} and high-freq {} vs. nyquist {}",
        low_freq,
        high_freq,
        nyquist
    );

    // FFT bin 宽度
    let fft_bin_width = sample_freq / (window_length_padded as f32);
    let mel_low_freq = hertz_to_mel(low_freq, MelScale::Kaldi);
    let mel_high_freq = hertz_to_mel(high_freq, MelScale::Kaldi);

    // 分频点之间的间隔
    let mel_freq_delta = (mel_high_freq - mel_low_freq) / ((num_bins + 1) as f32);

    // let mut vtln_high = vtln_high;
    // if vtln_high < 0.0 {
    //     vtln_high += nyquist;
    // }

    // if vtln_warp_factor != 1.0 {
    //     assert!(
    //         low_freq < vtln_low
    //             && vtln_low < high_freq
    //             && 0.0 < vtln_high
    //             && vtln_high < high_freq
    //             && vtln_low < vtln_high,
    //         "Bad values in options: vtln-low {} and vtln-high {}, versus low-freq {} and high-freq {}",
    //         vtln_low,
    //         vtln_high,
    //         low_freq,
    //         high_freq
    //     );
    // }

    // 创建 bin 索引张量
    let bins = Tensor::arange(0u32, num_bins as u32, device)?
        .to_dtype(candle_core::DType::F32)?
        .unsqueeze(1)?; // size(num_bins, 1)

    // 计算梅尔刻度下的边界频率
    let left_mel = bins.affine(mel_freq_delta as f64, mel_low_freq as f64)?;
    let center_mel = bins
        .affine(1.0, 1.0)?
        .affine(mel_freq_delta as f64, mel_low_freq as f64)?;
    let right_mel = bins
        .affine(1.0, 2.0)?
        .affine(mel_freq_delta as f64, mel_low_freq as f64)?;

    // 如果使用 VTLN，则对频率进行扭曲
    // let (left_mel, center_mel, right_mel) = if vtln_warp_factor != 1.0 {
    //     (
    //         vtln_warp_mel_freq(
    //             vtln_low,
    //             vtln_high,
    //             low_freq,
    //             high_freq,
    //             vtln_warp_factor,
    //             &left_mel,
    //         )?,
    //         vtln_warp_mel_freq(
    //             vtln_low,
    //             vtln_high,
    //             low_freq,
    //             high_freq,
    //             vtln_warp_factor,
    //             &center_mel,
    //         )?,
    //         vtln_warp_mel_freq(
    //             vtln_low,
    //             vtln_high,
    //             low_freq,
    //             high_freq,
    //             vtln_warp_factor,
    //             &right_mel,
    //         )?,
    //     )
    // } else {
    //     (left_mel, center_mel, right_mel)
    // };

    // 转换中心频率回赫兹单位
    let center_freqs = inverse_mel_scale(&center_mel)?;

    // 创建 FFT bin 频率
    let fft_bins = Tensor::arange(0u32, num_fft_bins as u32, device)?
        .to_dtype(candle_core::DType::F32)?
        .affine(fft_bin_width as f64, 0.0)?;
    let mel = mel_scale(&fft_bins)?.unsqueeze(0)?; // size(1, num_fft_bins)

    // 计算斜率
    let up_slope = mel
        .broadcast_sub(&left_mel)?
        .broadcast_div(&center_mel.broadcast_sub(&left_mel)?)?;
    let down_slope = right_mel
        .broadcast_sub(&mel)?
        .broadcast_div(&right_mel.broadcast_sub(&center_mel)?)?;

    // left_mel < center_mel < right_mel 所以我们可以取两个斜率的最小值并限制负值
    let min_slopes = up_slope.minimum(&down_slope)?;
    let zeros = Tensor::zeros(min_slopes.dims(), candle_core::DType::F32, device)?;
    let bins_tensor = min_slopes.maximum(&zeros)?;
    // let bins_tensor = if vtln_warp_factor == 1.0 {
    //     // left_mel < center_mel < right_mel 所以我们可以取两个斜率的最小值并限制负值
    //     let min_slopes = up_slope.minimum(&down_slope)?;
    //     let zeros = Tensor::zeros(min_slopes.dims(), candle_core::DType::F32, device)?;
    //     min_slopes.maximum(&zeros)?
    // } else {
    //     // 扭曲可能会改变 left_mel, center_mel, right_mel 的顺序
    //     let zeros = Tensor::zeros(up_slope.dims(), candle_core::DType::F32, device)?;
    //     let mut bins_tensor = zeros.clone();

    //     // 创建索引掩码
    //     let up_idx = mel
    //         .gt_tensor(&left_mel)?
    //         .and(&mel.le_tensor(&center_mel)?)?; // left_mel < mel <= center_mel
    //     let down_idx = mel
    //         .gt_tensor(&center_mel)?
    //         .and(&mel.lt_tensor(&right_mel)?)?; // center_mel < mel < right_mel

    //     bins_tensor = bins_tensor.where_cond(&up_idx, &up_slope)?;
    //     bins_tensor = bins_tensor.where_cond(&down_idx, &down_slope)?;
    //     bins_tensor
    // };

    Ok((bins_tensor, center_freqs))
}

pub fn spectrogram(
    waveform: &Tensor,
    window: &Tensor,
    frame_length: usize,
    hop_length: usize,
    fft_length: usize,
    power: Option<f32>,
    center: bool,
    preemphasis: f64,
    mel_filters: Option<&Tensor>,
    log_mel: Option<&str>,
    mel_floor: f32,
    remove_dc_offset: bool,
) -> Result<Tensor> {
    let waveform = if center {
        let pad = frame_length / 2;
        pad_reflect_last_dim(waveform, (pad, pad))?
    } else {
        waveform.clone()
    };
    let mut frames = extract_frames(&waveform, frame_length, hop_length)?;
    if remove_dc_offset {
        let row_means = frames.mean_keepdim(D::Minus1)?;
        frames = frames.broadcast_sub(&row_means)?;
    }

    if preemphasis != 0.0 {
        let buffer_0 = frames
            .i((.., .., 0))?
            .affine(1.0 - preemphasis, 0.0)?
            .unsqueeze(D::Minus1)?;
        let buffer_ = frames.i((.., .., 1..))?.sub(
            &frames
                .i((.., .., 0..frame_length - 1))?
                .affine(preemphasis, 0.0)?,
        )?;
        frames = Tensor::cat(&[buffer_0, buffer_], D::Minus1)?;
    }
    let mut frames = frames.broadcast_mul(&window)?;
    let pad_len = fft_length - frame_length;
    if pad_len > 0 {
        // (bs, nframes, frame_length) -> (bs, nframes, fft_length)
        frames = frames.pad_with_zeros(D::Minus1, 0, pad_len)?;
    }
    let mut spectrogram = apply_stft(&frames)?; // stft已经做了pow(2.0)
    spectrogram = spectrogram.transpose(D::Minus1, D::Minus2)?;
    if let Some(mel_filters) = mel_filters {
        let spect = mel_filters.t()?.broadcast_matmul(&spectrogram)?;
        spectrogram = spect.maximum(
            &Tensor::new(mel_floor, spect.device())?
                .to_dtype(spect.dtype())?
                .broadcast_as(spect.shape())?,
        )?;
    }
    if let Some(_) = power
        && let Some(log_mel) = log_mel
    {
        if log_mel == "log" {
            spectrogram = spectrogram.log()?;
        } else if log_mel == "log10" {
            spectrogram = log10(&spectrogram)?;
        } else {
            return Err(anyhow!(
                "dB not completed or Unknown log_mel option ".to_string()
            ));
        }
    }
    Ok(spectrogram)
}

pub fn split_audio_into_chunks(wav: &Tensor, sr: usize, max_chunk_sec: f32) -> Result<Vec<Tensor>> {
    // wav: (1, len)
    let total_len = wav.dim(1)?;
    let total_sec = total_len as f32 / sr as f32;
    let mut wavs = vec![];
    if total_sec <= max_chunk_sec {
        wavs.push(wav.clone());
    } else {
        let max_len = (max_chunk_sec * sr as f32).round() as usize;
        let split_len = total_len / max_len;
        let mut splits = vec![max_len; split_len];
        let remain_len = total_len % max_len;
        splits.push(remain_len);
        let split_wav = split_tensor(wav, &splits, 1)?;
        wavs.extend_from_slice(&split_wav);
    }
    Ok(wavs)
}
