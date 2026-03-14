use std::collections::HashMap;

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionParameters, ChatMessage, ChatMessageContent, ChatMessageContentPart,
};
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
#[cfg(feature = "ffmpeg")]
use ffmpeg_next as ffmpeg;
use image::DynamicImage;

#[cfg(feature = "ffmpeg")]
use crate::utils::video_utils::video_smart_resize;
use crate::{
    models::qwen3vl::config::PreprocessorConfig,
    utils::img_utils::{get_image, img_smart_resize, img_transform},
};

#[derive(Clone)]
pub struct VisionInput {
    pub data: Tensor,
    pub grid_thw: Tensor,
}

#[derive(Clone)]
pub struct GeneralInput {
    pub replace_text: String,
    pub pixel_values: Option<Tensor>,
    pub image_grid_thw: Option<Tensor>,
    pub pixel_values_video: Option<Tensor>,
    pub video_grid_thw: Option<Tensor>,
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub struct VideoMetadata {
    total_num_frames: u32,
    fps: f32,
    width: u32,
    height: u32,
    duration: f32,
    frame_indices: Vec<u32>,
}

#[allow(unused)]
pub struct Qwen3VLProcessor {
    img_process_cfg: PreprocessorConfig,
    video_process_cfg: PreprocessorConfig,
    device: Device,
    dtype: DType,
    image_token: String,
    video_token: String,
    vision_start_token: String,
    vision_end_token: String,
    fps: u32,
    min_frames: u32,
    max_frames: u32,
}

impl Qwen3VLProcessor {
    pub fn new(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let path = path.to_string();
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let img_process_cfg_file = path.clone() + "/preprocessor_config.json";
        assert!(
            std::path::Path::new(&img_process_cfg_file).exists(),
            "preprocessor_config.json not exists in model path"
        );
        let img_process_cfg: PreprocessorConfig =
            serde_json::from_slice(&std::fs::read(img_process_cfg_file)?)?;

        let video_process_cfg_file = path.clone() + "/video_preprocessor_config.json";
        assert!(
            std::path::Path::new(&video_process_cfg_file).exists(),
            "video_preprocessor_config.json not exists in model path"
        );
        let video_process_cfg: PreprocessorConfig =
            serde_json::from_slice(&std::fs::read(video_process_cfg_file)?)?;

        let image_token = "<|image_pad|>".to_string();
        let video_token = "<|video_pad|>".to_string();
        let vision_start_token = "<|vision_start|>".to_string();
        let vision_end_token = "<|vision_end|>".to_string();
        Ok(Self {
            img_process_cfg,
            video_process_cfg,
            device: device.clone(),
            dtype,
            image_token,
            video_token,
            vision_start_token,
            vision_end_token,
            fps: 2,
            min_frames: 4,
            max_frames: 768,
        })
    }

    pub fn new_qwen3_5_default(device: &Device, dtype: DType) -> Result<Self> {
        let img_process_cfg = PreprocessorConfig::qwen3_5_img_default();
        let video_process_cfg = PreprocessorConfig::qwen3_5_video_default();
        let image_token = "<|image_pad|>".to_string();
        let video_token = "<|video_pad|>".to_string();
        let vision_start_token = "<|vision_start|>".to_string();
        let vision_end_token = "<|vision_end|>".to_string();
        Ok(Self {
            img_process_cfg,
            video_process_cfg,
            device: device.clone(),
            dtype,
            image_token,
            video_token,
            vision_start_token,
            vision_end_token,
            fps: 2,
            min_frames: 4,
            max_frames: 768,
        })
    }

    pub fn extract_vision_info(
        &self,
        mes: &ChatCompletionParameters,
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut vision_map = HashMap::new();
        vision_map.insert("image".to_string(), Vec::new());
        vision_map.insert("video".to_string(), Vec::new());
        for chat_mes in mes.messages.clone() {
            if let ChatMessage::User { content, .. } = chat_mes
                && let ChatMessageContent::ContentPart(part_vec) = content
            {
                for part in part_vec {
                    if let ChatMessageContentPart::Image(img_part) = part {
                        let img_url = img_part.image_url;
                        vision_map.get_mut("image").unwrap().push(img_url.url);
                    } else if let ChatMessageContentPart::Video(video_part) = part {
                        let video_url = video_part.video_url;
                        vision_map.get_mut("video").unwrap().push(video_url.url);
                    }
                }
            }
        }
        Ok(vision_map)
    }

    pub fn process_img(
        &self,
        img: &DynamicImage,
        img_mean: &Tensor,
        img_std: &Tensor,
    ) -> Result<Tensor> {
        let img_h = img.height();
        let img_w = img.width();
        //  h,w resize成 32的倍数
        let (resize_h, resize_w) = img_smart_resize(
            img_h,
            img_w,
            (self.img_process_cfg.patch_size * self.img_process_cfg.merge_size) as u32,
            self.img_process_cfg.size.shortest_edge as u32,
            self.img_process_cfg.size.longest_edge as u32,
        )?;
        let img = img.resize_exact(resize_w, resize_h, image::imageops::FilterType::CatmullRom);
        let img_tensor = img_transform(&img, img_mean, img_std, &self.device, self.dtype)?;
        // (c, h, w) => (1, c, h, w)
        let img_tensor = img_tensor.unsqueeze(0)?;
        Ok(img_tensor)
    }

    pub fn process_vision_tensor(&self, img_tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        // Check that data have `num_frames` divisible by `temporal_patch_size`
        // img_tensor: (t, c, h, w)
        let t = img_tensor.dim(0)?;
        let img_tensor = if t % self.img_process_cfg.temporal_patch_size != 0 {
            let repeat_num = self.img_process_cfg.temporal_patch_size
                - t % self.img_process_cfg.temporal_patch_size;
            let repeats = img_tensor.i(t - 1)?.repeat((repeat_num, 1, 1, 1))?;
            Tensor::cat(&[img_tensor, &repeats], 0)?
        } else {
            img_tensor.clone()
        };
        let channel = img_tensor.dim(1)?;
        let grid_t = img_tensor.dim(0)? / self.img_process_cfg.temporal_patch_size;
        let grid_h = img_tensor.dim(2)? / self.img_process_cfg.patch_size;
        let grid_w = img_tensor.dim(3)? / self.img_process_cfg.patch_size;
        let shape = Shape::from(vec![
            grid_t,
            self.img_process_cfg.temporal_patch_size,
            channel,
            grid_h / self.img_process_cfg.merge_size,
            self.img_process_cfg.merge_size,
            self.img_process_cfg.patch_size,
            grid_w / self.img_process_cfg.merge_size,
            self.img_process_cfg.merge_size,
            self.img_process_cfg.patch_size,
        ]);
        let img_tensor = img_tensor.reshape(shape)?;
        // shape to // grid_t,
        // grid_h / merge_size,
        // grid_w / merge_size,
        // merge_size,
        // merge_size,
        // channel,
        // temporal_patch_size,
        // patch_size,
        // patch_size,
        let img_tensor = img_tensor.permute(vec![0, 3, 6, 4, 7, 2, 1, 5, 8])?;
        let img_tensor = img_tensor
            .reshape((
                grid_t * grid_h * grid_w,
                channel
                    * self.img_process_cfg.temporal_patch_size
                    * self.img_process_cfg.patch_size
                    * self.img_process_cfg.patch_size,
            ))?
            .contiguous()?;
        let grid_thw = Tensor::from_vec(
            vec![grid_t as u32, grid_h as u32, grid_w as u32],
            (1, 3),
            &self.device,
        )?;
        Ok((img_tensor, grid_thw))
    }

    pub fn process_images(
        &self,
        imgs: Vec<DynamicImage>,
        img_mean: &Tensor,
        img_std: &Tensor,
    ) -> Result<VisionInput> {
        let mut pixel_values_vec = Vec::new();
        let mut vision_grid_thws_vec = Vec::new();

        for img in imgs {
            let img_tensor = self.process_img(&img, img_mean, img_std)?;
            let img_tensor = Tensor::cat(&[&img_tensor, &img_tensor], 0)?.contiguous()?;
            let (img_tensor, grid_thw) = self.process_vision_tensor(&img_tensor)?;
            pixel_values_vec.push(img_tensor);
            vision_grid_thws_vec.push(grid_thw);
        }
        let pixel_values = Tensor::cat(&pixel_values_vec, 0)?;
        let vision_grid_thws = Tensor::cat(&vision_grid_thws_vec, 0)?;
        Ok(VisionInput {
            data: pixel_values,
            grid_thw: vision_grid_thws,
        })
    }

    pub fn process_videos(
        &self,
        data: Vec<Tensor>,
        img_mean: &Tensor,
        img_std: &Tensor,
    ) -> Result<VisionInput> {
        let mut pixel_values_vec = Vec::new();
        let mut vision_grid_thws_vec = Vec::new();
        for single_video in data {
            // 0-255 rescale to 0-1
            let video_tensor = single_video.to_dtype(self.dtype)?.affine(1.0 / 255.0, 0.)?;
            // normalize
            let video_tensor = video_tensor
                .broadcast_sub(img_mean)?
                .broadcast_div(img_std)?
                .contiguous()?;
            // t
            let (video_tensor, video_grid_thw) = self.process_vision_tensor(&video_tensor)?;
            pixel_values_vec.push(video_tensor);
            vision_grid_thws_vec.push(video_grid_thw);
        }
        let pixel_values = Tensor::cat(&pixel_values_vec, 0)?.contiguous()?;
        let vision_grid_thws = Tensor::cat(&vision_grid_thws_vec, 0)?.contiguous()?;
        Ok(VisionInput {
            data: pixel_values,
            grid_thw: vision_grid_thws,
        })
    }

    #[allow(unused)]
    fn calculate_timestamps(
        &self,
        frames_indices: Vec<u32>,
        fps: f32,
        t_merge_size: usize,
    ) -> Result<Vec<f32>> {
        let indices = if !frames_indices.len().is_multiple_of(t_merge_size) {
            let mut frames_indices = frames_indices.clone();
            let last = frames_indices[frames_indices.len() - 1];
            let pad_len = t_merge_size - frames_indices.len() % t_merge_size;
            for _ in 0..pad_len {
                frames_indices.push(last);
            }
            frames_indices
        } else {
            frames_indices.clone()
        };
        let timestamps: Vec<f32> = indices.iter().map(|&x| x as f32 / fps).collect();
        let mut stamps = Vec::new();
        for i in (0..timestamps.len()).step_by(t_merge_size) {
            let stamp = (timestamps[i] + timestamps[i + t_merge_size - 1]) / 2.0;
            stamps.push(stamp);
        }
        Ok(stamps)
    }

    #[allow(unused)]
    pub fn process_info(
        &self,
        messages: &ChatCompletionParameters,
        text: &str,
    ) -> Result<GeneralInput> {
        let mut pixel_values = None;
        let mut image_grid_thw = None;
        let mut pixel_values_video = None;
        let mut video_grid_thw: Option<Tensor> = None;
        let mut video_metadata: Option<Vec<VideoMetadata>> = None;
        let vision_map = self.extract_vision_info(messages)?;
        let img_mean =
            Tensor::from_slice(&self.img_process_cfg.image_mean, (3, 1, 1), &self.device)?
                .to_dtype(self.dtype)?;
        let img_std = Tensor::from_slice(&self.img_process_cfg.image_std, (3, 1, 1), &self.device)?
            .to_dtype(self.dtype)?;
        for (key, vec) in vision_map {
            // println!("key: {}, \nvalue: {:?}", key, vec);
            if key.eq("image") {
                let mut file_vec = Vec::new();
                for file in &vec {
                    let image = get_image(file);
                    match image {
                        Ok(img) => file_vec.push(img),
                        Err(e) => println!("get_image err: {e:?}"),
                    };
                }
                if !file_vec.is_empty() {
                    let vision_input = self.process_images(file_vec, &img_mean, &img_std);
                    match vision_input {
                        Ok(img_input) => {
                            pixel_values = Some(img_input.data);
                            image_grid_thw = Some(img_input.grid_thw);
                        }
                        Err(e) => println!("img process_images err: {e:?}"),
                    };
                }
            }
            #[cfg(feature = "ffmpeg")]
            if key.eq("video") {
                let mut file_vec = Vec::new();
                let mut video_infos = Vec::new();
                for file in &vec {
                    let video_data = get_video_data(
                        file,
                        self.video_process_cfg.patch_size as u32,
                        self.video_process_cfg.temporal_patch_size as u32,
                        self.video_process_cfg.merge_size as u32,
                        self.fps,
                        self.min_frames,
                        self.max_frames,
                        self.video_process_cfg.size.shortest_edge as u32,
                        self.video_process_cfg.size.longest_edge as u32,
                        &self.device,
                    );
                    match video_data {
                        Ok((tensor, video_info)) => {
                            file_vec.push(tensor);
                            video_infos.push(video_info);
                        }
                        Err(e) => println!("get_video_data err: {:?}", e),
                    };
                }
                if !file_vec.is_empty() {
                    let vision_input = self.process_videos(file_vec, &img_mean, &img_std);
                    match vision_input {
                        Ok(video_input) => {
                            pixel_values_video = Some(video_input.data);
                            video_grid_thw = Some(video_input.grid_thw);
                            video_metadata = Some(video_infos);
                        }
                        Err(e) => println!("video process_videos err: {:?}", e),
                    };
                }
            }
        }
        let merge_length = self.img_process_cfg.merge_size.pow(2);
        let mut text = text.to_string();
        if let Some(ref image_grid_thw) = image_grid_thw {
            let mut index = 0;
            while text.contains(&self.image_token) {
                let grid_i = image_grid_thw.i(index)?;
                let repeat_num =
                    grid_i.to_vec1::<u32>()?.iter().product::<u32>() as usize / merge_length;
                let replace = "<|placeholder|>".repeat(repeat_num);
                text = text.replacen(&self.image_token, &replace, 1);
                index += 1;
            }
            text = text.replace("<|placeholder|>", &self.image_token);
        }
        #[cfg(feature = "ffmpeg")]
        if let Some(ref video_grid_thw) = video_grid_thw {
            let mut index = 0;
            while text.contains(&self.video_token) {
                let grid_i = video_grid_thw.i(index)?;
                let video_info = &video_metadata.as_ref().unwrap()[index];
                let curr_timestamp = self.calculate_timestamps(
                    video_info.frame_indices.clone(),
                    video_info.fps,
                    self.img_process_cfg.merge_size,
                )?;
                let mut video_placeholder = "".to_string();
                let [t, h, w] = grid_i.to_vec1::<u32>()?[..] else {
                    return Err(anyhow!(format!("grid_thw Expected exactly 3 elements")));
                };
                let frame_seqlen = h * w / merge_length as u32;
                for frame_idx in 0..t {
                    let curr_time = curr_timestamp[frame_idx as usize];
                    video_placeholder += format!("<{:.1} seconds>", curr_time).as_str();
                    video_placeholder += self.vision_start_token.as_str();
                    video_placeholder += "<|placeholder|>".repeat(frame_seqlen as usize).as_str();
                    video_placeholder += self.vision_end_token.as_str();
                }
                let three_token = format!(
                    "{}{}{}",
                    self.vision_start_token, self.video_token, self.vision_end_token
                );
                if text.contains(&three_token) {
                    text = text.replacen(&three_token, &video_placeholder, 1);
                } else {
                    text = text.replacen(&self.video_token, &video_placeholder, 1);
                }
                index += 1;
            }
            text = text.replace("<|placeholder|>", &self.video_token);
        }
        let input = GeneralInput {
            replace_text: text,
            pixel_values,
            image_grid_thw,
            pixel_values_video,
            video_grid_thw,
        };
        Ok(input)
    }
}

#[cfg(feature = "ffmpeg")]
pub fn get_video_data(
    file: &String,
    patch_size: u32,
    temporal_patch_size: u32,
    merge_size: u32,
    fps: u32,
    min_frames: u32,
    max_frames: u32,
    min_pixels: u32,
    max_pixels: u32,
    device: &Device,
) -> Result<(Tensor, VideoMetadata)> {
    ffmpeg::init().map_err(|e| anyhow!(format!("Failed to initialize ffmpeg: {}", e)))?;

    let mut ictx = ffmpeg::format::input(&file)
        .map_err(|e| anyhow!(format!("Failed to open video file: {}", e)))?;
    let input = ictx
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| anyhow!(format!("No video stream found")))?;
    let video_stream_index = input.index();
    let context_decoder = ffmpeg::codec::context::Context::from_parameters(input.parameters())
        .map_err(|e| anyhow!(format!("Failed to crate decoder context: {}", e)))?;
    let mut decoder = context_decoder
        .decoder()
        .video()
        .map_err(|e| anyhow!(format!("Failed to decoder video: {}", e)))?;

    let video_h = decoder.height();
    let video_w = decoder.width();
    let format = decoder.format();

    let frames = input.frames();
    let rate = input.rate().0 as f32 / input.rate().1 as f32;
    let duration = frames as f32 * 1.0 / rate;
    // 1s取两帧
    let nframes = (frames as f32 / rate * fps as f32).round() as u32;
    let nframes = std::cmp::min(
        std::cmp::min(std::cmp::max(nframes, min_frames), max_frames),
        frames as u32,
    );
    let sample_interval = (frames as f32 / nframes as f32).round() as u32;
    let mut frame_indices = Vec::new();
    let mut frame_id = 0_u32;

    // 图片帧使用scaler reshape的时候需要保证宽高是16的倍数,不然reshape出来的是损坏的图片
    // 所以计算resize的目标宽高时,需要用16和image_factor的最小公倍数
    let (resize_h, resize_w) = video_smart_resize(
        nframes,
        video_h,
        video_w,
        temporal_patch_size,
        patch_size * merge_size,
        min_pixels,
        max_pixels,
        Some(16),
    )?;
    let mut scaler = ffmpeg::software::scaling::context::Context::get(
        format,
        video_w,
        video_h,
        ffmpeg::format::Pixel::RGB24,
        resize_w,
        resize_h,
        ffmpeg::software::scaling::flag::Flags::BILINEAR
            | ffmpeg::software::scaling::flag::Flags::ACCURATE_RND,
    )
    .map_err(|e| anyhow!(format!("Failed to crate scaler: {}", e)))?;

    let mut frames_vec = Vec::new();
    let mut receive_and_process_decoded_frames =
        |decoder: &mut ffmpeg::decoder::Video| -> Result<()> {
            let mut decoded = ffmpeg::frame::Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                if frame_id.is_multiple_of(sample_interval) {
                    frame_indices.push(frame_id);
                    let mut rgb_frame = ffmpeg::frame::Video::empty();
                    scaler
                        .run(&decoded, &mut rgb_frame)
                        .map_err(|e| anyhow!(format!("Failed to scaler run decoded: {}", e)))?;

                    // save_file(&rgb_frame, frame_id as usize);
                    let frame_data = rgb_frame.data(0);
                    let frame_tensor = Tensor::from_slice(
                        frame_data,
                        (resize_h as usize, resize_w as usize, 3),
                        device,
                    )?
                    .permute((2, 0, 1))?;
                    frames_vec.push(frame_tensor);
                }
                frame_id += 1;
            }
            Ok(())
        };

    for (stream, packet) in ictx.packets() {
        if stream.index() == video_stream_index {
            decoder
                .send_packet(&packet)
                .map_err(|e| anyhow!(format!("Failed to send packet: {}", e)))?;
            receive_and_process_decoded_frames(&mut decoder)?;
        }
    }
    decoder
        .send_eof()
        .map_err(|e| anyhow!(format!("Failed to decoder.send_eof(): {}", e)))?;
    receive_and_process_decoded_frames(&mut decoder)?;

    if frames_vec.is_empty() {
        return Err(anyhow!("No frames extracted from video".to_string()));
    }
    // (t, c, h, w)
    let frames_tensor = Tensor::stack(&frames_vec, 0)?.contiguous()?;
    let video_info = VideoMetadata {
        total_num_frames: frames as u32,
        fps: rate,
        width: video_w,
        height: video_h,
        duration,
        frame_indices,
    };
    Ok((frames_tensor, video_info))
}
