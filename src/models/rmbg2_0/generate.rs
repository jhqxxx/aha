use std::io::Cursor;

use aha_openai_dive::v1::resources::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse,
};
use anyhow::Result;
use base64::{Engine, prelude::BASE64_STANDARD};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::RgbaImage;
use rayon::prelude::*;
use rocket::futures::{Stream, stream};

use crate::{
    models::{GenerateModel, rmbg2_0::model::BiRefNet},
    utils::{
        build_img_completion_response, find_type_files, get_device, get_dtype,
        img_utils::{extract_images, float_tensor_to_dynamic_image, img_transform_with_resize},
    },
};

pub struct RMBG2_0Model {
    model: BiRefNet,
    h: u32,
    w: u32,
    img_mean: Tensor,
    img_std: Tensor,
    device: Device,
    dtype: DType,
    model_name: String,
}

impl RMBG2_0Model {
    pub fn init(path: &str, device: Option<&Device>, dtype: Option<DType>) -> Result<Self> {
        let device = get_device(device);
        let dtype = get_dtype(dtype, "float32");
        let model_list = find_type_files(path, "safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, &device)? };
        let model = BiRefNet::new(vb)?;
        let img_mean =
            Tensor::from_slice(&[0.485, 0.456, 0.406], (3, 1, 1), &device)?.to_dtype(dtype)?;
        let img_std =
            Tensor::from_slice(&[0.229, 0.224, 0.225], (3, 1, 1), &device)?.to_dtype(dtype)?;
        let model_name = std::path::Path::new(path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("rmbg2.0")
            .to_string();
        Ok(Self {
            model,
            h: 1024,
            w: 1024,
            img_mean,
            img_std,
            device,
            dtype,
            model_name,
        })
    }

    #[cfg(test)]
    pub fn h(&self) -> u32 {
        self.h
    }

    #[cfg(test)]
    pub fn w(&self) -> u32 {
        self.w
    }

    #[cfg(test)]
    pub fn img_mean(&self) -> &Tensor {
        &self.img_mean
    }

    #[cfg(test)]
    pub fn img_std(&self) -> &Tensor {
        &self.img_std
    }

    #[cfg(test)]
    pub fn device(&self) -> &Device {
        &self.device
    }
    #[cfg(test)]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[cfg(test)]
    pub fn model(&self) -> &BiRefNet {
        &self.model
    }
    pub fn inference(&self, mes: ChatCompletionParameters) -> Result<Vec<RgbaImage>> {
        let imgs = extract_images(&mes)?;
        if imgs.is_empty() {
            return Ok(vec![]);
        }

        // 并行预处理：提取原始尺寸、RGB 数据和转换为 tensor
        let preprocessed: Vec<_> = imgs
            .par_iter()
            .map(|img| {
                let height = img.height();
                let width = img.width();
                let rgb_img = img.to_rgb8();
                let tensor = img_transform_with_resize(
                    img,
                    self.h,
                    self.w,
                    &self.img_mean,
                    &self.img_std,
                    &self.device,
                    self.dtype,
                );
                (rgb_img, height, width, tensor)
            })
            .collect();

        // 检查预处理是否有错误
        let mut tensors = Vec::with_capacity(preprocessed.len());
        let mut meta: Vec<_> = Vec::with_capacity(preprocessed.len());
        for (rgb_img, height, width, tensor_result) in preprocessed {
            let tensor = tensor_result?;
            tensors.push(tensor);
            meta.push((rgb_img, height, width));
        }

        // 批量推理：将所有图片合并为一个 batch
        // to guobin211: 感谢你贡献的代码，不过现在模型中可变形卷积的实现只支持batch_size=1，所以推理还是用的循环QaQ
        // let batch_tensor = Tensor::stack(&tensors, 0)?;
        // let batch_output = self.model.forward(&batch_tensor)?;
        let mut batch_output = vec![];
        for img_tensor in tensors {
            let output = self.model.forward(&img_tensor.unsqueeze(0)?)?.squeeze(0)?;
            batch_output.push(output);
        }

        // 并行后处理：生成 RGBA 图像
        let results: Vec<Result<RgbaImage>> = meta
            .into_par_iter()
            .enumerate()
            .map(|(i, (rgb_img, height, width))| {
                // let rmbg_tensor = batch_output.i(i)?;
                let rmbg_tensor = &batch_output[i];
                let alpha_img = float_tensor_to_dynamic_image(rmbg_tensor)?;
                let alpha_img =
                    alpha_img.resize_exact(width, height, image::imageops::FilterType::CatmullRom);
                let alpha_gray = alpha_img.to_luma8();

                let rgb_raw = rgb_img.as_raw();
                let alpha_raw = alpha_gray.as_raw();
                let pixel_count = (width * height) as usize;
                let mut rgba_raw = vec![0u8; pixel_count * 4];

                // 并行分块写入
                rgba_raw
                    .par_chunks_mut(4)
                    .enumerate()
                    .for_each(|(idx, chunk)| {
                        let src = idx * 3;
                        chunk[0] = rgb_raw[src];
                        chunk[1] = rgb_raw[src + 1];
                        chunk[2] = rgb_raw[src + 2];
                        chunk[3] = alpha_raw[idx];
                    });

                RgbaImage::from_raw(width, height, rgba_raw)
                    .ok_or_else(|| anyhow::anyhow!("Failed to create RGBA image"))
            })
            .collect();

        results.into_iter().collect()
    }
}

impl GenerateModel for RMBG2_0Model {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let rmbg_png = self.inference(mes)?;
        let mut base64_vec = vec![];
        for img in rmbg_png {
            let mut png_bytes = Vec::new();
            img.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)?;
            let base64_string = BASE64_STANDARD.encode(png_bytes);
            base64_vec.push(base64_string);
        }
        let response = build_img_completion_response(&base64_vec, &self.model_name);
        Ok(response)
    }
    #[allow(unused_variables)]
    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        let error_stream = stream::once(async {
            Err(anyhow::anyhow!(format!(
                "{} model not support stream",
                self.model_name
            ))) as Result<ChatCompletionChunkResponse, anyhow::Error>
        });

        Ok(Box::new(Box::pin(error_stream)))
    }
}
