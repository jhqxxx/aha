use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use image::RgbaImage;
use rayon::prelude::*;

use crate::{
    models::rmbg2_0::model::BiRefNet,
    utils::{
        find_type_files, get_device, get_dtype,
        img_utils::{extract_images, float_tensor_to_dynamic_image, img_transform_with_resize},
    },
};

pub struct RMBG2_0 {
    model: BiRefNet,
    h: u32,
    w: u32,
    img_mean: Tensor,
    img_std: Tensor,
    device: Device,
    dtype: DType,
}

impl RMBG2_0 {
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
        Ok(Self {
            model,
            h: 1024,
            w: 1024,
            img_mean,
            img_std,
            device,
            dtype,
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

    pub fn generate(&self, mes: ChatCompletionParameters) -> Result<Vec<RgbaImage>> {
        let imgs = extract_images(&mes)?;
        if imgs.is_empty() {
            return Ok(vec![]);
        }

        // 并行预处理：提取原始尺寸和转换为 tensor
        let preprocessed: Vec<_> = imgs
            .par_iter()
            .map(|img| {
                let height = img.height();
                let width = img.width();
                let tensor = img_transform_with_resize(
                    img,
                    self.h,
                    self.w,
                    &self.img_mean,
                    &self.img_std,
                    &self.device,
                    self.dtype,
                );
                (img.clone(), height, width, tensor)
            })
            .collect();

        // 检查预处理是否有错误
        let mut tensors = Vec::with_capacity(preprocessed.len());
        let mut meta: Vec<_> = Vec::with_capacity(preprocessed.len());
        for (img, height, width, tensor_result) in preprocessed {
            let tensor = tensor_result?;
            tensors.push(tensor);
            meta.push((img, height, width));
        }

        // 批量推理：将所有图片合并为一个 batch
        let batch_tensor = Tensor::stack(&tensors, 0)?;
        let batch_output = self.model.forward(&batch_tensor)?;

        // 并行后处理：生成 RGBA 图像
        let results: Vec<Result<RgbaImage>> = meta
            .into_par_iter()
            .enumerate()
            .map(|(i, (img, height, width))| {
                let rmbg_tensor = batch_output.i(i)?;
                let alpha_img = float_tensor_to_dynamic_image(&rmbg_tensor)?;
                let alpha_img =
                    alpha_img.resize_exact(width, height, image::imageops::FilterType::CatmullRom);
                let alpha_gray = alpha_img.to_luma8();

                let rgb_img = img.to_rgb8();
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
