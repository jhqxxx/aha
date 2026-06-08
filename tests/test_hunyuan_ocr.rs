use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, hunyuan_ocr::generate::HunyuanOCRGenerateModel};
use aha::params::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn hunyuan_ocr_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_hunyuan_ocr hunyuan_ocr_generate -r -- --nocapture
    let message = r#"
    {
        "model": "Tencent-Hunyuan/HunyuanOCR",
        "messages": [
            {
                "role": "user",
                "content": [ 
                    {
                        "type": "image",
                        "image_url": 
                        {
                            "url": "file://./assets/img/ocr_test1.png"
                        }
                    },              
                    {
                        "type": "text", 
                        "text": "识别图片中的文字"
                    }
                ]
            }
        ]
    }
    "#;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Tencent-Hunyuan/HunyuanOCR/", save_dir);
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = HunyuanOCRGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let res = model.generate(mes)?;
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        println!("usage: \n {:?}", usage);
    }
    Ok(())
}

#[tokio::test]
async fn hunyuan_ocr_stream() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_hunyuan_ocr hunyuan_ocr_stream -r -- --nocapture

    let message = r#"
    {
        "model": "Tencent-Hunyuan/HunyuanOCR",
        "messages": [
            {
                "role": "user",
                "content": [ 
                    {
                        "type": "image",
                        "image_url": 
                        {
                            "url": "file://./assets/img/ocr_test1.png"
                        }
                    },              
                    {
                        "type": "text", 
                        "text": "检测并识别图片中的文字，将文本坐标格式化输出。"
                    }
                ]
            }
        ]
    }
    "#;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Tencent-Hunyuan/HunyuanOCR/", save_dir);
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = HunyuanOCRGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let mut stream = pin!(model.generate_stream(mes)?);
    while let Some(item) = stream.next().await {
        println!("generate: \n {:?}", item);
    }
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
