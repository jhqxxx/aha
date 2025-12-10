use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, hunyuan_ocr::generate::HunyuanOCRGenerateModel};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn hunyuan_ocr_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda hunyuan_ocr_generate -r -- --nocapture
    let message = r#"
    {
        "model": "hunyuan-ocr",
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
    let model_path = "/home/jhq/huggingface_model/Tencent-Hunyuan/HunyuanOCR/";
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = HunyuanOCRGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let i_start = Instant::now();
    let res = model.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("generate: \n {:?}", res);
    if res.usage.is_some() {
        let num_token = res.usage.as_ref().unwrap().total_tokens;
        let duration_secs = i_duration.as_secs_f64();
        let tps = num_token as f64 / duration_secs;
        println!("Tokens per second (TPS): {:.2}", tps);
    }
    println!("Time elapsed in generate is: {:?}", i_duration);
    
    Ok(())
}

#[tokio::test]
async fn hunyuan_ocr_stream() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda hunyuan_ocr_stream -r -- --nocapture

    let message = r#"
    {
        "model": "hunyuan-ocr",
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
    let model_path = "/home/jhq/huggingface_model/Tencent-Hunyuan/HunyuanOCR/";
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = HunyuanOCRGenerateModel::init(model_path, None, None)?;
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
