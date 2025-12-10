use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn qwen2_5vl_generate() -> Result<()> {
    // test with cpu :(太慢了, : RUST_BACKTRACE=1 cargo test qwen2_5vl_generate -r -- --nocapture
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda qwen2_5vl_generate -r -- --nocapture
    // test with cuda+flash-attn: RUST_BACKTRACE=1 cargo test -F cuda,flash-attn qwen2_5vl_generate -r -- --nocapture
    // let device = Device::cuda_if_available(0)?;
    // let dtype = DType::BF16;

    let model_path = "/home/jhq/huggingface_model/Qwen/Qwen2.5-VL-3B-Instruct/";

    let message = r#"
    {
        "model": "qwen2.5vl",
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
                        "text": "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = Qwen2_5VLGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let result = model.generate(mes)?;    
    let i_duration = i_start.elapsed();
    println!("generate: \n {:?}", result);
    if result.usage.is_some() {
        let num_token = result.usage.as_ref().unwrap().total_tokens;
        let duration_secs = i_duration.as_secs_f64();
        let tps = num_token as f64 / duration_secs;
        println!("Tokens per second (TPS): {:.2}", tps);
    }
    println!("Time elapsed in generate is: {:?}", i_duration);

    Ok(())
}

#[tokio::test]
async fn qwen2_5vl_stream() -> Result<()> {
    // test with cuda+flash-attn: RUST_BACKTRACE=1 cargo test -F cuda,flash-attn qwen2_5vl_generate -r -- --nocapture
    // let device = Device::cuda_if_available(0)?;
    // let dtype = DType::BF16;

    let model_path = "/home/jhq/huggingface_model/Qwen/Qwen2.5-VL-3B-Instruct/";

    let message = r#"
    {
        "model": "qwen2.5vl",
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
                        "text": "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = Qwen2_5VLGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let mut stream = pin!(model.generate_stream(mes)?);
    while let Some(item) = stream.next().await {
        println!("generate: \n {:?}", item);
    }

    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);

    Ok(())
}
