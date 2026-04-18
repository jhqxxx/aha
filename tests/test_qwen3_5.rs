use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, qwen3_5::generate::Qwen3_5GenerateModel};
use aha::params::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn qwen3_5_generate_no_visual() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_qwen3_5 qwen3_5_generate_no_visual -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Qwen/Qwen3.5-0.8B/", save_dir);

    let message = r#"
    {
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": "你好啊"
            }
        ]
    }
    "#;
    // "metadata": {"enable_thinking": "true"}
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut qwen3_5 = Qwen3_5GenerateModel::init_without_visual(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let text = qwen3_5.generate_text(mes)?;
    println!("text: {}", text);
    // let res = qwen3_5.generate(mes)?;
    // println!("generate: \n {:?}", res);
    // if let Some(usage) = &res.usage {
    //     println!("usage: \n {:?}", usage);
    // }
    Ok(())
}

#[test]
fn qwen3_5_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_qwen3_5 qwen3_5_generate -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Qwen/Qwen3.5-0.8B/", save_dir);

    let message = r#"
    {
        "model": "qwen3.5",
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
                        "text": "OCR"
                    }
                ]
            }
        ],
        "enable_thinking": true
    }
    "#;
    // "metadata": {"enable_thinking": "true"}
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut qwen3_5 = Qwen3_5GenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let res = qwen3_5.generate(mes)?;
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        println!("usage: \n {:?}", usage);
    }
    Ok(())
}

#[tokio::test]
async fn qwen3_5_stream() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_qwen3_5 qwen3_5_stream -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Qwen/Qwen3.5-0.8B/", save_dir);

    let message = r#"
    {
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_url": 
                        {
                            "url": "file://./assets/img/ocr_test3.png"
                        }
                    },             
                    {
                        "type": "text", 
                        "text": "OCR"
                    }
                ]
            }
        ],
        "enable_thinking": true
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut qwen3_5 = Qwen3_5GenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let mut stream = pin!(qwen3_5.generate_stream(mes)?);
    while let Some(item) = stream.next().await {
        println!("generate: \n {:?}", item);
    }
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
