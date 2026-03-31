use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, deepseek_ocr::generate::DeepseekOCRGenerateModel};
use aha::params::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn deepseek_ocr2_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_deepseek_ocr deepseek_ocr2_generate -r -- --nocapture
    let message = r#"
    {
        "model": "deepseek-ocr2",
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
                        "text": "<image>\nConvert the document to markdown. "
                    }
                ]
            }
        ],
        "metadata": {"crop_mode": "false"}
    }
    "#;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/deepseek-ai/DeepSeek-OCR-2/", save_dir);
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = DeepseekOCRGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let i_start = Instant::now();
    let res = model.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        let num_token = usage.total_tokens;
        let duration_secs = i_duration.as_secs_f64();
        let tps = num_token as f64 / duration_secs;
        println!("Tokens per second (TPS): {:.2}", tps);
    }
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}

#[test]
fn deepseek_ocr_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_deepseek_ocr deepseek_ocr_generate -r -- --nocapture
    let message = r#"
    {
        "model": "deepseek-ocr",
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
                        "text": "<image>\nConvert the document to markdown. "
                    }
                ]
            }
        ],
        "metadata": {"base_size": "640", "image_size": "640", "crop_mode": "false"}
    }
    "#;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/deepseek-ai/DeepSeek-OCR/", save_dir);
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = DeepseekOCRGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let i_start = Instant::now();
    let res = model.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        let num_token = usage.total_tokens;
        let duration_secs = i_duration.as_secs_f64();
        let tps = num_token as f64 / duration_secs;
        println!("Tokens per second (TPS): {:.2}", tps);
    }
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}

#[tokio::test]
async fn deepseek_ocr_stream() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_deepseek_ocr deepseek_ocr_stream -r -- --nocapture

    let message = r#"
    {
        "model": "deepseek-ocr",
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
                        "text": "<image>\n<|grounding|>Convert the document to markdown. "
                    }
                ]
            },
            {
                "role": "assistant",
                "content": ""
            }
        ],
        "metadata": {"base_size": "640", "image_size": "640", "crop_mode": "false"}
    }
    "#;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/deepseek-ai/DeepSeek-OCR/", save_dir);
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = DeepseekOCRGenerateModel::init(&model_path, None, None)?;
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
