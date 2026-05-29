use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, glm_ocr::generate::GlmOcrGenerateModel};
use aha::params::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn glm_ocr_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_glm_ocr glm_ocr_generate -r -- --nocapture
    let message = r#"
    {
        "model": "ZhipuAI/GLM-OCR",
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
                        "text": "Text Recognition:"
                    }
                ]
            }
        ],
        "repeat_penalty": 1.2,
        "repeat_last_n": 64
    }
    "#;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/ZhipuAI/GLM-OCR/", save_dir);
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = GlmOcrGenerateModel::init(&model_path, None, None)?;
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
async fn glm_ocr_stream() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_glm_ocr glm_ocr_stream -r -- --nocapture

    let message = r#"
    {
        "model": "ZhipuAI/GLM-OCR",
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
                        "text": "Text Recognition:"
                    }
                ]
            }
        ]
    }
    "#;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/ZhipuAI/GLM-OCR/", save_dir);
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = GlmOcrGenerateModel::init(&model_path, None, None)?;
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
