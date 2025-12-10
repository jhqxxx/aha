use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, deepseek_ocr::generate::DeepseekOCRGenerateModel};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn deepseek_ocr_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda deepseek_ocr_generate -r -- --nocapture
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
    let model_path = "/home/jhq/huggingface_model/deepseek-ai/DeepSeek-OCR/";
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = DeepseekOCRGenerateModel::init(model_path, None, None)?;
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
async fn deepseek_ocr_stream() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda deepseek_ocr_stream -r -- --nocapture

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
    let model_path = "/home/jhq/huggingface_model/deepseek-ai/DeepSeek-OCR/";
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = DeepseekOCRGenerateModel::init(model_path, None, None)?;
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
