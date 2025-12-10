use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, paddleocr_vl::generate::PaddleOCRVLGenerateModel};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn paddleocr_vl_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda paddleocr_vl_generate -r -- --nocapture
    let message = r#"
    {
        "model": "paddleocr_vl",
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
                        "text": "OCR:"
                    }
                ]
            }
        ],
        "stream": false
    }
    "#;
    let model_path = "/home/jhq/huggingface_model/PaddlePaddle/PaddleOCR-VL/";
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = PaddleOCRVLGenerateModel::init(model_path, None, None)?;
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
async fn paddleocr_vl_stream() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda paddleocr_vl_stream -r -- --nocapture

    let message = r#"
    {
        "model": "paddleocr_vl",
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
                        "text": "OCR:"
                    }
                ]
            }
        ]
    }
    "#;
    let model_path = "/home/jhq/huggingface_model/PaddlePaddle/PaddleOCR-VL/";
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = PaddleOCRVLGenerateModel::init(model_path, None, None)?;
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
