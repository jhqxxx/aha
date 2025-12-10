use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, qwen3vl::generate::Qwen3VLGenerateModel};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn qwen3vl_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda,ffmpeg qwen3vl_generate -r -- --nocapture

    let model_path = "/home/jhq/huggingface_model/Qwen/Qwen3-VL-2B-Instruct/";

    let message = r#"
    {
        "model": "qwen3vl",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video_url": 
                        {
                            "url": "./assets/video/video_test.mp4"
                        }
                    },              
                    {
                        "type": "text", 
                        "text": "视频中发生了什么？, 现在几点了"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut qwen3vl = Qwen3VLGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let res = qwen3vl.generate(mes)?;
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
async fn qwen3vl_stream() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda,ffmpeg qwen3vl_stream -r -- --nocapture

    let model_path = "/home/jhq/huggingface_model/Qwen/Qwen3-VL-2B-Instruct/";

    let message = r#"
    {
        "model": "qwen3vl",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video_url": 
                        {
                            "url": "./assets/video/video_test.mp4"
                        }
                    },              
                    {
                        "type": "text", 
                        "text": "视频中发生了什么？"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut qwen3vl = Qwen3VLGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let mut stream = pin!(qwen3vl.generate_stream(mes)?);
    while let Some(item) = stream.next().await {
        println!("generate: \n {:?}", item);
    }
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
