use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, minicpm4::generate::MiniCPMGenerateModel};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn minicpm_generate() -> Result<()> {
    // test with cpu :(太慢了, : RUST_BACKTRACE=1 cargo test minicpm_generate -r -- --nocapture
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda minicpm_generate -r -- --nocapture
    // test with cuda+flash-attn: RUST_BACKTRACE=1 cargo test -F cuda,flash-attn minicpm_generate -r -- --nocapture

    let model_path = "/home/jhq/huggingface_model/OpenBMB/MiniCPM4-0.5B/";
    let message = r#"
    {
        "temperature": 0.3,
        "top_p": 0.8,
        "model": "minicpm4",
        "messages": [
            {
                "role": "user",
                "content": "你吃饭了没"
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = MiniCPMGenerateModel::init(model_path, None, None)?;
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
async fn minicpm_stream() -> Result<()> {
    // test with cuda+flash-attn: RUST_BACKTRACE=1 cargo test -F cuda,flash-attn minicpm_stream -r -- --nocapture

    let model_path = "/home/jhq/huggingface_model/OpenBMB/MiniCPM4-0.5B/";

    let message = r#"
    {
        "model": "minicpm4",
        "messages": [
            {
                "role": "user",
                "content": "你是谁"
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = MiniCPMGenerateModel::init(model_path, None, None)?;
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
