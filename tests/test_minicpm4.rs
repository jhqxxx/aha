use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, minicpm4::generate::MiniCPM4GenerateModel};
use aha::params::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn minicpm_generate() -> Result<()> {
    // test with cpu :(太慢了, : RUST_BACKTRACE=1 cargo test minicpm_generate -r -- --nocapture
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_minicpm4 minicpm_generate -r -- --nocapture
    // test with cuda+flash-attn: RUST_BACKTRACE=1 cargo test -F cuda,flash-attn minicpm_generate -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/MiniCPM4-0.5B/", save_dir);
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
    let mut model = MiniCPM4GenerateModel::init(&model_path, None, None)?;
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
async fn minicpm_stream() -> Result<()> {
    // test with cuda+flash-attn: RUST_BACKTRACE=1 cargo test -F cuda,flash-attn minicpm_stream -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/MiniCPM4-0.5B/", save_dir);

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
    let mut model = MiniCPM4GenerateModel::init(&model_path, None, None)?;
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
