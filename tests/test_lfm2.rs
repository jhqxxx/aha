use aha::{
    models::{GenerateModel, lfm2::generate::Lfm2GenerateModel},
    params::chat::ChatCompletionParameters,
};
use anyhow::Result;
use rocket::futures::StreamExt;
use std::{pin::pin, time::Instant};

#[test]
fn lfm2_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_lfm2 lfm2_generate -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/LiquidAI/LFM2-1.2B/", save_dir);
    // let model_path = format!("{}/LiquidAI/LFM2.5-1.2B-Instruct/", save_dir);
    let message = r#"
    {
        "model": "lfm2",
        "messages": [
            {
                "role": "user",
                "content": "你是谁，你如何看待AI"
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = Lfm2GenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let result = model.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("generate: \n {:?}", result);
    if let Some(usage) = &result.usage {
        let num_token = usage.total_tokens;
        let duration_secs = i_duration.as_secs_f64();
        let tps = num_token as f64 / duration_secs;
        println!("Tokens per second (TPS): {:.2}", tps);
    }
    println!("Time elapsed in generate is: {:?}", i_duration);

    Ok(())
}

#[tokio::test]
async fn lfm2_stream() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_lfm2 lfm2_stream -r -- --nocapture
    // test with cuda+flash-attn: RUST_BACKTRACE=1 cargo test -F cuda,flash-attn qwen3_0_6b_generate -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    // let model_path = format!("{}/LiquidAI/LFM2-1.2B/", save_dir);
    let model_path = format!("{}/LiquidAI/LFM2.5-1.2B-Instruct/", save_dir);
    let message = r#"
    {
        "model": "lfm2",
        "messages": [
            {
                "role": "user",
                "content": "你如何看待AI"
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = Lfm2GenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    // let result = model.generate(mes)?;
    let mut stream = pin!(model.generate_stream(mes)?);
    let i_duration = i_start.elapsed();
    while let Some(token) = stream.next().await {
        println!("generate: \n {:?}", token);
    }
    // println!("generate: \n {:?}", result);
    // if let Some(usage) = &result.usage {
    //     let num_token = usage.total_tokens;
    //     let duration_secs = i_duration.as_secs_f64();
    //     let tps = num_token as f64 / duration_secs;
    //     println!("Tokens per second (TPS): {:.2}", tps);
    // }
    println!("Time elapsed in generate is: {:?}", i_duration);

    Ok(())
}
