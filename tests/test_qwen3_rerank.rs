use aha::models::{common::reranker::TextRerank, qwen3_reranker::Qwen3Reranker};
use anyhow::Result;
use std::time::Instant;

#[test]
fn qwen3_rerank() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_qwen3_rerank qwen3_rerank -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Qwen/Qwen3-Reranker-0.6B/", save_dir);

    let i_start = Instant::now();
    let mut model = Qwen3Reranker::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let docs = vec![
        "Rust async requests are commonly built with reqwest and tokio.".to_string(),
        "Paris is the capital of France.".to_string(),
    ];
    let input_texts = "How to make async HTTP calls in Rust?";
    let score = model.rerank(input_texts, &docs)?;
    println!("result: {:?}", score);

    Ok(())
}
