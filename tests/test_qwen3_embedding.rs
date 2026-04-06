use aha::models::{common::embedding::TextEmbedding, qwen3_embedding::Qwen3Embedding};
use anyhow::Result;
use std::time::Instant;

#[test]
fn qwen3_embedding() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_qwen3_embedding qwen3_embedding -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Qwen/Qwen3-Embedding-0.6B/", save_dir);

    let i_start = Instant::now();
    let mut model = Qwen3Embedding::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let input_texts = ["test Qwen3-Embedding-0.6B".to_string()];
    let result = model.embed_texts(&input_texts)?;
    println!("result: {:?}", result);

    Ok(())
}
