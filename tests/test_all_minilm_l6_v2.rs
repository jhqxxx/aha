use aha::models::{
    all_minilm_l6_v2::AllMiniLML6V2Embedding, common::embedding::TextEmbedding,
};
use anyhow::Result;
use std::time::Instant;

#[test]
fn all_minilm_l6_v2_embedding() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_all_minilm_l6_v2 all_minilm_l6_v2_embedding -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/sentence-transformers/all-MiniLM-L6-v2/", save_dir);

    let i_start = Instant::now();
    let mut model = AllMiniLML6V2Embedding::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let input_texts = ["test ALL_MINILM_L6_V2 embedding".to_string()];
    let result = model.embed_texts(&input_texts)?;
    println!("result: {:?}", result);

    Ok(())
}
