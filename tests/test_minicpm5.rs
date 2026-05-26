use std::time::Instant;

use aha::{
    models::{GenerateModel, minicpm5::generate::MiniCPM5GenerateModel},
    params::chat::ChatCompletionParameters,
};
use anyhow::Result;
#[test]
fn minicpm5_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_minicpm5 minicpm5_generate -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/MiniCPM5-1B/", save_dir);
    let message = r#"
    {
        "temperature": 0.3,
        "top_p": 0.8,
        "model": "minicpm4",
        "messages": [
            {
                "role": "user",
                "content": "什么是AI"
            }
        ],
        "enable_thinking": true
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = MiniCPM5GenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let res = model.generate(mes)?;
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        println!("usage: \n {:?}", usage);
    }
    Ok(())
}
