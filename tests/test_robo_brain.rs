use std::time::Instant;

use aha::models::{GenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

#[test]
fn robo_brain_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda robo_brain_generate -r -- --nocapture

    let model_path = "/home/jhq/huggingface_model/BAAI/RoboBrain2.0-3B/";

    let message = r#"
    {
        "model": "qwen2.5vl",
        "messages": [
            {
                "role": "user",
                "content": [           
                    {
                        "type": "text", 
                        "text": "hello RoboBrain"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = Qwen2_5VLGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let result = model.generate(mes)?;
    println!("generate: \n {:?}", result);
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);

    Ok(())
}
