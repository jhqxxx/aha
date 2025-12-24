use std::time::Instant;

use aha::models::rmbg2_0::generate::RMBG2_0;
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

#[test]
fn rmbg2_0_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda rmbg2_0_generate -r -- --nocapture

    let model_path = "/home/jhq/huggingface_model/AI-ModelScope/RMBG-2.0/";

    let message = r#"
    {
        "model": "rmbg2.0",
        "messages": [
            {
                "role": "user",
                "content": [ 
                    {
                        "type": "image",
                        "image_url": 
                        {
                            "url": "file://./assets/img/gougou.jpg"
                        }
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let model = RMBG2_0::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let result = model.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    for (i, img) in result.iter().enumerate() {
        let _ = img.save(format!("rmbg_{i}.png"));
    }

    Ok(())
}
