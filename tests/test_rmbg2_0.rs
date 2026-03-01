use std::time::Instant;

use aha::models::rmbg2_0::generate::RMBG2_0Model;
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

#[test]
fn rmbg2_0_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda --test test_rmbg2_0 rmbg2_0_generate -r -- --nocapture

    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/AI-ModelScope/RMBG-2.0/", save_dir);

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
    let model = RMBG2_0Model::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let result = model.inference(mes)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    for (i, img) in result.iter().enumerate() {
        let _ = img.save(format!("rmbg_{i}.png"));
    }

    Ok(())
}
