use std::time::Instant;

use aha::{
    models::{GenerateModel, voxcpm::generate::VoxCPMGenerate},
    params::chat::ChatCompletionParameters,
    utils::audio_utils::extract_and_save_audio_from_response,
};
use anyhow::Result;

#[test]
fn voxcpm2_use_message_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_voxcpm2  voxcpm2_use_message_generate -r -- --nocapture
    // control_instruction
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/VoxCPM2/", save_dir);
    let message = r#"
    {
        "model": "OpenBMB/VoxCPM2",
        "messages": [
            {
                "role": "user",
                "content": [  
                    {
                        "type": "text", 
                        "text": "老板儿，来碗担担面，多放海椒，再加个煎蛋哈"
                    }
                ]
            }
        ],
        "metadata": {"control_instruction": "萝莉音"}
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut voxcpm_generate = VoxCPMGenerate::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let generate = voxcpm_generate.generate(mes)?;
    let save_path = extract_and_save_audio_from_response(&generate, "./")?;
    for path in save_path {
        println!("save audio: {}", path);
    }
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
