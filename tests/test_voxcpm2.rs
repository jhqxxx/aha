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
                        "text": "哎呀姐姐，好久没看到你了，来来来坐坐坐，我给你摆个龙门阵。你莫看我这两天闲得很，上个月我可是搞了个大事情。"
                    }
                ]
            }
        ],
        "metadata": {"control_instruction": "四川话,男生"}
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
