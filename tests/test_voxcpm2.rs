use std::time::Instant;

use aha::{
    models::{
        GenerateModel, voxcpm::generate::VoxCPMGenerate,
        voxcpm_refact::generate::VoxCPMGenerateRefact,
    },
    params::chat::ChatCompletionParameters,
    utils::audio_utils::{extract_and_save_audio_from_response, save_wav_mono},
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
                        "type": "audio",
                        "audio_url": 
                        {
                            "url": "https://package-release.coderbox.cn/aiway/test/other/%E5%93%AA%E5%90%92.wav"
                        }
                    },  
                    {
                        "type": "text", 
                        "text": "aha是一个基于Rust和Candle框架的本地AI推理引擎，支持多模态模型（文本、视觉、语音、OCR）。"
                    }
                ]
            }
        ],
        "metadata": {"prompt_text": "天雷滚滚我好怕怕，劈得我浑身掉渣渣。突破天劫我笑哈哈，逆天改命我吹喇叭，滴答滴答滴滴答"}
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut voxcpm_generate = VoxCPMGenerateRefact::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    // for _ in 0..10 {
    //     let _ = voxcpm_generate.generate(mes.clone())?;
    // }
    // let mut times = vec![];
    // for _ in 0..100 {
    //     let start = Instant::now();
    //     let _ = voxcpm_generate.generate(mes.clone())?;
    //     times.push(start.elapsed());
    // }
    // let mean = times.iter().sum::<Duration>() / 100;
    // println!("mean: {:?}", mean);
    // times.sort();
    // println!("p99: {:?}", times[99]);
    let i_start = Instant::now();
    let generate = voxcpm_generate.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    let save_path = extract_and_save_audio_from_response(&generate, "./")?;
    for path in save_path {
        println!("save audio: {}", path);
    }
    Ok(())
}

#[test]
fn voxcpm2_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_voxcpm2 voxcpm2_generate -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/VoxCPM2/", save_dir);

    let i_start = Instant::now();
    let mut voxcpm_generate = VoxCPMGenerate::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let generate = voxcpm_generate.inference(
        "aha是一个基于Rust和Candle框架的本地AI推理引擎，支持多模态模型（文本、视觉、语音、OCR）。"
            .to_string(),
        None,
        None,
        2,
        1000,
        10,
        2.0,
        6.0,
    )?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);

    save_wav_mono(
        &generate,
        "voxcpm2.wav",
        voxcpm_generate.sample_rate() as u32,
    )?;
    Ok(())
}
