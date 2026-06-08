use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, glm_asr_nano::generate::GlmAsrNanoGenerateModel};
use aha::params::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;

#[test]
fn glm_asr_nano_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_glm_asr_nano glm_asr_nano_generate -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/ZhipuAI/GLM-ASR-Nano-2512/", save_dir);
    let message = r#"
    {
        "model": "glm-asr-nano",
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
                        "text": "Please transcribe this audio into text"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut glm_asr_model = GlmAsrNanoGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let res = glm_asr_model.generate(mes)?;
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        println!("usage: \n {:?}", usage);
    }
    Ok(())
}

#[tokio::test]
async fn glm_asr_nano_stream() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_glm_asr_nano glm_asr_nano_stream -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/ZhipuAI/GLM-ASR-Nano-2512/", save_dir);
    let message = r#"
    {
        "model": "glm-asr-nano",
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
                        "text": "Please transcribe this audio into text"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut glm_asr_model = GlmAsrNanoGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let i_start = Instant::now();
    let mut stream = pin!(glm_asr_model.generate_stream(mes)?);
    while let Some(item) = stream.next().await {
        println!("generate: \n {:?}", item);
    }
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
