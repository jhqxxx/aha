use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, fun_asr_nano::generate::FunAsrNanoGenerateModel};
use aha::params::chat::ChatCompletionParameters;
use anyhow::Result;
use rocket::futures::StreamExt;
#[test]
fn fun_asr_nano_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_fun_asr_nano fun_asr_nano_generate -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/FunAudioLLM/Fun-ASR-Nano-2512/", save_dir);
    let message = r#"
    {
        "model": "fun-asr-nano",
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
                        "text": "语音转写："
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut fun_asr_model = FunAsrNanoGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let res = fun_asr_model.generate(mes)?;
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        println!("usage: \n {:?}", usage);
    }
    Ok(())
}

#[tokio::test]
async fn fun_asr_nano_stream() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_fun_asr_nano fun_asr_nano_stream -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/FunAudioLLM/Fun-ASR-Nano-2512/", save_dir);
    let message = r#"
    {
        "model": "fun-asr-nano",
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
                        "text": "语音转写："
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut fun_asr_model = FunAsrNanoGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let i_start = Instant::now();
    let mut stream = pin!(fun_asr_model.generate_stream(mes)?);
    while let Some(item) = stream.next().await {
        println!("generate: \n {:?}", item);
    }
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
