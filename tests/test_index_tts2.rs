use std::time::Instant;

use aha::models::index_tts2::{generate::IndexTTS2Generate, utils::download_index_tts2_need_model};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

#[tokio::test]
async fn index_tts2_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda index_tts2_generate -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let _ = download_index_tts2_need_model(Some(&save_dir)).await?;
    let model_path = format!("{}/IndexTeam/IndexTTS-2", save_dir);
    let message = r#"
    {
        "model": "index-tts2",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": 
                        {
                            "url": "file:///home/jhq/Videos/voice_01.wav"
                        }
                    },              
                    {
                        "type": "text", 
                        "text": "你好啊"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut voxcpm_generate = IndexTTS2Generate::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let generate = voxcpm_generate.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
