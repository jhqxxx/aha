use std::time::Instant;

use aha::params::chat::ChatCompletionParameters;
use aha::{
    models::{
        GenerateModel,
        voxcpm::{generate::VoxCPMGenerate, tokenizer::SingleChineseTokenizer},
    },
    utils::audio_utils::{extract_and_save_audio_from_response, save_wav},
};
use anyhow::{Ok, Result};

#[test]
fn voxcpm_use_message_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda voxcpm_use_message_generate -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/VoxCPM-0.5B/", save_dir);
    let message = r#"
    {
        "model": "voxcpm",
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
                        "text": "VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech."
                    }
                ]
            }
        ],
        "metadata": {"prompt_text": "天雷滚滚我好怕怕，劈得我浑身掉渣渣。突破天劫我笑哈哈，逆天改命我吹喇叭，滴答滴答滴滴答"}
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
    // save_wav(&generate, "voxcpm.wav", 16000)?;
    Ok(())
}

#[test]
fn voxcpm_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_voxcpm voxcpm_generate -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/VoxCPM-0.5B/", save_dir);

    let i_start = Instant::now();
    let mut voxcpm_generate = VoxCPMGenerate::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    // let i_start = Instant::now();
    // let generate = voxcpm_generate.generate_simple("太阳当空照，花儿对我笑，小鸟说早早早".to_string())?;
    // let generate = voxcpm_generate.inference(
    //     "VoxCPM is an innovative end-to-end TTS model from ModelBest, designed to generate highly realistic speech.".to_string(),
    //     Some("啥子小师叔，打狗还要看主人，你再要继续，我，就是你的对手".to_string()),
    //     Some("file://./assets/audio/voice_01.wav".to_string()),
    //     // Some("一定被灰太狼给吃了，我已经为他准备好了花圈了".to_string()),
    //     // Some("file://./assets/audio/voice_05.wav".to_string()),
    //     2,
    //     100,
    //     10,
    //     2.0,
    //     // false,
    //     6.0,
    // )?;

    // 创建prompt_cache
    voxcpm_generate.build_prompt_cache(
        "啥子小师叔，打狗还要看主人，你再要继续，我，就是你的对手".to_string(),
        "file://./assets/audio/voice_01.wav".to_string(),
    )?;
    // 使用prompt_cache生成语音
    let i_start = Instant::now();
    let generate = voxcpm_generate.generate_use_prompt_cache(
        "太阳当空照，花儿对我笑，小鸟说早早早".to_string(),
        2,
        100,
        10,
        2.0,
        false,
        6.0,
    )?;

    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    save_wav(&generate, "voxcpm.wav", 16000)?;
    Ok(())
}

#[test]
fn voxcpm_tokenizer() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/VoxCPM-0.5B/", save_dir);
    let tokenizer = SingleChineseTokenizer::new(&model_path)?;
    let ids = tokenizer.encode("你好啊，你吃饭了吗".to_string())?;
    println!("ids: {:?}", ids);
    Ok(())
}
