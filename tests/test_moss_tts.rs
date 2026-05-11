use aha::models::moss_tts_nano::generate::MossTTSGenerate;
use anyhow::Result;

#[test]
fn moss_tts() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_moss_tts moss_tts -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let tts_path = format!("{}/openmoss/MOSS-TTS-Nano/", save_dir);
    let audio_tokenizer_path = format!("{}/openmoss/MOSS-Audio-Tokenizer-Nano/", save_dir);
    let mut model = MossTTSGenerate::init(&tts_path, &audio_tokenizer_path, None, None)?;
    let _ = model.generate(
        "你在干吗啊",
        // None,
        Some("file://./assets/audio/jiangjiang.wav"),
        None,
        // Some("哈喽大家好，我是蒋蒋"),
        // Some(aha::models::moss_tts_nano::model::MossTTSMode::Continuation),
        None,
    )?;
    Ok(())
}
