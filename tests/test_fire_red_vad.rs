use aha::models::fire_red_vad::vad::FireRedVad;
use anyhow::Result;

#[test]
fn aed() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_fire_red_vad aed -r -- --nocapture
    let audio_path = "file:///home/jhq/python_code/FireRedASR2S/assets/hello_zh.wav";
    let device = aha::Device::Cpu;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/xukaituo/FireRedVAD/AED/", save_dir);

    let vad = FireRedVad::init(&model_path, Some(&device), None)?;
    let res = vad.detect_file(audio_path)?;
    println!("vad res: {:?}", res);
    Ok(())
}

#[test]
fn stream_vad() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_fire_red_vad stream_vad -r -- --nocapture
    let audio_path = "file:///home/jhq/python_code/FireRedASR2S/assets/hello_zh.wav";
    let device = aha::Device::Cpu;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/jiangjiangaha/FireRedVAD-Stream-VAD/", save_dir);

    let vad = FireRedVad::init(&model_path, Some(&device), None)?;
    let res = vad.detect_file(audio_path)?;
    println!("vad res: {:?}", res);
    Ok(())
}

#[test]
fn vad() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_fire_red_vad vad -r -- --nocapture
    let audio_path = "file:///home/jhq/python_code/FireRedASR2S/assets/hello_zh.wav";
    let device = aha::Device::Cpu;
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/xukaituo/FireRedVAD/VAD/", save_dir);

    let vad = FireRedVad::init(&model_path, Some(&device), None)?;
    let res = vad.detect_file(audio_path)?;
    println!("vad res: {:?}", res);
    Ok(())
}
