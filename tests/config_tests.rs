use aha::models::{
    deepseek_ocr::config::DeepseekOCRConfig, hunyuan_ocr::config::HunYuanVLConfig,
    lfm2::config::Lfm2Config, minicpm4::config::MiniCPM4Config,
    paddleocr_vl::config::PaddleOCRVLConfig, qwen2_5vl::config::Qwen2_5VLConfig,
    qwen3vl::config::Qwen3VLConfig, voxcpm::config::VoxCPMConfig,
};
use anyhow::Result;

#[test]
fn qwen2_5_vl_config() -> Result<()> {
    // cargo test -F cuda,flash-attn qwen2_5vl_config -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/Qwen/Qwen2.5-VL-3B-Instruct/";
    let config_path = model_path.to_string() + "/config.json";
    let config: Qwen2_5VLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}

#[test]
fn minicpm4_config() -> Result<()> {
    // cargo test -F cuda,flash-attn minicpm4_config -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/OpenBMB/MiniCPM4-0.5B/";
    let config_path = model_path.to_string() + "/config.json";
    let config: MiniCPM4Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}

#[test]
fn voxcpm_config() -> Result<()> {
    // cargo test -F cuda,flash-attn voxcpm_config -r -- --nocapture
    // cargo test -F cuda voxcpm_config -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/openbmb/VoxCPM-0.5B/";
    let config_path = model_path.to_string() + "/config.json";
    let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}

#[test]
fn voxcpm1_5_config() -> Result<()> {
    // cargo test -F cuda voxcpm1_5_config -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/OpenBMB/VoxCPM1.5/";
    let config_path = model_path.to_string() + "/config.json";
    let config: VoxCPMConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}

#[test]
fn qwen3vl_config() -> Result<()> {
    // cargo test -F cuda qwen3vl_config -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/Qwen/Qwen3-VL-4B-Instruct/";
    let config_path = model_path.to_string() + "/config.json";
    let config: Qwen3VLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}

#[test]
fn deepseek_ocr_config() -> Result<()> {
    // cargo test -F cuda deepseek_ocr_config -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/deepseek-ai/DeepSeek-OCR/";
    let config_path = model_path.to_string() + "/config.json";
    let config: DeepseekOCRConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}

#[test]
fn hunyuan_ocr_config() -> Result<()> {
    // cargo test -F cuda hunyuan_ocr_config -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/Tencent-Hunyuan/HunyuanOCR/";
    let config_path = model_path.to_string() + "/config.json";
    let config: HunYuanVLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}
#[test]
fn paddleocr_vl_config() -> Result<()> {
    // cargo test -F cuda paddleocr_vl_config -r -- --nocapture
    let model_path = "/home/jhq/huggingface_model/PaddlePaddle/PaddleOCR-VL/";
    let config_path = model_path.to_string() + "/config.json";
    let config: PaddleOCRVLConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    Ok(())
}

#[test]
fn lfm2_config() -> Result<()> {
    // cargo test -F cuda --test config_tests lfm2_config -r -- --nocapture
    let model_path = "/home/jhq/.aha/LiquidAI/LFM2-1.2B/";
    // let model_path = "/home/jhq/.aha/LiquidAI/LFM2.5-1.2B-Instruct/";
    let config_path = model_path.to_string() + "/config.json";
    let mut config: Lfm2Config = serde_json::from_slice(&std::fs::read(config_path)?)?;
    println!("{:?}", config);
    config.full_attn_idx2layer_type();
    println!("{:?}", config);
    Ok(())
}
