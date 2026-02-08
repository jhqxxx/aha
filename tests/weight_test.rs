use std::collections::HashMap;

use aha::utils::{find_type_files, get_device, read_pth_tensor_info_cycle};
use anyhow::Result;
use candle_core::{Device, pickle::read_all_with_key, safetensors};
use candle_nn::VarBuilder;

#[test]
fn minicpm4_weight() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/MiniCPM4-0.5B/", save_dir);
    let model_list = find_type_files(&model_path, "safetensors")?;
    let device = Device::Cpu;
    for m in model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            println!("=== {} ===", key);
            println!("Shape: {:?}", tensor.shape());
            println!("DType: {:?}", tensor.dtype());
        }
    }
    Ok(())
}

#[test]
fn voxcpm_weight() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/VoxCPM-0.5B/", save_dir);
    let model_list = find_type_files(&model_path, "pth")?;
    println!("model_list: {:?}", model_list);
    let dev = get_device(None);
    let mut dict_to_hashmap = HashMap::new();
    let mut dtype = candle_core::DType::F16;
    for m in model_list {
        let dict = read_all_with_key(m, Some("state_dict"))?;
        dtype = dict[0].1.dtype();
        for (k, v) in dict {
            println!("key: {}, tensor shape: {:?}", k, v);
            dict_to_hashmap.insert(k, v);
        }
    }
    let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &dev);
    let contain_key = vb.contains_tensor("encoder.block.4.block.2.block.3.weight_g");
    println!(
        "contain encoder.block.4.block.2.block.3.weight_g: {}",
        contain_key
    );
    Ok(())
}

#[test]
fn voxcpm1_5_weight() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/OpenBMB/VoxCPM1.5/", save_dir);
    let model_list = find_type_files(&model_path, "pth")?;
    println!("model_list: {:?}", model_list);
    // let dev = get_device(None);
    let mut dict_to_hashmap = HashMap::new();
    // let mut dtype = candle_core::DType::F32;
    for m in model_list {
        let dict = read_all_with_key(m, Some("state_dict"))?;
        // dtype = dict[0].1.dtype();
        for (k, v) in dict {
            println!("key: {}, tensor shape: {:?}", k, v);
            dict_to_hashmap.insert(k, v);
        }
    }

    Ok(())
}

#[test]
fn qwen3vl_weight() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Qwen/Qwen3-VL-4B-Instruct/", save_dir);
    let model_list = find_type_files(&model_path, "safetensors")?;

    let device = Device::Cpu;
    for m in &model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            println!("=== {} === {:?}", key, tensor.shape());
        }
    }
    println!("model_list: {:?}", model_list);
    Ok(())
}

#[test]
fn deepseekocr_weight() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/deepseek-ai/DeepSeek-OCR/", save_dir);
    let model_list = find_type_files(&model_path, "safetensors")?;

    let device = Device::Cpu;
    for m in &model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            if key.contains("rel_pos_h") {
                println!("=== {} === {:?}", key, tensor.shape());
            }
            // println!("=== {} === {:?}", key, tensor.shape());
        }
    }
    println!("model_list: {:?}", model_list);
    Ok(())
}

#[test]
fn hunyuanocr_weight() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Tencent-Hunyuan/HunyuanOCR/", save_dir);
    let model_list = find_type_files(&model_path, "safetensors")?;

    let device = Device::Cpu;
    for m in &model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            if key.contains(".image_") {
                println!("=== {} === {:?}", key, tensor.shape());
            }
            // println!("=== {} === {:?}", key, tensor.shape());
        }
    }
    println!("model_list: {:?}", model_list);
    Ok(())
}

#[test]
fn glm_asr_nano_weight() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/ZhipuAI/GLM-ASR-Nano-2512/", save_dir);
    let model_list = find_type_files(&model_path, "safetensors")?;

    let device = Device::Cpu;
    for m in &model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            if key.contains(".embed_tokens") {
                println!("=== {} === {:?}", key, tensor.shape());
            }
            // println!("=== {} === {:?}", key, tensor.shape());
        }
    }
    println!("model_list: {:?}", model_list);
    Ok(())
}

#[test]
fn fun_asr_nano_weight() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/FunAudioLLM/Fun-ASR-Nano-2512/", save_dir);
    let model_list = find_type_files(&model_path, "pt")?;
    println!("model_list: {:?}", model_list);
    // let dev = get_device(None);
    let mut dict_to_hashmap = HashMap::new();
    // let mut dtype = candle_core::DType::F32;
    for m in model_list {
        // let dict = read_all_with_key(m, Some("state_dict"))?;
        let dict = read_all_with_key(m, None)?;
        // dtype = dict[0].1.dtype();
        for (k, v) in dict {
            if k.contains("model") {
                println!("key: {}, tensor shape: {:?}", k, v);
            }
            dict_to_hashmap.insert(k, v);
        }
    }
    Ok(())
}

#[test]
fn qwen3_weight() -> Result<()> {
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Qwen/Qwen3-0.6B/", save_dir);
    let model_list = find_type_files(&model_path, "safetensors")?;

    let device = Device::Cpu;
    for m in &model_list {
        let weights = safetensors::load(m, &device)?;
        for (key, tensor) in weights.iter() {
            // if key.contains(".embed_tokens") {
            //     println!("=== {} === {:?}", key, tensor.shape());
            // }
            println!("=== {} === {:?}", key, tensor.shape());
        }
    }
    println!("model_list: {:?}", model_list);
    Ok(())
}

#[test]
fn index_tts2_weight() -> Result<()> {
    let save_dir: String =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/IndexTeam/IndexTTS-2/", save_dir);
    let s2mel_path = model_path + "/s2mel.pth";
    // let wac2vec2_path = model_path+ "/wav2vec2bert_stats.pt";
    // let model_path = format!("{}/iic/speech_campplus_sv_zh-cn_16k-common/", save_dir);
    // let campplus_path = model_path+ "/campplus_cn_common.bin";
    // let model_list = find_type_files(&model_path, "safetensors")?;
    let model_list = vec![s2mel_path];
    // let mut dict_to_hashmap = HashMap::new();
    // let mut dtype = candle_core::DType::F32;
    for m in model_list {
        // let dict = read_all_with_key(m, Some("state_dict"))?;
        // let dict = read_all_with_key(m, Some("net"))?;
        let dict = read_pth_tensor_info_cycle(m, Some("net.cfm"))?;
        // dtype = dict[0].1.dtype();
        for (k, v) in dict {
            // if k.contains("model") {
            //     println!("key: {}, tensor shape: {:?}", k, v);
            // }
            // dict_to_hashmap.insert(k, v);
            println!("key: {}, tensor shape: {:?}", k, v);
        }
    }
    // let device = Device::Cpu;
    // let semantic_codec_path = save_dir.to_string() + "/amphion/MaskGCT/semantic_codec/model.safetensors" ;
    // let model_list = vec![semantic_codec_path];
    // for m in model_list {
    //     let weights = safetensors::load(m, &device)?;
    //     for (key, tensor) in weights.iter() {
    //         println!("=== {} === {:?}", key, tensor.shape());
    //     }
    // }
    Ok(())
}
