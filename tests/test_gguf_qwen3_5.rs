use std::time::Instant;

use aha::{
    models::{GenerateModel, qwen3_5::generate::Qwen3_5GenerateModel},
    params::chat::ChatCompletionParameters,
};
use anyhow::Result;
// use candle_core::{DType, Device, quantized::gguf_file};
#[test]
fn gguf_test() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -r -F cuda --test test_gguf_qwen3_5 gguf_test -- --nocapture
    // let model_path = "/home/jhq/.aha/Qwen/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q6_K.gguf"; // 有问题
    // let mmproj_path = "/home/jhq/.aha/Qwen/Qwen3.5-4B-GGUF/mmproj-F16.gguf";
    // let model_path = "/home/jhq/.aha/Qwen/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q6_K.gguf";
    // let mmproj_path = "/home/jhq/.aha/Qwen/Qwen3.5-2B-GGUF/mmproj-F16.gguf";
    let model_path = "/home/jhq/.aha/Qwen/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf";
    let mmproj_path = "/home/jhq/.aha/Qwen/Qwen3.5-0.8B-GGUF/mmproj-F16.gguf";
    // let mut model_file = std::fs::File::open(model_path)?;
    // let model = gguf_file::Content::read(&mut model_file)?;
    // for (key, value) in model.tensor_infos {
    //     if key.contains("blk.12.") {
    //         println!("{key}: {:#?}", value);
    //     }
    // }
    // for (key, value) in model.metadata {
    //     if key.contains("tokeni") {
    //         continue;
    //     }
    //     println!("{key}: {:#?}", value);
    // }
    // let mut mmproj_file = std::fs::File::open(mmproj_path)?;
    // let mmproj = gguf_file::Content::read(&mut mmproj_file)?;
    // println!("model: {:#?}", mmproj.tensor_infos.keys());
    // println!("group_count: {:?}", model.metadata.get("qwen35.ssm.group_count"));
    // println!("time_step_rank: {:?}", model.metadata.get("qwen35.ssm.time_step_rank"));
    // println!("state_size: {:?}", model.metadata.get("qwen35.ssm.state_size"));
    // for (key, value) in mmproj.metadata {
    //     println!("{key}: {:#?}", value);
    // }
    // let device = Device::new_cuda(0)?;
    // let mut mmproj_gguf = Gguf::new(mmproj, mmproj_file, device.clone());
    // let weight = mmproj_gguf.get_dequantized("v.position_embd.weight")?;
    // println!("weight: {:?}", weight);
    // let conv3d_weight_1 = mmproj_gguf.get_dequantized("v.patch_embd.weight.1")?;
    // println!("conv3d_weight_1: {}", conv3d_weight_1);
    // let conv3d_bias = mmproj_gguf.get_dequantized("v.patch_embd.bias")?;
    // println!("conv3d_bias: {}", conv3d_bias);
    // println!("model: {:?}", model.magic);
    // println!("generat.type: {:#?}", model.metadata.keys());
    // println!("tokenizer.ggml.eos_token_id: {:#?}", model.metadata.get("tokenizer.ggml.eos_token_id"));
    // println!("model: {:#?}", model.tensor_infos.keys());
    let message = r#"
    {
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_url":
                        {
                            "url": "file://./assets/img/ocr_test1.png"
                        }
                    },
                    {
                        "type": "text",
                        "text": "请分析图片并提取所有可见文本内容，按从左到右、从上到下的布局，返回纯文本"
                    }
                ]
            }
        ]
    }
    "#;

    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut gguf_qwen3_5 =
        Qwen3_5GenerateModel::init_from_gguf(model_path, mmproj_path.into(), None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let res = gguf_qwen3_5.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        let num_token = usage.total_tokens;
        let duration_secs = i_duration.as_secs_f64();
        let tps = num_token as f64 / duration_secs;
        println!("Tokens per second (TPS): {:.2}", tps);
    }
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
