use std::time::Instant;

use aha::{
    chat::ChatCompletionParameters,
    models::{GenerateModel, qwen3_5::generate::Qwen3_5GenerateModel},
};
use anyhow::Result;
// use candle_core::{Device, quantized::gguf_file};
#[test]
fn gguf_test() -> Result<()> {
    // cargo test -r -F cuda --test test_gguf_qwen3_5 gguf_test -- --nocapture
    // let path = "/home/jhq/.aha/Qwen/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q5_K_M.gguf"; // 有问题
    // let path = "/home/jhq/.aha/Qwen/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q6_K.gguf";
    let path = "/home/jhq/.aha/Qwen/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf";
    // let mut file = std::fs::File::open(path)?;
    // let model = gguf_file::Content::read(&mut file)?;
    // println!("group_count: {:?}", model.metadata.get("qwen35.ssm.group_count"));
    // println!("time_step_rank: {:?}", model.metadata.get("qwen35.ssm.time_step_rank"));
    // println!("state_size: {:?}", model.metadata.get("qwen35.ssm.state_size"));
    // for (key, value) in model.metadata {
    //     if key.contains("tokenizer") {
    //         continue;
    //     }
    //     println!("{key}: {:#?}", value);
    // }
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
                        "type": "text", 
                        "text": "你如何看待AI"
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut gguf_qwen3_5 = Qwen3_5GenerateModel::init_from_gguf(path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let res = gguf_qwen3_5.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("generate: \n {:?}", res);
    if res.usage.is_some() {
        let num_token = res.usage.as_ref().unwrap().total_tokens;
        let duration_secs = i_duration.as_secs_f64();
        let tps = num_token as f64 / duration_secs;
        println!("Tokens per second (TPS): {:.2}", tps);
    }
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}
