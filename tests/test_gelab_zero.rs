use std::time::Instant;

use aha::models::{GenerateModel, qwen3vl::generate::Qwen3VLGenerateModel};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

#[test]
fn gelab_zero_generate() -> Result<()> {
    // test with cuda: RUST_BACKTRACE=1 cargo test -F cuda gelab_zero_generate -r -- --nocapture

    let model_path = "/home/jhq/huggingface_model/stepfun-ai/GELab-Zero-4B-preview";

    let message = r#"
    {
        "model": "gelab-zero",
        "messages": [
            {
                "role": "user",
                "content": [    
                    {
                        "type": "text", 
                        "text": "Hello, GELab-Zero!, 现在几点了"
                    }
                ]
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "当你想知道现在的时间时非常有用。",
                    "parameters": {}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "当你想查询指定城市的天气时非常有用。",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市或县区，比如北京市、杭州市、余杭区等。"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "tool_choice": null
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut qwen3vl = Qwen3VLGenerateModel::init(model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let res = qwen3vl.generate(mes)?;
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
