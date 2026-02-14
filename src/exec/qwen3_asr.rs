//! Fun-ASR-Nano-2512 exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::GenerateModel;
use crate::models::qwen3_asr::generate::Qwen3AsrGenerateModel;

pub struct Qwen3ASRExec;

impl ExecModel for Qwen3ASRExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let i_start = Instant::now();
        let mut model = Qwen3AsrGenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        // Create ChatCompletionParameters for ASR
        let url = &input[0];
        let input_url = if url.starts_with("http://")
            || url.starts_with("https://")
            || url.starts_with("file://")
        {
            url.clone()
        } else {
            format!("file://{}", url)
        };

        let message = format!(
            r#"{{
            "model": "fun-asr-nano",
            "messages": [
                {{
                    "role": "user",
                    "content": [
                        {{
                            "type": "audio",
                            "audio_url": {{
                                "url": "{}"
                            }}
                        }}
                    ]
                }}
            ]
        }}"#,
            input_url
        );
        let mes = serde_json::from_str(&message)?;

        let i_start = Instant::now();
        let res = model.generate(mes)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in generate is: {:?}", i_duration);

        println!("Result: {:?}", res);

        if let Some(out) = output {
            std::fs::write(out, format!("{:?}", res))?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}
