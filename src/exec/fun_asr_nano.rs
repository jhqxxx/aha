//! Fun-ASR-Nano-2512 exec implementation for CLI `run` subcommand

use crate::exec::ExecModel;
use crate::models::{GenerateModel, fun_asr_nano::generate::FunAsrNanoGenerateModel};
use anyhow::{Ok, Result};
use std::time::Instant;

pub struct FunASRNanoExec;

impl ExecModel for FunASRNanoExec {
    fn run(input: &str, output: Option<&str>, weight_path: &str) -> Result<()> {
        let target_text = if input.starts_with("file://") {
            let path = &input[7..];
            std::fs::read_to_string(path)?
        } else {
            input.to_string()
        };

        let i_start = Instant::now();
        let mut model = FunAsrNanoGenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        // Create ChatCompletionParameters for ASR
        let input_url = if input.starts_with("http://") || input.starts_with("https://") || input.starts_with("file://") {
            input.to_string()
        } else {
            format!("file://{}", input)
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
                        }},
                        {{
                            "type": "text",
                            "text": "{}"
                        }}
                    ]
                }}
            ]
        }}"#,
            input_url, target_text
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
