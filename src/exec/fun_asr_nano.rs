//! Fun-ASR-Nano-2512 exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::{GenerateModel, fun_asr_nano::generate::FunAsrNanoGenerateModel};
use crate::utils::get_file_path;

pub struct FunASRNanoExec;

impl ExecModel for FunASRNanoExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let input_text = &input[0];
        let target_text = if input_text.starts_with("file://") {
            // let path = &input[7..];
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        let i_start = Instant::now();
        let mut model = FunAsrNanoGenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        // Create ChatCompletionParameters for ASR
        let url = input.get(1).ok_or_else(|| {
            anyhow::anyhow!("fun-asr-nano requires a second input: audio path or URL")
        })?;
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
