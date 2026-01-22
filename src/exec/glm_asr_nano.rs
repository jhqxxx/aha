//! GLM-ASR-Nano-2512 exec implementation for CLI `run` subcommand

use crate::exec::ExecModel;
use crate::models::{GenerateModel, glm_asr_nano::generate::GlmAsrNanoGenerateModel};
use anyhow::{Ok, Result};
use std::time::Instant;

pub struct GlmASRNanoExec;

impl ExecModel for GlmASRNanoExec {
    fn run(input: &str, output: Option<&str>, weight_path: &str) -> Result<()> {
        let target_text = if input.starts_with("file://") {
            let path = &input[7..];
            std::fs::read_to_string(path)?
        } else {
            input.to_string()
        };

        let i_start = Instant::now();
        let mut model = GlmAsrNanoGenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        // Create ChatCompletionParameters for ASR
        // Input should be an audio file path
        let input_url = if input.starts_with("http://") || input.starts_with("https://") || input.starts_with("file://") {
            input.to_string()
        } else {
            format!("file://{}", input)
        };

        let message = format!(
            r#"{{
            "model": "glm-asr-nano",
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
