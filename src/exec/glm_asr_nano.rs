//! GLM-ASR-Nano-2512 exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::{GenerateModel, glm_asr_nano::generate::GlmAsrNanoGenerateModel};
use crate::utils::get_file_path;

pub struct GlmASRNanoExec;

impl ExecModel for GlmASRNanoExec {
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
        let mut model = GlmAsrNanoGenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        // Create ChatCompletionParameters for ASR
        // Input should be an audio file path
        let url = input.get(1).ok_or_else(|| {
            anyhow::anyhow!("glm-asr-nano requires a second input: audio path or URL")
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
