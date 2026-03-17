//! Qwen3VL-2B exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::{GenerateModel, qwen3vl::generate::Qwen3VLGenerateModel};
use crate::utils::get_file_path;

pub struct Qwen3vlExec;

impl ExecModel for Qwen3vlExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let input_text = &input[0];
        let target_text = if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        let i_start = Instant::now();
        let mut model = Qwen3VLGenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);
        let url = input.get(1);
        let input_url = if let Some(url) = url
            && (url.starts_with("http://")
                || url.starts_with("https://")
                || url.starts_with("file://"))
        {
            Some(url.clone())
        } else {
            url.map(|url| format!("file://{}", url))
        };
        let message = if let Some(input_url) = &input_url
            && input_url.ends_with("mp4")
        {
            format!(
                r#"{{
            "model": "qwen3vl",
            "messages": [
                {{
                    "role": "user",
                    "content": [
                        {{
                            "type": "video",
                            "video_url": 
                            {{
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
            )
        } else if let Some(input_url) = &input_url {
            format!(
                r#"{{
            "model": "qwen3vl",
            "messages": [
                {{
                    "role": "user",
                    "content": [
                        {{
                            "type": "image",
                            "image_url": {{
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
            )
        } else {
            format!(
                r#"{{
            "model": "qwen3vl",
            "messages": [
                {{
                    "role": "user",
                    "content": [
                        {{
                            "type": "text", 
                            "text": "{}"
                        }}
                    ]
                }}
            ]
        }}"#,
                target_text
            )
        };
        let mes = serde_json::from_str(&message)?;

        let i_start = Instant::now();
        let result = model.generate(mes)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in generate is: {:?}", i_duration);

        println!("Result: {:?}", result);

        if let Some(out) = output {
            std::fs::write(out, format!("{:?}", result))?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}
