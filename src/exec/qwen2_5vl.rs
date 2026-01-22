//! Qwen2.5VL-3B exec implementation for CLI `run` subcommand

use crate::exec::ExecModel;
use crate::models::{GenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel};
use anyhow::{Ok, Result};
use std::time::Instant;

pub struct Qwen2_5vlExec;

impl ExecModel for Qwen2_5vlExec {
    fn run(input: &str, output: Option<&str>, weight_path: &str) -> Result<()> {
        let target_text = if input.starts_with("file://") {
            let path = &input[7..];
            std::fs::read_to_string(path)?
        } else {
            input.to_string()
        };

        let i_start = Instant::now();
        let mut model = Qwen2_5VLGenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let message = format!(
            r#"{{
            "model": "qwen2.5vl",
            "messages": [
                {{
                    "role": "user",
                    "content": "{}"
                }}
            ]
        }}"#,
            target_text.replace('"', "\\\"")
        );
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
