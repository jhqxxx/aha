//! MiniCPM4-0.5B exec implementation for CLI `run` subcommand

use crate::exec::ExecModel;
use crate::models::{GenerateModel, minicpm4::generate::MiniCPMGenerateModel};
use anyhow::{Ok, Result};
use std::time::Instant;

pub struct MiniCPM4Exec;

impl ExecModel for MiniCPM4Exec {
    fn run(input: &str, output: Option<&str>, weight_path: &str) -> Result<()> {
        let target_text = if input.starts_with("file://") {
            let path = &input[7..];
            std::fs::read_to_string(path)?
        } else {
            input.to_string()
        };

        let i_start = Instant::now();
        let mut model = MiniCPMGenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let message = format!(
            r#"{{
            "temperature": 0.3,
            "top_p": 0.8,
            "model": "minicpm4",
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

        // Print result
        println!("Result: {:?}", result);

        if let Some(out) = output {
            std::fs::write(out, format!("{:?}", result))?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}
