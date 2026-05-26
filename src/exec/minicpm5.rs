//! MiniCPM4-0.5B exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::GenerateModel;
use crate::models::minicpm5::generate::MiniCPM5GenerateModel;
use crate::utils::get_file_path;

pub struct MiniCPM5Exec;

impl ExecModel for MiniCPM5Exec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let input_text = &input[0];
        let target_text = if input_text.starts_with("file://") {
            // let path = &input[7..];
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.to_string()
        };

        let i_start = Instant::now();
        let mut model = MiniCPM5GenerateModel::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let message = format!(
            r#"{{
            "temperature": 0.3,
            "top_p": 0.8,
            "model": "minicpm5",
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
