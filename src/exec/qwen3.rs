//! Qwen3-0.6B exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::{GenerateModel, LoadSpec, qwen3::generate::Qwen3GenerateModel};
use crate::utils::get_file_path;

pub struct Qwen3Exec;

impl Qwen3Exec {
    pub fn run_with_spec(input: &[String], output: Option<&str>, spec: &LoadSpec) -> Result<()> {
        let input_text = &input[0];
        let target_text = if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        let i_start = Instant::now();
        let mut model = Qwen3GenerateModel::init_from_spec(spec, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let message = format!(
            r#"{{
            "model": "qwen3",
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

impl ExecModel for Qwen3Exec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let spec = LoadSpec::for_safetensors(crate::models::WhichModel::Qwen3_0_6B, weight_path);
        Self::run_with_spec(input, output, &spec)
    }
}
