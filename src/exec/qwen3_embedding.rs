use std::time::Instant;

use anyhow::Result;

use crate::exec::ExecModel;
use crate::models::qwen3_embedding::generate::Qwen3EmbeddingModel;
use crate::utils::get_file_path;

pub struct Qwen3EmbeddingExec;

impl ExecModel for Qwen3EmbeddingExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let input_text = input
            .first()
            .ok_or_else(|| anyhow::anyhow!("embedding run requires one text input"))?;
        let text = if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        let i_start = Instant::now();
        let mut model = Qwen3EmbeddingModel::init(weight_path, None, None)?;
        println!("Time elapsed in load model is: {:?}", i_start.elapsed());

        let i_start = Instant::now();
        let embedding = model.embed(&[text])?;
        println!("Time elapsed in embedding is: {:?}", i_start.elapsed());

        let output_json = serde_json::to_string_pretty(&embedding)?;
        println!("{}", output_json);
        if let Some(out) = output {
            std::fs::write(out, output_json)?;
            println!("Output saved to: {}", out);
        }
        Ok(())
    }
}
