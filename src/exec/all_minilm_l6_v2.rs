use std::time::Instant;

use crate::{
    exec::ExecModel,
    models::{all_minilm_l6_v2::AllMiniLML6V2Embedding, common::embedding::TextEmbedding},
    utils::get_file_path,
};
use anyhow::Result;

pub struct AllMiniLML6V2Exec;

impl ExecModel for AllMiniLML6V2Exec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let input_text = &input[0];
        let target_text = if input_text.starts_with("file://") {
            let path = get_file_path(input_text)?;
            std::fs::read_to_string(path)?
        } else {
            input_text.clone()
        };

        let i_start = Instant::now();
        let mut model = AllMiniLML6V2Embedding::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let i_start = Instant::now();
        let result = model.embed_texts(&[target_text])?;
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
