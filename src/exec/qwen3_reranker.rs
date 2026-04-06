use crate::{
    exec::ExecModel,
    models::{common::reranker::TextRerank, qwen3_reranker::Qwen3Reranker},
};
use anyhow::{Result, anyhow};
use std::time::Instant;

pub struct Qwen3RerankerExec;

impl ExecModel for Qwen3RerankerExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        if input.len() < 2 {
            return Err(anyhow!(
                "reranker run requires two inputs: <query> <documents-source>"
            ));
        }
        let query = input[0].clone();
        let docs_source = input[1].clone();
        let documents = parse_documents_source(&docs_source)?;
        if documents.is_empty() {
            return Err(anyhow!("documents list is empty"));
        }

        let i_start = Instant::now();
        let mut model = Qwen3Reranker::init(weight_path, None, None)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in load model is: {:?}", i_duration);

        let i_start = Instant::now();
        let scores = model.rerank(&query, &documents)?;
        let i_duration = i_start.elapsed();
        println!("Time elapsed in generate is: {:?}", i_duration);

        println!("Result: {:?}", scores);

        if let Some(out) = output {
            std::fs::write(out, format!("{:?}", scores))?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}

fn parse_documents_source(source: &str) -> Result<Vec<String>> {
    if source.starts_with("file://") {
        let path = source.trim_start_matches("file://");
        return read_documents_file(path);
    }
    if std::path::Path::new(source).exists() {
        return read_documents_file(source);
    }

    let docs = source
        .split("|||")
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(|x| x.to_string())
        .collect::<Vec<_>>();
    Ok(docs)
}

fn read_documents_file(path: &str) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let docs = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.to_string())
        .collect::<Vec<_>>();
    Ok(docs)
}
