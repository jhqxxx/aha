use std::time::Instant;

use anyhow::{Result, anyhow};
use serde::Serialize;

use crate::exec::ExecModel;
use crate::models::{LoadSpec, qwen3_reranker::generate::Qwen3RerankerModel};

#[derive(Debug, Serialize)]
struct RerankItem {
    index: usize,
    score: f32,
    document: String,
}

pub struct Qwen3RerankerExec;

impl Qwen3RerankerExec {
    pub fn run_with_spec(input: &[String], output: Option<&str>, spec: &LoadSpec) -> Result<()> {
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

        let mut model = Qwen3RerankerModel::init_from_spec(spec, None, None)?;
        let i_start = Instant::now();
        let scores = model.rerank(&query, &documents)?;
        println!("Time elapsed in rerank is: {:?}", i_start.elapsed());

        let mut ranked = scores
            .into_iter()
            .enumerate()
            .map(|(index, score)| RerankItem {
                index,
                score,
                document: documents[index].clone(),
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.score.total_cmp(&a.score));

        let output_json = serde_json::to_string_pretty(&ranked)?;
        println!("{}", output_json);

        let output_path = output
            .map(|o| o.to_string())
            .unwrap_or_else(|| format!("qwen3-rerank-{}.json", chrono::Utc::now().timestamp()));
        std::fs::write(&output_path, output_json.as_bytes())?;
        println!("Generate rerank output to {}", output_path);
        Ok(())
    }
}

impl ExecModel for Qwen3RerankerExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let spec =
            LoadSpec::for_safetensors(crate::models::WhichModel::Qwen3Reranker0_6B, weight_path);
        Self::run_with_spec(input, output, &spec)
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
