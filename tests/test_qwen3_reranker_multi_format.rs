use std::path::Path;

use aha::models::{
    ArtifactKind, LoadSpec, ModelPaths, WhichModel, qwen3_reranker::generate::Qwen3RerankerModel,
};
use anyhow::{Result, anyhow};

const QWEN3_RERANKER_ONNX_DIR: &str = r"D:\model_download\Qwen3-Reranker-0.6B-ONNX";
const QWEN3_RERANKER_SAFETENSORS_DIR: &str = r"D:\model_download\Qwen3-Reranker-0.6B";
const QWEN3_RERANKER_GGUF_DIR: &str = r"D:\model_download\Qwen3-Reranker-0.6B-Q8_0-GGUF";

fn require_existing_dir(path: &str) -> Result<()> {
    let dir = Path::new(path);
    if !dir.exists() {
        return Err(anyhow!("model dir not found: {}", path));
    }
    if !dir.is_dir() {
        return Err(anyhow!("path is not a directory: {}", path));
    }
    Ok(())
}

fn first_file_with_extension(dir: &str, extension: &str) -> Result<String> {
    require_existing_dir(dir)?;
    let mut files = std::fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
        })
        .map(|path| path.to_string_lossy().to_string())
        .collect::<Vec<_>>();
    files.sort();
    files
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .{} file found in {}", extension, dir))
}

fn top_score_index(scores: &[f32]) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .max_by(|lhs, rhs| lhs.1.total_cmp(rhs.1))
        .map(|(idx, _)| idx)
}

#[test]
fn qwen3_reranker_onnx_init_from_spec_can_rerank() -> Result<()> {
    if let Err(err) = aha::models::common::onnx::ensure_ort_dylib_path() {
        println!("skip reranker onnx test: {err}");
        return Ok(());
    }

    if let Err(err) = require_existing_dir(QWEN3_RERANKER_ONNX_DIR) {
        println!("skip reranker onnx test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::Qwen3Reranker0_6B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(QWEN3_RERANKER_ONNX_DIR.to_string()),
            // Intentionally left None to validate tokenizer auto-fallback from *-ONNX to sibling dir.
            tokenizer_dir: None,
            ..Default::default()
        },
    };

    let mut model = Qwen3RerankerModel::init_from_spec(&spec, None, None)?;
    let docs = vec![
        "Rust async requests are commonly built with reqwest and tokio.".to_string(),
        "Paris is the capital of France.".to_string(),
    ];
    let scores = model.rerank("How to make async HTTP calls in Rust?", &docs)?;
    assert_eq!(scores.len(), docs.len());
    Ok(())
}

#[test]
fn qwen3_reranker_safetensors_init_from_spec_can_rerank() -> Result<()> {
    if let Err(err) = require_existing_dir(QWEN3_RERANKER_SAFETENSORS_DIR) {
        println!("skip reranker safetensors test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::Qwen3Reranker0_6B,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(QWEN3_RERANKER_SAFETENSORS_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = Qwen3RerankerModel::init_from_spec(&spec, None, None)?;
    let docs = vec![
        "Rust async requests are commonly built with reqwest and tokio.".to_string(),
        "Paris is the capital of France.".to_string(),
    ];
    let scores = model.rerank("How to make async HTTP calls in Rust?", &docs)?;
    assert_eq!(scores.len(), docs.len());
    Ok(())
}

#[test]
fn qwen3_reranker_load_spec_accepts_gguf() {
    let spec = LoadSpec {
        model: WhichModel::Qwen3Reranker0_6B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some("D:/model_download/Qwen3-Reranker-0.6B-GGUF/model.gguf".to_string()),
            ..Default::default()
        },
    };
    spec.validate()
        .expect("qwen3 reranker should accept gguf artifact");
}

#[test]
fn qwen3_reranker_init_from_spec_gguf_can_rerank() -> Result<()> {
    let gguf_path = match first_file_with_extension(QWEN3_RERANKER_GGUF_DIR, "gguf") {
        Ok(path) => path,
        Err(err) => {
            println!("skip reranker gguf test: {err}");
            return Ok(());
        }
    };

    let spec = LoadSpec {
        model: WhichModel::Qwen3Reranker0_6B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path),
            tokenizer_dir: Some(QWEN3_RERANKER_GGUF_DIR.to_string()),
            ..Default::default()
        },
    };
    let mut model = Qwen3RerankerModel::init_from_spec(&spec, None, None)?;
    let docs = vec![
        "Rust async requests are commonly built with reqwest and tokio.".to_string(),
        "Paris is the capital of France.".to_string(),
    ];
    let scores = model.rerank("How to make async HTTP calls in Rust?", &docs)?;
    assert_eq!(scores.len(), docs.len());
    Ok(())
}

#[test]
fn qwen3_reranker_top_doc_consistent_across_formats() -> Result<()> {
    if let Err(err) = require_existing_dir(QWEN3_RERANKER_SAFETENSORS_DIR) {
        println!("skip reranker consistency test (safetensors): {err}");
        return Ok(());
    }
    if let Err(err) = require_existing_dir(QWEN3_RERANKER_ONNX_DIR) {
        println!("skip reranker consistency test (onnx): {err}");
        return Ok(());
    }
    if let Err(err) = aha::models::common::onnx::ensure_ort_dylib_path() {
        println!("skip reranker consistency test (onnxruntime): {err}");
        return Ok(());
    }
    let gguf_path = match first_file_with_extension(QWEN3_RERANKER_GGUF_DIR, "gguf") {
        Ok(path) => path,
        Err(err) => {
            println!("skip reranker consistency test (gguf): {err}");
            return Ok(());
        }
    };

    let query = "How to make async HTTP calls in Rust?";
    let docs = vec![
        "Rust async requests are commonly built with reqwest and tokio.".to_string(),
        "Paris is the capital of France.".to_string(),
        "Database index tuning can improve SQL query speed.".to_string(),
    ];

    let safetensors_spec = LoadSpec {
        model: WhichModel::Qwen3Reranker0_6B,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(QWEN3_RERANKER_SAFETENSORS_DIR.to_string()),
            ..Default::default()
        },
    };
    let onnx_spec = LoadSpec {
        model: WhichModel::Qwen3Reranker0_6B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(QWEN3_RERANKER_ONNX_DIR.to_string()),
            tokenizer_dir: None,
            ..Default::default()
        },
    };
    let gguf_spec = LoadSpec {
        model: WhichModel::Qwen3Reranker0_6B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path),
            tokenizer_dir: Some(QWEN3_RERANKER_GGUF_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut safetensors_model = Qwen3RerankerModel::init_from_spec(&safetensors_spec, None, None)?;
    let mut onnx_model = Qwen3RerankerModel::init_from_spec(&onnx_spec, None, None)?;
    let mut gguf_model = Qwen3RerankerModel::init_from_spec(&gguf_spec, None, None)?;

    let safetensors_scores = safetensors_model.rerank(query, &docs)?;
    let onnx_scores = onnx_model.rerank(query, &docs)?;
    let gguf_scores = gguf_model.rerank(query, &docs)?;

    let safetensors_top = top_score_index(&safetensors_scores)
        .ok_or_else(|| anyhow!("safetensors scores are empty"))?;
    let onnx_top = top_score_index(&onnx_scores).ok_or_else(|| anyhow!("onnx scores are empty"))?;
    let gguf_top = top_score_index(&gguf_scores).ok_or_else(|| anyhow!("gguf scores are empty"))?;

    assert_eq!(
        safetensors_top, 0,
        "expected safetensors top doc to be rust doc"
    );
    assert_eq!(onnx_top, 0, "expected onnx top doc to be rust doc");
    assert_eq!(gguf_top, 0, "expected gguf top doc to be rust doc");
    Ok(())
}
