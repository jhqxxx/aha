use std::{
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use aha::models::{
    ArtifactKind, LoadSpec, ModelPaths,
    common::{gguf::Gguf, onnx::ensure_ort_dylib_path, retrieval::cosine_similarity},
    qwen3_embedding::generate::Qwen3EmbeddingModel,
};
use anyhow::{Context, Result, anyhow};
use candle_core::{Device, quantized::gguf_file};
#[cfg(feature = "onnx-runtime")]
use ort::session::Session;

const QWEN3_EMBEDDING_SAFETENSORS_DIR: &str = r"D:\model_download\Qwen3-Embedding-0.6B";
const QWEN3_EMBEDDING_GGUF_DIR: &str = r"D:\model_download\Qwen3-Embedding-0.6B-GGUF";
const QWEN3_EMBEDDING_ONNX_DIR: &str = r"D:\model_download\Qwen3-Embedding-0.6B-ONNX";

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

fn first_file_with_extension(dir: &str, extension: &str) -> Result<PathBuf> {
    require_existing_dir(dir)?;

    let mut candidates = std::fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
        })
        .collect::<Vec<_>>();

    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .{} file found in {}", extension, dir))
}

fn first_file_with_extension_recursive(dir: &str, extension: &str) -> Result<PathBuf> {
    require_existing_dir(dir)?;

    let mut stack = vec![PathBuf::from(dir)];
    let mut matches = Vec::new();

    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
            {
                matches.push(path);
            }
        }
    }

    matches.sort();
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .{} file found (recursive) in {}", extension, dir))
}

#[test]
fn qwen3_embedding_safetensors_can_load() -> Result<()> {
    // Run this test only:
    // cargo test --test test_qwen3_embedding_multi_format qwen3_embedding_safetensors_can_load -- --nocapture
    require_existing_dir(QWEN3_EMBEDDING_SAFETENSORS_DIR)?;

    let _model = Qwen3EmbeddingModel::init(QWEN3_EMBEDDING_SAFETENSORS_DIR, None, None)
        .with_context(|| {
            format!(
                "failed to init safetensors model from {}",
                QWEN3_EMBEDDING_SAFETENSORS_DIR
            )
        })?;
    Ok(())
}

#[test]
fn qwen3_embedding_gguf_can_load() -> Result<()> {
    // Run this test only:
    // cargo test --test test_qwen3_embedding_multi_format qwen3_embedding_gguf_can_load -- --nocapture
    let gguf_path = first_file_with_extension(QWEN3_EMBEDDING_GGUF_DIR, "gguf")?;

    let file = File::open(&gguf_path)
        .with_context(|| format!("failed to open gguf file: {}", gguf_path.display()))?;
    let mut reader = BufReader::new(file);
    let content = gguf_file::Content::read(&mut reader)
        .with_context(|| format!("failed to parse gguf file: {}", gguf_path.display()))?;

    if content.tensor_infos.is_empty() {
        return Err(anyhow!(
            "gguf tensor_infos is empty: {}",
            gguf_path.display()
        ));
    }

    let gguf = Gguf::new(content, reader, Device::Cpu);
    let tokenizer = gguf
        .build_tokenizer(Some(false), Some(false), Some(false))
        .context("failed to build tokenizer from gguf metadata")?;

    let vocab_size = tokenizer.tokenizer.get_vocab_size(false);
    if vocab_size == 0 {
        return Err(anyhow!("gguf tokenizer vocab is empty"));
    }
    Ok(())
}

#[test]
fn qwen3_embedding_onnx_can_load() -> Result<()> {
    // Run this test only:
    // cargo test --test test_qwen3_embedding_multi_format qwen3_embedding_onnx_can_load -- --nocapture
    let onnx_path = first_file_with_extension_recursive(QWEN3_EMBEDDING_ONNX_DIR, "onnx")?;
    // Basic artifact-level smoke check: ONNX file can be discovered and read.
    let metadata = std::fs::metadata(&onnx_path)
        .with_context(|| format!("failed to read onnx metadata: {}", onnx_path.display()))?;
    if metadata.len() == 0 {
        return Err(anyhow!("onnx file is empty: {}", onnx_path.display()));
    }
    let _bytes = std::fs::read(&onnx_path)
        .with_context(|| format!("failed to read onnx file: {}", onnx_path.display()))?;

    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn qwen3_embedding_onnxruntime_can_create_session() -> Result<()> {
    // Run this test only:
    // cargo test --test test_qwen3_embedding_multi_format qwen3_embedding_onnxruntime_can_create_session -- --nocapture
    let onnx_path = first_file_with_extension_recursive(QWEN3_EMBEDDING_ONNX_DIR, "onnx")?;

    // ort with `load-dynamic` requires ONNX Runtime dynamic library path to be configured.
    // Example on Windows:
    // $env:ORT_DYLIB_PATH = "D:\\onnxruntime\\onnxruntime.dll"
    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip onnxruntime session test: {err}");
        return Ok(());
    }

    let session = Session::builder()
        .context("failed to create onnxruntime session builder")?
        .commit_from_file(&onnx_path)
        .with_context(|| {
            format!(
                "failed to create onnxruntime session from {}",
                onnx_path.display()
            )
        })?;

    if session.inputs().is_empty() {
        return Err(anyhow!("onnxruntime session has no inputs"));
    }
    if session.outputs().is_empty() {
        return Err(anyhow!("onnxruntime session has no outputs"));
    }

    Ok(())
}

#[test]
fn qwen3_embedding_onnx_init_from_spec_can_embed() -> Result<()> {
    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip onnx init test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: aha::models::WhichModel::Qwen3Embedding0_6B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(QWEN3_EMBEDDING_ONNX_DIR.to_string()),
            tokenizer_dir: Some(QWEN3_EMBEDDING_ONNX_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = Qwen3EmbeddingModel::init_from_spec(&spec, None, None)?;
    let output = model.embed(&["test onnx embedding".to_string()])?;
    assert_eq!(output.len(), 1);
    assert!(!output[0].is_empty());
    Ok(())
}

#[test]
fn qwen3_embedding_safetensors_init_from_spec_can_embed() -> Result<()> {
    if let Err(err) = require_existing_dir(QWEN3_EMBEDDING_SAFETENSORS_DIR) {
        println!("skip safetensors init_from_spec test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: aha::models::WhichModel::Qwen3Embedding0_6B,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(QWEN3_EMBEDDING_SAFETENSORS_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = Qwen3EmbeddingModel::init_from_spec(&spec, None, None)?;
    let output = model.embed(&["test safetensors embedding".to_string()])?;
    assert_eq!(output.len(), 1);
    assert!(!output[0].is_empty());
    Ok(())
}

#[test]
fn qwen3_embedding_load_spec_accepts_gguf() {
    let spec = LoadSpec {
        model: aha::models::WhichModel::Qwen3Embedding0_6B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some("D:/model_download/Qwen3-Embedding-0.6B-GGUF/model.gguf".to_string()),
            ..Default::default()
        },
    };
    spec.validate()
        .expect("qwen3 embedding should accept gguf artifact");
}

#[test]
fn qwen3_embedding_init_from_spec_gguf_can_embed() -> Result<()> {
    let gguf_path = match first_file_with_extension(QWEN3_EMBEDDING_GGUF_DIR, "gguf") {
        Ok(path) => path,
        Err(err) => {
            println!("skip gguf init_from_spec test: {err}");
            return Ok(());
        }
    };
    let spec = LoadSpec {
        model: aha::models::WhichModel::Qwen3Embedding0_6B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path.to_string_lossy().to_string()),
            tokenizer_dir: Some(QWEN3_EMBEDDING_GGUF_DIR.to_string()),
            ..Default::default()
        },
    };
    let mut model = Qwen3EmbeddingModel::init_from_spec(&spec, None, None)?;
    let output = model.embed(&["test gguf embedding".to_string()])?;
    assert_eq!(output.len(), 1);
    assert!(!output[0].is_empty());
    Ok(())
}

#[test]
fn qwen3_embedding_real_texts_similarity() -> Result<()> {
    // Run this test only:
    // cargo test --test test_qwen3_embedding_multi_format qwen3_embedding_real_texts_similarity -- --exact --nocapture --test-threads=1
    require_existing_dir(QWEN3_EMBEDDING_SAFETENSORS_DIR)?;

    let query = "如何在 Rust 项目中做异步 HTTP 请求";
    let documents = vec![
        "Rust 中可以使用 reqwest + tokio 发起异步 HTTP 请求".to_string(),
        "今天天气很好，适合出去散步和拍照".to_string(),
        "在 Python 里可以用 requests 发送同步网络请求".to_string(),
        "数据库索引优化可以显著提升查询性能".to_string(),
    ];

    let mut model = Qwen3EmbeddingModel::init(QWEN3_EMBEDDING_SAFETENSORS_DIR, None, None)?;

    let mut inputs = vec![query.to_string()];
    inputs.extend(documents.clone());
    let embeddings = model.embed(&inputs)?;

    if embeddings.len() != inputs.len() {
        return Err(anyhow!(
            "embedding count mismatch: got {}, expect {}",
            embeddings.len(),
            inputs.len()
        ));
    }

    for (idx, emb) in embeddings.iter().enumerate() {
        if emb.len() != 1024 {
            return Err(anyhow!(
                "embedding dim mismatch at index {}: got {}, expect 1024",
                idx,
                emb.len()
            ));
        }
    }

    for (idx, text) in inputs.iter().enumerate() {
        println!("text[{idx}]: {text}");
        println!("embedding[{idx}] dim={}", embeddings[idx].len());
        println!(
            "embedding[{idx}]={}",
            serde_json::to_string(&embeddings[idx])?
        );
    }

    let query_embedding = &embeddings[0];
    let mut similarities = Vec::with_capacity(documents.len());
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (doc_idx, doc_emb) in embeddings.iter().enumerate().skip(1) {
        let score = cosine_similarity(query_embedding, doc_emb)?;
        similarities.push(score);
        println!(
            "similarity(query, doc_{}) = {:.6}, doc = {}",
            doc_idx - 1,
            score,
            documents[doc_idx - 1]
        );
        if score > best_score {
            best_score = score;
            best_idx = doc_idx - 1;
        }
    }

    println!("best_match_doc_index={}", best_idx);
    println!("best_match_doc={}", documents[best_idx]);
    println!("best_match_score={:.6}", best_score);

    let result_json = serde_json::json!({
        "query": query,
        "documents": documents,
        "embeddings": embeddings,
        "similarities": similarities,
        "best_match_doc_index": best_idx,
        "best_match_doc": documents[best_idx],
        "best_match_score": best_score
    });
    let output_path = Path::new("target").join("qwen3_embedding_similarity_output.json");
    std::fs::write(&output_path, serde_json::to_string_pretty(&result_json)?)?;
    println!("result_json_saved_to={}", output_path.display());

    Ok(())
}
