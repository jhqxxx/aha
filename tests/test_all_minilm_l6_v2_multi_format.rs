use std::path::{Path, PathBuf};

use aha::models::{
    ArtifactKind, LoadSpec, ModelPaths, WhichModel,
    all_minilm_l6_v2::generate::AllMiniLML6V2Model,
    common::{onnx::ensure_ort_dylib_path, retrieval::cosine_similarity},
};
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "onnx-runtime")]
use ort::session::Session;

const ALL_MINILM_L6_V2_DIR: &str = r"D:\model_download\all-MiniLM-L6-v2";
const ALL_MINILM_L6_V2_GGUF_DIR: &str = r"D:\model_download\All-MiniLM-L6-v2-Embedding-GGUF";
const ALL_MINILM_L6_V2_ONNX_DIR: &str = r"D:\model_download\all-MiniLM-L6-v2\onnx";

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

fn first_file_with_extension(dir: &str, extension: &str) -> Result<PathBuf> {
    require_existing_dir(dir)?;

    let mut matches = std::fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
        })
        .collect::<Vec<_>>();

    matches.sort();
    matches
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("no .{} file found in {}", extension, dir))
}

#[test]
fn all_minilm_l6_v2_safetensors_can_load() -> Result<()> {
    require_existing_dir(ALL_MINILM_L6_V2_DIR)?;

    let _model = AllMiniLML6V2Model::init(ALL_MINILM_L6_V2_DIR, None, None).with_context(|| {
        format!(
            "failed to init all-minilm-l6-v2 safetensors model from {}",
            ALL_MINILM_L6_V2_DIR
        )
    })?;
    Ok(())
}

#[test]
fn all_minilm_l6_v2_gguf_file_can_load() -> Result<()> {
    let gguf_path = first_file_with_extension(ALL_MINILM_L6_V2_GGUF_DIR, "gguf")?;
    let metadata = std::fs::metadata(&gguf_path)
        .with_context(|| format!("failed to read gguf metadata: {}", gguf_path.display()))?;
    if metadata.len() == 0 {
        return Err(anyhow!("gguf file is empty: {}", gguf_path.display()));
    }
    let _bytes = std::fs::read(&gguf_path)
        .with_context(|| format!("failed to read gguf file: {}", gguf_path.display()))?;
    Ok(())
}

#[test]
fn all_minilm_l6_v2_onnx_file_can_load() -> Result<()> {
    let onnx_path = first_file_with_extension_recursive(ALL_MINILM_L6_V2_ONNX_DIR, "onnx")?;
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
fn all_minilm_l6_v2_onnxruntime_can_create_session() -> Result<()> {
    let onnx_path = first_file_with_extension_recursive(ALL_MINILM_L6_V2_ONNX_DIR, "onnx")?;

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
fn all_minilm_l6_v2_safetensors_init_from_spec_can_embed() -> Result<()> {
    require_existing_dir(ALL_MINILM_L6_V2_DIR)?;

    let spec = LoadSpec {
        model: WhichModel::AllMiniLML6V2,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(ALL_MINILM_L6_V2_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = AllMiniLML6V2Model::init_from_spec(&spec, None, None)?;
    let output = model.embed(&["test safetensors embedding".to_string()])?;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 384);
    Ok(())
}

#[test]
fn all_minilm_l6_v2_gguf_init_from_spec_can_embed() -> Result<()> {
    let gguf_path = match first_file_with_extension(ALL_MINILM_L6_V2_GGUF_DIR, "gguf") {
        Ok(path) => path,
        Err(err) => {
            println!("skip gguf init_from_spec test: {err}");
            return Ok(());
        }
    };

    let spec = LoadSpec {
        model: WhichModel::AllMiniLML6V2,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path.to_string_lossy().to_string()),
            tokenizer_dir: Some(ALL_MINILM_L6_V2_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = AllMiniLML6V2Model::init_from_spec(&spec, None, None)?;
    let output = model.embed(&["test gguf embedding".to_string()])?;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 384);
    Ok(())
}

#[test]
fn all_minilm_l6_v2_onnx_init_from_spec_can_embed() -> Result<()> {
    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip onnx init test: {err}");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::AllMiniLML6V2,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(ALL_MINILM_L6_V2_ONNX_DIR.to_string()),
            tokenizer_dir: Some(ALL_MINILM_L6_V2_DIR.to_string()),
            ..Default::default()
        },
    };

    let mut model = AllMiniLML6V2Model::init_from_spec(&spec, None, None)?;
    let output = model.embed(&["test onnx embedding".to_string()])?;
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].len(), 384);
    Ok(())
}

#[test]
fn all_minilm_l6_v2_load_spec_accepts_gguf() {
    let spec = LoadSpec {
        model: WhichModel::AllMiniLML6V2,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(format!(
                "{ALL_MINILM_L6_V2_GGUF_DIR}\\all-MiniLM-L6-v2-ggml-model-f16.gguf"
            )),
            tokenizer_dir: Some(ALL_MINILM_L6_V2_DIR.to_string()),
            ..Default::default()
        },
    };

    spec.validate()
        .expect("all-minilm-l6-v2 should accept gguf artifact");
}

#[test]
fn all_minilm_l6_v2_load_spec_accepts_onnx() {
    let spec = LoadSpec {
        model: WhichModel::AllMiniLML6V2,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(ALL_MINILM_L6_V2_ONNX_DIR.to_string()),
            tokenizer_dir: Some(ALL_MINILM_L6_V2_DIR.to_string()),
            ..Default::default()
        },
    };

    spec.validate()
        .expect("all-minilm-l6-v2 should accept onnx artifact");
}

#[test]
fn all_minilm_l6_v2_native_and_gguf_embeddings_are_close() -> Result<()> {
    let gguf_path = match first_file_with_extension(ALL_MINILM_L6_V2_GGUF_DIR, "gguf") {
        Ok(path) => path,
        Err(err) => {
            println!("skip native/gguf similarity test: {err}");
            return Ok(());
        }
    };

    let text = "Rust provides strong ownership guarantees for concurrent systems.";
    let mut native_model = AllMiniLML6V2Model::init(ALL_MINILM_L6_V2_DIR, None, None)?;
    let mut gguf_model = AllMiniLML6V2Model::init_gguf(
        &gguf_path.to_string_lossy(),
        Some(ALL_MINILM_L6_V2_DIR),
        None,
        None,
    )?;

    let native_embedding = native_model.embed(&[text.to_string()])?;
    let gguf_embedding = gguf_model.embed(&[text.to_string()])?;

    let similarity = cosine_similarity(&native_embedding[0], &gguf_embedding[0])?;
    assert!(
        similarity > 0.98,
        "native/gguf embedding similarity too low: {similarity}"
    );
    Ok(())
}

#[test]
fn all_minilm_l6_v2_native_and_onnx_embeddings_are_close() -> Result<()> {
    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip cross-backend similarity test: {err}");
        return Ok(());
    }

    let text = "Rust provides strong ownership guarantees for concurrent systems.";
    let mut native_model = AllMiniLML6V2Model::init(ALL_MINILM_L6_V2_DIR, None, None)?;
    let mut onnx_model =
        AllMiniLML6V2Model::init_onnx(ALL_MINILM_L6_V2_ONNX_DIR, Some(ALL_MINILM_L6_V2_DIR))?;

    let native_embedding = native_model.embed(&[text.to_string()])?;
    let onnx_embedding = onnx_model.embed(&[text.to_string()])?;

    let similarity = cosine_similarity(&native_embedding[0], &onnx_embedding[0])?;
    assert!(
        similarity > 0.98,
        "native/onnx embedding similarity too low: {similarity}"
    );
    Ok(())
}

#[test]
fn all_minilm_l6_v2_real_texts_similarity() -> Result<()> {
    require_existing_dir(ALL_MINILM_L6_V2_DIR)?;

    let query = "How do I send asynchronous HTTP requests in Rust?";
    let documents = vec![
        "In Rust, reqwest with tokio is a common way to send async HTTP requests.".to_string(),
        "The weather is sunny today, which is great for a walk in the park.".to_string(),
        "PostgreSQL indexes can improve database query latency on large tables.".to_string(),
        "A guitar usually has six strings and is used in many styles of music.".to_string(),
    ];

    let mut model = AllMiniLML6V2Model::init(ALL_MINILM_L6_V2_DIR, None, None)?;

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

    let query_embedding = &embeddings[0];
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (doc_idx, doc_emb) in embeddings.iter().enumerate().skip(1) {
        let score = cosine_similarity(query_embedding, doc_emb)?;
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

    assert_eq!(best_idx, 0, "unexpected top match for query");
    Ok(())
}
