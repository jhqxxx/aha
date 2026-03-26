use aha::models::{ArtifactKind, LoadSpec, ModelPaths, WhichModel};

#[test]
fn load_spec_auto_resolves_to_safetensors_default() {
    let spec = LoadSpec {
        model: WhichModel::Qwen3Embedding0_6B,
        artifact: ArtifactKind::Auto,
        paths: ModelPaths {
            weight_dir: Some("D:/model_download/Qwen3-Embedding-0.6B".to_string()),
            ..Default::default()
        },
    };

    assert_eq!(spec.resolved_artifact(), ArtifactKind::Safetensors);
    assert!(spec.validate().is_ok());
}

#[test]
fn load_spec_all_minilm_accepts_onnx() {
    let spec = LoadSpec {
        model: WhichModel::AllMiniLML6V2,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some("D:/model_download/all-MiniLM-L6-v2/onnx".to_string()),
            tokenizer_dir: Some("D:/model_download/all-MiniLM-L6-v2".to_string()),
            ..Default::default()
        },
    };

    assert!(spec.validate().is_ok());
}

#[test]
fn load_spec_all_minilm_accepts_gguf() {
    let spec = LoadSpec {
        model: WhichModel::AllMiniLML6V2,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(
                "D:/model_download/All-MiniLM-L6-v2-Embedding-GGUF/all-MiniLM-L6-v2-ggml-model-f16.gguf"
                    .to_string(),
            ),
            tokenizer_dir: Some("D:/model_download/all-MiniLM-L6-v2".to_string()),
            ..Default::default()
        },
    };

    assert!(spec.validate().is_ok());
}

#[test]
fn load_spec_gguf_requires_gguf_path() {
    let spec = LoadSpec {
        model: WhichModel::Qwen3_5Gguf,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths::default(),
    };

    let err = spec.validate().unwrap_err().to_string();
    assert!(err.contains("gguf_path is required"));
}

#[test]
fn load_spec_onnx_requires_onnx_path() {
    let spec = LoadSpec {
        model: WhichModel::Qwen3Embedding0_6B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths::default(),
    };

    let err = spec.validate().unwrap_err().to_string();
    assert!(err.contains("onnx_path is required"));
}

#[test]
fn load_spec_rejects_unsupported_artifact() {
    let spec = LoadSpec {
        model: WhichModel::MiniCPM4_0_5B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some("D:/tmp/model.onnx".to_string()),
            ..Default::default()
        },
    };

    let err = spec.validate().unwrap_err().to_string();
    assert!(err.contains("does not support artifact"));
}

#[test]
fn load_spec_accepts_qwen3_5_onnx() {
    let spec = LoadSpec {
        model: WhichModel::Qwen3_5_0_8B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some("D:/model_download/Qwen3.5-0.8B-ONNX".to_string()),
            ..Default::default()
        },
    };

    assert!(spec.validate().is_ok());
}
