use std::{fs, path::Path};

use aha::models::{
    ArtifactKind, GenerateModel, LoadSpec, ModelPaths, WhichModel,
    qwen3::generate::Qwen3GenerateModel,
};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

const DEFAULT_QWEN3_SAFETENSORS_DIRS: &[&str] = &[
    r"D:\model_download\Qwen3-0.6B",
    r"D:\model_download\Qwen\Qwen3-0.6B",
];
const DEFAULT_QWEN3_GGUF_FILES: &[&str] = &[
    r"D:\model_download\Qwen3-0.6B-GGUF\Qwen3-0.6B-Q8_0.gguf",
    r"D:\model_download\Qwen\Qwen3-0.6B-GGUF\Qwen3-0.6B-Q8_0.gguf",
];
#[cfg(feature = "onnx-runtime")]
const DEFAULT_QWEN3_ONNX_DIRS: &[&str] = &[
    r"D:\model_download\Qwen3-0.6B-ONNX",
    r"D:\model_download\Qwen\Qwen3-0.6B-ONNX",
];

fn find_first_gguf_file(dir: &Path) -> Option<String> {
    let entries = fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path
            .extension()
            .and_then(|ext| ext.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            return Some(path.to_string_lossy().to_string());
        }
    }
    None
}

fn resolve_qwen3_safetensors_dir() -> Option<String> {
    if let Ok(path) = std::env::var("AHA_QWEN3_SAFETENSORS_DIR")
        && Path::new(&path).is_dir()
    {
        return Some(path);
    }

    for path in DEFAULT_QWEN3_SAFETENSORS_DIRS {
        if Path::new(path).is_dir() {
            return Some((*path).to_string());
        }
    }

    let save_dir = aha::utils::get_default_save_dir()?;
    let managed_path = format!("{save_dir}/Qwen/Qwen3-0.6B");
    if Path::new(&managed_path).is_dir() {
        return Some(managed_path);
    }
    None
}

fn resolve_qwen3_gguf_path() -> Option<String> {
    if let Ok(path) = std::env::var("AHA_QWEN3_GGUF_PATH")
        && Path::new(&path).is_file()
    {
        return Some(path);
    }

    for path in DEFAULT_QWEN3_GGUF_FILES {
        if Path::new(path).is_file() {
            return Some((*path).to_string());
        }
    }

    let save_dir = aha::utils::get_default_save_dir()?;
    let managed_dir = format!("{save_dir}/Qwen/Qwen3-0.6B-GGUF");
    find_first_gguf_file(Path::new(&managed_dir))
}

#[cfg(feature = "onnx-runtime")]
fn resolve_qwen3_onnx_dir() -> Option<String> {
    if let Ok(path) = std::env::var("AHA_QWEN3_ONNX_DIR")
        && Path::new(&path).is_dir()
    {
        return Some(path);
    }

    for path in DEFAULT_QWEN3_ONNX_DIRS {
        if Path::new(path).is_dir() {
            return Some((*path).to_string());
        }
    }
    None
}

fn build_text_request() -> Result<ChatCompletionParameters> {
    let payload = serde_json::json!({
        "model": "qwen3-0.6b",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": "请用一句话介绍 Rust。"
            }
        ]
    });
    Ok(serde_json::from_value(payload)?)
}

#[test]
fn qwen3_safetensors_init_from_spec_can_generate() -> Result<()> {
    let Some(weight_dir) = resolve_qwen3_safetensors_dir() else {
        println!("skip qwen3 safetensors test: model dir not found, set AHA_QWEN3_SAFETENSORS_DIR");
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::Qwen3_0_6B,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(weight_dir),
            ..Default::default()
        },
    };

    let mut model = Qwen3GenerateModel::init_from_spec(&spec, None, None)?;
    let response = model.generate(build_text_request()?)?;
    let value = serde_json::to_value(response)?;
    let choices_len = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .map_or(0, |choices| choices.len());
    assert!(choices_len > 0, "expected at least one generated choice");
    Ok(())
}

#[test]
fn qwen3_load_spec_accepts_gguf() {
    let spec = LoadSpec {
        model: WhichModel::Qwen3_0_6B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some("D:/model_download/Qwen3-0.6B-GGUF/model.gguf".to_string()),
            ..Default::default()
        },
    };

    spec.validate().expect("qwen3 should accept gguf artifact");
}

#[test]
fn qwen3_load_spec_accepts_onnx() {
    let spec = LoadSpec {
        model: WhichModel::Qwen3_0_6B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some("D:/model_download/Qwen3-0.6B-ONNX".to_string()),
            ..Default::default()
        },
    };

    spec.validate().expect("qwen3 should accept onnx artifact");
}

#[test]
fn qwen3_gguf_init_from_spec_can_generate() -> Result<()> {
    let Some(gguf_path) = resolve_qwen3_gguf_path() else {
        println!("skip qwen3 gguf test: model file not found, set AHA_QWEN3_GGUF_PATH");
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::Qwen3_0_6B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path),
            ..Default::default()
        },
    };

    let mut model = Qwen3GenerateModel::init_from_spec(&spec, None, None)?;
    let response = model.generate(build_text_request()?)?;
    let value = serde_json::to_value(response)?;
    let choices_len = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .map_or(0, |choices| choices.len());
    assert!(choices_len > 0, "expected at least one generated choice");
    Ok(())
}

#[test]
#[cfg(feature = "onnx-runtime")]
fn qwen3_onnx_init_from_spec_can_generate() -> Result<()> {
    use aha::models::common::onnx::ensure_ort_dylib_path;

    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip qwen3 onnx test: {err}");
        return Ok(());
    }

    let Some(onnx_dir) = resolve_qwen3_onnx_dir() else {
        println!("skip qwen3 onnx test: onnx dir not found, set AHA_QWEN3_ONNX_DIR");
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::Qwen3_0_6B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(onnx_dir.clone()),
            tokenizer_dir: Some(onnx_dir),
            ..Default::default()
        },
    };

    let mut model = Qwen3GenerateModel::init_from_spec(&spec, None, None)?;
    let response = model.generate(build_text_request()?)?;
    let value = serde_json::to_value(response)?;
    let choices_len = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .map_or(0, |choices| choices.len());
    assert!(choices_len > 0, "expected at least one generated choice");
    Ok(())
}
