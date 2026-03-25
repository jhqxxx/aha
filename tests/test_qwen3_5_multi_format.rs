use std::path::{Path, PathBuf};

use aha::models::{
    ArtifactKind, GenerateModel, LoadSpec, ModelPaths, WhichModel,
    qwen3_5::generate::Qwen3_5GenerateModel,
};
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;

const DEFAULT_QWEN3_5_SAFETENSORS_DIR: &str = r"D:\model_download\Qwen3.5-0.8B";
#[cfg(feature = "onnx-runtime")]
const DEFAULT_QWEN3_5_ONNX_DIR: &str = r"D:\model_download\Qwen3.5-0.8B-ONNX";
const DEFAULT_QWEN3_5_GGUF_DIRS: &[&str] = &[
    r"D:\model_download\Qwen3.5-0.8B-GGUF",
    r"D:\model_download\Qwen3.5-0.8B-gguf",
    r"D:\model_download\Qwen3.5-2B-GGUF",
    r"D:\model_download\Qwen3.5-4B-GGUF",
];

fn env_or_default(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn existing_dir(path: &str) -> bool {
    let p = Path::new(path);
    p.exists() && p.is_dir()
}

fn first_file_with_extension_recursive(dir: &str, extension: &str) -> Result<Option<PathBuf>> {
    if !existing_dir(dir) {
        return Ok(None);
    }
    let mut stack = vec![PathBuf::from(dir)];
    let mut matches = Vec::new();
    while let Some(current) = stack.pop() {
        for entry in std::fs::read_dir(&current)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case(extension))
            {
                matches.push(path);
            }
        }
    }
    matches.sort();
    Ok(matches.into_iter().next())
}

fn resolve_gguf_path() -> Result<Option<String>> {
    if let Ok(path) = std::env::var("AHA_QWEN3_5_GGUF_PATH")
        && Path::new(&path).exists()
    {
        return Ok(Some(path));
    }
    for dir in DEFAULT_QWEN3_5_GGUF_DIRS {
        if let Some(path) = first_file_with_extension_recursive(dir, "gguf")? {
            return Ok(Some(path.to_string_lossy().to_string()));
        }
    }
    Ok(None)
}

fn build_text_request() -> Result<ChatCompletionParameters> {
    let payload = serde_json::json!({
        "model": "qwen3.5-0.8b",
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
fn qwen3_5_safetensors_init_from_spec_can_generate() -> Result<()> {
    let weight_dir = env_or_default(
        "AHA_QWEN3_5_SAFETENSORS_DIR",
        DEFAULT_QWEN3_5_SAFETENSORS_DIR,
    );
    if !existing_dir(&weight_dir) {
        println!("skip safetensors test: dir not found, set AHA_QWEN3_5_SAFETENSORS_DIR to run");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::Qwen3_5_0_8B,
        artifact: ArtifactKind::Safetensors,
        paths: ModelPaths {
            weight_dir: Some(weight_dir),
            ..Default::default()
        },
    };
    let mut model = Qwen3_5GenerateModel::init_from_spec(&spec, None, None)?;
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
fn qwen3_5_gguf_init_from_spec_can_generate() -> Result<()> {
    let Some(gguf_path) = resolve_gguf_path()? else {
        println!("skip gguf test: no gguf file found, set AHA_QWEN3_5_GGUF_PATH to run explicitly");
        return Ok(());
    };

    let spec = LoadSpec {
        model: WhichModel::Qwen3_5_0_8B,
        artifact: ArtifactKind::Gguf,
        paths: ModelPaths {
            gguf_path: Some(gguf_path),
            ..Default::default()
        },
    };
    let mut model = Qwen3_5GenerateModel::init_from_spec(&spec, None, None)?;
    let response = model.generate(build_text_request()?)?;
    let value = serde_json::to_value(response)?;
    let choices_len = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .map_or(0, |choices| choices.len());
    assert!(choices_len > 0, "expected at least one generated choice");
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn qwen3_5_onnx_init_from_spec_can_generate() -> Result<()> {
    use aha::models::common::onnx::ensure_ort_dylib_path;

    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip onnx test: {err}");
        return Ok(());
    }

    let onnx_dir = env_or_default("AHA_QWEN3_5_ONNX_DIR", DEFAULT_QWEN3_5_ONNX_DIR);
    if !existing_dir(&onnx_dir) {
        println!("skip onnx test: dir not found, set AHA_QWEN3_5_ONNX_DIR to run");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::Qwen3_5_0_8B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(onnx_dir.clone()),
            tokenizer_dir: Some(onnx_dir),
            ..Default::default()
        },
    };
    let mut model = Qwen3_5GenerateModel::init_from_spec(&spec, None, None)?;
    let response = model.generate(build_text_request()?)?;
    let value = serde_json::to_value(response)?;
    let choices_len = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .map_or(0, |choices| choices.len());
    assert!(choices_len > 0, "expected at least one generated choice");
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn qwen3_5_onnx_image_multimodal_can_generate() -> Result<()> {
    use aha::models::common::onnx::ensure_ort_dylib_path;

    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip onnx multimodal image test: {err}");
        return Ok(());
    }

    let onnx_dir = env_or_default("AHA_QWEN3_5_ONNX_DIR", DEFAULT_QWEN3_5_ONNX_DIR);
    if !existing_dir(&onnx_dir) {
        println!("skip onnx multimodal image test: dir not found");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::Qwen3_5_0_8B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(onnx_dir.clone()),
            tokenizer_dir: Some(onnx_dir),
            ..Default::default()
        },
    };
    let mut model = Qwen3_5GenerateModel::init_from_spec(&spec, None, None)?;
    let image_path = std::env::current_dir()?
        .join("assets")
        .join("img")
        .join("ocr_test1.png");
    if !image_path.exists() {
        println!(
            "skip onnx multimodal image test: local image not found at {}",
            image_path.display()
        );
        return Ok(());
    }
    let image_url = format!(
        "file:///{}",
        image_path.to_string_lossy().replace('\\', "/")
    );
    let payload = serde_json::json!({
        "model": "qwen3.5-0.8b",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": "识别图像中的文字"
                    }
                ]
            }
        ]
    });
    let request: ChatCompletionParameters = serde_json::from_value(payload)?;
    let response = model.generate(request)?;
    let value = serde_json::to_value(response)?;
    let choices_len = value
        .get("choices")
        .and_then(|choices| choices.as_array())
        .map_or(0, |choices| choices.len());
    assert!(
        choices_len > 0,
        "expected at least one generated choice for multimodal image request"
    );
    Ok(())
}

#[cfg(feature = "onnx-runtime")]
#[test]
fn qwen3_5_onnx_video_multimodal_is_rejected() -> Result<()> {
    use aha::models::common::onnx::ensure_ort_dylib_path;

    if let Err(err) = ensure_ort_dylib_path() {
        println!("skip onnx multimodal video rejection test: {err}");
        return Ok(());
    }

    let onnx_dir = env_or_default("AHA_QWEN3_5_ONNX_DIR", DEFAULT_QWEN3_5_ONNX_DIR);
    if !existing_dir(&onnx_dir) {
        println!("skip onnx multimodal video rejection test: dir not found");
        return Ok(());
    }

    let spec = LoadSpec {
        model: WhichModel::Qwen3_5_0_8B,
        artifact: ArtifactKind::Onnx,
        paths: ModelPaths {
            onnx_path: Some(onnx_dir.clone()),
            tokenizer_dir: Some(onnx_dir),
            ..Default::default()
        },
    };
    let mut model = Qwen3_5GenerateModel::init_from_spec(&spec, None, None)?;
    let payload = serde_json::json!({
        "model": "qwen3.5-0.8b",
        "max_tokens": 8,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video_url": {"url": "file://./assets/video/dummy.mp4"}
                    },
                    {
                        "type": "text",
                        "text": "描述视频内容"
                    }
                ]
            }
        ]
    });
    let request: ChatCompletionParameters = serde_json::from_value(payload)?;
    let err = model
        .generate(request)
        .expect_err("onnx backend should reject video multimodal input for now");
    assert!(
        err.to_string().contains("audio/video") || err.to_string().contains("video"),
        "unexpected error: {err}"
    );
    Ok(())
}
