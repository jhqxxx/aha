use anyhow::{Result, anyhow};

use crate::models::WhichModel;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactKind {
    Auto,
    Safetensors,
    Gguf,
    Onnx,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelPaths {
    pub weight_dir: Option<String>,
    pub gguf_path: Option<String>,
    pub mmproj_path: Option<String>,
    pub onnx_path: Option<String>,
    pub tokenizer_dir: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LoadSpec {
    pub model: WhichModel,
    pub artifact: ArtifactKind,
    pub paths: ModelPaths,
}

pub fn supported_artifacts(model: WhichModel) -> &'static [ArtifactKind] {
    match model {
        WhichModel::AllMiniLML6V2 => &[
            ArtifactKind::Safetensors,
            ArtifactKind::Gguf,
            ArtifactKind::Onnx,
        ],
        WhichModel::Qwen3_0_6B => &[
            ArtifactKind::Safetensors,
            ArtifactKind::Gguf,
            ArtifactKind::Onnx,
        ],
        WhichModel::Qwen3Embedding0_6B
        | WhichModel::Qwen3Embedding4B
        | WhichModel::Qwen3Embedding8B
        | WhichModel::Qwen3Reranker0_6B
        | WhichModel::Qwen3Reranker4B
        | WhichModel::Qwen3Reranker8B => &[
            ArtifactKind::Safetensors,
            ArtifactKind::Gguf,
            ArtifactKind::Onnx,
        ],
        WhichModel::Qwen3_5_0_8B
        | WhichModel::Qwen3_5_2B
        | WhichModel::Qwen3_5_4B
        | WhichModel::Qwen3_5_9B => &[
            ArtifactKind::Safetensors,
            ArtifactKind::Gguf,
            ArtifactKind::Onnx,
        ],
        WhichModel::Qwen3_5Gguf
        | WhichModel::Qwen3_5_0_8BUnslothGguf
        | WhichModel::Qwen3_5_2BUnslothGguf
        | WhichModel::Qwen3_5_4BUnslothGguf
        | WhichModel::Qwen3_5_0_8BLmstudioGguf
        | WhichModel::Qwen3_5_2BLmstudioGguf
        | WhichModel::Qwen3_5_4BLmstudioGguf => &[ArtifactKind::Gguf],
        _ => &[ArtifactKind::Safetensors],
    }
}

pub fn default_artifact(model: WhichModel) -> ArtifactKind {
    match model {
        WhichModel::Qwen3_5Gguf
        | WhichModel::Qwen3_5_0_8BUnslothGguf
        | WhichModel::Qwen3_5_2BUnslothGguf
        | WhichModel::Qwen3_5_4BUnslothGguf
        | WhichModel::Qwen3_5_0_8BLmstudioGguf
        | WhichModel::Qwen3_5_2BLmstudioGguf
        | WhichModel::Qwen3_5_4BLmstudioGguf => ArtifactKind::Gguf,
        _ => ArtifactKind::Safetensors,
    }
}

impl LoadSpec {
    pub fn for_safetensors(model: WhichModel, weight_dir: impl Into<String>) -> Self {
        Self {
            model,
            artifact: ArtifactKind::Safetensors,
            paths: ModelPaths {
                weight_dir: Some(weight_dir.into()),
                ..Default::default()
            },
        }
    }

    pub fn resolved_artifact(&self) -> ArtifactKind {
        match self.artifact {
            ArtifactKind::Auto => default_artifact(self.model),
            artifact => artifact,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let artifact = self.resolved_artifact();
        if !supported_artifacts(self.model).contains(&artifact) {
            return Err(anyhow!(
                "model {} does not support artifact {:?}",
                self.model.openai_model_id(),
                artifact
            ));
        }

        match artifact {
            ArtifactKind::Auto => unreachable!("artifact should be resolved before validation"),
            ArtifactKind::Safetensors => {
                if self.paths.weight_dir.is_none() {
                    return Err(anyhow!(
                        "weight_path is required for {} with safetensors",
                        self.model.openai_model_id()
                    ));
                }
            }
            ArtifactKind::Gguf => {
                if self.paths.gguf_path.is_none() {
                    return Err(anyhow!(
                        "gguf_path is required for {} with gguf",
                        self.model.openai_model_id()
                    ));
                }
            }
            ArtifactKind::Onnx => {
                if self.paths.onnx_path.is_none() {
                    return Err(anyhow!(
                        "onnx_path is required for {} with onnx",
                        self.model.openai_model_id()
                    ));
                }
            }
        }
        Ok(())
    }
}
