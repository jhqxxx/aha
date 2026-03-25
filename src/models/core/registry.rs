use anyhow::Result;

use crate::models::{
    ModelInstance, WhichModel, load_model_legacy, qwen3::generate::Qwen3GenerateModel,
    qwen3_5::generate::Qwen3_5GenerateModel, qwen3_embedding::generate::Qwen3EmbeddingModel,
    qwen3_reranker::generate::Qwen3RerankerModel,
};

use super::artifact::LoadSpec;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelLoaderFamily {
    Qwen3,
    Qwen3Embedding,
    Qwen3Reranker,
    Qwen3_5,
    Legacy,
}

fn resolve_model_loader_family(model: WhichModel) -> ModelLoaderFamily {
    match model {
        WhichModel::Qwen3_0_6B => ModelLoaderFamily::Qwen3,
        WhichModel::Qwen3Embedding0_6B
        | WhichModel::Qwen3Embedding4B
        | WhichModel::Qwen3Embedding8B => ModelLoaderFamily::Qwen3Embedding,
        WhichModel::Qwen3Reranker0_6B
        | WhichModel::Qwen3Reranker4B
        | WhichModel::Qwen3Reranker8B => ModelLoaderFamily::Qwen3Reranker,
        WhichModel::Qwen3_5_0_8B
        | WhichModel::Qwen3_5_2B
        | WhichModel::Qwen3_5_4B
        | WhichModel::Qwen3_5_9B
        | WhichModel::Qwen3_5Gguf
        | WhichModel::Qwen3_5_0_8BUnslothGguf
        | WhichModel::Qwen3_5_2BUnslothGguf
        | WhichModel::Qwen3_5_4BUnslothGguf
        | WhichModel::Qwen3_5_0_8BLmstudioGguf
        | WhichModel::Qwen3_5_2BLmstudioGguf
        | WhichModel::Qwen3_5_4BLmstudioGguf => ModelLoaderFamily::Qwen3_5,
        _ => ModelLoaderFamily::Legacy,
    }
}

pub fn load_model_from_spec<'a>(spec: &LoadSpec) -> Result<ModelInstance<'a>> {
    spec.validate()?;
    let model = match resolve_model_loader_family(spec.model) {
        ModelLoaderFamily::Qwen3 => {
            ModelInstance::Qwen3(Qwen3GenerateModel::init_from_spec(spec, None, None)?)
        }
        ModelLoaderFamily::Qwen3Embedding => {
            ModelInstance::Qwen3Embedding(Qwen3EmbeddingModel::init_from_spec(spec, None, None)?)
        }
        ModelLoaderFamily::Qwen3Reranker => {
            ModelInstance::Qwen3Reranker(Qwen3RerankerModel::init_from_spec(spec, None, None)?)
        }
        ModelLoaderFamily::Qwen3_5 => {
            ModelInstance::Qwen3_5(Qwen3_5GenerateModel::init_from_spec(spec, None, None)?)
        }
        ModelLoaderFamily::Legacy => {
            let weight_path = spec.paths.weight_dir.as_deref().unwrap_or_default();
            let gguf = spec.paths.gguf_path.as_deref();
            let mmproj = spec.paths.mmproj_path.as_deref();
            return load_model_legacy(spec.model, weight_path, gguf, mmproj);
        }
    };
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::{ModelLoaderFamily, resolve_model_loader_family};
    use crate::models::WhichModel;

    #[test]
    fn registry_routes_qwen3_family_models() {
        assert_eq!(
            resolve_model_loader_family(WhichModel::Qwen3_0_6B),
            ModelLoaderFamily::Qwen3
        );
        assert_eq!(
            resolve_model_loader_family(WhichModel::Qwen3Embedding0_6B),
            ModelLoaderFamily::Qwen3Embedding
        );
        assert_eq!(
            resolve_model_loader_family(WhichModel::Qwen3Reranker0_6B),
            ModelLoaderFamily::Qwen3Reranker
        );
    }

    #[test]
    fn registry_routes_qwen3_5_variants_to_unified_loader() {
        assert_eq!(
            resolve_model_loader_family(WhichModel::Qwen3_5_0_8B),
            ModelLoaderFamily::Qwen3_5
        );
        assert_eq!(
            resolve_model_loader_family(WhichModel::Qwen3_5_9B),
            ModelLoaderFamily::Qwen3_5
        );
        assert_eq!(
            resolve_model_loader_family(WhichModel::Qwen3_5_2BLmstudioGguf),
            ModelLoaderFamily::Qwen3_5
        );
    }

    #[test]
    fn registry_keeps_non_family_models_on_legacy_path() {
        assert_eq!(
            resolve_model_loader_family(WhichModel::MiniCPM4_0_5B),
            ModelLoaderFamily::Legacy
        );
        assert_eq!(
            resolve_model_loader_family(WhichModel::DeepSeekOCR),
            ModelLoaderFamily::Legacy
        );
    }
}
