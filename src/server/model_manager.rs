use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use candle_core::{DType, Device};
use tokenizers::Tokenizer;

use crate::models::{ModelInstance, common::model_mapping::WhichModel};

/// 共享资源配置 - 所有模型共用
pub struct SharedResources {
    /// 计算设备 (CPU/GPU)
    pub device: Device,
    /// 数据类型 (F16/F32等)
    pub dtype: DType,
    /// Tokenizer 缓存池 (避免重复加载)
    pub tokenizer_cache: RwLock<HashMap<String, Arc<Tokenizer>>>,
}

impl SharedResources {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self {
            device,
            dtype,
            tokenizer_cache: RwLock::new(HashMap::new()),
        }
    }

    /// 获取或缓存 tokenizer
    pub async fn get_or_load_tokenizer(&self, model_id: &str, loader: impl FnOnce() -> anyhow::Result<Tokenizer>) -> anyhow::Result<Arc<Tokenizer>> {
        // 先尝试从缓存读取
        {
            let cache = self.tokenizer_cache.read().await;
            if let Some(tokenizer) = cache.get(model_id) {
                return Ok(tokenizer.clone());
            }
        }
        
        // 缓存未命中，加载并存储
        let tokenizer = loader()?;
        let tokenizer = Arc::new(tokenizer);
        
        {
            let mut cache = self.tokenizer_cache.write().await;
            cache.insert(model_id.to_string(), tokenizer.clone());
        }
        
        Ok(tokenizer)
    }
}

/// 单个模型的完整信息
pub struct ModelEntry {
    pub which_model: WhichModel,
    pub instance: ModelInstance<'static>,
    pub model_id: String,
}

/// 多模型管理器 - 支持同时运行多个模型
pub struct MultiModelManager {
    /// 共享资源
    pub shared: Arc<SharedResources>,
    /// 已加载的模型注册表: model_id -> ModelEntry
    pub models: RwLock<HashMap<String, Arc<ModelEntry>>>,
}

impl MultiModelManager {
    pub fn new(shared: Arc<SharedResources>) -> Self {
        Self {
            shared,
            models: RwLock::new(HashMap::new()),
        }
    }

    /// 注册一个新模型
    pub async fn register_model(&self, model_id: String, entry: ModelEntry) {
        let mut models = self.models.write().await;
        models.insert(model_id, Arc::new(entry));
    }

    /// 获取指定模型
    pub async fn get_model(&self, model_id: &str) -> Option<Arc<ModelEntry>> {
        let models = self.models.read().await;
        models.get(model_id).cloned()
    }

    /// 列出所有已加载的模型
    pub async fn list_models(&self) -> Vec<String> {
        let models = self.models.read().await;
        models.keys().cloned().collect()
    }

    /// 卸载指定模型
    pub async fn unload_model(&self, model_id: &str) -> bool {
        let mut models = self.models.write().await;
        models.remove(model_id).is_some()
    }

    /// 检查模型是否存在
    pub async fn has_model(&self, model_id: &str) -> bool {
        let models = self.models.read().await;
        models.contains_key(model_id)
    }
}
