use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use crate::models::load_gguf_model;
use crate::models::{GenerateModel, ModelInstance, common::model_mapping::WhichModel, load_model};
use crate::params::chat::ChatCompletionParameters;
use crate::utils::string_to_static_str;
use anyhow::anyhow;
use rocket::futures::StreamExt;
use rocket::serde::{Serialize, json::Json};
use rocket::{
    Request, State,
    futures::Stream,
    get,
    http::{ContentType, Status},
    post,
    response::{Responder, stream::TextStream},
};
use tokio::sync::RwLock;

use crate::server::model_manager::{ModelEntry, MultiModelManager, SharedResources};
use crate::server::process::cleanup_pid_file;

/// Wrapper to store model type together with the model instance
pub(crate) struct StoredModel {
    pub which_model: WhichModel,
    pub instance: ModelInstance<'static>,
}

// 保留旧的单模型接口用于向后兼容
pub(crate) static MODEL: OnceLock<Arc<RwLock<StoredModel>>> = OnceLock::new();

// 新的多模型管理器
pub(crate) static MULTI_MODEL_MANAGER: OnceLock<Arc<MultiModelManager>> = OnceLock::new();

static SHUTDOWN_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static SERVER_PORT: OnceLock<u16> = OnceLock::new();
static ALLOW_REMOTE_SHUTDOWN: OnceLock<bool> = OnceLock::new();

/// 初始化共享资源（只调用一次）
pub fn init_shared_resources() -> Arc<SharedResources> {
    use candle_core::{Device, DType};
    
    // 检测可用的设备
    let device = if cfg!(feature = "cuda") {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    } else if cfg!(feature = "metal") {
        Device::new_metal(0).unwrap_or(Device::Cpu)
    } else {
        Device::Cpu
    };
    
    // 根据设备选择最优数据类型
    let dtype = match &device {
        Device::Cuda(_) => DType::F16,  // GPU 使用 F16 节省显存
        Device::Metal(_) => DType::F16,
        Device::Cpu => DType::F32,
    };
    
    let shared = Arc::new(SharedResources::new(device, dtype));
    MULTI_MODEL_MANAGER.get_or_init(|| {
        Arc::new(MultiModelManager::new(shared.clone()))
    });
    
    shared
}

/// 加载单个模型并注册到管理器
pub fn load_and_register_model(
    model_type: WhichModel,
    path: String,
    gguf: Option<String>,
    mmproj: Option<String>,
) -> anyhow::Result<()> {
    let manager = MULTI_MODEL_MANAGER.get().ok_or_else(|| {
        anyhow!("Multi-model manager not initialized. Call init_shared_resources first.")
    })?;
    
    let model_id = format!("{:?}", model_type);
    
    // 检查模型是否已加载
    if tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(manager.has_model(&model_id))
    }) {
        println!("Model {} already loaded, skipping...", model_id);
        return Ok(());
    }
    
    // 加载模型
    let instance = if model_type.is_gguf() {
        if let Some(gguf_path) = gguf {
            let gguf_path = string_to_static_str(gguf_path);
            let mmproj_path = mmproj.map(string_to_static_str);
            load_gguf_model(model_type, None, gguf_path, mmproj_path, None)?
        } else {
            return Err(anyhow!("gguf model need gguf model path"));
        }
    } else if model_type.is_onnx() {
        return Err(anyhow!("onnx comming soon but now not support"));
    } else {
        let model_path = string_to_static_str(path);
        load_model(model_type, model_path, None, None)?
    };
    
    // 注册模型
    let entry = ModelEntry {
        which_model: model_type,
        instance,
        model_id: model_id.clone(),
    };
    
    tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(manager.register_model(model_id.clone(), entry))
    });
    
    println!("Model {} registered successfully", model_id);
    Ok(())
}

pub fn init(
    model_type: WhichModel,
    path: String,
    gguf: Option<String>,
    mmproj: Option<String>,
) -> anyhow::Result<()> {
    let model = if model_type.is_gguf() {
        if let Some(gguf_path) = gguf {
            let gguf_path = string_to_static_str(gguf_path);
            let mmproj_path = mmproj.map(string_to_static_str);
            load_gguf_model(model_type, None, gguf_path, mmproj_path, None)?
        } else {
            return Err(anyhow!("gguf model need gguf model path"));
        }
    } else if model_type.is_onnx() {
        return Err(anyhow!("onnx comming soon but now not support"));
    } else {
        let model_path = string_to_static_str(path);
        load_model(model_type, model_path, None, None)?
    };

    MODEL.get_or_init(|| {
        Arc::new(RwLock::new(StoredModel {
            which_model: model_type,
            instance: model,
        }))
    });
    Ok(())
}

pub fn set_server_port(port: u16, allow_remote_shutdown: bool) {
    SHUTDOWN_FLAG.get_or_init(|| Arc::new(AtomicBool::new(false)));
    SERVER_PORT.get_or_init(|| port);
    ALLOW_REMOTE_SHUTDOWN.get_or_init(|| allow_remote_shutdown);
}

#[allow(unused)]
pub fn get_shutdown_flag() -> Arc<AtomicBool> {
    SHUTDOWN_FLAG
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

pub(crate) enum Response<R: Stream<Item = String> + Send> {
    Stream(TextStream<R>),
    Text(String),
    Error(String),
}

impl<'r, 'o: 'r, R> Responder<'r, 'o> for Response<R>
where
    R: Stream<Item = String> + Send + 'o,
    'r: 'o,
{
    fn respond_to(self, req: &'r Request<'_>) -> rocket::response::Result<'o> {
        match self {
            Response::Stream(stream) => stream.respond_to(req),
            Response::Text(text) => text.respond_to(req),
            Response::Error(e) => {
                let mut res = rocket::response::Response::new();
                res.set_status(Status::InternalServerError);
                res.set_header(ContentType::JSON);
                res.set_sized_body(e.len(), std::io::Cursor::new(e));
                Ok(res)
            }
        }
    }
}

/// 根据 model 参数获取对应的模型实例
async fn get_model_by_name(model_name: &str) -> anyhow::Result<Arc<crate::server::model_manager::ModelEntry>> {
    let manager = MULTI_MODEL_MANAGER.get().ok_or_else(|| {
        anyhow::anyhow!("Multi-model manager not initialized")
    })?;
    
    // 尝试直接匹配 model_id
    if let Some(entry) = manager.get_model(model_name).await {
        return Ok(entry);
    }
    
    // 如果只有一个模型，直接使用它（向后兼容）
    let models = manager.list_models().await;
    if models.len() == 1 {
        if let Some(entry) = manager.get_model(&models[0]).await {
            return Ok(entry);
        }
    }
    
    Err(anyhow::anyhow!(
        "Model '{}' not found. Available models: {:?}",
        model_name,
        models
    ))
}

#[post("/completions", data = "<req>")]
pub(crate) async fn chat(
    req: Json<ChatCompletionParameters>,
) -> (ContentType, Response<impl Stream<Item = String> + Send>) {
    let params = req.into_inner();
    let model_name = params.model.clone();
    
    match params.stream {
        Some(false) => {
            let response = async {
                let entry = get_model_by_name(&model_name).await?;
                // TODO: use entry.instance directly once multi-model chat is fully implemented
                drop(entry);
                
                let model_ref = MODEL
                    .get()
                    .cloned()
                    .ok_or_else(|| anyhow::anyhow!("model not init"))?;
                let mut guard = model_ref.write().await;
                guard.instance.generate(params)
            }.await;
            
            match response {
                Ok(res) => {
                    let response_str = serde_json::to_string(&res).unwrap();
                    (ContentType::Text, Response::Text(response_str))
                }
                Err(e) => (ContentType::Text, Response::Error(e.to_string())),
            }
        }
        _ => {
            let text_stream = TextStream! {
                let model_ref = MODEL.get().cloned().ok_or_else(|| anyhow::anyhow!("model not init")).unwrap();
                let mut guard = model_ref.write().await;
                let stream_result = guard.instance.generate_stream(params);
                match stream_result {
                    Ok(stream) => {
                        let mut stream = pin!(stream);
                        while let Some(result) = stream.next().await {
                            match result {
                                Ok(chunk) => {
                                    if let Ok(json_str) = serde_json::to_string(&chunk) {
                                        yield format!("data: {}\n\n", json_str);
                                    }
                                }
                                Err(e) => {
                                    yield format!("data: {{\"error\": \"{}\"}}\n\n", e);
                                    break;
                                }
                            }
                        }
                        yield "data: [DONE]\n\n".to_string();
                    },
                    Err(e) => {
                        yield format!("event: error\ndata: {}\n\n", e.to_string());
                    }
                }
            };
            (ContentType::EventStream, Response::Stream(text_stream))
        }
    }
}

/// 列出所有已加载的模型
#[get("/models/list")]
pub(crate) async fn list_loaded_models() -> (ContentType, String) {
    let manager = match MULTI_MODEL_MANAGER.get() {
        Some(m) => m,
        None => {
            return (ContentType::JSON, serde_json::json!({"error": "Manager not initialized"}).to_string());
        }
    };
    
    let models = manager.list_models().await;
    let response = serde_json::json!({
        "models": models,
        "count": models.len()
    });
    
    (ContentType::JSON, response.to_string())
}

#[post("/remove_background", data = "<req>")]
pub(crate) async fn remove_background(req: Json<ChatCompletionParameters>) -> (Status, String) {
    let response = {
        let model_ref = MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("model not init"))
            .unwrap();
        let mut guard = model_ref.write().await;
        guard.instance.generate(req.into_inner())
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (Status::Ok, response_str)
        }
        Err(e) => (Status::InternalServerError, e.to_string()),
    }
}

#[post("/speech", data = "<req>")]
pub(crate) async fn speech(req: Json<ChatCompletionParameters>) -> (Status, String) {
    let response = {
        let model_ref = MODEL
            .get()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("model not init"))
            .unwrap();
        let mut guard = model_ref.write().await;
        guard.instance.generate(req.into_inner())
    };
    match response {
        Ok(res) => {
            let response_str = serde_json::to_string(&res).unwrap();
            (Status::Ok, response_str)
        }
        Err(e) => (Status::InternalServerError, e.to_string()),
    }
}

// Health check endpoint

#[derive(Serialize)]
pub(crate) struct HealthResponse {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[get("/health")]
pub(crate) async fn health() -> (Status, (ContentType, Json<HealthResponse>)) {
    if MODEL.get().is_some() {
        let response = HealthResponse {
            status: "ok".to_string(),
            error: None,
        };
        (Status::Ok, (ContentType::JSON, Json(response)))
    } else {
        let response = HealthResponse {
            status: "unhealthy".to_string(),
            error: Some("model not initialized".to_string()),
        };
        (
            Status::ServiceUnavailable,
            (ContentType::JSON, Json(response)),
        )
    }
}

// Models endpoint (OpenAI-compatible format)

/// OpenAI-compatible model object
#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: String,
    created: Option<i64>,
    owned_by: String,
}

/// OpenAI-compatible models list response
#[derive(Serialize)]
struct ModelsListResponse {
    object: String,
    data: Vec<ModelObject>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[get("/models")]
pub(crate) async fn models() -> (Status, (ContentType, Json<serde_json::Value>)) {
    if let Some(model_ref) = MODEL.get() {
        let guard = model_ref.read().await;
        let which_model = guard.which_model;

        let model_obj = ModelObject {
            id: which_model.as_string(),
            object: "model".to_string(),
            created: None, // We don't track creation time
            owned_by: which_model.model_owner(),
        };
        drop(guard);

        let response = ModelsListResponse {
            object: "list".to_string(),
            data: vec![model_obj],
        };
        (
            Status::Ok,
            (
                ContentType::JSON,
                Json(serde_json::to_value(response).unwrap()),
            ),
        )
    } else {
        let response = ErrorResponse {
            error: "model not initialized".to_string(),
        };
        (
            Status::ServiceUnavailable,
            (
                ContentType::JSON,
                Json(serde_json::to_value(response).unwrap()),
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test health endpoint when model is not initialized
    #[tokio::test]
    async fn test_health_endpoint_uninitialized() {
        let (status, (content_type, response)) = health().await;
        assert_eq!(status, Status::ServiceUnavailable);
        assert_eq!(content_type, ContentType::JSON);
        assert_eq!(response.status, "unhealthy");
        assert_eq!(response.error, Some("model not initialized".to_string()));
    }

    // Test health endpoint when model is initialized
    // Note: This test requires a model to be initialized, which may not be feasible
    // in unit tests without access to model files. This is a placeholder for integration tests.
    //
    // #[tokio::test]
    // async fn test_health_endpoint_initialized() {
    //     // This would require model initialization
    //     // Consider moving to integration tests
    // }

    // Test models endpoint when model is not initialized
    #[tokio::test]
    async fn test_models_endpoint_uninitialized() {
        let (status, (content_type, response)) = models().await;
        assert_eq!(status, Status::ServiceUnavailable);
        assert_eq!(content_type, ContentType::JSON);
        let error = response.get("error").and_then(|v| v.as_str());
        assert_eq!(error, Some("model not initialized"));
    }

    // Test model type classification
    #[test]
    fn test_get_model_type_llm() {
        assert_eq!(WhichModel::Qwen3_0_6B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen3VL2B.model_type(), "llm");
        assert_eq!(WhichModel::MiniCPM4_0_5B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen2_5VL3B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen2_5VL7B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen3VL4B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen3VL8B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen3VL32B.model_type(), "llm");
    }

    #[test]
    fn test_get_model_type_ocr() {
        assert_eq!(WhichModel::DeepSeekOCR.model_type(), "ocr");
        assert_eq!(WhichModel::HunyuanOCR.model_type(), "ocr");
        assert_eq!(WhichModel::PaddleOCRVL.model_type(), "ocr");
    }

    #[test]
    fn test_get_model_type_asr() {
        assert_eq!(WhichModel::Qwen3ASR0_6B.model_type(), "asr");
        assert_eq!(WhichModel::Qwen3ASR1_7B.model_type(), "asr");
        assert_eq!(WhichModel::GlmASRNano2512.model_type(), "asr");
        assert_eq!(WhichModel::FunASRNano2512.model_type(), "asr");
    }

    #[test]
    fn test_get_model_type_image() {
        assert_eq!(WhichModel::RMBG2_0.model_type(), "image");
    }

    #[test]
    fn test_get_model_type_tts() {
        assert_eq!(WhichModel::VoxCPM.model_type(), "image");
        assert_eq!(WhichModel::VoxCPM1_5.model_type(), "image");
    }
}

// Shutdown endpoint
#[derive(Serialize)]
struct ShutdownResponse {
    message: String,
}

#[post("/shutdown")]
pub(crate) async fn shutdown(
    shutdown_flag: &State<Arc<AtomicBool>>,
) -> (Status, (ContentType, Json<serde_json::Value>)) {
    // Check if remote shutdown is allowed
    let allow_remote = ALLOW_REMOTE_SHUTDOWN.get().copied().unwrap_or(false);

    // Log the shutdown request
    eprintln!(
        "[SHUTDOWN] Shutdown requested (remote_allowed: {})",
        allow_remote
    );

    // Note: Rocket 0.5 doesn't provide easy access to client IP in request guards
    // For proper IP-based filtering, you would need to use custom request guards
    // or middleware. For now, we rely on the --allow-remote-shutdown flag.

    shutdown_flag.store(true, Ordering::SeqCst);

    // Cleanup PID file in a background task
    if let Some(&port) = SERVER_PORT.get() {
        let _ = cleanup_pid_file(port);
    }

    // Schedule shutdown after a short delay to allow response to be sent
    let _flag = shutdown_flag.inner().clone();
    tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        std::process::exit(0);
    });

    let response = ShutdownResponse {
        message: "Shutting down...".to_string(),
    };
    (
        Status::Ok,
        (
            ContentType::JSON,
            Json(serde_json::to_value(response).unwrap()),
        ),
    )
}
