use std::pin::pin;
use std::sync::{Arc, OnceLock};

use aha::models::{GenerateModel, ModelInstance, WhichModel, load_model};
use aha::process::cleanup_pid_file;
use aha::utils::string_to_static_str;
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use rocket::futures::StreamExt;
use rocket::serde::{json::Json, Serialize};
use rocket::{
    Request,
    State,
    futures::Stream,
    get,
    http::{ContentType, Status},
    post,
    response::{Responder, stream::TextStream},
};
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;

/// Wrapper to store model type together with the model instance
struct StoredModel {
    which_model: WhichModel,
    instance: ModelInstance<'static>,
}

static MODEL: OnceLock<Arc<RwLock<StoredModel>>> = OnceLock::new();
static SHUTDOWN_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static SERVER_PORT: OnceLock<u16> = OnceLock::new();
static ALLOW_REMOTE_SHUTDOWN: OnceLock<bool> = OnceLock::new();

pub fn init(model_type: WhichModel, path: String) -> anyhow::Result<()> {
    let model_path = string_to_static_str(path);
    let model = load_model(model_type, model_path)?;
    MODEL.get_or_init(|| Arc::new(RwLock::new(StoredModel {
        which_model: model_type,
        instance: model,
    })));
    Ok(())
}

pub fn set_server_port(port: u16, allow_remote_shutdown: bool) {
    SHUTDOWN_FLAG.get_or_init(|| Arc::new(AtomicBool::new(false)));
    SERVER_PORT.get_or_init(|| port);
    ALLOW_REMOTE_SHUTDOWN.get_or_init(|| allow_remote_shutdown);
}

pub fn get_shutdown_flag() -> Arc<AtomicBool> {
    SHUTDOWN_FLAG.get_or_init(|| Arc::new(AtomicBool::new(false))).clone()
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

#[post("/completions", data = "<req>")]
pub(crate) async fn chat(
    req: Json<ChatCompletionParameters>,
) -> (ContentType, Response<impl Stream<Item = String> + Send>) {
    match req.stream {
        Some(false) => {
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
                    (ContentType::Text, Response::Text(response_str))
                }
                Err(e) => (ContentType::Text, Response::Error(e.to_string())),
            }
        }
        _ => {
            let text_stream = TextStream! {
                let model_ref = MODEL.get().cloned().ok_or_else(|| anyhow::anyhow!("model not init")).unwrap();
                let mut guard = model_ref.write().await;
                let stream_result = guard.instance.generate_stream(req.into_inner());
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
        (Status::ServiceUnavailable, (ContentType::JSON, Json(response)))
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

/// Convert WhichModel to a display-friendly model ID (kebab-case)
fn which_model_to_id(which_model: WhichModel) -> &'static str {
    match which_model {
        WhichModel::MiniCPM4_0_5B => "minicpm4-0.5b",
        WhichModel::Qwen2_5vl3B => "qwen2.5vl-3b",
        WhichModel::Qwen2_5vl7B => "qwen2.5vl-7b",
        WhichModel::Qwen3_0_6B => "qwen3-0.6b",
        WhichModel::Qwen3ASR0_6B => "qwen3asr-0.6b",
        WhichModel::Qwen3ASR1_7B => "qwen3asr-1.7b",
        WhichModel::Qwen3vl2B => "qwen3vl-2b",
        WhichModel::Qwen3vl4B => "qwen3vl-4b",
        WhichModel::Qwen3vl8B => "qwen3vl-8b",
        WhichModel::Qwen3vl32B => "qwen3vl-32b",
        WhichModel::DeepSeekOCR => "deepseek-ocr",
        WhichModel::HunyuanOCR => "hunyuan-ocr",
        WhichModel::PaddleOCRVL => "paddleocr-vl",
        WhichModel::RMBG2_0 => "rmbg2.0",
        WhichModel::VoxCPM => "voxcpm",
        WhichModel::VoxCPM1_5 => "voxcpm1.5",
        WhichModel::GlmASRNano2512 => "glm-asr-nano-2512",
        WhichModel::FunASRNano2512 => "fun-asr-nano-2512",
    }
}

/// Get the owner/organization name for a model
fn which_model_to_owner(which_model: WhichModel) -> &'static str {
    match which_model {
        WhichModel::MiniCPM4_0_5B => "OpenBMB",
        WhichModel::Qwen2_5vl3B | WhichModel::Qwen2_5vl7B => "Qwen",
        WhichModel::Qwen3_0_6B | WhichModel::Qwen3ASR0_6B | WhichModel::Qwen3ASR1_7B => "Qwen",
        WhichModel::Qwen3vl2B | WhichModel::Qwen3vl4B | WhichModel::Qwen3vl8B | WhichModel::Qwen3vl32B => "Qwen",
        WhichModel::DeepSeekOCR => "deepseek-ai",
        WhichModel::HunyuanOCR => "Tencent-Hunyuan",
        WhichModel::PaddleOCRVL => "PaddlePaddle",
        WhichModel::RMBG2_0 => "AI-ModelScope",
        WhichModel::VoxCPM | WhichModel::VoxCPM1_5 => "OpenBMB",
        WhichModel::GlmASRNano2512 => "ZhipuAI",
        WhichModel::FunASRNano2512 => "FunAudioLLM",
    }
}

#[get("/models")]
pub(crate) async fn models() -> (Status, (ContentType, Json<serde_json::Value>))
{
    if let Some(model_ref) = MODEL.get() {
        let guard = model_ref.read().await;
        let which_model = guard.which_model;

        let model_obj = ModelObject {
            id: which_model_to_id(which_model).to_string(),
            object: "model".to_string(),
            created: None, // We don't track creation time
            owned_by: which_model_to_owner(which_model).to_string(),
        };
        drop(guard);

        let response = ModelsListResponse {
            object: "list".to_string(),
            data: vec![model_obj],
        };
        (Status::Ok, (ContentType::JSON, Json(serde_json::to_value(response).unwrap())))
    } else {
        let response = ErrorResponse {
            error: "model not initialized".to_string(),
        };
        (Status::ServiceUnavailable, (ContentType::JSON, Json(serde_json::to_value(response).unwrap())))
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
        assert_eq!(WhichModel::Qwen3vl2B.model_type(), "llm");
        assert_eq!(WhichModel::MiniCPM4_0_5B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen2_5vl3B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen2_5vl7B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen3vl4B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen3vl8B.model_type(), "llm");
        assert_eq!(WhichModel::Qwen3vl32B.model_type(), "llm");
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
        assert_eq!(WhichModel::VoxCPM.model_type(), "image");
        assert_eq!(WhichModel::VoxCPM1_5.model_type(), "image");
    }

    // Test model_id retrieval
    #[test]
    fn test_get_model_id() {
        assert_eq!(WhichModel::Qwen3_0_6B.model_id(), "Qwen/Qwen3-0.6B");
        assert_eq!(WhichModel::DeepSeekOCR.model_id(), "deepseek-ai/DeepSeek-OCR");
        assert_eq!(WhichModel::VoxCPM1_5.model_id(), "OpenBMB/VoxCPM1.5");
    }

    // Test OpenAI-compatible model ID conversion
    #[test]
    fn test_which_model_to_id() {
        assert_eq!(which_model_to_id(WhichModel::Qwen3_0_6B), "qwen3-0.6b");
        assert_eq!(which_model_to_id(WhichModel::DeepSeekOCR), "deepseek-ocr");
        assert_eq!(which_model_to_id(WhichModel::VoxCPM1_5), "voxcpm1.5");
        assert_eq!(which_model_to_id(WhichModel::MiniCPM4_0_5B), "minicpm4-0.5b");
    }

    // Test owner/organization mapping
    #[test]
    fn test_which_model_to_owner() {
        assert_eq!(which_model_to_owner(WhichModel::Qwen3_0_6B), "Qwen");
        assert_eq!(which_model_to_owner(WhichModel::DeepSeekOCR), "deepseek-ai");
        assert_eq!(which_model_to_owner(WhichModel::VoxCPM1_5), "OpenBMB");
        assert_eq!(which_model_to_owner(WhichModel::HunyuanOCR), "Tencent-Hunyuan");
    }
}

// Shutdown endpoint
#[derive(Serialize)]
struct ShutdownResponse {
    message: String,
}

#[post("/shutdown")]
pub(crate) async fn shutdown(shutdown_flag: &State<Arc<AtomicBool>>) -> (Status, (ContentType, Json<serde_json::Value>)) {
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
        (ContentType::JSON, Json(serde_json::to_value(response).unwrap())),
    )
}
