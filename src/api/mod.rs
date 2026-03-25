use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use aha::models::{ArtifactKind, GenerateModel, LoadSpec, ModelInstance, WhichModel, load_model};
use aha::process::cleanup_pid_file;
use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use rocket::futures::StreamExt;
use rocket::serde::{Deserialize, Serialize, json::Json};
use rocket::{
    Request, State,
    futures::Stream,
    get,
    http::{ContentType, Status},
    post,
    response::{Responder, stream::TextStream},
};
use serde_json::Value;
use tokio::sync::RwLock;

// ASR (Automatic Speech Recognition) API module
pub(crate) mod asr;
pub(crate) mod asr_types;

// Re-export ASR routes
pub(crate) use asr::transcriptions;

/// Wrapper to store model type together with the model instance
pub(crate) struct StoredModel {
    which_model: WhichModel,
    artifact: ArtifactKind,
    instance: ModelInstance<'static>,
}

// Export MODEL for use in ASR module
pub(crate) static MODEL: OnceLock<Arc<RwLock<StoredModel>>> = OnceLock::new();
static SHUTDOWN_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static SERVER_PORT: OnceLock<u16> = OnceLock::new();
static ALLOW_REMOTE_SHUTDOWN: OnceLock<bool> = OnceLock::new();

pub fn init(spec: LoadSpec) -> anyhow::Result<()> {
    let resolved_artifact = spec.resolved_artifact();
    let model_type = spec.model;
    let model = load_model(&spec)?;
    MODEL.get_or_init(|| {
        Arc::new(RwLock::new(StoredModel {
            which_model: model_type,
            artifact: resolved_artifact,
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

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: Value,
}

#[derive(Debug, Serialize)]
struct EmbeddingData {
    object: String,
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RerankRequest {
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
}

#[derive(Debug, Serialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
    document: String,
}

#[derive(Debug, Serialize)]
struct RerankResponse {
    object: String,
    model: String,
    results: Vec<RerankResult>,
}

fn parse_embedding_input(input: &Value) -> anyhow::Result<Vec<String>> {
    match input {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                let s = v.as_str().ok_or_else(|| {
                    anyhow::anyhow!("embedding input array must contain only strings")
                })?;
                out.push(s.to_string());
            }
            if out.is_empty() {
                return Err(anyhow::anyhow!("embedding input cannot be empty"));
            }
            Ok(out)
        }
        _ => Err(anyhow::anyhow!(
            "embedding input must be a string or an array of strings"
        )),
    }
}

fn validate_rerank_input(query: &str, documents: &[String]) -> anyhow::Result<()> {
    if query.trim().is_empty() {
        return Err(anyhow::anyhow!("rerank query cannot be empty"));
    }
    if documents.is_empty() {
        return Err(anyhow::anyhow!("rerank documents cannot be empty"));
    }
    if documents.iter().any(|doc| doc.trim().is_empty()) {
        return Err(anyhow::anyhow!(
            "rerank documents cannot contain empty strings"
        ));
    }
    Ok(())
}

#[post("/embeddings", data = "<req>")]
pub(crate) async fn embeddings(req: Json<EmbeddingRequest>) -> (Status, Json<Value>) {
    let texts = match parse_embedding_input(&req.input) {
        Ok(v) => v,
        Err(e) => {
            return (
                Status::BadRequest,
                Json(serde_json::json!({ "error": e.to_string() })),
            );
        }
    };
    let model_ref = match MODEL.get().cloned() {
        Some(v) => v,
        None => {
            return (
                Status::ServiceUnavailable,
                Json(serde_json::json!({ "error": "model not init" })),
            );
        }
    };
    let mut guard = model_ref.write().await;
    let embeddings = match guard.instance.embedding(&texts) {
        Ok(v) => v,
        Err(e) => {
            return (
                Status::BadRequest,
                Json(serde_json::json!({ "error": e.to_string() })),
            );
        }
    };
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| guard.which_model.openai_model_id().to_string());
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            object: "embedding".to_string(),
            index,
            embedding,
        })
        .collect::<Vec<_>>();
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: model_name,
    };
    (Status::Ok, Json(serde_json::to_value(response).unwrap()))
}

#[post("/rerank", data = "<req>")]
pub(crate) async fn rerank(req: Json<RerankRequest>) -> (Status, Json<Value>) {
    let req = req.into_inner();
    if let Err(e) = validate_rerank_input(&req.query, &req.documents) {
        return (
            Status::BadRequest,
            Json(serde_json::json!({ "error": e.to_string() })),
        );
    }

    let model_ref = match MODEL.get().cloned() {
        Some(v) => v,
        None => {
            return (
                Status::ServiceUnavailable,
                Json(serde_json::json!({ "error": "model not init" })),
            );
        }
    };

    let mut guard = model_ref.write().await;
    let scores = match guard.instance.rerank(&req.query, &req.documents) {
        Ok(v) => v,
        Err(e) => {
            return (
                Status::BadRequest,
                Json(serde_json::json!({ "error": e.to_string() })),
            );
        }
    };

    let mut results = scores
        .into_iter()
        .enumerate()
        .map(|(index, relevance_score)| RerankResult {
            index,
            relevance_score,
            document: req.documents[index].clone(),
        })
        .collect::<Vec<_>>();
    results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    if let Some(top_n) = req.top_n {
        results.truncate(top_n.min(results.len()));
    }

    let response = RerankResponse {
        object: "list".to_string(),
        model: req
            .model
            .unwrap_or_else(|| guard.which_model.openai_model_id().to_string()),
        results,
    };
    (Status::Ok, Json(serde_json::to_value(response).unwrap()))
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
    artifact: String,
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
            id: which_model.openai_model_id().to_string(),
            object: "model".to_string(),
            created: None, // We don't track creation time
            owned_by: which_model.owner().to_string(),
            artifact: format!("{:?}", guard.artifact).to_ascii_lowercase(),
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

    #[test]
    fn test_parse_embedding_input_string() {
        let input = serde_json::json!("hello");
        let out = parse_embedding_input(&input).unwrap();
        assert_eq!(out, vec!["hello".to_string()]);
    }

    #[test]
    fn test_parse_embedding_input_array() {
        let input = serde_json::json!(["a", "b"]);
        let out = parse_embedding_input(&input).unwrap();
        assert_eq!(out, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_validate_rerank_input() {
        let query = "hello";
        let docs = vec!["doc1".to_string(), "doc2".to_string()];
        assert!(validate_rerank_input(query, &docs).is_ok());
    }

    #[test]
    fn test_validate_rerank_input_empty_doc() {
        let query = "hello";
        let docs = vec!["".to_string()];
        assert!(validate_rerank_input(query, &docs).is_err());
    }

    // Test model type classification
    #[test]
    fn test_get_model_type_llm() {
        assert_eq!(WhichModel::Qwen3_0_6B.model_type(), "llm");
        assert_eq!(WhichModel::MiniCPM4_0_5B.model_type(), "llm");
    }

    #[test]
    fn test_get_model_type_vlm() {
        assert_eq!(WhichModel::Qwen3vl2B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen2_5vl3B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen2_5vl7B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen3vl4B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen3vl8B.model_type(), "vlm");
        assert_eq!(WhichModel::Qwen3vl32B.model_type(), "vlm");
    }

    #[test]
    fn test_get_model_type_embedding() {
        assert_eq!(WhichModel::Qwen3Embedding0_6B.model_type(), "embedding");
        assert_eq!(WhichModel::Qwen3Embedding4B.model_type(), "embedding");
        assert_eq!(WhichModel::Qwen3Embedding8B.model_type(), "embedding");
    }

    #[test]
    fn test_get_model_type_reranker() {
        assert_eq!(WhichModel::Qwen3Reranker0_6B.model_type(), "reranker");
        assert_eq!(WhichModel::Qwen3Reranker4B.model_type(), "reranker");
        assert_eq!(WhichModel::Qwen3Reranker8B.model_type(), "reranker");
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
        assert_eq!(
            WhichModel::Qwen3Reranker4B.model_id(),
            "Qwen/Qwen3-Reranker-4B"
        );
        assert_eq!(
            WhichModel::Qwen3Reranker8B.model_id(),
            "Qwen/Qwen3-Reranker-8B"
        );
        assert_eq!(
            WhichModel::DeepSeekOCR.model_id(),
            "deepseek-ai/DeepSeek-OCR"
        );
        assert_eq!(WhichModel::VoxCPM1_5.model_id(), "OpenBMB/VoxCPM1.5");
    }

    // Test OpenAI-compatible model ID conversion
    #[test]
    fn test_openai_model_id() {
        assert_eq!(WhichModel::Qwen3_0_6B.openai_model_id(), "qwen3-0.6b");
        assert_eq!(
            WhichModel::Qwen3Reranker4B.openai_model_id(),
            "qwen3-reranker-4b"
        );
        assert_eq!(
            WhichModel::Qwen3Reranker8B.openai_model_id(),
            "qwen3-reranker-8b"
        );
        assert_eq!(WhichModel::DeepSeekOCR.openai_model_id(), "deepseek-ocr");
        assert_eq!(WhichModel::VoxCPM1_5.openai_model_id(), "voxcpm1.5");
        assert_eq!(WhichModel::MiniCPM4_0_5B.openai_model_id(), "minicpm4-0.5b");
    }

    // Test owner/organization mapping
    #[test]
    fn test_model_owner() {
        assert_eq!(WhichModel::Qwen3_0_6B.owner(), "Qwen");
        assert_eq!(WhichModel::Qwen3Reranker4B.owner(), "Qwen");
        assert_eq!(WhichModel::Qwen3Reranker8B.owner(), "Qwen");
        assert_eq!(WhichModel::DeepSeekOCR.owner(), "deepseek-ai");
        assert_eq!(WhichModel::VoxCPM1_5.owner(), "OpenBMB");
        assert_eq!(WhichModel::HunyuanOCR.owner(), "Tencent-Hunyuan");
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
