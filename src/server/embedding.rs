use rocket::{http::Status, post, serde::json::Json};
use serde_json::Value;

use crate::{
    params::embedding::{EmbeddingData, EmbeddingRequest, EmbeddingResponse},
    server::api::get_model_by_name,
};

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

#[utoipa::path(
    post,
    path = "/embeddings",
    request_body = EmbeddingRequest,
    responses(
        (status = 200, description = "Embedding vectors for the input", body = EmbeddingResponse, content_type = "application/json"),
    ),
    tag = "embeddings",
)]
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
    let model_name = req.model.clone().unwrap_or_default();
    let entry = match get_model_by_name(&model_name).await {
        Ok(e) => e,
        Err(_) => {
            return (
                Status::ServiceUnavailable,
                Json(serde_json::json!({ "error": "model not init" })),
            );
        }
    };
    let mut guard = entry.instance.write().await;
    let embeddings = match guard.embedding(&texts) {
        Ok(v) => v,
        Err(e) => {
            return (
                Status::BadRequest,
                Json(serde_json::json!({ "error": e.to_string() })),
            );
        }
    };
    let model_name = entry.which_model.as_string();
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
