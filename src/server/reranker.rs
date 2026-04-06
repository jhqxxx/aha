use rocket::{http::Status, post, serde::json::Json};
use serde_json::Value;

use crate::{
    params::rerank::{RerankRequest, RerankResponse, RerankResult},
    server::api::MODEL,
};

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
        model: guard.which_model.as_string(),
        results,
    };
    (Status::Ok, Json(serde_json::to_value(response).unwrap()))
}
