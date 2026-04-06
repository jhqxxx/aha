use serde::{Deserialize, Serialize};

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
