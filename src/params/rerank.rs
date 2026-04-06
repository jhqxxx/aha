use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub(crate) struct RerankRequest {
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    pub document: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct RerankResponse {
    pub object: String,
    pub model: String,
    pub results: Vec<RerankResult>,
}
