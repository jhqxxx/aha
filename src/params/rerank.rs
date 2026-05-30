use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

#[derive(Debug, Deserialize, ToSchema)]
pub struct RerankRequest {
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    pub document: String,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct RerankResponse {
    pub object: String,
    pub model: String,
    pub results: Vec<RerankResult>,
}
