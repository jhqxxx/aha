use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub(crate) struct EmbeddingRequest {
    pub model: Option<String>,
    pub input: Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct EmbeddingResponse {
    pub object: String,
    pub model: String,
    pub data: Vec<EmbeddingData>,
}
