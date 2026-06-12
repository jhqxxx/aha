// ASR API data types for OpenAI-compatible transcription endpoint

use rocket::form::FromForm;
use serde::Serialize;
use utoipa::ToSchema;

/// Request parameters for audio transcription
#[derive(Debug, FromForm)]
pub struct TranscriptionRequest<'r> {
    /// The audio file to transcribe
    pub file: rocket::fs::TempFile<'r>,

    /// ID of the model to use (ignored, always uses loaded model)
    pub model: Option<String>,

    /// Language code (e.g., "zh", "en")
    pub language: Option<String>,

    /// Optional text to guide the transcription (not implemented, ignored)
    #[allow(dead_code)]
    pub prompt: Option<String>,

    /// Response format (only "json" supported)
    pub response_format: Option<String>,

    /// Sampling temperature (0.0 to 1.0)
    pub temperature: Option<f32>,
}

/// Standard transcription response
#[derive(Debug, Serialize, ToSchema)]
pub struct TranscriptionResponse {
    pub text: String,
}

/// Error response following OpenAI format
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}
