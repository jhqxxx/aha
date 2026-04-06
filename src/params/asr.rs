// ASR API data types for OpenAI-compatible transcription endpoint

use rocket::form::FromForm;
use serde::Serialize;

/// Request parameters for audio transcription
#[derive(Debug, FromForm)]
pub(crate) struct TranscriptionRequest<'r> {
    /// The audio file to transcribe
    pub(crate) file: rocket::fs::TempFile<'r>,

    /// ID of the model to use (ignored, always uses loaded model)
    pub(crate) model: Option<String>,

    /// Language code (e.g., "zh", "en")
    pub(crate) language: Option<String>,

    /// Optional text to guide the transcription (not implemented, ignored)
    #[allow(dead_code)]
    pub(crate) prompt: Option<String>,

    /// Response format (only "json" supported)
    pub(crate) response_format: Option<String>,

    /// Sampling temperature (0.0 to 1.0)
    pub(crate) temperature: Option<f32>,
}

/// Standard transcription response
#[derive(Debug, Serialize)]
pub(crate) struct TranscriptionResponse {
    pub(crate) text: String,
}

/// Error response following OpenAI format
#[derive(Debug, Serialize)]
pub(crate) struct ErrorResponse {
    pub(crate) error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorDetail {
    pub(crate) message: String,
    #[serde(rename = "type")]
    pub(crate) error_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) code: Option<String>,
}
