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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transcription_response_serialization() {
        let response = TranscriptionResponse {
            text: "Hello, world!".to_string(),
        };
        let json = serde_json::to_string(&response).unwrap();
        assert_eq!(json, r#"{"text":"Hello, world!"}"#);
    }

    #[test]
    fn test_error_response_serialization() {
        let error = ErrorResponse {
            error: ErrorDetail {
                message: "Invalid audio file".to_string(),
                error_type: "invalid_request_error".to_string(),
                code: Some("invalid_audio".to_string()),
            },
        };
        let json = serde_json::to_string(&error).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["error"]["message"], "Invalid audio file");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["code"], "invalid_audio");
    }
}
