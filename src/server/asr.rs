// OpenAI-compatible ASR (Automatic Speech Recognition) API endpoint
// Implements POST /audio/transcriptions and /v1/audio/transcriptions

use aha::models::GenerateModel;
use aha::params::chat::{
    AudioUrlType, ChatCompletionParameters, ChatMessage, ChatMessageAudioContentPart,
    ChatMessageContent, ChatMessageContentPart,
};
use aha::utils::{clean_asr_response, map_language_code};
use rocket::http::Status;
use rocket::serde::json::Json;
use rocket::{form::Form, post};

use crate::server::api::MODEL;
use crate::server::asr_types::{
    ErrorDetail, ErrorResponse, TranscriptionRequest, TranscriptionResponse,
};

/// Handle audio transcription requests
///
/// This endpoint accepts multipart/form-data with an audio file and returns
/// the transcription text in OpenAI-compatible format.
///
/// # Supported Parameters
/// - `file`: Audio file (required) - wav, mp3, m4a, etc.
/// - `model`: Model name (optional, ignored)
/// - `language`: Language code (optional) - zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, ro, hu, mk
/// - `prompt`: Optional prompt text (ignored in this implementation)
/// - `response_format`: Response format (only "json" supported)
/// - `temperature`: Sampling temperature (0.0 to 1.0, default 0.0)
///
/// # Returns
/// JSON response with format: `{"text": "transcribed text"}`
#[post("/transcriptions", data = "<req>")]
pub(crate) async fn transcriptions(
    req: Form<TranscriptionRequest<'_>>,
) -> (Status, Json<serde_json::Value>) {
    // Validate response_format (only JSON and text supported)
    #[allow(clippy::collapsible_if)]
    if let Some(ref format) = req.response_format {
        if format != "json" && format != "text" {
            return error_response(
                Status::BadRequest,
                "invalid_request_error",
                "Only 'json' and 'text' response formats are supported",
                Some("unsupported_format".to_string()),
            );
        }
    }

    // Get the audio file path
    let file_path = match req.file.path() {
        Some(path) => path,
        None => {
            return error_response(
                Status::BadRequest,
                "invalid_request_error",
                "Audio file is required",
                Some("missing_file".to_string()),
            );
        }
    };

    // Build file:// URL for the model
    let file_url = format!("file://{}", file_path.display());

    // Map language code to full language name
    let language_name = req
        .language
        .as_ref()
        .and_then(|code| map_language_code(code));

    // Build ChatCompletionParameters for the ASR model
    let audio_part = ChatMessageContentPart::Audio(ChatMessageAudioContentPart {
        r#type: "audio".to_string(),
        audio_url: AudioUrlType { url: file_url },
    });

    let params = ChatCompletionParameters {
        messages: vec![ChatMessage::User {
            content: ChatMessageContent::ContentPart(vec![audio_part]),
            name: None,
        }],
        model: req.model.clone().unwrap_or_else(|| "asr".to_string()),
        temperature: req.temperature.or(Some(0.0)),
        max_tokens: None,
        stream: None,
        top_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        stop: None,
        n: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        metadata: language_name.map(|lang| {
            let mut map = std::collections::HashMap::new();
            map.insert("language".to_string(), lang);
            map
        }),
        ..Default::default()
    };

    // Get the model and generate transcription
    let model_ref = match MODEL.get() {
        Some(m) => m,
        None => {
            return error_response(
                Status::ServiceUnavailable,
                "service_unavailable",
                "Model not initialized",
                Some("model_not_loaded".to_string()),
            );
        }
    };

    let response = {
        let mut guard = model_ref.write().await;
        guard.instance.generate(params)
    };

    match response {
        Ok(chat_response) => {
            // Extract the transcription text from the response
            let raw_text = chat_response
                .choices
                .first()
                .and_then(|choice| {
                    if let ChatMessage::Assistant { content, .. } = &choice.message {
                        content.as_ref().and_then(|c| {
                            if let ChatMessageContent::Text(text) = c {
                                Some(text.clone())
                            } else {
                                None
                            }
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or_else(String::new);

            // Clean the response (remove "language English<asr_text>" prefix)
            let cleaned_text = clean_asr_response(&raw_text);

            // Return OpenAI-compatible transcription response
            let transcription = TranscriptionResponse { text: cleaned_text };
            (
                Status::Ok,
                Json(serde_json::to_value(transcription).unwrap()),
            )
        }
        Err(e) => {
            // Determine appropriate error status based on error message
            let error_msg = e.to_string();
            let (status, error_type, code) =
                if error_msg.contains("audio") || error_msg.contains("decode") {
                    (
                        Status::BadRequest,
                        "invalid_request_error",
                        Some("audio_decode_error".to_string()),
                    )
                } else {
                    (
                        Status::InternalServerError,
                        "server_error",
                        Some("inference_error".to_string()),
                    )
                };

            error_response(status, error_type, &error_msg, code)
        }
    }
}

/// Helper function to create error responses in OpenAI format
fn error_response(
    status: Status,
    error_type: &str,
    message: &str,
    code: Option<String>,
) -> (Status, Json<serde_json::Value>) {
    let error_response = ErrorResponse {
        error: ErrorDetail {
            message: message.to_string(),
            error_type: error_type.to_string(),
            code,
        },
    };
    (status, Json(serde_json::to_value(error_response).unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_response_serialization() {
        let (status, json) = error_response(
            Status::BadRequest,
            "invalid_request_error",
            "Test error message",
            Some("test_code".to_string()),
        );

        assert_eq!(status, Status::BadRequest);
        let parsed: serde_json::Value = serde_json::from_str(&json.to_string()).unwrap();
        assert_eq!(parsed["error"]["message"], "Test error message");
        assert_eq!(parsed["error"]["type"], "invalid_request_error");
        assert_eq!(parsed["error"]["code"], "test_code");
    }
}
