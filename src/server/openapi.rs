use utoipa::OpenApi;

use crate::params::chat::{
    ApproximateUserLocation, AudioDataIdParameter, AudioFormat, AudioParameters, AudioUrlType,
    ChatCompletionChunkChoice, ChatCompletionChunkResponse, ChatCompletionChoice,
    ChatCompletionFunction, ChatCompletionParameters, ChatCompletionResponse,
    ChatCompletionResponseFormat, ChatCompletionStreamOptions, ChatCompletionTool,
    ChatCompletionToolChoice, ChatCompletionToolChoiceFunction,
    ChatCompletionToolChoiceFunctionName, ChatCompletionToolType, ChatMessage,
    ChatMessageAudioContentPart, ChatMessageContent, ChatMessageContentPart,
    ChatMessageImageContentPart, ChatMessageImageUrl, ChatMessageTextContentPart,
    ChatMessageVideoContentPart, DeltaChatMessage, DeltaFunction, DeltaToolCall, Function,
    ImageUrlDetail, ImageUrlType, InputAudioData, JsonSchema, LogProbsContentInfo, LogProps,
    LogPropsContent, Modality, PredictedOutput, PredictedOutputArrayPart,
    PredictedOutputContent, PredictedOutputType, ToolCall, UserLocationType, VideoUrlType, Voice,
    WebSearchOptions, WebSearchUserLocation,
};
use crate::params::shared::{
    CompletionTokensDetails, FinishReason, PromptTokensDetails, ReasoningEffort, StopToken, Usage,
    WebSearchContextSize,
};
use crate::params::asr::{ErrorDetail, ErrorResponse, TranscriptionResponse};
use crate::params::embedding::{EmbeddingData, EmbeddingRequest, EmbeddingResponse};
use crate::params::rerank::{RerankRequest, RerankResponse, RerankResult};

use super::api;
use super::asr;
use super::embedding;
use super::reranker;

#[derive(OpenApi)]
#[openapi(
    info(
        title = "aha API",
        description = "aha model inference server API — supports LLM chat, ASR, embeddings, reranking",
        version = "0.2.5",
        license(name = "Apache-2.0")
    ),
    paths(
        api::chat,
        api::speech,
        api::remove_background,
        api::health,
        api::models,
        api::list_loaded_models,
        api::shutdown,
        asr::transcriptions,
        embedding::embeddings,
        reranker::rerank,
    ),
    components(
        schemas(
            // Chat params
            ChatCompletionParameters,
            ChatCompletionResponse,
            ChatCompletionChunkResponse,
            ChatCompletionChoice,
            ChatCompletionChunkChoice,
            ChatMessage,
            DeltaChatMessage,
            ChatMessageContent,
            ChatMessageContentPart,
            ChatMessageTextContentPart,
            ChatMessageImageContentPart,
            ChatMessageAudioContentPart,
            ChatMessageVideoContentPart,
            ChatMessageImageUrl,
            ChatCompletionStreamOptions,
            ChatCompletionTool,
            ChatCompletionToolChoice,
            ChatCompletionToolChoiceFunction,
            ChatCompletionToolChoiceFunctionName,
            ChatCompletionToolType,
            ChatCompletionFunction,
            ChatCompletionResponseFormat,
            JsonSchema,
            ToolCall,
            DeltaToolCall,
            Function,
            DeltaFunction,
            LogProps,
            LogPropsContent,
            LogProbsContentInfo,
            AudioDataIdParameter,
            AudioParameters,
            AudioFormat,
            Voice,
            AudioUrlType,
            ImageUrlType,
            ImageUrlDetail,
            VideoUrlType,
            InputAudioData,
            WebSearchOptions,
            WebSearchUserLocation,
            ApproximateUserLocation,
            UserLocationType,
            Modality,
            PredictedOutput,
            PredictedOutputType,
            PredictedOutputContent,
            PredictedOutputArrayPart,
            // Shared
            Usage,
            FinishReason,
            StopToken,
            ReasoningEffort,
            WebSearchContextSize,
            PromptTokensDetails,
            CompletionTokensDetails,
            // ASR
            TranscriptionResponse,
            ErrorResponse,
            ErrorDetail,
            // Embedding
            EmbeddingRequest,
            EmbeddingResponse,
            EmbeddingData,
            // Rerank
            RerankRequest,
            RerankResponse,
            RerankResult,
        )
    ),
    tags(
        (name = "chat", description = "Chat Completion (text generation)"),
        (name = "audio", description = "Speech & Transcription"),
        (name = "embeddings", description = "Text Embeddings"),
        (name = "rerank", description = "Document Reranking"),
        (name = "images", description = "Image Processing"),
        (name = "models", description = "Model Info & Health"),
        (name = "admin", description = "Admin Operations (shutdown, model list)"),
    )
)]
pub struct ApiDoc;
