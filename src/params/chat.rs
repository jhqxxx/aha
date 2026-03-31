use super::shared::{ReasoningEffort, WebSearchContextSize};
use crate::params::shared::StopToken;
use crate::params::shared::{FinishReason, Usage};
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::Display;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionResponse {
    /// A unique identifier for the chat completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// A list of chat completion choices. Can be more than one if n is greater than 1.
    pub choices: Vec<ChatCompletionChoice>,
    /// The Unix timestamp (in seconds) of when the chat completion was created.
    pub created: u32,
    /// The model used for the chat completion.
    pub model: String,
    /// The service tier used for processing the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// This fingerprint represents the backend configuration that the model runs with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// The object type, which is always chat.completion.
    pub object: String,
    /// Usage statistics for the completion request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionChunkResponse {
    /// A unique identifier for the chat completion. Each chunk has the same ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// A list of chat completion choices. Can be more than one if n is greater than 1.
    pub choices: Vec<ChatCompletionChunkChoice>,
    /// The Unix timestamp (in seconds) of when the chat completion was created. Each chunk has the same timestamp.
    pub created: u32,
    /// The model to generate the completion.
    pub model: String,
    /// This fingerprint represents the backend configuration that the model runs with.
    /// Can be used in conjunction with the seed request parameter to understand when backend changes have been made that might impact determinism.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    /// The object type, which is always chat.completion.chunk.
    pub object: String,
    /// An optional field that will only be present when you set stream_options: {"include_usage": true} in your request. When present, it contains a null value except for the last chunk which contains the token usage statistics for the entire request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Serialize, Deserialize, Debug, Default, Builder, Clone, PartialEq)]
#[builder(name = "ChatCompletionParametersBuilder")]
#[builder(setter(into, strip_option), default)]
pub struct ChatCompletionParameters {
    /// A list of messages comprising the conversation so far.
    pub messages: Vec<ChatMessage>,
    /// ID of the model to use.
    pub model: String,
    /// Whether or not to store the output of this chat completion request for use in our model distillation or evals products.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Constrains effort on reasoning for reasoning models.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Developer-defined tags and values used for filtering completions in the dashboard.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far,
    /// decreasing the model's likelihood to repeat the same line verbatim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Modify the likelihood of specified tokens appearing in the completion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<HashMap<String, i32>>,
    /// Whether to return log probabilities of the output tokens or not.
    /// If true, returns the log probabilities of each output token returned in the 'content' of 'message'.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    /// An integer between 0 and 5 specifying the number of most likely tokens to return at each token position,
    /// each with an associated log probability. 'logprobs' must be set to 'true' if this parameter is used.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    /// Max completion tokens, deprecated (still used by vllm)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    /// How many chat completion choices to generate for each input message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// Output types that you would like the model to generate for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<Modality>>,
    /// Configuration for a Predicted Output, which can greatly improve response times when large parts of the model response are known ahead of time.
    /// This is most common when you are regenerating a file with only minor changes to most of the content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictedOutput>,
    /// Parameters for audio output. Required when audio output is requested with modalities: ["audio"].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioParameters>,
    /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far,
    /// increasing the model's likelihood to talk about new topics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    /// An object specifying the format that the model must output.
    /// Compatible with GPT-4o, GPT-4o mini, GPT-4 Turbo and all GPT-3.5 Turbo models newer than gpt-3.5-turbo-1106.
    /// Setting to { "type": "json_schema", "json_schema": {...} } enables Structured Outputs which ensures the model will match your supplied JSON schema.
    /// Setting to { "type": "json_object" } enables JSON mode, which ensures the message the model generates is valid JSON.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ChatCompletionResponseFormat>,
    /// Deprecated (still used by vllm)
    /// This feature is in Beta. If specified, our system will make a best effort to sample deterministically,
    /// such that repeated requests with the same seed and parameters should return the same result.
    /// Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter to monitor changes in the backend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    /// Up to 4 sequences where the API will stop generating further tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<StopToken>,
    /// If set, partial messages will be sent, like in ChatGPT. Tokens will be sent as data-only server-sent events
    /// as they become available, with the stream terminated by a data: [DONE] message.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Options for streaming response. Only set this when you set stream: true.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random,
    /// while lower values like 0.2 will make it more focused and deterministic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.
    /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// A list of tools the model may call. Currently, only functions are supported as a tool.
    /// Use this to provide a list of functions the model may generate JSON inputs for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTool>>,
    /// Controls which (if any) tool is called by the model. none means the model will not call any tool and instead generates a message.
    /// auto means the model can pick between generating a message or calling one or more tools.
    /// required means the model must call one or more tools.
    /// Specifying a particular tool via {"type": "function", "function": {"name": "my_function"}} forces the model to call that tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoice>,
    /// Whether to enable parallel function calling during tool use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    /// A stable identifier used to help detect users of your application that may be violating OpenAI's usage policies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    /// Used by OpenAI to cache responses for similar requests to optimize your cache hit rates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    /// This tool searches the web for relevant results to use in a response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web_search_options: Option<WebSearchOptions>,
    /// Allows to pass arbitrary json as an extra_body parameter, for specific features/openai-compatible endpoints.
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra_body: Option<Value>,
    /// Azure OpenAI and some other providers may require special query parameters to be set on the request URL.
    /// This field allows you to specify those query parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_params: Option<HashMap<String, String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionStreamOptions {
    /// If set, an additional chunk will be streamed before the data: [DONE] message.
    pub include_usage: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub continuous_usage_stats: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionToolChoiceFunction {
    /// The type of the tool. Currently, only 'function' is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<ChatCompletionToolType>,
    /// Name of the function.
    pub function: ChatCompletionToolChoiceFunctionName,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionToolChoiceFunctionName {
    /// Name of the function.
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionFunction {
    /// Name of the function.
    pub name: String,
    /// Optional description of the function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The parameters the function takes. The model will generate JSON inputs for these parameters.
    pub parameters: serde_json::Value,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatCompletionResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: JsonSchema },
}

#[derive(Serialize, Deserialize, Debug, Default, Builder, Clone, PartialEq)]
#[builder(name = "JsonSchemaBuilder")]
#[builder(setter(into, strip_option), default)]
pub struct JsonSchema {
    /// A description of what the response format is for, used by the model to determine how to respond in the format.
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    /// The name of the response format. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.
    name: String,
    /// The schema for the response format, described as a JSON Schema object.
    #[serde(skip_serializing_if = "Option::is_none")]
    schema: Option<serde_json::Value>,
    /// Whether to enable strict schema adherence when generating the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionTool {
    /// The type of the tool. Currently, only 'function' is supported.
    pub r#type: ChatCompletionToolType,
    /// The name of the function to call.
    pub function: ChatCompletionFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum ChatMessage {
    Developer {
        /// The contents of the developer message.
        content: ChatMessageContent,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    System {
        /// The contents of the system message.
        content: ChatMessageContent,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        /// The contents of the user message.
        content: ChatMessageContent,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        /// The contents of the assistant message. Required unless tool_calls is specified.
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ChatMessageContent>,
        /// The reasoning content by the assistant. (DeepSeek API only)
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        /// The refusal message by the assistant.
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// Data about a previous audio response from the model.
        #[serde(skip_serializing_if = "Option::is_none")]
        audio: Option<AudioDataIdParameter>,
        /// The tool calls generated by the model, such as function calls.
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    Tool {
        /// The contents of the tool message.
        content: ChatMessageContent,
        /// Tool call that this message is responding to.
        tool_call_id: String,
    },
}

impl ChatMessage {
    /// Get the ChatMessageContent data, if it exists.
    pub fn message(&self) -> Option<&ChatMessageContent> {
        match self {
            ChatMessage::Developer { content, .. }
            | ChatMessage::System { content, .. }
            | ChatMessage::User { content, .. }
            | ChatMessage::Assistant {
                content: Some(content),
                ..
            } => Some(content),
            ChatMessage::Assistant { content: None, .. } => None,
            ChatMessage::Tool { .. } => None,
        }
    }

    /// Get the content of the message as text, if it is a simple text message.
    pub fn text(&self) -> Option<&str> {
        match self {
            ChatMessage::Developer { content, .. }
            | ChatMessage::System { content, .. }
            | ChatMessage::User { content, .. }
            | ChatMessage::Tool { content, .. }
            | ChatMessage::Assistant {
                content: Some(content),
                ..
            } => {
                if let ChatMessageContent::Text(text) = content {
                    Some(text)
                } else {
                    None
                }
            }
            ChatMessage::Assistant { content: None, .. } => None,
        }
    }

    /// Get the name of the message sender, if it exists.
    pub fn name(&self) -> Option<&str> {
        match self {
            ChatMessage::Developer { name, .. }
            | ChatMessage::System { name, .. }
            | ChatMessage::User { name, .. }
            | ChatMessage::Assistant { name, .. } => name.as_deref(),
            ChatMessage::Tool { .. } => None,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum DeltaChatMessage {
    Developer {
        /// The contents of the developer message.
        content: ChatMessageContent,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    System {
        /// The contents of the system message.
        content: ChatMessageContent,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    User {
        /// The contents of the user message.
        content: ChatMessageContent,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    Assistant {
        /// The contents of the assistant message. Required unless tool_calls is specified.
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ChatMessageContent>,
        /// The reasoning content by the assistant. (DeepSeek API only)
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        /// The refusal message by the assistant.
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        /// The tool calls generated by the model, such as function calls.
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<DeltaToolCall>>,
    },
    Tool {
        /// The contents of the tool message.
        content: String,
        /// Tool call that this message is responding to.
        tool_call_id: String,
    },
    #[serde(untagged)]
    Untagged {
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<ChatMessageContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        refusal: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<DeltaToolCall>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool. Currently, only function is supported.
    pub r#type: String,
    /// The function that the model called.
    pub function: Function,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DeltaToolCall {
    /// The index of the tool call in the list of tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// /// The ID of the tool call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The type of the tool. Currently, only 'function' is supported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    /// The function that the model called.
    pub function: DeltaFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Function {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to call the function with, as generated by the model in JSON format.
    pub arguments: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct DeltaFunction {
    /// The name of the function to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The arguments to call the function with, as generated by the model in JSON format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionChoice {
    /// The index of the choice in the list of choices.
    pub index: u32,
    /// A chat completion message generated by the model.
    pub message: ChatMessage,
    /// The reason the model stopped generating tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// Log probability information for the choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProps>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AudioDataIdParameter {
    /// Unique identifier for a previous audio response from the model.
    pub id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AudioParameters {
    /// The voice the model uses to respond.
    pub voice: Voice,
    /// Specifies the output audio format.
    pub format: AudioFormat,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LogProps {
    /// A list of message content tokens with log probability information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<LogPropsContent>>,
    /// A list of message refusal tokens with log probability information.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Vec<LogPropsContent>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LogPropsContent {
    /// Token information
    #[serde(flatten)]
    pub token_info: LogProbsContentInfo,
    /// List of the most likely tokens and their log probability, at this token position.
    pub top_logprobs: Vec<LogProbsContentInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LogProbsContentInfo {
    /// The token.
    pub token: String,
    /// The log probability of this token, if it is within the top 20 most likely tokens.
    /// Otherwise, the value -9999.0 is used to signify that the token is very unlikely.
    pub logprob: f32,
    /// A list of integers representing the UTF-8 bytes representation of the token.
    pub bytes: Option<Vec<u8>>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatCompletionChunkChoice {
    /// The index of the choice in the list of choices.
    pub index: Option<u32>,
    /// A chat completion delta generated by streamed model responses.
    pub delta: DeltaChatMessage,
    /// The reason the model stopped generating tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    /// Log probability information for the choice.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogProps>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ImageUrlType {
    /// Either a URL of the image or the base64 encoded image data.
    pub url: String,
    /// Specifies the detail level of the image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageUrlDetail>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AudioUrlType {
    /// audio  URL or base64
    pub url: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct VideoUrlType {
    /// video  URL
    pub url: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PredictedOutput {
    /// The type of the predicted content you want to provide.
    pub r#type: PredictedOutputType,
    /// The content that should be matched when generating a model response.
    /// If generated tokens would match this content, the entire model response can be returned much more quickly.
    pub content: PredictedOutputContent,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum PredictedOutputContent {
    String(String),
    Array(Vec<PredictedOutputArrayPart>),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct PredictedOutputArrayPart {
    /// The type of the content part.
    pub r#type: String,
    /// The text content.
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PredictedOutputType {
    Content,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Audio,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ImageUrlDetail {
    Auto,
    High,
    Low,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    ContentPart(Vec<ChatMessageContentPart>),
    None,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum ChatMessageContentPart {
    Text(ChatMessageTextContentPart),
    Image(ChatMessageImageContentPart),
    Audio(ChatMessageAudioContentPart),
    Video(ChatMessageVideoContentPart),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatMessageTextContentPart {
    /// The type of the content part.
    pub r#type: String,
    /// The text content.
    pub text: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatMessageImageContentPart {
    /// The type of the content part.
    pub r#type: String,
    /// The text content.
    pub image_url: ImageUrlType,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatMessageAudioContentPart {
    /// The type of the content part. Always input_audio.
    pub r#type: String,
    /// The input audio data.
    pub audio_url: AudioUrlType,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatMessageVideoContentPart {
    /// The type of the content part.
    pub r#type: String,
    /// The text content.
    pub video_url: VideoUrlType,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ChatMessageImageUrl {
    /// Either a URL of the image or the base64 encoded image data.
    pub url: String,
    /// Specifies the detail level of the image.
    pub detail: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct InputAudioData {
    /// Base64 encoded audio data.
    pub data: String,
    /// The format of the encoded audio data. Currently supports "wav" and "mp3".
    pub format: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WebSearchOptions {
    /// High level guidance for the amount of context window space to use for the search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<WebSearchContextSize>,
    /// Approximate location parameters for the search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<ApproximateUserLocation>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ApproximateUserLocation {
    pub r#type: UserLocationType,
    /// Approximate location parameters for the search.
    pub approximate: WebSearchUserLocation,
}

impl Display for ChatMessageContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatMessageContent::Text(text) => write!(f, "{text}"),
            ChatMessageContent::ContentPart(tcp) => {
                for part in tcp {
                    write!(f, "{part:?}")?;
                }
                Ok(())
            }
            ChatMessageContent::None => write!(f, ""),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionToolType {
    Function,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ChatCompletionToolChoice {
    None,
    Auto,
    Required,
    #[serde(untagged)]
    ChatCompletionToolChoiceFunction(ChatCompletionToolChoiceFunction),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WebSearchUserLocation {
    pub city: Option<String>,
    pub country: Option<String>,
    pub region: Option<String>,
    pub timezone: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum UserLocationType {
    Approximate,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Voice {
    Alloy,
    Ash,
    Ballad,
    Coral,
    Echo,
    Sage,
    Shimmer,
    Verse,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    Wav,
    Mp3,
    Flac,
    Opus,
    Pcm16,
}

impl Default for ChatMessageContent {
    fn default() -> Self {
        ChatMessageContent::Text("".to_string())
    }
}

impl DeltaFunction {
    pub fn merge(&mut self, other: &Self) {
        if self.name.is_none() && other.name.is_some() {
            self.name.clone_from(&other.name);
        }

        if let Some(arguments) = &other.arguments {
            if let Some(self_arguments) = &mut self.arguments {
                self_arguments.push_str(arguments);
            } else {
                self.arguments = Some(arguments.clone());
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.name.is_none() && self.arguments.is_none()
    }
}
