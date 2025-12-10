pub mod audio_utils;
pub mod img_utils;
pub mod tensor_utils;
pub mod video_utils;

use std::process::Command;

use aha_openai_dive::v1::resources::{
    chat::{
        ChatCompletionChoice, ChatCompletionChunkChoice, ChatCompletionChunkResponse,
        ChatCompletionParameters, ChatCompletionResponse, ChatMessage, ChatMessageContent,
        ChatMessageContentPart, DeltaChatMessage, DeltaFunction, DeltaToolCall, Function, ToolCall,
    },
    shared::{FinishReason, Usage},
};
use anyhow::Result;
use candle_core::{DType, Device};
use candle_transformers::generation::{LogitsProcessor, Sampling};

pub fn get_device(device: Option<&Device>) -> Device {
    match device {
        Some(d) => d.clone(),
        None => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Device::Cpu
            }
        }
    }
}

pub fn get_gpu_sm_arch() -> Result<f32> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
        .map_err(|e| anyhow::anyhow!(format!("Failed to execute nvidia-smi: {}", e)))?;
    if !output.status.success() {
        return Err(anyhow::anyhow!(format!(
            "nvidia-smi failed with status: {}\nError: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let output_str = String::from_utf8_lossy(&output.stdout);
    let output_str = output_str.trim();
    let sm_float = match output_str.parse::<f32>() {
        Ok(num) => num,
        Err(_) => {
            return Err(anyhow::anyhow!(format!(
                "gpr sm arch: {} parse float32 error",
                output_str
            )));
        }
    };
    Ok(sm_float)
}

pub fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    match dtype {
        Some(d) => d,
        None => {
            #[cfg(feature = "cuda")]
            {
                match cfg_dtype {
                    "float32" | "float" => DType::F32,
                    "float64" | "double" => DType::F64,
                    "float16" => DType::F16,
                    "bfloat16" => {
                        let arch = get_gpu_sm_arch();
                        match arch {
                            Err(_) => DType::F16,
                            Ok(a) => {
                                // nvidia显卡sm架构>=8.0的才支持BF16
                                if a >= 8.0 { DType::BF16 } else { DType::F16 }
                            }
                        }
                    }
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                match cfg_dtype {
                    "float32" | "float" => DType::F32,
                    "float64" | "double" => DType::F64,
                    "float16" | "bfloat16" => DType::F16, // cpu上bfloat16有问题
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            }
        }
    }
}

pub fn string_to_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

pub fn find_type_files(path: &str, extension_type: &str) -> Result<Vec<String>> {
    let mut files = Vec::new();

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        if file_path.is_file()
            && let Some(extension) = file_path.extension()
            && extension == extension_type
        {
            files.push(file_path.to_string_lossy().to_string());
        }
    }

    Ok(files)
}

pub fn round_by_factor(num: u32, factor: u32) -> u32 {
    let round = (num as f32 / factor as f32).round() as u32;
    round * factor
}

pub fn floor_by_factor(num: f32, factor: u32) -> u32 {
    let floor = (num / factor as f32).floor() as u32;
    floor * factor
}

pub fn ceil_by_factor(num: f32, factor: u32) -> u32 {
    let ceil = (num / factor as f32).ceil() as u32;
    ceil * factor
}

pub fn build_completion_response(res: String, model_name: &str, num_tokens: Option<u32>) -> ChatCompletionResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let usage = match num_tokens {
        Some(num) => {
            Some(Usage {
                input_tokens: None,
                input_tokens_details: None,
                output_tokens: None,
                output_tokens_details: None,
                prompt_tokens: None,
                completion_tokens: None,
                total_tokens: num,
                prompt_tokens_details: None,
                completion_tokens_details: None
            })
        },
        None => None,
    };
    let mut response = ChatCompletionResponse {
        id: Some(id),
        choices: vec![],
        created: chrono::Utc::now().timestamp() as u32,
        model: model_name.to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage
    };
    let choice = if res.contains("<tool_call>") {
        let mes: Vec<&str> = res.split("<tool_call>").collect();
        let content = mes[0].to_string();
        let mut tool_vec = Vec::new();
        for (i, m) in mes.iter().enumerate().skip(1) {
            let tool_mes = m.replace("</tool_call>", "");
            let function = match serde_json::from_str::<serde_json::Value>(&tool_mes) {
                Ok(json_value) => {
                    let name = json_value
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                        .unwrap_or_default();

                    let arguments = json_value
                        .get("arguments")
                        .map(|v| v.to_string())
                        .unwrap_or_default();

                    Function { name, arguments }
                }
                Err(_) => Function {
                    name: "".to_string(),
                    arguments: "".to_string(),
                },
            };
            let tool_call = ToolCall {
                id: (i - 1).to_string(),
                r#type: "function".to_string(),
                function,
            };
            tool_vec.push(tool_call);
        }
        ChatCompletionChoice {
            index: 0,
            message: ChatMessage::Assistant {
                content: Some(ChatMessageContent::Text(content)),
                reasoning_content: None,
                refusal: None,
                name: None,
                audio: None,
                tool_calls: Some(tool_vec),
            },
            finish_reason: Some(FinishReason::ToolCalls),
            logprobs: None,
        }
    } else {
        ChatCompletionChoice {
            index: 0,
            message: ChatMessage::Assistant {
                content: Some(ChatMessageContent::Text(res)),
                reasoning_content: None,
                refusal: None,
                name: None,
                audio: None,
                tool_calls: None,
            },
            finish_reason: Some(FinishReason::StopSequenceReached),
            logprobs: None,
        }
    };
    response.choices.push(choice);
    response
}

pub fn build_completion_chunk_response(
    res: String,
    model_name: &str,
    tool_call_id: Option<String>,
    tool_call_content: Option<String>,
) -> ChatCompletionChunkResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let mut response = ChatCompletionChunkResponse {
        id: Some(id),
        choices: vec![],
        created: chrono::Utc::now().timestamp() as u32,
        model: model_name.to_string(),
        system_fingerprint: None,
        object: "chat.completion.chunk".to_string(),
        usage: None,
    };
    let choice = if let Some(tool_call_id) = tool_call_id {
        let function = if let Some(content) = tool_call_content {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(json_value) => {
                    let name = json_value
                        .get("name")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string());

                    let arguments = json_value.get("arguments").map(|v| v.to_string());

                    DeltaFunction { name, arguments }
                }
                Err(_) => DeltaFunction {
                    name: None,
                    arguments: Some(content),
                },
            }
        } else {
            DeltaFunction {
                name: None,
                arguments: None,
            }
        };
        ChatCompletionChunkChoice {
            index: Some(0),
            delta: DeltaChatMessage::Assistant {
                content: None,
                reasoning_content: None,
                refusal: None,
                name: None,
                tool_calls: Some(vec![DeltaToolCall {
                    index: Some(0),
                    id: Some(tool_call_id),
                    r#type: Some("function".to_string()),
                    function,
                }]),
            },
            finish_reason: None,
            logprobs: None,
        }
    } else {
        ChatCompletionChunkChoice {
            index: Some(0),
            delta: DeltaChatMessage::Assistant {
                content: Some(ChatMessageContent::Text(res)),
                reasoning_content: None,
                refusal: None,
                name: None,
                tool_calls: None,
            },
            finish_reason: None,
            logprobs: None,
        }
    };
    response.choices.push(choice);
    response
}

pub fn get_logit_processor(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: u64,
) -> LogitsProcessor {
    let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
    match top_k {
        None => LogitsProcessor::new(
            seed,
            temperature.map(|temp| temp as f64),
            top_p.map(|tp| tp as f64),
        ),
        Some(k) => {
            let sampling = match temperature {
                None => Sampling::ArgMax,
                Some(temperature) => match top_p {
                    None => Sampling::TopK {
                        k,
                        temperature: temperature as f64,
                    },
                    Some(p) => Sampling::TopKThenTopP {
                        k,
                        p: p as f64,
                        temperature: temperature as f64,
                    },
                },
            };
            LogitsProcessor::from_sampling(seed, sampling)
        }
    }
}

pub fn extract_mes(mes: &ChatCompletionParameters) -> Result<Vec<(String, String)>> {
    let mut mes_vec = Vec::new();
    for chat_mes in mes.messages.clone() {
        if let ChatMessage::User { content, .. } = chat_mes.clone()
            && let ChatMessageContent::ContentPart(part_vec) = content
        {
            for part in part_vec {
                if let ChatMessageContentPart::Text(text_part) = part {
                    let text = text_part.text;
                    mes_vec.push(("<|User|>".to_string(), text));
                }
            }
        } else if let ChatMessage::Assistant { content, .. } = chat_mes.clone()
            && let Some(cont) = content
            && let ChatMessageContent::Text(c) = cont
        {
            mes_vec.push(("<|Assistant|>".to_string(), c));
        }
    }
    Ok(mes_vec)
}
