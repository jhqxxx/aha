pub mod audio_utils;
pub mod img_utils;
pub mod tensor_utils;
pub mod video_utils;

use std::io::Read;
use std::{collections::HashMap, fs, path::PathBuf, process::Command, time::Duration};

use aha_openai_dive::v1::resources::{
    chat::{
        AudioUrlType, ChatCompletionChoice, ChatCompletionChunkChoice, ChatCompletionChunkResponse,
        ChatCompletionParameters, ChatCompletionResponse, ChatMessage, ChatMessageAudioContentPart,
        ChatMessageContent, ChatMessageContentPart, ChatMessageImageContentPart, DeltaChatMessage,
        DeltaFunction, DeltaToolCall, Function, ImageUrlType, ToolCall,
    },
    shared::{FinishReason, Usage},
};
use anyhow::{Result, anyhow};
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{
    Context, DType, Device, Shape, Tensor,
    pickle::{Object, PthTensors, Stack, TensorInfo, read_all_with_key},
};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use dirs::home_dir;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use modelscope::ModelScope;
use tokio::time::sleep;

pub fn get_device(device: Option<&Device>) -> Device {
    match device {
        Some(d) => d.clone(),
        None => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            }
            #[cfg(all(not(feature = "cuda"), feature = "metal"))]
            {
                Device::new_metal(0).unwrap_or(Device::Cpu)
            }
            #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
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

pub fn get_vb_model_path(
    model_path: String,
    dtype: DType,
    device: Device,
    key: Option<&'_ str>,
) -> Result<VarBuilder<'_>> {
    let mut dict_to_hashmap = HashMap::new();
    let dict = read_all_with_key(&model_path, key)?;
    for (k, v) in dict {
        dict_to_hashmap.insert(k, v);
    }
    let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &device);
    Ok(vb)
}

pub fn get_vb_extension(
    path: String,
    extension_type: String,
    dtype: DType,
    device: Device,
    key: Option<&'_ str>,
) -> Result<VarBuilder<'_>> {
    let model_list = find_type_files(&path, &extension_type)?;
    let mut dict_to_hashmap = HashMap::new();
    for m in model_list {
        let dict = read_all_with_key(m, key)?;
        for (k, v) in dict {
            dict_to_hashmap.insert(k, v);
        }
    }
    let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &device);
    Ok(vb)
}

pub fn crate_tensor_from_reader<R: std::io::Read>(
    shape: Shape,
    dtype: DType,
    reader: &mut R,
) -> Result<Tensor> {
    let elem_count = shape.elem_count();
    match dtype {
        DType::BF16 => {
            let mut data_t = vec![bf16::ZERO; elem_count];
            reader.read_u16_into::<LittleEndian>(data_t.reinterpret_cast_mut())?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::F16 => {
            let mut data_t = vec![f16::ZERO; elem_count];
            reader.read_u16_into::<LittleEndian>(data_t.reinterpret_cast_mut())?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::F32 => {
            let mut data_t = vec![0f32; elem_count];
            reader.read_f32_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::F64 => {
            let mut data_t = vec![0f64; elem_count];
            reader.read_f64_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::U8 => {
            let mut data_t = vec![0u8; elem_count];
            reader.read_exact(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::U32 => {
            let mut data_t = vec![0u32; elem_count];
            reader.read_u32_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::I64 => {
            let mut data_t = vec![0i64; elem_count];
            reader.read_i64_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
    }
}

pub fn read_pth_tensor_info_cycle<P: AsRef<std::path::Path>>(
    path: P,
    key: Option<&str>,
) -> Result<Vec<(String, Tensor)>> {
    let file = std::fs::File::open(path.as_ref())?;
    let zip_reader = std::io::BufReader::new(file);
    let mut zip = zip::ZipArchive::new(zip_reader)?;
    let zip_file_names = zip
        .file_names()
        .map(|f| f.to_string())
        .collect::<Vec<String>>();

    let mut tensor_infos = vec![];
    for file_name in zip_file_names.iter() {
        if !file_name.ends_with("data.pkl") {
            continue;
        }
        let dir_name = std::path::PathBuf::from(file_name.strip_suffix(".pkl").context("no .pkl")?);
        let reader = zip.by_name(file_name)?;
        let mut reader = std::io::BufReader::new(reader);
        let mut stack = Stack::empty();
        stack.read_loop(&mut reader)?;
        let obj = stack.finalize()?;

        let obj = match obj {
            Object::Build { callable, args } => match *callable {
                Object::Reduce { callable, args: _ } => match *callable {
                    Object::Class {
                        module_name,
                        class_name,
                    } if module_name == "__torch__" && class_name == "Module" => *args,
                    _ => continue,
                },
                _ => continue,
            },
            obj => obj,
        };

        // If key is provided, then we need to extract the state_dict from the object.
        let obj = if let Some(key) = key {
            let multi_key: Vec<&str> = key.split(".").collect();
            if multi_key.len() > 1 {
                let mut current_obj = obj;
                for k in multi_key.iter() {
                    if let Object::Dict(key_values) = current_obj {
                        current_obj = key_values
                            .into_iter()
                            .find(|(key_obj, _)| *key_obj == Object::Unicode(k.to_string()))
                            .map(|(_, v)| v)
                            .ok_or_else(|| anyhow!(format!("key '{}' not found", k)))?;
                    } else {
                        return Err(anyhow!(format!(
                            "Expected dictionary at key '{}', but found other type",
                            k
                        )));
                    }
                }
                current_obj
            } else {
                if let Object::Dict(key_values) = obj {
                    key_values
                        .into_iter()
                        .find(|(k, _)| *k == Object::Unicode(key.to_owned()))
                        .map(|(_, v)| v)
                        .ok_or_else(|| anyhow!(format!("key {key} not found")))?
                } else {
                    obj
                }
            }
        } else {
            obj
        };

        // If the object is a dict, then we can extract the tensor info from it.
        // NOTE: We are assuming that the `obj` is state_dict by this stage.
        if let Object::Dict(key_values) = obj {
            for (name, value) in key_values.into_iter() {
                match value.into_tensor_info(name, &dir_name) {
                    Ok(Some(tensor_info)) => tensor_infos.push(tensor_info),
                    Ok(None) => {}
                    Err(err) => eprintln!("skipping: {err:?}"),
                }
            }
        }
    }
    let tensor_infos: HashMap<String, TensorInfo> = tensor_infos
        .into_iter()
        .map(|ti| (ti.name.to_string(), ti))
        .collect();

    let tensor_names = tensor_infos.keys();
    let mut tensors = Vec::with_capacity(tensor_names.len());
    for name in tensor_names {
        let _ = match tensor_infos.get(name) {
            None => {}
            Some(tensor_info) => {
                let zip_reader = std::io::BufReader::new(std::fs::File::open(&path)?);
                let mut zip = zip::ZipArchive::new(zip_reader)?;
                let mut reader = zip.by_name(&tensor_info.path)?;
                let is_fortran_contiguous = tensor_info.layout.is_fortran_contiguous();
                let rank = tensor_info.layout.shape().rank();

                // Reading the data is a bit tricky as it can be strided, for now only support the basic
                // case and when the tensor is fortran contiguous.
                if !tensor_info.layout.is_contiguous() && !is_fortran_contiguous {
                    return Err(anyhow!(format!(
                        "cannot retrieve non-contiguous tensors {:?}",
                        tensor_info.layout
                    )));
                }
                let start_offset = tensor_info.layout.start_offset();
                if start_offset > 0 {
                    std::io::copy(
                        &mut reader.by_ref().take(start_offset as u64),
                        &mut std::io::sink(),
                    )?;
                }
                let tensor = crate_tensor_from_reader(
                    tensor_info.layout.shape().clone(),
                    tensor_info.dtype,
                    &mut reader,
                )?;

                if rank > 1 && is_fortran_contiguous {
                    // Reverse the shape, e.g. Shape(2, 3, 4) -> Shape(4, 3, 2)
                    let shape_reversed: Vec<_> =
                        tensor_info.layout.dims().iter().rev().cloned().collect();
                    let tensor = tensor.reshape(shape_reversed)?;

                    // Permute (transpose) the dimensions, e.g. Shape(4, 3, 2) -> Shape(2, 3, 4)
                    let dim_indeces_reversed: Vec<_> = (0..rank).rev().collect();
                    let tensor = tensor.permute(dim_indeces_reversed)?;
                    // Ok(Some(tensor))
                    tensors.push((name.clone(), tensor));
                } else {
                    tensors.push((name.clone(), tensor));
                }
            }
        };
    }
    Ok(tensors)
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

pub fn build_img_completion_response(
    base64vec: &Vec<String>,
    model_name: &str,
) -> ChatCompletionResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let mut response = ChatCompletionResponse {
        id: Some(id),
        choices: vec![],
        created: chrono::Utc::now().timestamp() as u32,
        model: model_name.to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage: None,
    };
    let mut conten_part_vec = vec![];
    for img_bas64 in base64vec {
        let img_base64_prefix = "data:image/png;base64,".to_string() + img_bas64;
        let part = ChatMessageContentPart::Image(ChatMessageImageContentPart {
            r#type: "image".to_string(),
            image_url: ImageUrlType {
                url: img_base64_prefix,
                detail: None,
            },
        });
        conten_part_vec.push(part);
    }
    let choice = ChatCompletionChoice {
        index: 0,
        message: ChatMessage::Assistant {
            content: Some(ChatMessageContent::ContentPart(conten_part_vec)),
            reasoning_content: None,
            refusal: None,
            name: None,
            audio: None,
            tool_calls: None,
        },
        finish_reason: Some(FinishReason::StopSequenceReached),
        logprobs: None,
    };
    response.choices.push(choice);
    response
}

pub fn build_audio_completion_response(
    base64_audio: &String,
    model_name: &str,
) -> ChatCompletionResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let mut response = ChatCompletionResponse {
        id: Some(id),
        choices: vec![],
        created: chrono::Utc::now().timestamp() as u32,
        model: model_name.to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage: None,
    };

    let base64_audio = format!("data:audio/wav;base64,{}", base64_audio);
    let conten_part_vec = vec![ChatMessageContentPart::Audio(ChatMessageAudioContentPart {
        r#type: "audio".to_string(),
        audio_url: AudioUrlType {
            url: base64_audio.to_string(),
        },
    })];
    let choice = ChatCompletionChoice {
        index: 0,
        message: ChatMessage::Assistant {
            content: Some(ChatMessageContent::ContentPart(conten_part_vec)),
            reasoning_content: None,
            refusal: None,
            name: None,
            audio: None,
            tool_calls: None,
        },
        finish_reason: Some(FinishReason::StopSequenceReached),
        logprobs: None,
    };
    response.choices.push(choice);
    response
}

pub fn build_completion_response(
    res: String,
    model_name: &str,
    num_tokens: Option<u32>,
) -> ChatCompletionResponse {
    let id = uuid::Uuid::new_v4().to_string();
    let usage = num_tokens.map(|num| Usage {
        prompt_tokens: None,
        completion_tokens: None,
        total_tokens: num,
        prompt_tokens_details: None,
        completion_tokens_details: None,
    });
    let mut response = ChatCompletionResponse {
        id: Some(id),
        choices: vec![],
        created: chrono::Utc::now().timestamp() as u32,
        model: model_name.to_string(),
        service_tier: None,
        system_fingerprint: None,
        object: "chat.completion".to_string(),
        usage,
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

pub fn extract_metadata_value<T>(
    metadata: &Option<std::collections::HashMap<String, String>>,
    key: &str,
) -> Option<T>
where
    T: std::str::FromStr + Clone + PartialEq,
{
    if let Some(map) = metadata
        && let Some(value_str) = map.get(key)
        && let Ok(value) = value_str.parse::<T>()
    {
        return Some(value);
    }
    None
}

pub fn extract_user_text(mes: &ChatCompletionParameters) -> Result<String> {
    let mut ret = "".to_string();
    for chat_mes in mes.messages.clone() {
        if let ChatMessage::User { content, .. } = chat_mes.clone()
            && let ChatMessageContent::ContentPart(part_vec) = content
        {
            for part in part_vec {
                if let ChatMessageContentPart::Text(text_part) = part {
                    let text = text_part.text;
                    if text.chars().count() > 0 {
                        ret = ret + &text + "\n"
                    }
                }
            }
        }
    }
    ret = ret.trim().to_string();
    Ok(ret)
}

pub fn extract_user_text_vec(mes: &ChatCompletionParameters) -> Result<Vec<String>> {
    let mut ret = vec![];
    for chat_mes in mes.messages.clone() {
        if let ChatMessage::User { content, .. } = chat_mes.clone()
            && let ChatMessageContent::ContentPart(part_vec) = content
        {
            for part in part_vec {
                if let ChatMessageContentPart::Text(text_part) = part {
                    let text = text_part.text;
                    if text.chars().count() > 0 {
                        ret.push(text);
                    }
                }
            }
        }
    }
    Ok(ret)
}

pub fn get_default_save_dir() -> Option<String> {
    home_dir().map(|mut path| {
        path.push(".aha");
        if let Err(e) = fs::create_dir_all(&path) {
            eprintln!("Failed to create directory {:?}: {}", path, e);
        }
        path.to_string_lossy().to_string()
    })
}

pub async fn download_model(
    model_id: &str,
    save_dir: &str,
    max_retries: u32,
) -> anyhow::Result<()> {
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        println!(
            "Attempting to download model (attempt {}/{})",
            attempts, max_retries
        );

        match ModelScope::download(model_id, save_dir).await {
            Ok(()) => {
                println!("Model downloaded successfully");
                return Ok(());
            }
            Err(e) => {
                if attempts >= max_retries {
                    return Err(anyhow::anyhow!(
                        "Failed to download model after {} attempts. Last error: {}",
                        max_retries,
                        e
                    ));
                }

                println!(
                    "Download failed (attempt {}): {}. Retrying in 2 seconds...",
                    attempts, e
                );
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
}

pub fn get_file_path(file: &str) -> Result<PathBuf> {
    let path = url::Url::parse(file)?;
    let path = path.to_file_path();
    let path = match path {
        Ok(path) => path,
        Err(_) => {
            let mut path = file.to_owned();
            path = path.split_off(7);
            PathBuf::from(path)
        }
    };
    Ok(path)
}

pub fn capitalize_first_letter(input: &str) -> String {
    if input.is_empty() {
        return input.to_string();
    }

    let mut chars = input.chars();
    let first_char = chars.next().unwrap().to_uppercase().collect::<String>();
    let remaining = chars.as_str().to_lowercase();
    format!("{}{}", first_char, remaining)
}
