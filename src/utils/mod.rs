pub mod audio_utils;
pub mod img_utils;
pub mod interpolate;
pub mod response_utils;
pub mod tensor_utils;
pub mod video_utils;

use std::fs::File;
use std::io::{Cursor, Read};
use std::time::{SystemTime, UNIX_EPOCH};
use std::{collections::HashMap, fs, path::PathBuf, process::Command, time::Duration};

use crate::models::common::model_mapping::WhichModel;
use crate::params::chat::{
    ChatCompletionParameters, ChatMessage, ChatMessageContent, ChatMessageContentPart,
};
use anyhow::{Result, anyhow};
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{
    Context, DType, Device, Shape, Tensor,
    pickle::{Object, Stack, TensorInfo, read_all_with_key},
};
use candle_nn::VarBuilder;
use dirs::home_dir;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use modelscope::ModelScope;
use tokio::time::sleep;
use zip::ZipArchive;

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
        DType::I16 => {
            let mut data_t = vec![0i16; elem_count];
            reader.read_i16_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::I32 => {
            let mut data_t = vec![0i32; elem_count];
            reader.read_i32_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::I64 => {
            let mut data_t = vec![0i64; elem_count];
            reader.read_i64_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::F8E4M3 | DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 => {
            Err(anyhow!(format!("UnsupportedDTypeForOp '{:?}'", dtype)))
        }
    }
}

pub fn read_pth_tensor_info_cycle<P: AsRef<std::path::Path>>(
    path: P,
    key: Option<&str>,
) -> Result<HashMap<String, Tensor>> {
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
            } else if let Object::Dict(key_values) = obj {
                key_values
                    .into_iter()
                    .find(|(k, _)| *k == Object::Unicode(key.to_owned()))
                    .map(|(_, v)| v)
                    .ok_or_else(|| anyhow!(format!("key {key} not found")))?
            } else {
                obj
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
        match tensor_infos.get(name) {
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
    let mut dict_to_hashmap = HashMap::new();
    for (k, v) in tensors {
        dict_to_hashmap.insert(k, v);
    }
    Ok(dict_to_hashmap)
}

// 获取秒级时间戳
pub fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// 获取毫秒级时间戳
pub fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
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
        if let ChatMessage::User { content, .. } = chat_mes.clone() {
            match content {
                ChatMessageContent::Text(text) => ret = ret + &text + "\n",
                ChatMessageContent::ContentPart(part_vec) => {
                    for part in part_vec {
                        if let ChatMessageContentPart::Text(text_part) = part {
                            let text = text_part.text;
                            if text.chars().count() > 0 {
                                ret = ret + &text + "\n"
                            }
                        }
                    }
                }
                _ => {}
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

pub fn load_tensor_from_pt(
    path: &str,
    zip_name: &str,
    shape: Shape,
    device: &Device,
) -> Result<Tensor> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    // // 列出所有文件（调试用）
    // for i in 0..archive.len() {
    //     let file = archive.by_index(i)?;
    //     println!("File: {} ({} bytes)", file.name(), file.size());
    // }
    // 读取原始字节数据
    let mut data_file = archive.by_name(zip_name)?;
    let mut buffer = Vec::new();
    data_file.read_to_end(&mut buffer)?;
    // 将字节转换为 f32 (little endian)
    let mut cursor = Cursor::new(buffer);
    let num_elements = shape.elem_count();
    let mut data = Vec::with_capacity(num_elements);

    for _ in 0..num_elements {
        let val = cursor.read_f32::<LittleEndian>()?;
        data.push(val);
    }
    let t = Tensor::from_vec(data, shape, device)?;
    Ok(t)
}

/// Map OpenAI language code to full language name for ASR models
///
/// Supports 29 languages as per Qwen3ASR specification
pub fn map_language_code(code: &str) -> Option<String> {
    match code.to_lowercase().as_str() {
        "zh" => Some("Chinese".to_string()),
        "en" => Some("English".to_string()),
        "yue" => Some("Cantonese".to_string()),
        "ar" => Some("Arabic".to_string()),
        "de" => Some("German".to_string()),
        "fr" => Some("French".to_string()),
        "es" => Some("Spanish".to_string()),
        "pt" => Some("Portuguese".to_string()),
        "id" => Some("Indonesian".to_string()),
        "it" => Some("Italian".to_string()),
        "ko" => Some("Korean".to_string()),
        "ru" => Some("Russian".to_string()),
        "th" => Some("Thai".to_string()),
        "vi" => Some("Vietnamese".to_string()),
        "ja" => Some("Japanese".to_string()),
        "tr" => Some("Turkish".to_string()),
        "hi" => Some("Hindi".to_string()),
        "ms" => Some("Malay".to_string()),
        "nl" => Some("Dutch".to_string()),
        "sv" => Some("Swedish".to_string()),
        "da" => Some("Danish".to_string()),
        "fi" => Some("Finnish".to_string()),
        "pl" => Some("Polish".to_string()),
        "cs" => Some("Czech".to_string()),
        "fil" => Some("Filipino".to_string()),
        "fa" => Some("Persian".to_string()),
        "el" => Some("Greek".to_string()),
        "ro" => Some("Romanian".to_string()),
        "hu" => Some("Hungarian".to_string()),
        "mk" => Some("Macedonian".to_string()),
        _ => None,
    }
}

/// Clean ASR model output by extracting pure text from model-specific format
///
/// Qwen3ASR outputs format: "language English<asr_text>The morning sun..."
/// This function extracts the text after "<asr_text>" marker.
/// If no marker is found, returns the original text trimmed (for compatibility).
pub fn clean_asr_response(raw: &str) -> String {
    if let Some(start) = raw.find("<asr_text>") {
        raw[start + "<asr_text>".len()..].trim().to_string()
    } else {
        raw.trim().to_string()
    }
}

/// Get the default weight path for a given model
/// Returns ~/.aha/{model_id} e.g., ~/.aha/OpenBMB/VoxCPM1.5
pub fn get_default_weight_path(model: WhichModel) -> String {
    let model_id = model.as_string();
    let save_dir = get_default_save_dir().expect("Failed to get home directory");
    format!("{}/{}", save_dir, model_id)
}

/// Check if a model is downloaded by verifying the model directory exists
/// Returns true if ~/.aha/{model_id} directory exists, false otherwise
pub fn is_model_downloaded(model: WhichModel) -> bool {
    let model_id = model.as_string();
    let save_dir = match get_default_save_dir() {
        Some(dir) => dir,
        None => return false,
    };
    let model_path = format!("{}/{}", save_dir, model_id);
    std::path::Path::new(&model_path).exists()
}

/// Calculate total size of a directory recursively
pub fn dir_size(path: &std::path::Path) -> anyhow::Result<u64> {
    let mut total = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                total += dir_size(&entry_path)?;
            } else {
                total += entry.metadata()?.len();
            }
        }
    } else {
        total = std::fs::metadata(path)?.len();
    }
    Ok(total)
}

/// Convert bytes to human readable format
pub fn bytes_to_human(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

pub fn bucketize(input: usize, boundaries: &[usize]) -> Result<usize> {
    if boundaries.is_empty() {
        return Err(anyhow!("bucketize param boundaries can not be empty"));
    }
    match boundaries.binary_search(&input) {
        Ok(i) => Ok(i),
        Err(i) => Ok(i),
    }
    // let mut index = 0;
    // let mut change = false;
    // for i in 0..boundaries.len() {
    //     if input <= boundaries[i] {
    //         index = i;
    //         change = true;
    //         break;
    //     }
    // }
    // if !change {
    //     index = boundaries.len();
    // }
    // Ok(index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_language_code_chinese() {
        assert_eq!(map_language_code("zh"), Some("Chinese".to_string()));
    }

    #[test]
    fn test_map_language_code_english() {
        assert_eq!(map_language_code("en"), Some("English".to_string()));
    }

    #[test]
    fn test_map_language_code_case_insensitive() {
        assert_eq!(map_language_code("ZH"), Some("Chinese".to_string()));
        assert_eq!(map_language_code("EN"), Some("English".to_string()));
    }

    #[test]
    fn test_map_language_code_invalid() {
        assert_eq!(map_language_code("xx"), None);
    }

    #[test]
    fn test_clean_asr_response_standard_format() {
        let raw = "language English<asr_text>The morning sun cast golden light";
        let cleaned = clean_asr_response(raw);
        assert_eq!(cleaned, "The morning sun cast golden light");
    }

    #[test]
    fn test_clean_asr_response_chinese_format() {
        let raw = "language Chinese<asr_text>科技不断改变着我们的生活";
        let cleaned = clean_asr_response(raw);
        assert_eq!(cleaned, "科技不断改变着我们的生活");
    }

    #[test]
    fn test_clean_asr_response_with_newlines() {
        let raw = "language English<asr_text>\n\n  Hello world\n  ";
        let cleaned = clean_asr_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_asr_response_no_marker() {
        let raw = "  Plain text without marker  ";
        let cleaned = clean_asr_response(raw);
        assert_eq!(cleaned, "Plain text without marker");
    }
}
