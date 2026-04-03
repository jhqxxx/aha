use crate::{
    cli::args::{CliArgs, DeleteArgs, DownloadArgs, ListArgs, RunArgs, ServArgs, ServListArgs},
    server::{
        api::init,
        process::{ServiceStatus, find_aha_services},
        start_http_server,
    },
};
use aha::exec::*;
use aha::{
    models::common::model_mapping::WhichModel,
    utils::{
        bytes_to_human, dir_size, download_model, get_default_save_dir, get_default_weight_path,
        is_model_downloaded,
    },
};
use anyhow::anyhow;
use serde::Serialize;

pub mod args;

/// Model information for JSON output
#[derive(Serialize)]
struct ModelInfo {
    model_id: String,
    owner: String,
    #[serde(rename = "type")]
    model_type: String,
    downloaded: bool,
}

/// List all supported models
pub(crate) fn run_list(args: ListArgs) -> anyhow::Result<()> {
    let models = WhichModel::model_list();

    if args.json {
        // JSON output
        let model_infos: Vec<ModelInfo> = models
            .iter()
            .map(|model| ModelInfo {
                model_id: model.as_string(),
                owner: model.model_owner(),
                model_type: model.model_type().to_string(),
                downloaded: is_model_downloaded(*model),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&model_infos)?);
    } else {
        // Table output (default)
        println!("Available models:");
        println!();
        println!(
            "{:<40} {:<20} {:<10} {:<10}",
            "Model ID", "Owner", "type", "Download"
        );
        println!("{}", "-".repeat(80));
        for model in models {
            let model_id = model.as_string();
            let owner = model.model_owner();
            let model_type = model.model_type();
            let download_status = if is_model_downloaded(model) {
                "  ✔"
            } else {
                ""
            };
            println!(
                "{:<40} {:<20} {:<10} {:<10}",
                model_id, owner, model_type, download_status
            );
        }
    }

    Ok(())
}

/// Run the 'cli' subcommand: download model (if needed) and start service
pub(crate) async fn run_cli(args: CliArgs) -> anyhow::Result<()> {
    let CliArgs {
        model,
        server_common,
        save_dir,
        download_retries,
        path_common,
    } = args;
    let model_id = model.as_string();

    let (model_path, gguf, mmproj) = if model.is_gguf() {
        if path_common.gguf_path.is_none() {
            return Err(anyhow!("gguf model path is required"));
        }
        (
            "GGUF".to_string(),
            path_common.gguf_path,
            path_common.mmproj_path,
        )
    } else if model.is_onnx() {
        return Err(anyhow!("onnx model not support now"));
    } else {
        let model_path = match path_common.weight_path {
            Some(path) => path,
            None => {
                let save_dir = match save_dir {
                    Some(dir) => dir,
                    None => get_default_save_dir().expect("Failed to get home directory"),
                };
                let max_retries = download_retries.unwrap_or(3);
                download_model(&model_id, &save_dir, max_retries).await?;
                save_dir + "/" + &model_id
            }
        };
        (model_path, None, None)
    };

    init(model, model_path, gguf, mmproj)?;
    start_http_server(
        server_common.address,
        server_common.port,
        server_common.allow_remote_shutdown,
    )
    .await?;

    Ok(())
}

/// Run the 'serv' subcommand: start service only (no download)
pub(crate) async fn run_serv(args: ServArgs) -> anyhow::Result<()> {
    let ServArgs {
        model,
        server_common,
        path_common,
    } = args;
    let (model_path, gguf, mmproj) = if model.is_gguf() {
        if path_common.gguf_path.is_none() {
            return Err(anyhow!("gguf model path is required"));
        }
        (
            "GGUF".to_string(),
            path_common.gguf_path,
            path_common.mmproj_path,
        )
    } else if model.is_onnx() {
        return Err(anyhow!("onnx model not support now"));
    } else {
        let model_path = match path_common.weight_path {
            Some(path) => path,
            None => get_default_weight_path(model),
        };
        if !std::path::Path::new(&model_path).exists() {
            return Err(anyhow!(
                "serv subcommand will not download model, use `weight-path` to pass the model path"
            ));
        }
        (model_path, None, None)
    };

    init(model, model_path, gguf, mmproj)?;
    start_http_server(
        server_common.address,
        server_common.port,
        server_common.allow_remote_shutdown,
    )
    .await?;

    Ok(())
}

/// Run the 'ps' subcommand: list running AHA services
pub(crate) fn run_ps(args: ServListArgs) -> anyhow::Result<()> {
    let services = find_aha_services()?;

    if services.is_empty() {
        println!("No aha services found running.");
        return Ok(());
    }

    if args.compact {
        // Compact format: one service per line
        for svc in services {
            println!("{}", svc.service_id);
        }
    } else {
        // Table format
        println!(
            "{:<20} {:<10} {:<20} {:<10} {:<15} {:<10}",
            "Service ID", "PID", "Model", "Port", "Address", "Status"
        );
        println!("{}", "-".repeat(85));

        for svc in services {
            let model = svc.model.as_deref().unwrap_or("N/A");
            let status = match svc.status {
                ServiceStatus::Running => "Running",
                ServiceStatus::Stopping => "Stopping",
                ServiceStatus::Unknown => "Unknown",
            };
            println!(
                "{:<20} {:<10} {:<20} {:<10} {:<15} {:<10}",
                svc.service_id, svc.pid, model, svc.port, svc.address, status,
            );
        }
    }

    Ok(())
}

/// Run the 'download' subcommand: download model only (no server)
pub(crate) async fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    let DownloadArgs {
        model,
        save_dir,
        download_retries,
    } = args;
    let model_id = model.as_string();

    let save_dir = match save_dir {
        Some(dir) => dir,
        None => get_default_save_dir().expect("Failed to get home directory"),
    };
    let max_retries = download_retries.unwrap_or(3);

    download_model(&model_id, &save_dir, max_retries).await?;

    Ok(())
}

/// Run the 'run' subcommand: direct model inference
pub(crate) fn run_run(args: RunArgs) -> anyhow::Result<()> {
    let RunArgs {
        model,
        input,
        output,
        path_common,
    } = args;

    // Use default weight path if not specified
    let weight_path = match path_common.weight_path {
        Some(path) => path,
        None => get_default_weight_path(model),
    };
    match model {
        WhichModel::MiniCPM4_0_5B => {
            minicpm4::MiniCPM4Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::LFM2_1_2B => {
            lfm2::Lfm2Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::LFM2_5_1_2BInstruct => {
            lfm2::Lfm2Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::LFM2_5VL1_6B => {
            lfm2vl::Lfm2VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::LFM2VL1_6B => {
            lfm2vl::Lfm2VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen2_5VL3B => {
            qwen2_5vl::Qwen2_5VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen2_5VL7B => {
            qwen2_5vl::Qwen2_5VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_0_6B => {
            qwen3::Qwen3Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_1_7B => {
            qwen3::Qwen3Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_4B => {
            qwen3::Qwen3Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5_0_8B => {
            qwen3_5::Qwen3_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5_2B => {
            qwen3_5::Qwen3_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5_4B => {
            qwen3_5::Qwen3_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5_9B => {
            qwen3_5::Qwen3_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5Gguf => {
            qwen3_5::Qwen3_5Exec::run_gguf(
                &input,
                output.as_deref(),
                path_common.gguf_path,
                path_common.mmproj_path,
            )?;
        }
        WhichModel::Qwen3ASR0_6B => {
            qwen3_asr::Qwen3ASRExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3ASR1_7B => {
            qwen3_asr::Qwen3ASRExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3VL2B => {
            qwen3vl::Qwen3VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3VL4B => {
            qwen3vl::Qwen3VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3VL8B => {
            qwen3vl::Qwen3VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3VL32B => {
            qwen3vl::Qwen3VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::DeepSeekOCR => {
            deepseek_ocr::DeepSeekORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::DeepSeekOCR2 => {
            deepseek_ocr::DeepSeekORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::HunyuanOCR => {
            hunyuan_ocr::HunyuanORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::PaddleOCRVL => {
            paddleocr_vl::PaddleOVLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::PaddleOCRVL1_5 => {
            paddleocr_vl::PaddleOVLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::RMBG2_0 => {
            rmbg2_0::RMBG2_0Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::VoxCPM => {
            voxcpm::VoxCPMExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::VoxCPM1_5 => {
            voxcpm::VoxCPMExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::GlmASRNano2512 => {
            glm_asr_nano::GlmASRNanoExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::FunASRNano2512 => {
            fun_asr_nano::FunASRNanoExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::GlmOCR => {
            glm_ocr::GlmOcrExec::run(&input, output.as_deref(), &weight_path)?;
        }
    }

    Ok(())
}

/// Run the 'delete' subcommand: delete model from default location
pub(crate) fn run_delete(args: DeleteArgs) -> anyhow::Result<()> {
    let DeleteArgs { model } = args;
    let model_id = model.as_string();
    let save_dir = get_default_save_dir().expect("Failed to get home directory");
    let model_path = format!("{}/{}", save_dir, model_id);

    let path = std::path::Path::new(&model_path);

    if !path.exists() {
        println!("Model not found: {} does not exist", model_path);
        return Ok(());
    }

    // Show model info
    println!("Model ID: {}", model_id);
    println!("Location: {}", model_path);

    // Calculate size if possible
    if let Ok(metadata) = std::fs::metadata(path)
        && metadata.is_dir()
        && let Ok(total_size) = dir_size(path)
    {
        println!("Size: {}", bytes_to_human(total_size));
    }

    // Confirm deletion
    print!("Are you sure you want to delete this model? (y/N): ");
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    let input = input.trim().to_lowercase();
    if input != "y" && input != "yes" {
        println!("Deletion cancelled.");
        return Ok(());
    }

    // Delete the directory
    std::fs::remove_dir_all(path)?;

    println!("Model deleted successfully: {}", model_path);

    Ok(())
}
