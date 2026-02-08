use std::{net::IpAddr, str::FromStr, sync::Arc};

use aha::{
    models::WhichModel,
    process::{create_pid_file, cleanup_pid_file},
    utils::{download_model, get_default_save_dir},
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rocket::{
    Config,
    data::{ByteUnit, Limits},
    routes,
};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::api::{init, set_server_port};
mod api;

#[derive(Parser, Debug)]
#[command(name = "aha")]
#[command(version, about, long_about = None)]
struct Cli {
    /// Service listen address
    #[arg(short, long, default_value = "127.0.0.1")]
    address: Option<String>,

    /// Service listen port
    #[arg(short, long)]
    port: Option<u16>,

    /// Model type (required for backward compatibility)
    #[arg(short, long)]
    model: Option<WhichModel>,

    /// Local model weight path
    #[arg(long)]
    weight_path: Option<String>,

    /// Model download save directory
    #[arg(long)]
    save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    download_retries: Option<u32>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Download model and start service (default)
    Cli(CliArgs),
    /// Start service only (--weight-path is optional, defaults to ~/.aha/{model_id})
    Serv(ServArgs),
    /// List all running aha services
    Ps(ServListArgs),
    /// Delete a downloaded model from the default location (~/.aha/{model_id})
    Delete(DeleteArgs),
    /// Download model only
    Download(DownloadArgs),
    /// Run model inference directly
    Run(RunArgs),
    /// List all supported models
    List,
}

/// Common/shared arguments for server operations
#[derive(Args, Debug)]
struct CommonArgs {
    /// Service listen address
    #[arg(short, long, default_value = "127.0.0.1")]
    address: String,

    /// Service listen port
    #[arg(short, long, default_value_t = 10100)]
    port: u16,

    /// Model type (required)
    #[arg(short, long)]
    model: WhichModel,

    /// Allow remote shutdown requests (default: local only, use with caution)
    #[arg(long)]
    allow_remote_shutdown: bool,
}

/// Arguments for the 'cli' subcommand (download + serve)
#[derive(Args, Debug)]
struct CliArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Local model weight path (skip download if provided)
    #[arg(long)]
    weight_path: Option<String>,

    /// Model download save directory
    #[arg(long)]
    save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    download_retries: Option<u32>,
}

/// Arguments for the 'serv start' subcommand
#[derive(Args, Debug)]
struct ServArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Local model weight path (defaults to ~/.aha/{model_id} if not specified)
    #[arg(long)]
    weight_path: Option<String>,
}

/// Arguments for the 'serv list' subcommand
#[derive(Args, Debug)]
struct ServListArgs {
    /// Compact output format
    #[arg(short, long)]
    compact: bool,
}

/// Arguments for the 'download' subcommand (download only)
#[derive(Args, Debug)]
struct DownloadArgs {
    /// Model type (required)
    #[arg(short, long)]
    model: WhichModel,

    /// Model download save directory
    #[arg(short, long)]
    save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    download_retries: Option<u32>,
}

/// Arguments for the 'run' subcommand (direct inference)
#[derive(Args, Debug)]
struct RunArgs {
    /// Model type (required)
    #[arg(short, long)]
    model: WhichModel,

    /// Input text or file path
    #[arg(short, long, num_args = 1..=2, value_delimiter = ' ')]
    input: Vec<String>,

    /// Output file path (optional)
    #[arg(short, long)]
    output: Option<String>,

    /// Local model weight path (defaults to ~/.aha/{model_id} if not specified)
    #[arg(long)]
    weight_path: Option<String>,
}

/// Arguments for the 'delete' subcommand (delete model from default location)
#[derive(Args, Debug)]
struct DeleteArgs {
    /// Model type (required)
    #[arg(short, long)]
    model: WhichModel,
}

/// Get the default weight path for a given model
/// Returns ~/.aha/{model_id} e.g., ~/.aha/OpenBMB/VoxCPM1.5
fn get_default_weight_path(model: WhichModel) -> String {
    let model_id = model.model_id();
    let save_dir = get_default_save_dir().expect("Failed to get home directory");
    format!("{}/{}", save_dir, model_id)
}

/// List all supported models
fn run_list() -> anyhow::Result<()> {
    let models = [
        WhichModel::MiniCPM4_0_5B,
        WhichModel::Qwen2_5vl3B,
        WhichModel::Qwen2_5vl7B,
        WhichModel::Qwen3_0_6B,
        WhichModel::Qwen3ASR0_6B,
        WhichModel::Qwen3ASR1_7B,
        WhichModel::Qwen3vl2B,
        WhichModel::Qwen3vl4B,
        WhichModel::Qwen3vl8B,
        WhichModel::Qwen3vl32B,
        WhichModel::DeepSeekOCR,
        WhichModel::HunyuanOCR,
        WhichModel::PaddleOCRVL,
        WhichModel::RMBG2_0,
        WhichModel::VoxCPM,
        WhichModel::VoxCPM1_5,
        WhichModel::GlmASRNano2512,
        WhichModel::FunASRNano2512,
    ];

    println!("Available models:");
    println!();
    println!("{:<30} ModelScope ID", "Model Name");
    println!("{}", "-".repeat(80));
    for model in models {
        let possible_value = model.to_possible_value().unwrap();
        let name = possible_value.get_name();
        let id = model.model_id();
        println!("{:<30} {}", name, id);
    }

    Ok(())
}

/// Run the 'cli' subcommand: download model (if needed) and start service
async fn run_cli(args: CliArgs) -> anyhow::Result<()> {
    let CliArgs {
        common,
        weight_path,
        save_dir,
        download_retries,
    } = args;
    let model_id = common.model.model_id();

    let model_path = match weight_path {
        Some(path) => path,
        None => {
            let save_dir = match save_dir {
                Some(dir) => dir,
                None => get_default_save_dir().expect("Failed to get home directory"),
            };
            let max_retries = download_retries.unwrap_or(3);
            download_model(model_id, &save_dir, max_retries).await?;
            save_dir + "/" + model_id
        }
    };

    init(common.model, model_path)?;
    start_http_server(common.address, common.port, common.allow_remote_shutdown).await?;

    Ok(())
}

/// Run the 'serv' subcommand: start service only (no download)
async fn run_serv(args: ServArgs) -> anyhow::Result<()> {
    let ServArgs {
        common,
        weight_path,
    } = args;

    let model_path = match weight_path {
        Some(path) => path,
        None => get_default_weight_path(common.model),
    };

    init(common.model, model_path)?;
    start_http_server(common.address, common.port, common.allow_remote_shutdown).await?;

    Ok(())
}

/// Run the 'ps' subcommand: list running AHA services
fn run_ps(args: ServListArgs) -> anyhow::Result<()> {
    use aha::process::find_aha_services;

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
        println!("{:<20} {:<10} {:<20} {:<10} {:<15} {:<10}",
            "Service ID", "PID", "Model", "Port", "Address", "Status");
        println!("{}", "-".repeat(85));

        for svc in services {
            let model = svc.model.as_deref().unwrap_or("N/A");
            let status = match svc.status {
                aha::process::ServiceStatus::Running => "Running",
                aha::process::ServiceStatus::Stopping => "Stopping",
                aha::process::ServiceStatus::Unknown => "Unknown",
            };
            println!("{:<20} {:<10} {:<20} {:<10} {:<15} {:<10}",
                svc.service_id,
                svc.pid,
                model,
                svc.port,
                svc.address,
                status,
            );
        }
    }

    Ok(())
}

/// Run the 'download' subcommand: download model only (no server)
async fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    let DownloadArgs {
        model,
        save_dir,
        download_retries,
    } = args;
    let model_id = model.model_id();

    let save_dir = match save_dir {
        Some(dir) => dir,
        None => get_default_save_dir().expect("Failed to get home directory"),
    };
    let max_retries = download_retries.unwrap_or(3);

    download_model(model_id, &save_dir, max_retries).await?;

    Ok(())
}

/// Run the 'run' subcommand: direct model inference
fn run_run(args: RunArgs) -> anyhow::Result<()> {
    use aha::exec::ExecModel;

    let RunArgs {
        model,
        input,
        output,
        weight_path,
    } = args;

    // Use default weight path if not specified
    let weight_path = match weight_path {
        Some(path) => path,
        None => get_default_weight_path(model),
    };

    match model {
        WhichModel::MiniCPM4_0_5B => {
            use aha::exec::minicpm4::MiniCPM4Exec;
            MiniCPM4Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen2_5vl3B => {
            use aha::exec::qwen2_5vl::Qwen2_5vlExec;
            Qwen2_5vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen2_5vl7B => {
            use aha::exec::qwen2_5vl::Qwen2_5vlExec;
            Qwen2_5vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_0_6B => {
            use aha::exec::qwen3::Qwen3Exec;
            Qwen3Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3ASR0_6B => {
            use aha::exec::qwen3_asr::Qwen3ASRExec;
            Qwen3ASRExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3ASR1_7B => {
            use aha::exec::qwen3_asr::Qwen3ASRExec;
            Qwen3ASRExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3vl2B => {
            use aha::exec::qwen3vl::Qwen3vlExec;
            Qwen3vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3vl4B => {
            use aha::exec::qwen3vl::Qwen3vlExec;
            Qwen3vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3vl8B => {
            use aha::exec::qwen3vl::Qwen3vlExec;
            Qwen3vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3vl32B => {
            use aha::exec::qwen3vl::Qwen3vlExec;
            Qwen3vlExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::DeepSeekOCR => {
            use aha::exec::deepseek_ocr::DeepSeekORExec;
            DeepSeekORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::HunyuanOCR => {
            use aha::exec::hunyuan_ocr::HunyuanORExec;
            HunyuanORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::PaddleOCRVL => {
            use aha::exec::paddleocr_vl::PaddleOVLExec;
            PaddleOVLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::RMBG2_0 => {
            use aha::exec::rmbg2_0::RMBG2_0Exec;
            RMBG2_0Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::VoxCPM => {
            use aha::exec::voxcpm::VoxCPMExec;
            VoxCPMExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::VoxCPM1_5 => {
            use aha::exec::voxcpm1_5::VoxCPM1_5Exec;
            VoxCPM1_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::GlmASRNano2512 => {
            use aha::exec::glm_asr_nano::GlmASRNanoExec;
            GlmASRNanoExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::FunASRNano2512 => {
            use aha::exec::fun_asr_nano::FunASRNanoExec;
            FunASRNanoExec::run(&input, output.as_deref(), &weight_path)?;
        }
    }

    Ok(())
}

/// Run the 'delete' subcommand: delete model from default location
fn run_delete(args: DeleteArgs) -> anyhow::Result<()> {
    let DeleteArgs { model } = args;
    let model_id = model.model_id();
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
    if let Ok(metadata) = std::fs::metadata(path) {
        if metadata.is_dir() {
            if let Ok(total_size) = dir_size(path) {
                println!("Size: {}", bytes_to_human(total_size));
            }
        }
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

/// Calculate total size of a directory recursively
fn dir_size(path: &std::path::Path) -> anyhow::Result<u64> {
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
fn bytes_to_human(bytes: u64) -> String {
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Cli(args)) => run_cli(args).await,
        Some(Commands::Serv(args)) => run_serv(args).await,
        Some(Commands::Ps(args)) => run_ps(args),
        Some(Commands::Delete(args)) => run_delete(args),
        Some(Commands::Download(args)) => run_download(args).await,
        Some(Commands::Run(args)) => run_run(args),
        Some(Commands::List) => run_list(),
        None => {
            // Backward compatibility: when no subcommand is provided, use 'cli' behavior
            let model = cli.model.expect("Model is required (use -m or --model)");
            let args = CliArgs {
                common: CommonArgs {
                    address: cli.address.unwrap_or_else(|| "127.0.0.1".to_string()),
                    port: cli.port.unwrap_or(10100),
                    model,
                    allow_remote_shutdown: false,
                },
                weight_path: cli.weight_path,
                save_dir: cli.save_dir,
                download_retries: cli.download_retries,
            };
            run_cli(args).await
        }
    }
}

pub(crate) async fn start_http_server(address: String, port: u16, allow_remote_shutdown: bool) -> anyhow::Result<()> {
    // Set server port for shutdown endpoint
    set_server_port(port, allow_remote_shutdown);

    // Create PID file for service tracking
    let pid = std::process::id();
    create_pid_file(pid, port)?;

    // Set up shutdown flag
    let shutdown_flag = Arc::new(AtomicBool::new(false));
    let shutdown_flag_clone = shutdown_flag.clone();

    // Configure Ctrl+C handler for graceful shutdown
    let port_for_cleanup = port;
    let shutdown_handler = tokio::spawn(async move {
        tokio::signal::ctrl_c().await.ok();
        println!("Received shutdown signal, gracefully shutting down...");
        shutdown_flag_clone.store(true, Ordering::SeqCst);
        // Give time for existing requests to complete
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        // Cleanup PID file
        let _ = cleanup_pid_file(port_for_cleanup);
        std::process::exit(0);
    });

    let mut builder = rocket::build().configure(Config {
        address: IpAddr::from_str(&address)?,
        port,
        limits: Limits::default()
            .limit("string", ByteUnit::Mebibyte(5))
            .limit("json", ByteUnit::Mebibyte(5))
            .limit("data-form", ByteUnit::Mebibyte(100))
            .limit("file", ByteUnit::Mebibyte(100)),
        ..Config::default()
    });

    builder = builder.mount("/chat", routes![api::chat]);
    // /images/remove_background
    builder = builder.mount("/images", routes![api::remove_background]);
    // /audio/speech
    builder = builder.mount("/audio", routes![api::speech]);
    // Health check and model info endpoints
    builder = builder.mount("/", routes![api::health, api::models]);
    // Shutdown endpoint
    builder = builder.manage(shutdown_flag);
    builder = builder.mount("/", routes![api::shutdown]);

    let _rocket = builder.launch().await?;

    // Cleanup PID file when server exits
    cleanup_pid_file(port)?;
    shutdown_handler.abort();

    Ok(())
}
