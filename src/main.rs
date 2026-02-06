use std::{net::IpAddr, str::FromStr};

use aha::{
    models::WhichModel,
    utils::{download_model, get_default_save_dir},
};
use clap::{Args, Parser, Subcommand, ValueEnum};
use rocket::{
    Config,
    data::{ByteUnit, Limits},
    routes,
};

use crate::api::init;
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

/// Arguments for the 'serv' subcommand (serve only)
#[derive(Args, Debug)]
struct ServArgs {
    #[command(flatten)]
    common: CommonArgs,

    /// Local model weight path (defaults to ~/.aha/{model_id} if not specified)
    #[arg(long)]
    weight_path: Option<String>,
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

/// Get the default weight path for a given model
/// Returns ~/.aha/{model_id} e.g., ~/.aha/OpenBMB/VoxCPM1.5
fn get_default_weight_path(model: WhichModel) -> String {
    let model_id = get_model_id(model);
    let save_dir = get_default_save_dir().expect("Failed to get home directory");
    format!("{}/{}", save_dir, model_id)
}

/// Get the ModelScope model ID for a given WhichModel variant
fn get_model_id(model: WhichModel) -> &'static str {
    match model {
        WhichModel::MiniCPM4_0_5B => "OpenBMB/MiniCPM4-0.5B",
        WhichModel::Qwen2_5vl3B => "Qwen/Qwen2.5-VL-3B-Instruct",
        WhichModel::Qwen2_5vl7B => "Qwen/Qwen2.5-VL-7B-Instruct",
        WhichModel::Qwen3_0_6B => "Qwen/Qwen3-0.6B",
        WhichModel::Qwen3ASR0_6B => "Qwen/Qwen3-ASR-0.6B",
        WhichModel::Qwen3ASR1_7B => "Qwen/Qwen3-ASR-1.7B",
        WhichModel::Qwen3vl2B => "Qwen/Qwen3-VL-2B-Instruct",
        WhichModel::Qwen3vl4B => "Qwen/Qwen3-VL-4B-Instruct",
        WhichModel::Qwen3vl8B => "Qwen/Qwen3-VL-8B-Instruct",
        WhichModel::Qwen3vl32B => "Qwen/Qwen3-VL-32B-Instruct",
        WhichModel::DeepSeekOCR => "deepseek-ai/DeepSeek-OCR",
        WhichModel::HunyuanOCR => "Tencent-Hunyuan/HunyuanOCR",
        WhichModel::PaddleOCRVL => "PaddlePaddle/PaddleOCR-VL",
        WhichModel::RMBG2_0 => "AI-ModelScope/RMBG-2.0",
        WhichModel::VoxCPM => "OpenBMB/VoxCPM-0.5B",
        WhichModel::VoxCPM1_5 => "OpenBMB/VoxCPM1.5",
        WhichModel::GlmASRNano2512 => "ZhipuAI/GLM-ASR-Nano-2512",
        WhichModel::FunASRNano2512 => "FunAudioLLM/Fun-ASR-Nano-2512",
    }
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
        let id = get_model_id(model);
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
    let model_id = get_model_id(common.model);

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
    start_http_server(common.address, common.port).await?;

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
    start_http_server(common.address, common.port).await?;

    Ok(())
}

/// Run the 'download' subcommand: download model only (no server)
async fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    let DownloadArgs {
        model,
        save_dir,
        download_retries,
    } = args;
    let model_id = get_model_id(model);

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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Cli(args)) => run_cli(args).await,
        Some(Commands::Serv(args)) => run_serv(args).await,
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
                },
                weight_path: cli.weight_path,
                save_dir: cli.save_dir,
                download_retries: cli.download_retries,
            };
            run_cli(args).await
        }
    }
}

pub(crate) async fn start_http_server(address: String, port: u16) -> anyhow::Result<()> {
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
    // /images/speech
    builder = builder.mount("/audio", routes![api::speech]);

    builder.launch().await?;
    Ok(())
}
