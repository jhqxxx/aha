use aha::models::common::model_mapping::WhichModel;
use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "aha")]
#[command(version, about, long_about = None)]
pub(crate) struct Cli {
    /// Service listen address
    #[arg(short, long, default_value = "127.0.0.1")]
    pub address: Option<String>,

    /// Service listen port
    #[arg(short, long)]
    pub port: Option<u16>,

    /// Model type (required for backward compatibility)
    #[arg(short, long)]
    pub model: Option<WhichModel>,

    /// Local model weight path
    #[arg(long)]
    pub weight_path: Option<String>,

    /// Model download save directory
    #[arg(long)]
    pub save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    pub download_retries: Option<u32>,

    /// Local GGUF model weight path (required for loading models with GGUF).
    #[arg(long)]
    pub gguf_path: Option<String>,

    /// Local path for mmproj GGUF model weights (required for loading with multimodel GGUF)
    #[arg(long)]
    pub mmproj_path: Option<String>,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
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
    List(ListArgs),
}

/// Common/shared arguments for server operations
#[derive(Args, Debug)]
pub(crate) struct CommonArgs {
    /// Service listen address
    #[arg(short, long, default_value = "127.0.0.1")]
    pub address: String,

    /// Service listen port
    #[arg(short, long, default_value_t = 10100)]
    pub port: u16,

    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,

    /// Allow remote shutdown requests (default: local only, use with caution)
    #[arg(long)]
    pub allow_remote_shutdown: bool,
}

/// Arguments for the 'cli' subcommand (download + serve)
#[derive(Args, Debug)]
pub(crate) struct CliArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Local model weight path (skip download if provided)
    #[arg(long)]
    pub weight_path: Option<String>,

    /// Model download save directory
    #[arg(long)]
    pub save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    pub download_retries: Option<u32>,

    /// Local GGUF model weight path (required for loading models with GGUF).
    #[arg(long)]
    pub gguf_path: Option<String>,

    /// Local path for mmproj GGUF model weights (required for loading with multimodel GGUF)
    #[arg(long)]
    pub mmproj_path: Option<String>,
}

/// Arguments for the 'serv start' subcommand
#[derive(Args, Debug)]
pub(crate) struct ServArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// Local model weight path (defaults to ~/.aha/{model_id} if not specified)
    #[arg(long)]
    pub weight_path: Option<String>,

    /// Local GGUF model weight path (required for loading models with GGUF).
    #[arg(long)]
    pub gguf_path: Option<String>,

    /// Local path for mmproj GGUF model weights (required for loading with multimodel GGUF)
    #[arg(long)]
    pub mmproj_path: Option<String>,
}

/// Arguments for the 'serv list' subcommand
#[derive(Args, Debug)]
pub(crate) struct ServListArgs {
    /// Compact output format
    #[arg(short, long)]
    pub compact: bool,
}

/// Arguments for the 'download' subcommand (download only)
#[derive(Args, Debug)]
pub(crate) struct DownloadArgs {
    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,

    /// Model download save directory
    #[arg(short, long)]
    pub save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    pub download_retries: Option<u32>,
}

/// Arguments for the 'run' subcommand (direct inference)
#[derive(Args, Debug)]
pub(crate) struct RunArgs {
    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,

    /// Input text or file path
    #[arg(short, long, num_args = 1..=2, value_delimiter = ' ')]
    pub input: Vec<String>,

    /// Output file path (optional)
    #[arg(short, long)]
    pub output: Option<String>,

    /// Local model weight path (defaults to ~/.aha/{model_id} if not specified)
    #[arg(long)]
    pub weight_path: Option<String>,

    /// Local GGUF model weight path (required for loading models with GGUF).
    #[arg(long)]
    pub gguf_path: Option<String>,

    /// Local path for mmproj GGUF model weights (required for loading with multimodel GGUF)
    #[arg(long)]
    pub mmproj_path: Option<String>,
}

/// Arguments for the 'delete' subcommand (delete model from default location)
#[derive(Args, Debug)]
pub(crate) struct DeleteArgs {
    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,
}

/// Arguments for the 'list' subcommand (list all supported models)
#[derive(Args, Debug)]
pub(crate) struct ListArgs {
    /// Output models in JSON format (includes name, model_id, and type fields)
    #[arg(short, long)]
    pub json: bool,
}
