use clap::Parser;

use crate::cli::{
    args::{Cli, CliArgs, Commands, CommonArgs},
    run_cli, run_delete, run_download, run_list, run_ps, run_run, run_serv,
};

mod cli;
mod server;

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
        Some(Commands::List(args)) => run_list(args),
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
                gguf_path: cli.gguf_path,
                mmproj_path: cli.mmproj_path,
            };
            run_cli(args).await
        }
    }
}
