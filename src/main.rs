use std::time::Duration;

use aha::models::WhichModel;
use clap::Parser;
use dirs::home_dir;
use modelscope::ModelScope;
use rocket::{
    Config,
    data::{ByteUnit, Limits},
    routes,
};
use tokio::time::sleep;

use crate::api::init;
mod api;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value_t = 10100)]
    port: u16,

    #[arg(short, long)]
    model: WhichModel,

    #[arg(long)]
    weight_path: Option<String>,

    #[arg(long)]
    save_dir: Option<String>,

    #[arg(long)]
    download_retries: Option<u32>,
}
async fn download_model(model_id: &str, save_dir: &str, max_retries: u32) -> anyhow::Result<()> {
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

fn get_default_save_dir() -> Option<String> {
    home_dir().map(|mut path| {
        path.push(".aha"); // 在 home 目录下创建 .aha 文件夹
        path.to_string_lossy().to_string()
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let model_id = match args.model {
        WhichModel::MiniCPM4_0_5B => "OpenBMB/MiniCPM4-0.5B",
        WhichModel::Qwen2_5vl3B => "Qwen/Qwen2.5-VL-3B-Instruct",
        WhichModel::Qwen3vl2B => "Qwen/Qwen3-VL-2B-Instruct",
    };
    let model_path = match args.weight_path {
        Some(path) => path,
        None => {
            let save_dir = match args.save_dir {
                Some(dir) => dir,
                None => get_default_save_dir().expect("Failed to get home directory"),
            };
            let max_retries = args.download_retries.unwrap_or(3);
            download_model(model_id, &save_dir, max_retries).await?;
            save_dir + "/" + model_id
        }
    };
    println!("-------------------download path: {}", model_path);
    init(args.model, model_path)?;
    start_http_server(args.port).await?;

    Ok(())
}

pub async fn start_http_server(port: u16) -> anyhow::Result<()> {
    let mut builder = rocket::build().configure(Config {
        port,
        limits: Limits::default()
            .limit("string", ByteUnit::Mebibyte(5))
            .limit("json", ByteUnit::Mebibyte(5))
            .limit("data-form", ByteUnit::Mebibyte(100))
            .limit("file", ByteUnit::Mebibyte(100)),
        ..Config::default()
    });

    builder = builder.mount("/chat", routes![api::chat]);

    builder.launch().await?;
    Ok(())
}

// fn main() {
//     println!("Hello, world!");
// }
