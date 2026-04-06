use crate::server::api::set_server_port;
use crate::server::process::{cleanup_pid_file, create_pid_file};
use rocket::data::{ByteUnit, Limits};
use rocket::{Config, routes};
use std::net::IpAddr;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

// ASR (Automatic Speech Recognition) API module
pub(crate) mod api;
pub(crate) mod asr;
pub(crate) mod embedding;
pub(crate) mod process;
pub(crate) mod reranker;

pub(crate) async fn start_http_server(
    address: String,
    port: u16,
    allow_remote_shutdown: bool,
) -> anyhow::Result<()> {
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

    builder = builder.mount("/v1/chat", routes![api::chat]);
    builder = builder.mount("/chat", routes![api::chat]);
    // /images/remove_background
    builder = builder.mount("/images", routes![api::remove_background]);
    // /audio/speech and /audio/transcriptions (ASR transcription endpoint)
    builder = builder.mount("/audio", routes![api::speech, asr::transcriptions]);
    // /v1/audio/transcriptions (OpenAI standard ASR transcription endpoint)
    builder = builder.mount("/v1/audio", routes![asr::transcriptions]);
    // /embeddings and /v1/embeddings (OpenAI-compatible embeddings endpoint)
    builder = builder.mount("/", routes![embedding::embeddings]);
    builder = builder.mount("/v1", routes![embedding::embeddings]);

    // /rerank and /v1/rerank (OpenAI-compatible embeddings endpoint)
    builder = builder.mount("/", routes![reranker::rerank]);
    builder = builder.mount("/v1", routes![reranker::rerank]);

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
