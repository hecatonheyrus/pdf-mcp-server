mod embeddings;
mod error;
mod server;
mod loader;
mod prompt_logger;
mod tokenizer;



use crate::{
    error::ServerError,
    server::PdfMcpServer,
};

use rmcp::transport::streamable_http_server::{
    StreamableHttpService, session::local::LocalSessionManager,
};

use clap::Parser;

use rmcp::{
    transport::io::stdio,
    ServiceExt,
};

#[cfg(not(target_os = "windows"))]
use xdg::BaseDirectories;

use pyo3::prelude::*;

use crate::tokenizer::Paragraph;

// --- CLI Argument Parsing ---

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg()]
    mode: String, // http/stdio

    /// Log user prompts to ./log/prompts.log (Windows only)
    #[arg(long, env = "PDF_MCP_LOG_PROMPTS")]
    log_prompts: bool,

}

const BIND_ADDRESS: &str = "127.0.0.1:8000";


#[tokio::main]
async fn main() -> Result<(), ServerError> {


    let cli = Cli::parse();
    let run_mode = cli.mode.trim().to_string();


    let service = PdfMcpServer::new(
        "PDF MCP Server initializing...".to_string(),

    )?;

    Python::initialize();

    if let Some(path) = prompt_logger::init(cli.log_prompts) {
        eprintln!("User prompts are logged to: {:?}", path);
    }


    if run_mode == "stdio".to_string(){

    eprintln!("PDF MCP server starting via stdio...");


    let server_handle = service.serve(stdio()).await.map_err(|e| {
        eprintln!("Failed to start server: {:?}", e);
        ServerError::McpRuntime(e.to_string())
    })?;

    eprintln!("MCP Server started: parameters stdio, IP address:port N/A");

    server_handle.waiting().await.map_err(|e| {
        eprintln!("Server encountered an error while running: {:?}", e);
        ServerError::McpRuntime(e.to_string())
    })?;

    eprintln!("Rust Docs MCP server stopped.");

    } else if run_mode == "https".to_string(){


    let http_service = StreamableHttpService::new(
        move || Ok(service.clone()),
        LocalSessionManager::default().into(),
        Default::default(),
    );

    let router = axum::Router::new().nest_service("/mcp", http_service);
    let tcp_listener = tokio::net::TcpListener::bind(BIND_ADDRESS).await?;

    let local_addr = tcp_listener.local_addr()?;

    eprintln!("MCP Server started: parameters http, {}", local_addr);

    let _ = axum::serve(tcp_listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await;

    eprintln!("PDF MCP server stopped.");

  } else{

    ServerError::Config("Invalid mode argument".to_string());

  }

  Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    }
