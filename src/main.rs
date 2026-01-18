mod embeddings;
mod error;
mod server;
mod loader;
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
    pdf_file_path: String,
    #[arg()]
    mode: String, // http/stdio
   
}

const BIND_ADDRESS: &str = "127.0.0.1:8000";


#[tokio::main]
async fn main() -> Result<(), ServerError> {


    let cli = Cli::parse();
    let path = cli.pdf_file_path.trim().to_string(); // Trim whitespace

    let run_mode = cli.mode.trim().to_string();


    let service = PdfMcpServer::new(
        path.clone(),   
        "PDF MCP Server initializing...".to_string(),

    )?; 

    Python::initialize();

  
    if run_mode == "stdio".to_string(){

    eprintln!("PDF MCP server starting via stdio...");


    let server_handle = service.serve(stdio()).await.map_err(|e| {
        eprintln!("Failed to start server: {:?}", e);
        ServerError::McpRuntime(e.to_string()) 
    })?;

    eprintln!("PDF MCP server running...");

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

    eprintln!("PDF MCP server starting via HTTP...");

    let _ = axum::serve(tcp_listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await;
        
    eprintln!("PDF MCP server running...");

  } else{

    ServerError::Config("Invalid mode argument".to_string());

  }

  Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    }
