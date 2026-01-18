use rmcp::ServiceError; 
use thiserror::Error;


#[derive(Debug, Error)]
pub enum ServerError {
    
    #[error("Configuration Error: {0}")]
    Config(String),

    #[error("MCP Service Error: {0}")]
    Mcp(#[from] ServiceError), 
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
   
    #[error("OpenAI Error: {0}")]
    OpenAI(#[from] async_openai::error::OpenAIError),
    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error), 
    #[error("Tiktoken Error: {0}")]
    Tiktoken(String),
    #[error("XDG Directory Error: {0}")]
    Xdg(String),
    #[error("MCP Runtime Error: {0}")]
    McpRuntime(String),
}