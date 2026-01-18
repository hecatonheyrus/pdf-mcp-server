use crate::error::ServerError;
use async_openai::{
    config::OpenAIConfig, error::ApiError as OpenAIAPIErr, types::CreateEmbeddingRequestArgs,
    Client as OpenAIClient,
};
use ndarray::{Array1, ArrayView1};
use std::sync::OnceLock;
use std::sync::Arc;
use tiktoken_rs::cl100k_base;


pub static OPENAI_CLIENT: OnceLock<OpenAIClient<OpenAIConfig>> = OnceLock::new();

use bincode::{Encode, Decode};
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize, Debug, Encode, Decode)]
pub struct CachedDocumentEmbedding {
    pub path: String,
    pub content: String, // Add the extracted document content
    pub vector: Vec<f32>,
    pub page_num: u32,
}


/// Calculates the cosine similarity between two vectors.
pub fn cosine_similarity(v1: ArrayView1<f32>, v2: ArrayView1<f32>) -> f32 {
    let dot_product = v1.dot(&v2);
    let norm_v1 = v1.dot(&v1).sqrt();
    let norm_v2 = v2.dot(&v2).sqrt();

    if norm_v1 == 0.0 || norm_v2 == 0.0 {
        0.0
    } else {
        dot_product / (norm_v1 * norm_v2)
    }
}

pub async fn generate_embeddings(
    client: &OpenAIClient<OpenAIConfig>,
    pages_texts: &Vec<String>,
    model: &str,
    ) -> Result<(Vec<Array1<f32>>, usize), ServerError> {

    let mut total_tokens: usize = 0;
    let mut embeddings_vector: Vec<Array1<f32>> = Vec::<Array1<f32>>::new();

    for page_text in pages_texts{

        let (embedding_array, token_count) = generate_embeddings_for_page(&client, &page_text, &model).await?;
        total_tokens += token_count;
        embeddings_vector.push(embedding_array);
    }

    Ok((embeddings_vector, total_tokens))
}

/// Generates embeddings for a list of documents using the OpenAI API.
pub async fn generate_embeddings_for_page(
    client: &OpenAIClient<OpenAIConfig>,
    document_text: &String,
    model: &str,
) -> Result<(Array1<f32>, usize), ServerError> { // Return tuple: (embeddings, total_tokens)


    // Get the tokenizer for the model and wrap in Arc
    let bpe = Arc::new(cl100k_base().map_err(|e| ServerError::Tiktoken(e.to_string()))?);

    const TOKEN_LIMIT: usize = 8000; // Keep a buffer below the 8192 limit

            // Clone client, model, doc, and Arc<BPE> for the async block
    let client = client.clone();
    let model = model.to_string();


        let document_text = Arc::clone(&Arc::new(document_text));

        let bpe = Arc::clone(&bpe); // Clone the Arc pointer

                // Calculate token count for this document
        let token_count = bpe.encode_with_special_tokens(&document_text).len();

        if token_count > TOKEN_LIMIT {
                    // Return Ok(None) to indicate skipping, with 0 tokens processed for this doc
            return Err(ServerError::McpRuntime("Actual tokens  exceed limit ".to_string())); // Include token count type
        }

                // Prepare input for this single document
        let inputs: Vec<String> = vec![document_text.to_string()];

        let request = CreateEmbeddingRequestArgs::default()
            .model(&model) // Use cloned model string
            .input(inputs)
            .build()?; // Propagates OpenAIError

        let response = client.embeddings().create(request).await?; // Propagates OpenAIError
      

        if response.data.len() != 1 {
            return Err(ServerError::OpenAI(
                async_openai::error::OpenAIError::ApiError(OpenAIAPIErr {
                    message: format!(
                        "Mismatch in response length  got {}.",
                        response.data.len()
                    ),
                    r#type: Some("sdk_error".to_string()),
                    param: None,
                    code: None,
                }),
            ));
        }

                // Process result
        let embedding_data = response.data.first().unwrap(); // Safe unwrap due to check above
        let embedding_array = Array1::from(embedding_data.embedding.clone());
            
        Ok((embedding_array, token_count)) // Include token count
   
}