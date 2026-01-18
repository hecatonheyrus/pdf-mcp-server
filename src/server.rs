
use crate::{
    loader::{pages_argument_to_vector, generate_embeddings_path, Load_embeddings_from_cache, store_embeddings_in_cache},
    embeddings::{cosine_similarity, generate_embeddings, CachedDocumentEmbedding},
    error::ServerError, // Keep ServerError for ::new()
};

use crate::Paragraph;

use rmcp::{
    ErrorData as McpError,
    ServerHandler, // Import necessary rmcp items
    model::*,
    tool, tool_handler, tool_router,
    handler::server::{router::tool::ToolRouter},
    handler::server::wrapper::Parameters,

    model::{
       Implementation,
       ProtocolVersion,
        ServerCapabilities,
        ServerInfo,
    },   
   
};

use async_openai::{
	Client as OpenAIClient, config::OpenAIConfig,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs,
    },

};

use ndarray::Array1;
use std::{env, sync::Arc}; 
use tokio::sync::Mutex;

use std::thread;

use std::sync::OnceLock;

use pyo3::prelude::*;

use std::ffi::CStr;
use std::ffi::CString;

use std::path::{Path,};


#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SumRequest {
    	#[schemars(description = "The first number to add")]
    	pub a: i32,
    	#[schemars(description = "The second number to add")]
    	pub b: i32,
}


#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct PdfPageTexRequest {
    #[schemars(description = "Page number to extract")]
    	pub page_number: u32,

}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct PdfContentRequest {
    #[schemars(description = "Text query to extract content")]
    pub query: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct PdfPagesContentRequest {
    #[schemars(description = "Text query to extract content of particular pages")]
    pub query: String,
    pub page_list: String,
}		

#[derive(Clone)]
pub struct PdfMcpServer{
	tool_router: ToolRouter<Self>,
    pdf_file_path: Arc<String>, // Use Arc for cheap cloning
    embeddings: Vec<(u32, Array1<f32>)>,
    startup_message: Arc<Mutex<Option<String>>>, // Keep the message itself
    startup_message_sent: Arc<Mutex<bool>>,     // Flag to track if sent (using tokio::sync::Mutex)
                                       
   
}

#[tool_router]
impl PdfMcpServer {

    pub fn new(
        pdf_file_path: String,
        startup_message: String,

    ) -> Result<Self, ServerError> {
        Ok(Self {
        	tool_router: Self::tool_router(),
            pdf_file_path: Arc::new(pdf_file_path),
            embeddings: Vec::<(u32, Array1<f32>)>::new(),
            startup_message: Arc::new(Mutex::new(Some(startup_message))), // Initialize message
            startup_message_sent: Arc::new(Mutex::new(false)), // Initialize flag to false
    
        })
    }


    fn split_by_paragaphs<'a>(&self, pdf_text: &'a str) -> Paragraph<'a>{
  
       let paragraphs = Paragraph::new(&pdf_text);  
       paragraphs
    }


   fn print_text_by_paragraphs(&self, paragraphs: Paragraph) {

   	 for (i, p) in paragraphs.enumerate() {
        println!("Paragraph #{i}: {p}");
       }
   } 

 

pub fn py_run_pdfminer_all_pages(&self, pdf_file: &str) -> PyResult<String>{

	let pages_num = self.py_run_pdfminer_get_pages_number(pdf_file);

	let pages_num_vector : Vec<u32> = (0..=pages_num?).step_by(1).collect();

	let mut pdf_text: String = String::new();

     Python::attach(|py| {
        let miner = PyModule::import(py, "pdfminer.high_level").unwrap();
        pdf_text = miner
            .getattr("extract_text").unwrap()
            .call1(((pdf_file),
            	"".to_string(),             //password - according to args order in high_level.extract_text()
            	(pages_num_vector),)).unwrap()   //pages vector 
            .extract().unwrap();
  
    });

   Ok(pdf_text)

}


pub fn py_run_pdfminer_get_pages_number(&self, pdf_file: &str) -> PyResult<u32> {

   let mut pages_number: u32 = 0;	

   Python::attach(|py| {
      
   	  let miner = PyModule::import(py, "pdfminer.high_level").unwrap();

   	  let pdf_generator = miner
   	    .getattr("extract_pages").unwrap()
   	    .call1(((pdf_file),)).unwrap();

      loop{
   	    match pdf_generator.getattr("__next__").unwrap().call0(){

   	    Ok(_)=> {	      
   	      pages_number += 1;
   	     }        
        Err(_) => break,   
   	   }
   	
   	} 	  

    });	

   Ok(pages_number)

}



   #[tool(description = "Calculate sum of two numbers")]
    fn sum_of_two(&self, Parameters(SumRequest {a, b}): Parameters<SumRequest>) -> String{
    	(a + b).to_string()
    }


    #[tool(description = "Get full text of PDF file")]
    fn get_pdf_text_of_page(&self, Parameters(PdfPageTexRequest{page_number}): Parameters<PdfPageTexRequest>)  -> String {

        let page_text: String = self.get_page_text_py(page_number).unwrap();

        page_text
     
    }

    #[tool(description = "Get summary of PDF document pages by page numbers")]
    async fn get_summary_of_pdf_document_pages(&self, 
    	  Parameters(PdfPagesContentRequest{query, page_list}): Parameters<PdfPagesContentRequest>)  -> Result<String, McpError>{ 

      let CLIENT: OnceLock<OpenAIClient<OpenAIConfig>> = OnceLock::new();
    	  // --- Initialize OpenAI Client (needed for question embedding even if cache hit) ---
      let openai_client = if let Ok(api_base) = env::var("OPENAI_API_BASE") {
        let config = OpenAIConfig::new().with_api_base(api_base);
        OpenAIClient::with_config(config)
       } else {
           OpenAIClient::new()
       };

         CLIENT
          .set(openai_client.clone()) // Clone the client for the OnceCell
          .expect("Failed to set OpenAI client"); 

        let embedding_model: String = env::var("EMBEDDING_MODEL")
          .unwrap_or_else(|_| "text-embedding-3-small".to_string());   

    	 let mut embeddings: Vec<(u32, Array1<f32>)> = Vec::<(u32, Array1<f32>)>::new();

         let mut pages_texts: Vec<String> = Vec::<String>::new();  

         let mut pages_numbers: Vec<u32> = Vec::<u32>::new(); 

         let mut pages: String = "".to_string();

         if page_list == "*"{

         	 let total_number_of_pages = self.py_run_pdfminer_get_pages_number(&self.pdf_file_path.to_string());
             for n in 0..total_number_of_pages.unwrap(){
             	 pages_numbers.push(n);
             }
            pages = "all".to_string();   
         } else {

         pages = page_list.trim().to_string();
         pages_numbers = pages_argument_to_vector(pages.clone()).unwrap();

         } 

         let full_embeddings_path = generate_embeddings_path(self.pdf_file_path.to_string(), pages).unwrap();

         eprintln!("Cache file path: {:?}", full_embeddings_path);
 

         if full_embeddings_path.exists(){
           thread::scope(|s| {
              s.spawn(|| {
    	    eprintln!("Attempting to load cached data from: {:?}", full_embeddings_path);
    	    let cached_data = Load_embeddings_from_cache(full_embeddings_path).unwrap();
     
             for item in cached_data {
     
               embeddings.push((item.page_num, Array1::from(item.vector.clone())));                      
               eprintln!("Page : {}-------------------------------------------", item.page_num);
               eprintln!("{}", item.content);
            }                            
                      
          });
         });
       } else {
       
         eprintln!("Cache file not found. Will generate.");

         for p_num in &pages_numbers{

           pages_texts.push(self.get_page_text_py(*p_num).unwrap());   //use Py
        }

    eprintln!("Generating embeddings...");
    
    match generate_embeddings(&openai_client, &pages_texts, &embedding_model).await {
    	Ok((generated_embeddings_vector, total_tokens_all)) => {
    		let cost_per_million = 0.02;
            let estimated_cost = (total_tokens_all as f64 / 1_000_000.0) * cost_per_million;
            eprintln!(
                "Embedding generation cost for {} tokens: ${:.6}",
                total_tokens_all, estimated_cost
            );

            eprintln!("Saving generated documents and embeddings to: {:?}", full_embeddings_path);

            let mut combined_cache_data: Vec<CachedDocumentEmbedding> = Vec::new();

            for i in 0..pages_texts.len(){  
     
             combined_cache_data.push(
    	       CachedDocumentEmbedding {
               path: full_embeddings_path.display().to_string(),
               content: pages_texts[i].clone(),
               vector: generated_embeddings_vector[i].clone().to_vec(),
               page_num: pages_numbers[i],
              }
            );

             embeddings.push((pages_numbers[i], Array1::from(generated_embeddings_vector[i].clone()))); 

           };

           thread::spawn(move || {
   	          store_embeddings_in_cache(combined_cache_data, full_embeddings_path);
          }).join().unwrap();

    	}
    	Err(_) => {
           return Err(
           	ErrorData{
           		code: ErrorCode(-42),
           		message: std::borrow::Cow::Borrowed("Some error in generate_embeddings()"),
           		data: None,
           	});
    	}
      }
      


       }

       let mut sent_guard = self.startup_message_sent.lock().await;
        if !*sent_guard {
            let mut msg_guard = self.startup_message.lock().await;
            if let Some(_) = msg_guard.take() {
                // Take the message out
                *sent_guard = true; // Mark as sent
            }
            // Drop guards explicitly to avoid holding locks longer than needed
            drop(msg_guard);
            drop(sent_guard);
        } else {
            // Drop guard if already sent
            drop(sent_guard);
        }


         let question_embedding_request = CreateEmbeddingRequestArgs::default()
            .model(embedding_model)
            .input(query.to_string())
            .build()
            .map_err(|e| {
                McpError::internal_error(format!("Failed to build embedding request: {}", e), None)
            })?;    


         let question_embedding_response = openai_client
            .embeddings()
            .create(question_embedding_request)
            .await
            .map_err(|e| McpError::internal_error(format!("OpenAI API error: {}", e), None))?;

        let question_embedding = question_embedding_response.data.first().ok_or_else(|| {
            McpError::internal_error("Failed to get embedding for question", None)
        })?;

        let question_vector = Array1::from(question_embedding.embedding.clone());

        // --- Find Best Matching Document ---
        let mut best_match: Option<f32> = None;

        let mut best_page_num: u32 = 0;

        for embeddings_array in embeddings.iter(){
           
            let score = cosine_similarity(question_vector.view(), embeddings_array.1.view());       
            if best_match.is_none() || score > best_match.unwrap(){
            	best_match = Some(score);
                best_page_num = embeddings_array.0;

            }

        }

         // --- Generate Response using LLM ---
        let response_text = match best_match {
            Some(_score) => {
                eprintln!("Best match found, page number {}", best_page_num);

                let doc = self.get_page_text_py(best_page_num).unwrap(); 

                let system_prompt = format!(
                        "You are an expert technical assistant for the ????? '{}'. \
                         Answer the user's question based *only* on the provided context. \
                         If the context does not contain the answer, say so. \
                         Do not make up information. Be clear, concise, and comprehensive providing example usage code when possible.",
                        self.pdf_file_path 
                    );
                    let user_prompt = format!(
                        "Context:\n---\n{}\n---\n\nQuestion: {}",
                        doc, query
                    );

                    let llm_model: String = env::var("LLM_MODEL")
                        .unwrap_or_else(|_| "gpt-4o-mini-2024-07-18".to_string());
                    let chat_request = CreateChatCompletionRequestArgs::default()
                        .model(llm_model)
                        .messages(vec![
                            ChatCompletionRequestSystemMessageArgs::default()
                                .content(system_prompt)
                                .build()
                                .map_err(|e| {
                                    McpError::internal_error(
                                        format!("Failed to build system message: {}", e),
                                        None,
                                    )
                                })?
                                .into(),
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(user_prompt)
                                .build()
                                .map_err(|e| {
                                    McpError::internal_error(
                                        format!("Failed to build user message: {}", e),
                                        None,
                                    )
                                })?
                                .into(),
                        ])
                        .build()
                        .map_err(|e| {
                            McpError::internal_error(
                                format!("Failed to build chat request: {}", e),
                                None,
                            )
                        })?;

                    let chat_response = openai_client.chat().create(chat_request).await.map_err(|e| {
                        McpError::internal_error(format!("OpenAI chat API error: {}", e), None)
                    })?;

                    chat_response
                        .choices
                        .first()
                        .and_then(|choice| choice.message.content.clone())
                        .unwrap_or_else(|| "Error: No response from LLM.".to_string())
               
            }
            None => "Could not find any relevant document context.".to_string(),
        }; 


         Ok(response_text)        

    } 	



pub fn set_embeddings(&mut self, embeddings_array: (u32, Array1<f32>)) {
   
    self.embeddings.push(embeddings_array.clone());
	
}  


pub fn get_page_text_py(&self, page_number: u32) -> PyResult<String>{

   let mut pdf_text: String = String::new();

   let pdf_path = Path::new(&*self.pdf_file_path);

     Python::attach(|py| {
        let miner = PyModule::import(py, "pdfminer.high_level").unwrap();
        pdf_text = miner
            .getattr("extract_text").unwrap()
            .call1(((pdf_path),
            	"".to_string(),             //password - according to args order in high_level.extract_text()
            	(vec![page_number]),)).unwrap()   //pages vector 
            .extract().unwrap();

    
    });


     Ok(pdf_text)

    }  


  }


#[tool_handler]
impl ServerHandler for PdfMcpServer {
    fn get_info(&self) -> ServerInfo {
        // Define capabilities using the builder
        let capabilities = ServerCapabilities::builder()
            .enable_tools() // Enable tools capability
            .build();

        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05, // Use latest known version
            capabilities,
            server_info: Implementation {
                name: "pdf-mcp-server".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            // Provide instructions based on the specific crate
            instructions: Some(format!(
                "This server provides tools to query documentation for PDF file '{}' ",
                self.pdf_file_path
            )),
        }
    }  
    

 }     


