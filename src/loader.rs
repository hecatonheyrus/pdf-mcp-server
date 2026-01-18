
use std::io::{Error, ErrorKind,};
use std::path::PathBuf;

use std::str::FromStr;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::fs::{self, File};
use std::io::BufReader;

use crate::error::ServerError;

use crate::embeddings::CachedDocumentEmbedding;
use bincode::config;



fn hash_features(value: String) -> String{

   let mut hasher = DefaultHasher::new(); // TO DO: use SHA-256
   value.hash(&mut hasher);
   format!("{:x}", hasher.finish())
}


pub fn pages_argument_to_vector(pages: String) -> Result<Vec<u32>, Error> {

   let mut pages_numbers : Vec::<u32> = Vec::<u32>::new();

   if pages.contains(","){ //p1, p2, p3 ....
   	  let pages_str = pages.split(',');
   	  for p in pages_str{
       pages_numbers.push(p.parse().unwrap());      
     }
   }else{   // p1-p2

     let pages_str: Vec<&str> = pages.split('-').collect();

     if pages_str.len() > 2{
       let err_msg = format!("Invalid argumnets length for - delimeiter: should be 2!");
       return Err(Error::new(ErrorKind::InvalidData, err_msg));
     };

     let p_start : u32 = FromStr::from_str(pages_str[0]).unwrap();
     let p_end : u32 = FromStr::from_str(pages_str[1]).unwrap();

     if p_start > p_end {
       let err_msg = format!("First page number must be less than second!");
       return Err(Error::new(ErrorKind::InvalidData, err_msg));
     };

     if p_start == p_end{
     	pages_numbers.push(p_start);
     } else if p_start < p_end{

       for p in p_start..p_end{
       	pages_numbers.push(p);
       }	

     };


   };



   Ok(pages_numbers)

}


pub fn generate_embeddings_path(pdf_file_path: String, pages: String) -> Result<PathBuf, ServerError>{

    let path_hash = hash_features(pdf_file_path.clone());
    let embeddings_relative_path = PathBuf::from(path_hash)
        .join(pages.to_string())
        .join("_vec_")
        .join("embeddings.bin");


   #[cfg(not(target_os = "windows"))]
    let full_embeddings_path = {
        let xdg_dirs = BaseDirectories::with_prefix("pdf-mcp-server")
            .map_err(|e| ServerError::Xdg(format!("Failed to get XDG directories: {}", e)))?;
        xdg_dirs
            .place_data_file(embeddings_relative_path)
            .map_err(ServerError::Io)?
    };  
    
   
   #[cfg(target_os = "windows")]
    let full_embeddings_path = {
        let cache_dir = dirs::cache_dir().ok_or_else(|| {
            ServerError::Config("Could not determine cache directory on Windows".to_string())
        })?;
        let app_cache_dir = cache_dir.join("pdf-mcp-server");
        // Ensure the base app cache directory exists
        fs::create_dir_all(&app_cache_dir).map_err(ServerError::Io)?;
        app_cache_dir.join(&embeddings_relative_path)
    };


   Ok(full_embeddings_path)    

}


pub fn Load_embeddings_from_cache(cache_path: PathBuf) -> Result<Vec<CachedDocumentEmbedding>, ServerError>{

       match File::open(&cache_path) {
            
            Ok(file) => {
              let reader = BufReader::new(file);

              match bincode::decode_from_reader::<Vec<CachedDocumentEmbedding>, _, _>(
                    reader,
                    config::standard(),
                ){


              Ok(cached_data) => {
                  Ok(cached_data)    
               }
              
              Err(e) => {
                  Err(ServerError::Xdg(format!("Failed to decode cache file: {}. Will regenerate.", e)))
               }
               

            }

        }

            Err(e) => {
                Err(ServerError::Xdg(format!("Failed to open embeddings file: {}", e)))
            }


       }
   

    }

    pub fn store_embeddings_in_cache(combined_cache_data: Vec<CachedDocumentEmbedding>, full_embeddings_path: PathBuf) -> (){
        
       match bincode::encode_to_vec(&combined_cache_data, config::standard()) {
                Ok(encoded_bytes) => {
                    if let Some(parent_dir) = full_embeddings_path.parent() {
                        if !parent_dir.exists() {
                            if let Err(e) = fs::create_dir_all(parent_dir) {
                                eprintln!(
                                    "Warning: Failed to create cache directory {}: {}",
                                    parent_dir.display(),
                                    e
                                );
                            }
                        }
                    }
                    if let Err(e) = fs::write(&full_embeddings_path, encoded_bytes) {
                        eprintln!("Warning: Failed to write cache file: {}", e);
                    } else {
                        eprintln!(
                            "Cache saved successfully ({} items).",
                            combined_cache_data.len()
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to encode data for cache: {}", e);
                }
     }


 }
      

    





