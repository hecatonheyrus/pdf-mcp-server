# pdf-mcp-server
The MCP server to work with PDF files using LLMs, namely, OpenAI. It is written in Rust and includes interrfaces to Python libraries(using PyO3).

The following command runs the server:

       cargo run [path to PDF file] [connection mode]
   
 where connection mode is one of 2 : stdio or https. The command above will also load required PDF document. Once done, the document is ready to work with using MCP tools shipped with the MCP server. 

 Recommended tool to test the MCP server is IBM MCP Context Gateway https://github.com/IBM/mcp-context-forge.git. Testing is pretty easy and straighforward.

 Once run, MCP Context Gateway can connect to the runnig PDF MCP server via https:

 
 
 
   ![context_forge](https://github.com/user-attachments/assets/e3237754-177a-4f32-8d9c-9b760df90709)

  In the picture above, the MCP server is running at https://localhost:8000 and contains 2 MCP tools: 
  
  1) to get plain text of a PDF page - mostly for testing purposes.

  2) to  get summary of PDF pages provided by the user using OpenAI LLM.

  Once tool #2 is selected, further interactions with the server using IBM MCP Context Gateway is as easy as filling up 2 input text fields:

  -  list of pages of the PDF document to be summarized. The list can be comma-separated (i.e. 25,26,27) or 'begin-end' format (i.e. 25-27).
  -  prompt to the LLM to extract summary of the pages privided at the previous step. For example: "summary of PDF document".

  JSON containing summary of the pages requested will be returned by the server.

  ![context_forge_result](https://github.com/user-attachments/assets/4783c7f5-f215-4f45-8a92-c62dbaaba677)


  

  

  
