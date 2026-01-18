# pdf-mcp-server
The MCP server to extract content (summary) of PDF files using LLMs.

The following command runs the server:

       cargo run [path to PDF file] [connection mode]
   
 where connection mode is one of 2 options: stdio or http

 Recommended way to test the MCP server is to use IBM MCP Context Gateway https://github.com/IBM/mcp-context-forge.git. Testing using it is pretty easy and straighforward.

 Once run, MCP Context Gateway can be connected to the runnig PDF MCP server via https protocol:

 
 
 
   ![context_forge](https://github.com/user-attachments/assets/e3237754-177a-4f32-8d9c-9b760df90709)

  In the picture above, the MCP server is running at https://localhost:8000 and contains 2 MCP tools: 
  
  1) to get plain text of a PDF page

  2) to  get summary of PDF pages privided by the user using OpenAI LLM.

  If tool #2 is selected, further intreactions with the server using IBM MCP Context Gateway contain only 2 steps: 

  - provide list of pages of the PDF document to be summarized. This list can be comma-separated (i.e. 25,26,27) or can have 'begin-end' format (i.e. 25-27).
  - privide a prompt to the LLM to extract summary of the pages privided at the previous step. For example: "summary of PDF document".

  A JSON containing summary of the pages requested will be returned by the server.

  ![context_forge_result](https://github.com/user-attachments/assets/4783c7f5-f215-4f45-8a92-c62dbaaba677)


  

  

  
