# pdf-mcp-server
The MCP server to extract content (summary) of PDF files using LLMs.

The following command runs the server:

       cargo run [path to pdffile] [connection mode]
   
 where connection mode is one of 2 options: stdio or http

 The recommended way to test the MCP server is to use IBM MCP Context Gateway https://github.com/IBM/mcp-context-forge.git

 Once run, MCP Context Gateway can be connected to the runnig PDF MCP server via https protocol:

 
 
 
   
