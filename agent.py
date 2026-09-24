import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import AsyncAnthropic

class MCPAgent:
    def __init__(self):
        self.llm = AsyncAnthropic()
        self.exit_stack = AsyncExitStack()
        self.session = None

    async def connect_to_server(self, server_path: str):
        """Spins up an MCP server process and connects via standard I/O."""
        server_params = StdioServerParameters(
            command="python", 
            args=[server_path]
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

async def get_mcp_tools(self):
        response = await self.session.list_tools()
        return [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]

async def run_loop(self, user_query: str):
        messages = [{"role": "user", "content": user_query}]
        mcp_tools = await self.get_mcp_tools()
        
        while True:
            response = await self.llm.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=1000,
                messages=messages,
                tools=mcp_tools
            )
            
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return next(c.text for c in response.content if c.type == "text")
            
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Agent requested tool: {block.name}")
                    result = await self.session.call_tool(block.name, arguments=block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.content
                    })
            messages.append({"role": "user", "content": tool_results})

async def main():
    agent = MCPAgent()
    
    try:
        await agent.connect_to_server("path/to/your/mcp_server.py")
        final_answer = await agent.run_loop("Analyze the data using your tools.")
        print("Final Output:", final_answer)
        
    finally:
        await agent.exit_stack.aclose()

if __name__ == "__main__":
    asyncio.run(main())