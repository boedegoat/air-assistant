from typing import Optional, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

import os

class MCPClient:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        
        self.messages = []

    async def connect_to_stdio_server(self, server_name: str, config: Dict[str, Any]):
        """
        Connect to an MCP server using STDIO.
        """
        command = config.get("command")
        args = config.get("args", [])

        if not command:
            print(f"Skipping {server_name}: 'command' not found in config.")
            return

        server_params = StdioServerParameters(command=command, args=args)
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read_stream, write_stream = stdio_transport

        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        
        response = await session.list_tools()
        tools = response.tools
        
        self.sessions[server_name] = session
        print(f"\nConnected to STDIO server '{server_name}' with tools:", [tool.name for tool in tools])

    async def connect_to_servers(self, stdio_configs: Optional[Dict[str, Any]] = None, connect_sse: bool = True):
        """
        Connect to MCP servers.
        """
        if stdio_configs:
            for server_name, config in stdio_configs.items():
                await self.connect_to_stdio_server(server_name, config)

        if connect_sse:
            sse_url = os.environ.get("OMNIPARSER_SSE_URL")
            try:
                sse_transport = await self.exit_stack.enter_async_context(sse_client(url=sse_url))
                read_stream, write_stream = sse_transport

                session = await self.exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                
                response = await session.list_tools()
                tools = response.tools
                
                self.sessions["sse_default"] = session
                print("\nConnected to SSE server with tools:", [tool.name for tool in tools])
            except Exception as e:
                print(f"Failed to connect to SSE server at {sse_url}: {e}")
