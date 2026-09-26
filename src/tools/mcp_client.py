"""Connect to the web-search MCP server using the stdio transport."""
import sys
import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_PATH = str(Path(__file__).resolve().parent.parent.parent / "mcp_server" / "search_server.py")

_client = MultiServerMCPClient(
    {
        "web_search": {
            "transport": "stdio",
            "command": sys.executable,  # Guarantees the active virtualenv Python binary is used
            "args": [_SERVER_PATH],
        }
    }
)

_tools_cache = None
_tools_lock = asyncio.Lock()

async def get_mcp_search_tool():
    global _tools_cache
    async with _tools_lock:
        if _tools_cache is None:
            tools = await _client.get_tools()
            _tools_cache = {t.name: t for t in tools}
    return _tools_cache["web_search"]