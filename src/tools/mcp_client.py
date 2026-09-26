"""Connect to the web-search MCP server using the stdio transport.

The client and discovered tools are cached so the server connection can be
reused across research queries within the process.
"""
import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

_SERVER_PATH = str(Path(__file__).resolve().parent.parent.parent / "mcp_server" / "search_server.py")

_client = MultiServerMCPClient(
    {
        "web_search": {
            "transport": "stdio",
            "command": "python",
            "args": [_SERVER_PATH],
        }
    }
)

_tools_cache = None
_tools_lock = asyncio.Lock()


async def get_mcp_search_tool():
    """Returns the web_search tool loaded live from the MCP server over stdio.
    Cached after first call so we don't re-handshake with the subprocess
    on every single researcher_node invocation."""
    global _tools_cache
    async with _tools_lock:
        if _tools_cache is None:
            tools = await _client.get_tools()
            _tools_cache = {t.name: t for t in tools}
    return _tools_cache["web_search"]
