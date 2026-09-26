"""Expose public web search as an MCP tool over the stdio transport.

The LangGraph workflow launches this server as a subprocess through
``src/tools/mcp_client.py``.
"""
from typing import List, Dict
from fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchResults

mcp = FastMCP("web-search-server")

_search = DuckDuckGoSearchResults(output_format="list")


@mcp.tool()
def web_search(query: str) -> List[Dict[str, str]]:
    """Search the web and return structured results with title, url, and snippet
    for each hit. Use specific, targeted queries rather than full sentences."""
    try:
        results = _search.invoke(query)
        if not results:
            return [{"title": "No results", "url": "", "snippet": "No specific search results found."}]
        return [
            {
                "title": r.get("title", "Untitled"),
                "url": r.get("link", r.get("url", "")),
                "snippet": r.get("snippet", r.get("body", "")),
            }
            for r in results
        ]
    except Exception as e:
        return [{"title": "Search error", "url": "", "snippet": f"Search error: {str(e)}"}]


if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport
