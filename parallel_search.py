"""Optional anonymous Parallel Search MCP tool for the research agent."""

import asyncio

import httpx
from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "LangGraph-Agent/1.0 (+https://github.com/Pratyush1110/LangGraph-Agent)"
TIMEOUT_SECONDS = 30


async def _call_search(query: str) -> str:
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS
    ) as client:
        async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "web_search", {"objective": query, "search_queries": [query]}
                )
                text = "\n".join(
                    block.text for block in result.content if isinstance(block, TextContent)
                )
                if result.isError:
                    return f"Search failed: {text or 'Parallel returned a tool error.'}"
                return text or "No results found."


def _error_text(error: Exception) -> str:
    # The MCP transport can wrap useful HTTP errors in task-group exceptions.
    nested = getattr(error, "exceptions", None)
    if nested:
        return "; ".join(_error_text(item) for item in nested)
    return str(error)


async def _asearch(query: str) -> str:
    try:
        return await asyncio.wait_for(_call_search(query), timeout=TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return f"Search failed: Parallel search timed out after {TIMEOUT_SECONDS} seconds."
    except Exception as error:
        return f"Search failed: {_error_text(error)}"


def _search(query: str) -> str:
    return asyncio.run(_asearch(query))


parallel_web_search = StructuredTool.from_function(
    func=_search,
    coroutine=_asearch,
    name="web_search",
    description="Search the web using Parallel for current events and facts. Returns source URLs and excerpts.",
)
