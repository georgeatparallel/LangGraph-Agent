"""Exercise provider selection and the graph through the real MCP client."""

import asyncio
import json
import os
import unittest
from unittest.mock import patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent
import parallel_search


RealAsyncClient = httpx.AsyncClient
SEARCH_TEXT = "Title: LangGraph releases\nURL: https://github.com/langchain-ai/langgraph/releases\nSummary: Current release notes."


class MCPServer:
    def __init__(self, content=None, is_error=False, status=200, wait=False):
        self.content = content if content is not None else [{"type": "text", "text": SEARCH_TEXT}]
        self.is_error = is_error
        self.status = status
        self.wait = wait
        self.requests = []
        self.clients = []
        self.called = asyncio.Event()
        self.cancelled = False

    def client(self, **kwargs):
        client = RealAsyncClient(transport=httpx.MockTransport(self.handle), **kwargs)
        self.clients.append(client)
        return client

    async def handle(self, request):
        message = json.loads(request.content)
        self.requests.append((request, message))
        method = message["method"]
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1.0"},
            }
        elif method == "notifications/initialized":
            return httpx.Response(202)
        elif method == "tools/list":
            result = {"tools": [{
                "name": "web_search", "description": "Search current information",
                "inputSchema": {
                    "type": "object", "properties": {
                        "objective": {"type": "string"},
                        "search_queries": {"type": "array", "items": {"type": "string"}},
                    }, "required": ["objective", "search_queries"],
                },
            }]}
        elif method == "tools/call":
            self.called.set()
            if self.wait:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            if self.status != 200:
                return httpx.Response(self.status, text="Service unavailable")
            result = {"content": self.content, "isError": self.is_error}
        else:
            raise AssertionError(f"Unexpected MCP method: {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def assert_requests(self, test):
        methods = [m["method"] for _, m in self.requests]
        test.assertEqual(methods[:3], ["initialize", "notifications/initialized", "tools/call"])
        test.assertTrue(all(m == "tools/list" for m in methods[3:]))
        for request, _ in self.requests:
            test.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
            test.assertEqual(request.headers["User-Agent"], parallel_search.USER_AGENT)
            test.assertNotIn("Authorization", request.headers)
        call = next(m["params"] for _, m in self.requests if m["method"] == "tools/call")
        test.assertEqual(call["name"], "web_search")
        test.assertEqual(call["arguments"], {
            "objective": "LangGraph recent release notes",
            "search_queries": ["LangGraph recent release notes"],
        })
        test.assertTrue(all(c.is_closed for c in self.clients))


class ResearchModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        if isinstance(messages[-1], ToolMessage):
            self.result = messages[-1].content
            return AIMessage(content=f"Research result:\n{self.result}")
        return AIMessage(content="", tool_calls=[{
            "id": "search-1", "name": "web_search",
            "args": {"query": "LangGraph recent release notes"},
        }])


class ProviderTests(unittest.TestCase):
    def test_default_and_explicit_duckduckgo_preserve_tools(self):
        for setting in (None, "duckduckgo"):
            model = ResearchModel()
            with patch.dict(os.environ, {}, clear=True), patch("agent.ChatGroq", return_value=model):
                if setting is not None:
                    os.environ["WEB_SEARCH_PROVIDER"] = setting
                agent.build_agent()
            self.assertEqual(model.tools, [agent.web_search, agent.read_pdf])

    def test_invalid_provider_fails_before_creating_model(self):
        with patch.dict(os.environ, {"WEB_SEARCH_PROVIDER": "unknown"}, clear=True), \
                patch("agent.ChatGroq") as model:
            with self.assertRaisesRegex(ValueError, "duckduckgo or parallel"):
                agent.build_agent()
            model.assert_not_called()

    def test_selected_provider_runs_through_graph_and_returns_sources(self):
        server = MCPServer()
        model = ResearchModel()
        with patch.dict(os.environ, {"WEB_SEARCH_PROVIDER": "parallel"}, clear=True), \
                patch("agent.ChatGroq", return_value=model), \
                patch("parallel_search.httpx.AsyncClient", side_effect=server.client):
            graph = agent.build_agent()
            result = graph.invoke({"messages": [HumanMessage(content="Find release notes")]})
        self.assertEqual(model.tools, [parallel_search.parallel_web_search, agent.read_pdf])
        self.assertEqual(model.result, SEARCH_TEXT)
        self.assertIn(SEARCH_TEXT, result["messages"][-1].content)
        server.assert_requests(self)

    def test_tool_errors_and_empty_results(self):
        for content, error, expected in (
            ([], False, "No results found."),
            ([{"type": "text", "text": "Free-tier quota reached; retry later."}], True,
             "Search failed: Free-tier quota reached; retry later."),
        ):
            server = MCPServer(content=content, is_error=error)
            with patch("parallel_search.httpx.AsyncClient", side_effect=server.client):
                result = parallel_search.parallel_web_search.invoke({"query": "LangGraph recent release notes"})
            self.assertEqual(result, expected)
            server.assert_requests(self)

    def test_http_errors_return_failure_to_graph(self):
        server = MCPServer(status=503)
        model = ResearchModel()
        with patch.dict(os.environ, {"WEB_SEARCH_PROVIDER": "parallel"}, clear=True), \
                patch("agent.ChatGroq", return_value=model), \
                patch("parallel_search.httpx.AsyncClient", side_effect=server.client):
            result = agent.build_agent().invoke({"messages": [HumanMessage(content="Find release notes")]})
        self.assertIn("Search failed:", model.result)
        self.assertIn("503", result["messages"][-1].content)
        server.assert_requests(self)


class AsyncProviderTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_graph_consumes_search_result(self):
        server = MCPServer()
        model = ResearchModel()
        with patch.dict(os.environ, {"WEB_SEARCH_PROVIDER": "parallel"}, clear=True), \
                patch("agent.ChatGroq", return_value=model), \
                patch("parallel_search.httpx.AsyncClient", side_effect=server.client):
            result = await agent.build_agent().ainvoke({"messages": [HumanMessage(content="Find release notes")]})
        self.assertIn(SEARCH_TEXT, result["messages"][-1].content)
        server.assert_requests(self)

    async def test_timeout_stops_request_and_closes_client(self):
        server = MCPServer(wait=True)
        with patch("parallel_search.httpx.AsyncClient", side_effect=server.client), \
                patch("parallel_search.TIMEOUT_SECONDS", 0.2):
            result = await parallel_search.parallel_web_search.ainvoke({"query": "LangGraph recent release notes"})
        self.assertIn("Search failed: Parallel search timed out", result)
        self.assertTrue(server.cancelled)
        server.assert_requests(self)

    async def test_cancellation_propagates_and_closes_client(self):
        server = MCPServer(wait=True)
        with patch("parallel_search.httpx.AsyncClient", side_effect=server.client):
            task = asyncio.create_task(parallel_search.parallel_web_search.ainvoke(
                {"query": "LangGraph recent release notes"}
            ))
            await asyncio.wait_for(server.called.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(server.cancelled)
        server.assert_requests(self)


if __name__ == "__main__":
    unittest.main()
