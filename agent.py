import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from tools import web_search, read_pdf
from logger import AgentLogger

load_dotenv()

def build_agent():
    provider = os.getenv('WEB_SEARCH_PROVIDER', 'duckduckgo').strip().lower()
    if provider == 'parallel':
        from parallel_search import parallel_web_search
        search_tool = parallel_web_search
    elif provider == 'duckduckgo':
        search_tool = web_search
    else:
        raise ValueError('WEB_SEARCH_PROVIDER must be duckduckgo or parallel')

    llm = ChatGroq(
        model='llama-3.1-8b-instant',
        temperature=0,
        max_tokens=1024,
        callbacks=[AgentLogger()]
    )

    tools = [search_tool, read_pdf]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        system_msg = SystemMessage(
            content=(
                "You are a helpful and professional assistant.\n"
                "Always provide responses in a polished, well-structured, and detailed format.\n"
                "Use clear sections, concise headings, and readable formatting where appropriate.\n"
                "Every final answer MUST be at least 100 words, even for simple questions.\n"
                "If the user asks for a short answer, still provide at least 100 words while staying relevant.\n"
                "Use web_search ONLY when necessary.\n"
                "When using a tool, call it EXACTLY once with a simple query.\n"
                "Do not repeat tool calls.\n"
                "After receiving tool results, give a final answer.\n"
                "Do NOT output anything except valid tool calls when calling tools."
            )
        )
        messages = [system_msg] + state['messages']
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}

    builder = StateGraph(MessagesState)

    builder.add_node('agent', agent_node)
    builder.add_node('tools', ToolNode(tools))

    builder.add_edge(START, 'agent')
    builder.add_conditional_edges(
        'agent',
        tools_condition,
        {
            'tools': 'tools',
            '__end__': END
        }
    )
    builder.add_edge('tools', 'agent')

    return builder.compile()
