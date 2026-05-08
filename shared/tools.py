"""
shared/tools.py — Single source of truth for the web search tool.

All four framework implementations (LangGraph, CrewAI, AutoGen, LangChain) use
DuckDuckGo for search. Instead of copy-pasting the setup and the brave_search
fix in every file, we define the raw search function here once and import it.

Each framework wraps it in its own tool protocol:
    LangGraph → DuckDuckGoSearchResults(name="web_search")  via bind_tools()
    CrewAI    → @tool("web_search") decorator around search_web()
    AutoGen   → FunctionTool(search_web, description=...)

WHY THE NAME "web_search" MATTERS:
    Llama 3.1 8B has a strong training prior to call search tools "brave_search".
    Naming the tool "web_search" (generic) prevents the model from defaulting
    to that hallucinated name and causing a Groq tool_use_failed error.
    This fix is applied at every layer: tool name, docstring, and agent prompts.
"""

from langchain_community.tools import DuckDuckGoSearchResults

# ---------------------------------------------------------------------------
# Raw DuckDuckGo instance — used directly by LangGraph via bind_tools()
# ---------------------------------------------------------------------------
ddg_search = DuckDuckGoSearchResults(name="web_search", num_results=2)


# ---------------------------------------------------------------------------
# Plain Python function — wrapped by CrewAI (@tool) and AutoGen (FunctionTool)
# ---------------------------------------------------------------------------
def search_web(query: str) -> str:
    """
    web_search: Search the web for real-time financial data.
    Use this to find current stock news, prices, PE ratios, market cap,
    earnings reports, and other information not in training data.
    IMPORTANT: This tool is named web_search. Always call web_search.
    Never use brave_search or any other tool name.
    """
    return ddg_search.invoke(query)
