"""
===================================================================================
03_crewai — Same Multi-Agent System, Declarative Approach
===================================================================================

Implements the same 3-agent financial research system (news, price, risk)
using CrewAI's declarative API.

Instead of building graphs with nodes and edges (LangGraph), you describe:
    WHO  → Agent(role, goal, backstory)
    WHAT → Task(description, expected_output, agent)
    HOW  → Crew(agents, tasks, process)

Two execution modes:
    [1] Sequential    — news → price → risk, fixed order, always all 3
    [2] Hierarchical  — manager LLM delegates dynamically (closest to LangGraph supervisor)

Key difference from LangGraph multi_agent.py:
    - ~150 lines vs ~300 lines for the same system
    - No TypedDict State, no graph builder, no add_node/add_edge
    - Task.context=[news_task, price_task] replaces agent_outputs in shared state
    - Supervisor prompt is auto-generated (black box) vs written yourself

Key limitation vs LangGraph:
    - No Human-in-the-Loop (no interrupt/resume)
    - All 3 tasks always run — no general-knowledge shortcut
    - Less control over execution flow
===================================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

from shared.tools import search_web as _search_web_fn
from shared.config import GROQ_MODEL, GROQ_API_KEY

load_dotenv()


# =================================================================================
# LLM — CrewAI uses its own LLM wrapper (backed by LiteLLM)
# Format: "provider/model-name"
# =================================================================================
llm = LLM(
    model=f"groq/{GROQ_MODEL}",
    api_key=GROQ_API_KEY,
    temperature=0.3
)


# =================================================================================
# Tool — wrap shared search function with CrewAI's @tool decorator
# =================================================================================
# Brave_search fix: tool name + docstring + agent goal all say "web_search"
# so Llama 3.1 8B never drifts to its "brave_search" training prior.
@tool("web_search")
def web_search(query: str) -> str:
    """
    web_search: Search the web for real-time financial data.
    Use this tool to find current stock news, prices, PE ratios, market cap,
    earnings reports, and other information not in your training data.
    IMPORTANT: This is the ONLY search tool available. Always call web_search,
    never brave_search or any other tool name.
    """
    return _search_web_fn(query)


# =================================================================================
# AGENTS — plain-English declarative definitions
# =================================================================================
news_agent = Agent(
    role="Financial News Analyst",
    goal=(
        "Use the web_search tool to find the latest financial news, announcements, "
        "and events about the stock in question. Provide a structured report with "
        "key headlines, overall sentiment, and important events. "
        "Always call web_search — it is your only search tool."
    ),
    backstory=(
        "You are a specialist in financial journalism with 10 years of experience "
        "tracking market-moving news. You structure your findings clearly so other "
        "analysts on your team can build on your work. "
        "You only use the web_search tool — no other search tools exist."
    ),
    tools=[web_search],
    llm=llm,
    verbose=False,
)

price_agent = Agent(
    role="Stock Price & Metrics Specialist",
    goal=(
        "Use the web_search tool to find current stock price, P/E ratio, market cap, "
        "52-week range, and recent percentage change. Provide a structured metrics "
        "report for other analysts. "
        "Always call web_search — it is your only search tool."
    ),
    backstory=(
        "You are a quantitative analyst specializing in stock valuation metrics. "
        "You are precise with numbers and always note when data may be delayed. "
        "You only use the web_search tool — no other search tools exist."
    ),
    tools=[web_search],
    llm=llm,
    verbose=False,
)

risk_agent = Agent(
    role="Investment Risk Assessor",
    goal=(
        "Synthesize the news and price reports from teammates into a balanced risk "
        "assessment. Do NOT search the web — you have no search tools."
    ),
    backstory=(
        "You are a senior investment risk analyst who combines qualitative news signals "
        "with quantitative price data. You rely entirely on your team's findings. "
        "You always include a past-performance disclaimer."
    ),
    # No tools — reads conversation context only
    # Equivalent to risk_agent_node in LangGraph reading state["agent_outputs"]
    tools=[],
    llm=llm,
    verbose=False,
)


# =================================================================================
# TASKS — assign work, wire context for agent-to-agent sharing
# =================================================================================
# {question} placeholder is filled at crew.kickoff(inputs={"question": "..."})

news_task = Task(
    description=(
        "Research the latest financial news about: {question}\n\n"
        "Search for recent news, announcements, earnings, leadership changes, launches.\n\n"
        "Format:\nNEWS REPORT:\n"
        "- Key headlines: (2-3 bullets)\n"
        "- Overall sentiment: (Positive / Negative / Neutral)\n"
        "- Important events: (earnings, launches, scandals, partnerships)"
    ),
    expected_output="Structured NEWS REPORT — 150 words or less.",
    agent=news_agent,
)

price_task = Task(
    description=(
        "Research current stock price and metrics for: {question}\n\n"
        "Find current price, P/E ratio, market cap, 52-week range, recent % change.\n\n"
        "Format:\nPRICE REPORT:\n"
        "- Current Price:\n- P/E Ratio:\n- Market Cap:\n"
        "- 52-week High/Low:\n- Recent % Change:\n- Valuation: (Overvalued/Fair/Undervalued)"
    ),
    expected_output="Structured PRICE REPORT — 120 words or less.",
    agent=price_agent,
)

risk_task = Task(
    description=(
        "Using the news and price reports from your teammates, assess investment risk "
        "for: {question}\n\nDo NOT search the web. Use ONLY the provided context.\n\n"
        "Format:\nRISK ASSESSMENT:\n"
        "- Risk Level: (Low / Medium / High)\n"
        "- Key risk factors: (2-3 bullets)\n"
        "- Key positive factors: (2-3 bullets)\n"
        "- Recommendation:\n"
        "- Disclaimer: past performance does not guarantee future results"
    ),
    expected_output="Structured RISK ASSESSMENT — 200 words or less.",
    agent=risk_agent,
    # context feeds news_task + price_task outputs to risk_agent automatically
    # No shared state dict needed — CrewAI handles the wiring
    context=[news_task, price_task],
)


# =================================================================================
# CREWS — two modes
# =================================================================================
sequential_crew = Crew(
    agents=[news_agent, price_agent, risk_agent],
    tasks=[news_task, price_task, risk_task],
    process=Process.sequential,
    verbose=True,
)

hierarchical_crew = Crew(
    agents=[news_agent, price_agent, risk_agent],
    tasks=[news_task, price_task, risk_task],
    process=Process.hierarchical,
    manager_llm=llm,
    verbose=True,
)


# =================================================================================
# Run
# =================================================================================
def run_sequential(question: str):
    print("\n" + "=" * 60)
    print("  CREWAI — SEQUENTIAL  (news → price → risk, always all 3)")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")
    result = sequential_crew.kickoff(inputs={"question": question})
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result.raw)


def run_hierarchical(question: str):
    print("\n" + "=" * 60)
    print("  CREWAI — HIERARCHICAL  (manager delegates dynamically)")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")
    result = hierarchical_crew.kickoff(inputs={"question": question})
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result.raw)


def main():
    print("\n" + "=" * 60)
    print("  CREWAI — MULTI-AGENT FINANCIAL RESEARCH")
    print("=" * 60)
    print("\n[1] Sequential   — news → price → risk (fixed, always all 3)")
    print("[2] Hierarchical — manager decides order dynamically\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break
        if not question:
            continue
        mode = input("Mode? [1=sequential / 2=hierarchical, default=1]: ").strip()
        try:
            if mode == "2":
                run_hierarchical(question)
            else:
                run_sequential(question)
        except Exception as e:
            print(f"\nError: {e}\n")
        print()


if __name__ == "__main__":
    main()
