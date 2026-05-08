"""
===================================================================================
04_autogen — Same Multi-Agent System, Conversational Approach
===================================================================================

Implements the same 3-agent financial research system (news, price, risk)
using AutoGen's conversational agent model.

Core idea: agents TALK TO EACH OTHER until the problem is solved.
No graph edges. No task lists. The conversation IS the shared memory.

Key insight:
    LangGraph: risk_agent reads state["agent_outputs"]    ← explicit state wiring
    CrewAI:    risk_task reads context=[news_task, ...]   ← explicit context wiring
    AutoGen:   risk_agent reads messages above it in chat ← automatic, no wiring needed

Two execution modes:
    [1] RoundRobin  — agents take turns in fixed order (news → price → risk)
    [2] Selector    — selector LLM reads conversation and picks who speaks next

Important: AutoGen 0.4+ is ASYNC — all team.run() calls need asyncio.run().
This is the only framework in this project that requires async.
===================================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

from shared.tools import search_web as _search_web_fn
from shared.config import GROQ_MODEL, GROQ_API_KEY, GROQ_BASE_URL

load_dotenv()


# =================================================================================
# Model client — Groq via OpenAI-compatible endpoint
# =================================================================================
model_client = OpenAIChatCompletionClient(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    base_url=GROQ_BASE_URL,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
        "structured_output": False,
    }
)


# =================================================================================
# Tool — FunctionTool wraps shared search function
# =================================================================================
# Brave_search fix: description starts with "web_search:" and explicitly warns
# against "brave_search". Agents also repeat the name in system_message.
web_search_tool = FunctionTool(
    _search_web_fn,
    description=(
        "web_search: Search the web for real-time financial data including "
        "stock news, prices, PE ratios, market cap, and earnings. "
        "Always use web_search. Never call brave_search."
    )
)


# =================================================================================
# AGENTS — AssistantAgent with system_message (persona)
# =================================================================================
# Each agent receives the FULL conversation history before speaking.
# risk_agent reads news + price reports simply by reading the messages above it.
# No state dict, no context=[] — the chat IS the shared memory.

news_agent = AssistantAgent(
    name="news_agent",
    model_client=model_client,
    tools=[web_search_tool],
    system_message=(
        "You are a financial news analyst. Your job: search for the latest news "
        "about the stock being discussed. "
        "Always use the web_search tool — never use brave_search. "
        "Format:\nNEWS REPORT:\n"
        "- Key headlines: (2-3 bullets)\n"
        "- Overall sentiment: (Positive / Negative / Neutral)\n"
        "- Important events: (earnings, launches, leadership changes)\n"
        "After delivering your report, stop — let the next agent speak."
    ),
)

price_agent = AssistantAgent(
    name="price_agent",
    model_client=model_client,
    tools=[web_search_tool],
    system_message=(
        "You are a stock price and metrics specialist. Your job: search for current "
        "price, P/E ratio, market cap, 52-week range, and recent % change. "
        "Always use the web_search tool — never use brave_search. "
        "Format:\nPRICE REPORT:\n"
        "- Current Price:\n- P/E Ratio:\n- Market Cap:\n"
        "- 52-week High/Low:\n- Recent % Change:\n"
        "After delivering your report, stop — let the next agent speak."
    ),
)

risk_agent = AssistantAgent(
    name="risk_agent",
    model_client=model_client,
    tools=[],   # No tools — reads conversation history only
    system_message=(
        "You are an investment risk assessor. You do NOT search the web. "
        "Read the NEWS REPORT and PRICE REPORT already in the conversation "
        "and synthesize a risk assessment. "
        # The full conversation history is visible to every agent in AutoGen.
        # risk_agent simply reads what news_agent and price_agent wrote above it.
        # No explicit context wiring needed — this is AutoGen's key advantage.
        "Format:\nRISK ASSESSMENT:\n"
        "- Risk Level: (Low / Medium / High)\n"
        "- Key risk factors: (2-3 bullets)\n"
        "- Key positive factors: (2-3 bullets)\n"
        "- Recommendation:\n"
        "- Disclaimer: past performance does not guarantee future results\n\n"
        "End your message with TERMINATE on its own line to signal completion."
    ),
)


# =================================================================================
# TEAMS — two modes
# =================================================================================
termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(20)

# RoundRobin: fixed turn order — news → price → risk → [TERMINATE] → stop
roundrobin_team = RoundRobinGroupChat(
    participants=[news_agent, price_agent, risk_agent],
    termination_condition=termination,
)

# Selector: LLM reads conversation and picks who speaks next
# Equivalent to LangGraph supervisor + conditional edges
SELECTOR_PROMPT = """You are a research team coordinator.

Available team members:
{roles}

Conversation so far:
{history}

Rules:
- No NEWS REPORT yet → select news_agent
- No PRICE REPORT yet → select price_agent
- Both reports present → select risk_agent
- Once RISK ASSESSMENT delivered → done

Reply with ONLY the agent name: news_agent, price_agent, or risk_agent"""

selector_team = SelectorGroupChat(
    participants=[news_agent, price_agent, risk_agent],
    model_client=model_client,
    selector_prompt=SELECTOR_PROMPT,
    termination_condition=termination,
    allow_repeated_speaker=False,
)


# =================================================================================
# Run — async because AutoGen 0.4+ is fully async
# =================================================================================
def _print_messages(messages):
    """Print conversation messages, collapsing tool call noise."""
    for msg in messages:
        content = str(msg.content)
        if "[FunctionCall" in content or "[FunctionExecutionResult" in content:
            print(f"  [{msg.source}] [called web_search...]")
        else:
            print(f"\n[{msg.source.upper()}]")
            print(content.replace("TERMINATE", "").strip())


async def run_roundrobin(question: str):
    print("\n" + "=" * 60)
    print("  AUTOGEN — ROUND ROBIN  (news → price → risk, fixed order)")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")
    result = await roundrobin_team.run(task=question)
    _print_messages(result.messages)
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    for msg in reversed(result.messages):
        content = str(msg.content).replace("TERMINATE", "").strip()
        if content and "[Function" not in content:
            print(content)
            break


async def run_selector(question: str):
    print("\n" + "=" * 60)
    print("  AUTOGEN — SELECTOR  (selector LLM picks who speaks next)")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")
    result = await selector_team.run(task=question)
    _print_messages(result.messages)
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    for msg in reversed(result.messages):
        content = str(msg.content).replace("TERMINATE", "").strip()
        if content and "[Function" not in content:
            print(content)
            break


def main():
    print("\n" + "=" * 60)
    print("  AUTOGEN — MULTI-AGENT FINANCIAL RESEARCH")
    print("=" * 60)
    print("\nAgents CONVERSE until done. Conversation = shared memory.")
    print("No task lists. No graph edges. No explicit context wiring.\n")
    print("[1] RoundRobin — news → price → risk (fixed turns)")
    print("[2] Selector   — selector LLM routes dynamically\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break
        if not question:
            continue
        mode = input("Mode? [1=roundrobin / 2=selector, default=1]: ").strip()
        try:
            if mode == "2":
                asyncio.run(run_selector(question))
            else:
                asyncio.run(run_roundrobin(question))
        except Exception as e:
            print(f"\nError: {e}\n")
        print()


if __name__ == "__main__":
    main()
