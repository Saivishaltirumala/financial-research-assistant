"""
===================================================================================
FILE 7: SAME MULTI-AGENT SYSTEM — NOW WITH AUTOGEN
===================================================================================

WHAT IS AUTOGEN?
    AutoGen is Microsoft's open-source multi-agent framework. Its core idea is
    CONVERSATIONAL AGENTS — instead of tasks flowing through a pipeline (CrewAI)
    or nodes firing in a graph (LangGraph), agents TALK TO EACH OTHER in a
    back-and-forth conversation until the problem is solved.

    Every agent is a chat participant. They send messages, receive replies, and
    decide what to do next based on what others said. The "work" emerges from
    the conversation, not from a predefined task list or graph structure.

THE THREE MENTAL MODELS — side by side:

    LangGraph  → You DRAW A GRAPH: nodes + edges + state
                 Execution follows the graph you wired up
                 "You control everything"

    CrewAI     → You DESCRIBE A CREW: agents with roles, tasks with descriptions
                 CrewAI runs the pipeline
                 "Tell me WHO and WHAT, I'll figure out HOW"

    AutoGen    → You CREATE A GROUP CHAT: agents with personas + termination rules
                 Agents message each other until done
                 "Put agents in a room and let them talk it out"

===================================================================================
AUTOGEN CORE CONCEPTS:
===================================================================================

┌──────────────────────────────────────────────────────────────────────────────┐
│  AssistantAgent                                                              │
│  ──────────────                                                              │
│  An LLM-powered agent that can reason, write text, call tools, and          │
│  send messages. Defined by a name, a system_message (its persona), and      │
│  optionally a list of tools it can call.                                     │
│                                                                              │
│  ↳ LangGraph equivalent: a node function + its system prompt + LLM           │
│  ↳ CrewAI equivalent:    an Agent(role=..., goal=..., backstory=...)         │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Team (the "group chat room")                                                │
│  ────────────────────────────                                                │
│  A Team holds multiple agents and orchestrates who speaks next.              │
│  AutoGen 0.4+ has several Team types:                                        │
│                                                                              │
│  RoundRobinGroupChat  → agents take turns IN ORDER (agent1, agent2, agent3,  │
│                         agent1, agent2, ...) regardless of content           │
│                         Simplest team. Good when you always need each agent. │
│                                                                              │
│  SelectorGroupChat    → a SELECTOR LLM reads all agent descriptions and      │
│                         decides WHO should speak next after each message.    │
│                         Dynamic — closest to LangGraph's supervisor pattern. │
│                         Like a smart meeting moderator who calls on the      │
│                         right person based on what just happened.            │
│                                                                              │
│  ↳ LangGraph equivalent: RoundRobin = fixed sequential edges                 │
│                          Selector   = supervisor with add_conditional_edges  │
│  ↳ CrewAI equivalent:    RoundRobin = Process.sequential                     │
│                          Selector   = Process.hierarchical                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  Termination Conditions                                                      │
│  ──────────────────────                                                      │
│  AutoGen conversations don't stop on their own — you define WHEN to stop.   │
│                                                                              │
│  TextMentionTermination("TERMINATE")  → stop when any agent says "TERMINATE" │
│  MaxMessageTermination(10)            → stop after 10 messages total         │
│  TextMentionTermination | MaxMessage  → stop on EITHER condition (safety net)│
│                                                                              │
│  This is unique to AutoGen's conversational model — LangGraph uses END node, │
│  CrewAI stops when all tasks complete. AutoGen needs explicit stop signals   │
│  because the conversation could loop forever otherwise.                      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  FunctionTool                                                                │
│  ────────────                                                                │
│  Wraps a Python function so AutoGen agents can call it as a tool.            │
│  The function's name + description is what the agent reads to decide         │
│  when and how to call it — same docstring-as-schema concept as               │
│  LangGraph's bind_tools() and CrewAI's @tool decorator.                      │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│  IMPORTANT: AutoGen 0.4+ is ASYNC                                            │
│  ─────────────────────────────────                                           │
│  All team.run() calls are coroutines — you must use asyncio.run() or         │
│  await them inside an async function. This is different from:                │
│  - LangGraph: graph.invoke({...})  ← synchronous by default                 │
│  - CrewAI:    crew.kickoff(...)    ← synchronous                             │
│  AutoGen: await team.run(...)      ← async, must wrap in asyncio.run()       │
└──────────────────────────────────────────────────────────────────────────────┘

===================================================================================
THE CONVERSATION FLOW (what actually happens at runtime):
===================================================================================

RoundRobin mode:
    user (task) → news_agent → price_agent → risk_agent → [TERMINATE detected] → stop

    Each agent receives the FULL conversation history before speaking.
    risk_agent sees everything news_agent and price_agent already said —
    that's how it knows what to synthesize. No explicit context=[task1, task2]
    needed (CrewAI) and no shared agent_outputs list (LangGraph). The
    conversation itself IS the shared memory.

SelectorGroupChat mode:
    user (task) → [selector LLM picks who speaks] → agent speaks
                → [selector LLM re-evaluates] → next agent speaks
                → ... → [TERMINATE detected] → stop

    The selector LLM reads all agent descriptions and the latest message,
    then decides: "given what we just heard, who should speak next?"
    This is dynamic — closer to LangGraph's supervisor re-evaluating after
    each agent.

===================================================================================
ADVANTAGES OF AUTOGEN vs CREWAI and LANGGRAPH:
===================================================================================

✅  NATURAL ITERATION — agents can go back and forth until satisfied.
    News agent can say "I found limited data, searching again..." and do
    multiple tool calls naturally within its turn. No task retries needed.

✅  CONVERSATION IS SHARED MEMORY — every agent sees the full history.
    No need for shared state (LangGraph's agent_outputs), no need for
    context=[] (CrewAI). Information flows automatically through messages.

✅  HUMAN-IN-THE-LOOP IS NATURAL — adding a UserProxyAgent that requires
    human input at each turn is trivial. Pause/resume is built into the
    conversational model. (LangGraph needs interrupt() machinery; CrewAI
    has no equivalent.)

✅  CODE EXECUTION BUILT-IN — UserProxyAgent can execute Python/shell code
    locally. LLM writes code → executor runs it → LLM sees output → fixes
    errors → repeat. This loop is AutoGen's original killer feature.

✅  FLEXIBLE TERMINATION — you compose termination conditions with |  and &.
    TextMentionTermination("DONE") | MaxMessageTermination(20)

===================================================================================
DISADVANTAGES OF AUTOGEN vs CREWAI and LANGGRAPH:
===================================================================================

❌  ASYNC-ONLY API — everything requires asyncio. Adds boilerplate compared
    to CrewAI's synchronous crew.kickoff() and LangGraph's graph.invoke().

❌  LESS PREDICTABLE OUTPUT STRUCTURE — output is a conversation (list of
    messages). Extracting a clean final answer requires parsing the last
    message. CrewAI's expected_output gives you a structured result; LangGraph's
    state gives you typed fields.

❌  CONVERSATIONS CAN DRIFT — in RoundRobin mode, agents always speak even
    when they have nothing useful to add. In Selector mode the selector LLM
    can make wrong decisions. Hard to guarantee a clean flow.

❌  HARDER TO ENFORCE STRICT TASK ORDER — you can't force "news agent must
    ALWAYS run before risk agent" as cleanly as LangGraph's conditional edges
    or CrewAI's context=[]. You rely on prompting and selector logic.

❌  VERBOSE DEBUGGING — conversation logs are long. Finding which message
    contained the error is harder than LangGraph's node-by-node stream events.

===================================================================================
WHEN TO USE AUTOGEN vs CREWAI vs LANGGRAPH:
===================================================================================

Use AUTOGEN when:
    • Writing + running code iteratively (LLM writes → executor runs → LLM fixes)
    • Multiple agents need to DEBATE or CRITIQUE each other's work
    • Human needs to approve or intervene at conversational turns
    • Workflow is open-ended — you don't know the exact steps upfront
    • Research tasks where agents refine work through back-and-forth dialogue

Use CREWAI when:
    • Prototyping quickly with minimal boilerplate
    • Workflow is a fixed pipeline (news → price → risk, always the same)
    • Team is non-technical — plain-English agent definitions matter
    • Structured output is important (expected_output shapes the response)

Use LANGGRAPH when:
    • Production systems — precise flow control, full observability
    • Human-in-the-Loop with graph freeze/resume (interrupt + Command(resume))
    • Parallel execution (fan-out/fan-in) for performance
    • Subgraph isolation — internal agent state must not leak
    • Custom supervisor logic with full ownership of reasoning prompt

===================================================================================
"""

import asyncio
import os
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchResults

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()


# =================================================================================
# STEP 1: Create the model client — Groq via OpenAI-compatible API
# =================================================================================
# AutoGen 0.4+ uses OpenAIChatCompletionClient to talk to any OpenAI-compatible
# endpoint. Groq exposes one at https://api.groq.com/openai/v1.
#
# model_info is required for non-standard endpoints — it tells AutoGen which
# capabilities the model supports so it formats requests correctly.
#
# Compare with how we set up LLMs in other files:
#   LangGraph:  ChatGroq(model_name="llama-3.1-8b-instant")
#   CrewAI:     LLM(model="groq/llama-3.1-8b-instant", api_key=...)
#   AutoGen:    OpenAIChatCompletionClient(model=..., base_url=groq_url, ...)

model_client = OpenAIChatCompletionClient(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model_info={
        "vision": False,
        "function_calling": True,   # Required: enables tool use
        "json_output": False,
        "family": "unknown",
        "structured_output": False,
    }
)


# =================================================================================
# STEP 2: Define the Tool — FunctionTool wraps a plain Python function
# =================================================================================
# AutoGen uses FunctionTool(fn, description=...) to make a Python function
# callable by agents as a tool.
#
# IMPORTANT — same Llama 3.1 8B brave_search hallucination issue applies here.
# Same three-layer fix used in 6_crewai_approach.py:
# 1. Tool description starts with "web_search:" — anchors the model to the name
# 2. Description explicitly says never use brave_search
# 3. Agent system_message repeats "use web_search" (see STEP 3)
#
# How FunctionTool differs from other frameworks:
#   LangGraph: DuckDuckGoSearchResults(name="web_search") — LangChain tool directly
#   CrewAI:    @tool("web_search") decorator on a wrapper function
#   AutoGen:   FunctionTool(fn, description=...) — plain function, no decorator

_ddg = DuckDuckGoSearchResults(name="web_search", num_results=2)

def _web_search_fn(query: str) -> str:
    """
    web_search: Search the web for real-time financial data.
    Use this tool to find current stock news, prices, PE ratios, and market cap.
    IMPORTANT: Always call web_search. Never use brave_search or any other name.
    """
    return _ddg.invoke(query)

web_search_tool = FunctionTool(
    _web_search_fn,
    description=(
        "web_search: Search the web for real-time financial data including "
        "stock news, prices, PE ratios, market cap, and earnings. "
        "Always use web_search. Never call brave_search."
    )
)


# =================================================================================
# STEP 3: Define AGENTS — AssistantAgent with system_message (persona)
# =================================================================================
# AutoGen agents are defined by:
#   name           → identifier used in conversation logs and selector routing
#   model_client   → the LLM powering this agent
#   tools          → list of FunctionTools this agent can call
#   system_message → the agent's persona, expertise, and behavioral rules
#
# The system_message here plays the same role as:
#   LangGraph: SystemMessage(content="...") inside the node function
#   CrewAI:    role + goal + backstory combined
#
# KEY AUTOGEN PATTERN: each agent's system_message must include instructions
# on WHEN TO STOP SPEAKING (say "TERMINATE" when done, or hand off clearly).
# Without this, agents keep adding to the conversation even when finished.
#
# Also note: in AutoGen there is NO explicit "task assignment" like CrewAI's
# Task(agent=news_agent). Any agent can respond to any message. The system_message
# shapes what the agent focuses on, and the Team type controls who speaks when.

news_agent = AssistantAgent(
    name="news_agent",
    model_client=model_client,
    tools=[web_search_tool],        # This agent CAN search the web
    system_message=(
        "You are a financial news analyst. Your only job is to search for the "
        "latest news about the stock or company being discussed. "
        "Always use the web_search tool — never use brave_search. "
        "Format your findings as:\n"
        "NEWS REPORT:\n"
        "- Key headlines: (2-3 bullet points)\n"
        "- Overall sentiment: (Positive / Negative / Neutral)\n"
        "- Important events: (earnings, launches, leadership changes)\n"
        "After delivering your report, write nothing else — let the next agent speak."
    ),
)

price_agent = AssistantAgent(
    name="price_agent",
    model_client=model_client,
    tools=[web_search_tool],        # This agent CAN search the web
    system_message=(
        "You are a stock price and metrics specialist. Your only job is to search "
        "for current stock price, P/E ratio, market cap, 52-week range, and recent "
        "percentage change. "
        "Always use the web_search tool — never use brave_search. "
        "Format your findings as:\n"
        "PRICE REPORT:\n"
        "- Current Price: (value or N/A)\n"
        "- P/E Ratio: (value or N/A)\n"
        "- Market Cap: (value or N/A)\n"
        "- 52-week High/Low: (value or N/A)\n"
        "- Recent % Change: (value or N/A)\n"
        "After delivering your report, write nothing else — let the next agent speak."
    ),
)

risk_agent = AssistantAgent(
    name="risk_agent",
    model_client=model_client,
    tools=[],                       # NO tools — reads conversation history only
    system_message=(
        "You are an investment risk assessor. You do NOT search the web. "
        "You read the NEWS REPORT and PRICE REPORT already in the conversation "
        "and synthesize a risk assessment. "
        # The conversation history IS the shared memory — risk_agent reads
        # everything news_agent and price_agent wrote above it in the chat.
        # This is AutoGen's equivalent of LangGraph's agent_outputs list and
        # CrewAI's context=[news_task, price_task]. No explicit wiring needed —
        # the full conversation is automatically visible to every agent.
        "Format your assessment as:\n"
        "RISK ASSESSMENT:\n"
        "- Risk Level: (Low / Medium / High)\n"
        "- Key risk factors: (2-3 bullets)\n"
        "- Key positive factors: (2-3 bullets)\n"
        "- Recommendation: (suitable investor profile)\n"
        "- Disclaimer: past performance does not guarantee future results\n"
        "End your message with the word TERMINATE on its own line to signal "
        "that the research is complete."
    ),
)


# =================================================================================
# STEP 4a: RoundRobinGroupChat — agents take turns in fixed order
# =================================================================================
# Every agent speaks once per round, in the order they are listed.
# Round: news_agent → price_agent → risk_agent → [TERMINATE found] → stop
#
# Termination: TextMentionTermination("TERMINATE") stops the chat as soon as
# any agent's message contains the word "TERMINATE". We also add
# MaxMessageTermination(20) as a safety net to prevent infinite loops.
# The | operator means EITHER condition can stop the run.
#
# HOW THIS COMPARES:
#   LangGraph sequential: hardcoded edges news_node → price_node → risk_node → END
#   CrewAI sequential:    tasks=[news_task, price_task, risk_task], Process.sequential
#   AutoGen RoundRobin:   participants=[news_agent, price_agent, risk_agent]
#                         agents take turns automatically — no edges, no task list

termination = TextMentionTermination("TERMINATE") | MaxMessageTermination(20)

roundrobin_team = RoundRobinGroupChat(
    participants=[news_agent, price_agent, risk_agent],
    termination_condition=termination,
)


# =================================================================================
# STEP 4b: SelectorGroupChat — a selector LLM decides who speaks next
# =================================================================================
# After each message, a SELECTOR LLM reads all agent names + descriptions and
# the conversation so far, then picks who should speak next.
#
# This is the AutoGen equivalent of:
#   LangGraph: supervisor_node with add_conditional_edges
#   CrewAI:    Process.hierarchical with manager_llm
#
# KEY DIFFERENCE from LangGraph supervisor:
#   LangGraph: YOU write the supervisor prompt (SUPERVISOR_PROMPT in 5_multi_agent.py)
#              → full control over reasoning logic, general-knowledge shortcut, etc.
#   AutoGen:   you provide selector_prompt with {roles} and {history} placeholders
#              → AutoGen fills in agent names+descriptions and conversation history
#              → selector LLM decides dynamically — less code, less control
#
# The selector_prompt below tells the selector:
# - What agents are available (injected via {roles})
# - What has been said so far (injected via {history})
# - The rule: news and price before risk
# - When to pick risk_agent (only after both reports are in the conversation)

SELECTOR_PROMPT = """You are a research team coordinator managing a financial research conversation.

Available team members and what they do:
{roles}

Conversation so far:
{history}

Rules for selecting who speaks next:
- If no NEWS REPORT is in the conversation yet → select news_agent
- If no PRICE REPORT is in the conversation yet → select price_agent
- If both NEWS REPORT and PRICE REPORT are present → select risk_agent
- Once risk_agent has delivered its RISK ASSESSMENT → the conversation is done

Reply with ONLY the name of the agent who should speak next.
Choose from: news_agent, price_agent, risk_agent"""

selector_team = SelectorGroupChat(
    participants=[news_agent, price_agent, risk_agent],
    model_client=model_client,      # LLM that powers the selector decisions
    selector_prompt=SELECTOR_PROMPT,
    termination_condition=termination,
    allow_repeated_speaker=False,   # Prevent same agent speaking twice in a row
)


# =================================================================================
# STEP 5: Run it — ASYNC because AutoGen 0.4+ is fully async
# =================================================================================
# AutoGen's team.run() is a coroutine. You must wrap it with asyncio.run()
# or await it inside an async function.
#
# result.messages → list of all messages exchanged (the full conversation)
# result.messages[-1] → typically the last agent's message (risk assessment)
#
# Compare to:
#   LangGraph: result = graph.invoke({...})      → synchronous, returns state dict
#   CrewAI:    result = crew.kickoff({...})       → synchronous, returns CrewOutput
#   AutoGen:   result = await team.run(task=...) → async, returns TaskResult

async def run_roundrobin(question: str):
    """Fixed order: news_agent → price_agent → risk_agent (always all 3)."""
    print("\n" + "=" * 60)
    print("  AUTOGEN — ROUND ROBIN TEAM")
    print("  (Fixed order: news → price → risk)")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")
    print("Watch agents take turns in the conversation:\n")

    result = await roundrobin_team.run(task=question)

    # Print each message with its source so the conversation flow is visible
    for msg in result.messages:
        source = msg.source
        content = str(msg.content)
        # Skip internal tool call/result messages for readability
        if "[FunctionCall" in content or "[FunctionExecutionResult" in content:
            print(f"  [{source}] [called web_search tool...]")
        else:
            print(f"\n[{source.upper()}]")
            # Remove TERMINATE from display
            print(content.replace("TERMINATE", "").strip())

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    # Find the last substantive message (risk agent's report)
    for msg in reversed(result.messages):
        content = str(msg.content).replace("TERMINATE", "").strip()
        if content and "[Function" not in content:
            print(content)
            break


async def run_selector(question: str):
    """Selector LLM dynamically decides who speaks next after each message."""
    print("\n" + "=" * 60)
    print("  AUTOGEN — SELECTOR GROUP CHAT")
    print("  (Selector LLM routes to the right agent after each turn)")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")
    print("Watch the selector pick agents dynamically:\n")

    result = await selector_team.run(task=question)

    for msg in result.messages:
        source = msg.source
        content = str(msg.content)
        if "[FunctionCall" in content or "[FunctionExecutionResult" in content:
            print(f"  [{source}] [called web_search tool...]")
        else:
            print(f"\n[{source.upper()}]")
            print(content.replace("TERMINATE", "").strip())

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
    print("\nAutoGen: agents CONVERSE until done — no task lists, no graph edges.")
    print("Shared memory = the conversation itself (every agent reads full history).\n")
    print("Mode options:")
    print("  [1] RoundRobin  — news → price → risk in fixed turn order")
    print("  [2] Selector    — selector LLM dynamically picks who speaks next")
    print()

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
