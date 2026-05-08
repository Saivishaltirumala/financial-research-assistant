"""
===================================================================================
FILE 5: MULTI-AGENT SYSTEM IN LANGGRAPH
===================================================================================

WHY THIS IS DIFFERENT FROM 4_subgraphs.py:
---------------------------------------------------------------------------
4_subgraphs.py — Modular Pipeline (NOT truly multi-agent):
    - Routing decision is made ONCE upfront by classify_query
    - Subgraphs follow a FIXED sequence of steps (no LLM deciding inside)
    - Subgraphs are ISOLATED — news subgraph never talks to price subgraph
    - Parent just dispatches and waits — no re-evaluation

5_multi_agent.py — True Multi-Agent System:
    - A SUPERVISOR AGENT (with its own LLM) continuously decides what to do next
    - Each SPECIALIST AGENT has its own LLM that thinks independently
    - Agents SEE each other's outputs — risk agent reads what news + price agents found
    - Supervisor RE-EVALUATES after every agent runs — dynamic, not fixed
    - Agents can be called in ANY ORDER based on what the supervisor learns

ANALOGY:
    4_subgraphs.py = A manager hands a fixed checklist to a department
    5_multi_agent.py = A smart team lead who monitors progress, reads each
                       specialist's report, and decides who to call next

THE SUPERVISOR PATTERN — Core of Multi-Agent Systems:
    Every agent reports BACK to the supervisor after completing its work.
    The supervisor sees the full picture and decides the next move.
    This loop continues until the supervisor says "FINISH".

    START
      │
      ▼
    SUPERVISOR (LLM) ──────────────────────────────────────┐
      │                                                     │
      │  Decides who to call next:                         │
      │                                                     │
      ├─► NEWS AGENT (own LLM + search) ──────────────────►│
      │      "Here's what I found about news..."           │
      │                                                     │
      ├─► PRICE AGENT (own LLM + search) ────────────────►│
      │      "Here's the price data I found..."            │
      │                                                     │
      ├─► RISK AGENT (own LLM, reads others' output) ─────►│
      │      "Based on news + price, here's my risk view..." │
      │                                                     │
      └─► FINISH ──► END (supervisor writes final answer)  │
                                                            │
            ◄──── Supervisor sees ALL outputs, re-evaluates┘

KEY MULTI-AGENT PROPERTIES DEMONSTRATED:
    1. SUPERVISOR RE-EVALUATES: After each agent reports, supervisor decides next step
    2. OWN LLM PER AGENT: Each agent has its own LLM + system prompt + specialization
    3. SHARED MEMORY: All agents write to agent_outputs — they build on each other's work
    4. DYNAMIC ORDERING: Supervisor can call agents in any order, skip some, repeat some
    5. EMERGENT BEHAVIOR: The flow is NOT hardcoded — it emerges from supervisor's decisions
===================================================================================
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END

load_dotenv()


# =================================================================================
# STEP 1: Define Shared State
# =================================================================================
# Unlike 4_subgraphs.py where each subgraph had its OWN isolated state,
# here ALL agents share the SAME state. This is what enables communication:
# - News agent writes its findings → Risk agent reads them
# - Price agent writes its findings → Supervisor reads them and decides next step
# This shared, accumulating state is what makes it a TEAM, not isolated modules.

class MultiAgentState(TypedDict):
    question:      str                            # The user's original question

    # THE COMMUNICATION BUS — agents write here, other agents read from here
    # Annotated with operator.add so each agent's output APPENDS (not overwrites)
    # After news + price agents run: ["News: ...", "Price: ..."]
    # After risk agent runs:         ["News: ...", "Price: ...", "Risk: ..."]
    agent_outputs: Annotated[list, operator.add]

    # Tracks which agents have already run — passed to supervisor so it
    # never calls the same agent twice. Also used as a loop guard.
    called_agents: Annotated[list, operator.add]

    # Supervisor's decision — which agent to call next (or "FINISH")
    # This is re-set by the supervisor after EVERY agent completes
    next_agent:    str

    final_answer:  str                            # Written only when supervisor says FINISH


# =================================================================================
# STEP 2: Initialize LLMs — ONE PER AGENT (key multi-agent property!)
# =================================================================================
# In 4_subgraphs.py, there was ONE shared LLM for all subgraph nodes.
# Here, each agent conceptually has its OWN LLM instance with its own identity.
# In production, you'd use different models per agent (e.g., fast model for
# routing, powerful model for risk analysis). Here we use the same base model
# but each agent has its own system prompt = its own "personality and expertise".

supervisor_llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1)
# Low temperature for supervisor — it should make consistent routing decisions

news_llm       = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)
price_llm      = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)
risk_llm       = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
# Lower temperature for risk — risk analysis should be measured and consistent

search_tool = DuckDuckGoSearchResults(name="web_search", num_results=2)


# =================================================================================
# STEP 3: SUPERVISOR AGENT — The orchestrator with its own LLM
# =================================================================================
# This is the heart of multi-agent systems.
# The supervisor is NOT just a routing function (like in 4_subgraphs.py).
# It's a THINKING AGENT that:
# - Reads the question AND all agent outputs accumulated so far
# - Decides intelligently what to do next
# - Re-evaluates EVERY time an agent completes (dynamic, not one-time)
# - Knows when enough information has been gathered → says FINISH

# =================================================================================
# AGENT DESCRIPTIONS — Single source of truth
# =================================================================================
# WHY THIS APPROACH?
# You might wonder: "can the supervisor just read node docstrings like tools do?"
# Answer: NO — tools work via bind_tools() which has a specific mechanism to read
# @tool docstrings and inject them as schemas into the LLM's context automatically.
# Nodes are just Python functions — there is no bind_nodes() equivalent.
# LangGraph never sends node docstrings to any LLM. The supervisor has ZERO
# awareness of news_agent_node, price_agent_node etc. as Python functions.
#
# SOLUTION: Define descriptions here in one dict, then inject them into the
# supervisor prompt dynamically. This way:
# - Descriptions live in ONE place (not duplicated in docstring + prompt)
# - Adding a new agent = add one entry here, prompt updates automatically
# - Same descriptions could also be shown to users, logged, or used elsewhere
AGENT_DESCRIPTIONS = {
    "news_agent":  "searches for latest news, announcements, and events about a stock",
    "price_agent": "searches for stock price, PE ratio, market cap, and financial metrics",
    "risk_agent":  "assesses investment risk — requires BOTH news AND price data to work properly",
}

# Build the agents section of the prompt dynamically from the dict above
# Output: "- news_agent: searches for latest news...\n- price_agent: ..."
_agents_info = "\n".join([f"- {name}: {desc}" for name, desc in AGENT_DESCRIPTIONS.items()])

# =================================================================================
# SUPERVISOR PROMPT — Built from AGENT_DESCRIPTIONS, no hardcoded order
# =================================================================================
# Previously our prompt was a lookup table disguised as intelligence:
#   "For news questions → call news_agent"
#   "For investment questions → call news_agent, then price_agent, then risk_agent"
# That's NOT dynamic — it's just IF/ELSE rules written in English.
# The supervisor wasn't reasoning, it was pattern-matching.
#
# Now: supervisor reads what's been gathered, thinks about what's STILL MISSING,
# and picks the most useful next agent. Order is decided at runtime through reasoning.
# We only give it one hard constraint (risk needs news+price) and let it reason freely.
SUPERVISOR_PROMPT = f"""You are a supervisor managing a team of financial research agents.

Your available agents:
{_agents_info}

Your job:
1. Read the user's question carefully
2. Read what agents have already reported
3. Think: what information is still MISSING to fully answer the question?
4. Pick the agent that fills the most important gap — or say FINISH if nothing is missing

Rules:
- NEVER call an agent already listed under "Agents already called"
- Only hard rule: never call risk_agent unless both news_agent AND price_agent have reported
- Say FINISH as soon as the question is fully answerable from existing reports

Reply with ONLY one word: news_agent, price_agent, risk_agent, or FINISH"""


def supervisor_node(state: MultiAgentState) -> dict:
    """
    SUPERVISOR AGENT — Re-runs after EVERY agent completes.

    This is the critical difference from 4_subgraphs.py:
    - In subgraphs: routing happened ONCE (classify_query ran once, done)
    - Here: supervisor runs AGAIN after each agent, reads their output,
            and makes a NEW decision about what to do next

    The supervisor sees:
    - The original question
    - Which agents have already been called (so it never repeats)
    - Everything agents have reported so far (agent_outputs)
    - And decides: who should I call next? or is this enough?

    We also guard against infinite loops: if all 3 agents have been called,
    force FINISH regardless of what the supervisor decides.
    """
    called = state.get("called_agents", [])
    reports_so_far = "\n\n".join(state["agent_outputs"]) if state["agent_outputs"] else "No reports yet."

    print(f"\n[SUPERVISOR] Re-evaluating... ({len(called)} agents called so far: {called})")

    # Safety guard: if all possible agents have been called, force FINISH
    all_agents = {"news_agent", "price_agent", "risk_agent"}
    if all_agents.issubset(set(called)):
        print(f"[SUPERVISOR] All agents have reported → forcing FINISH")
        return {"next_agent": "FINISH"}

    response = supervisor_llm.invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=f"""User Question: {state['question']}

Agents already called (DO NOT call these again): {called if called else 'None'}

Agent Reports So Far:
{reports_so_far}

What should happen next? Reply with ONLY one word.""")
    ])

    decision = response.content.strip().lower()

    # Normalize to valid agent names, and skip if already called
    if "news" in decision and "news_agent" not in called:
        next_agent = "news_agent"
    elif "price" in decision and "price_agent" not in called:
        next_agent = "price_agent"
    elif "risk" in decision and "risk_agent" not in called:
        next_agent = "risk_agent"
    else:
        next_agent = "FINISH"

    print(f"[SUPERVISOR] Decision: → {next_agent}")
    return {"next_agent": next_agent}


# =================================================================================
# STEP 4: SPECIALIST AGENTS — Each has its own LLM and expertise
# =================================================================================

def news_agent_node(state: MultiAgentState) -> dict:
    """
    NEWS AGENT — Independent specialist for financial news.

    What makes this a true AGENT (not just a subgraph node):
    - Has its OWN LLM (news_llm) with its own specialized system prompt
    - Makes its own decision on what to search for
    - Writes a structured report to the SHARED state (agent_outputs)
    - Other agents (especially risk_agent) will READ this report

    In 4_subgraphs.py, the news subgraph nodes were just sequential functions.
    Here, news_agent is a THINKING specialist that produces a report for the team.
    """
    print(f"\n[NEWS AGENT] Activated — researching news independently...")
    search_results = search_tool.invoke(f"{state['question']} stock news latest")

    response = news_llm.invoke([
        SystemMessage(content="""You are a financial news specialist.
Analyze the search results and write a concise structured report.
Format:
NEWS REPORT:
- Key headlines: (2-3 bullet points)
- Overall news sentiment: (Positive/Negative/Neutral)
- Important events: (any earnings, launches, scandals, partnerships)"""),
        HumanMessage(content=f"Question: {state['question']}\n\nSearch Results: {search_results}")
    ])

    report = f"[NEWS AGENT REPORT]\n{response.content}"
    print(f"[NEWS AGENT] Report written. Reporting back to supervisor...")

    # Append report to shared state AND mark this agent as called
    # Supervisor reads called_agents to avoid calling news_agent again
    return {"agent_outputs": [report], "called_agents": ["news_agent"]}


def price_agent_node(state: MultiAgentState) -> dict:
    """
    PRICE AGENT — Independent specialist for stock price and metrics.

    Independently searches for price data, writes a structured report.
    Risk agent will READ this report when assessing investment risk.
    """
    print(f"\n[PRICE AGENT] Activated — researching price data independently...")
    search_results = search_tool.invoke(f"{state['question']} stock price PE ratio valuation")

    response = price_llm.invoke([
        SystemMessage(content="""You are a financial metrics specialist.
Analyze the search results and write a concise structured report.
Format:
PRICE REPORT:
- Current Price: (value or N/A)
- P/E Ratio: (value or N/A)
- Market Cap: (value or N/A)
- Recent % Change: (value or N/A)
- Valuation assessment: (Overvalued/Undervalued/Fair)"""),
        HumanMessage(content=f"Question: {state['question']}\n\nSearch Results: {search_results}")
    ])

    report = f"[PRICE AGENT REPORT]\n{response.content}"
    print(f"[PRICE AGENT] Report written. Reporting back to supervisor...")

    return {"agent_outputs": [report], "called_agents": ["price_agent"]}


def risk_agent_node(state: MultiAgentState) -> dict:
    """
    RISK AGENT — Reads other agents' work, assesses investment risk.

    THIS IS THE KEY MULTI-AGENT BEHAVIOR:
    Risk agent does NOT search the web. Instead, it reads what news_agent
    and price_agent already found (from shared agent_outputs) and synthesizes
    a risk assessment.

    This is AGENT COLLABORATION — one agent building on another's work.
    In 4_subgraphs.py, subgraphs were completely isolated from each other.
    Here, risk_agent explicitly reads the team's accumulated knowledge.
    """
    print(f"\n[RISK AGENT] Activated — reading team reports and assessing risk...")

    # Read ALL previous agent reports from shared state
    # This is what makes it a team: risk agent sees what news + price agents found
    team_reports = "\n\n".join(state["agent_outputs"])

    response = risk_llm.invoke([
        SystemMessage(content="""You are a financial risk assessment specialist.
Based on the news and price reports from your team members, assess the investment risk.
Format:
RISK ASSESSMENT:
- Risk Level: (Low/Medium/High)
- Key risk factors: (2-3 bullet points)
- Key positive factors: (2-3 bullet points)
- Overall recommendation: (what type of investor this suits)
- Disclaimer: (always include past performance disclaimer)"""),
        HumanMessage(content=f"""Original Question: {state['question']}

Team Reports:
{team_reports}

Based on the above team reports, assess the investment risk.""")
    ])

    report = f"[RISK AGENT REPORT]\n{response.content}"
    print(f"[RISK AGENT] Report written. Reporting back to supervisor...")

    return {"agent_outputs": [report], "called_agents": ["risk_agent"]}


# =================================================================================
# STEP 5: Supervisor's FINISH — Write the final answer
# =================================================================================

def write_final_answer(state: MultiAgentState) -> dict:
    """
    Called when supervisor decides FINISH.
    Synthesizes all agent reports into one coherent final answer.
    """
    print(f"\n[SUPERVISOR] All agents done. Synthesizing final answer...")
    all_reports = "\n\n".join(state["agent_outputs"])

    response = supervisor_llm.invoke([
        SystemMessage(content="You are a financial supervisor. Synthesize all agent reports "
                              "into a clear, concise final answer for the user. "
                              "Be direct and structured. Keep it under 200 words."),
        HumanMessage(content=f"User Question: {state['question']}\n\nAgent Reports:\n{all_reports}")
    ])
    return {"final_answer": response.content}


# =================================================================================
# STEP 6: Build the Multi-Agent Graph
# =================================================================================
# Notice the LOOP structure — every agent points BACK to supervisor.
# The supervisor is the hub. Agents are the spokes.
# This is the "supervisor pattern" in multi-agent systems.

graph_builder = StateGraph(MultiAgentState)

# Add all agent nodes
graph_builder.add_node("supervisor",   supervisor_node)
graph_builder.add_node("news_agent",   news_agent_node)
graph_builder.add_node("price_agent",  price_agent_node)
graph_builder.add_node("risk_agent",   risk_agent_node)
graph_builder.add_node("final_answer", write_final_answer)

# START → supervisor (supervisor always goes first)
graph_builder.add_edge(START, "supervisor")

# Supervisor → (conditional) → agent OR FINISH
# Supervisor's `next_agent` value determines where to go.
# "FINISH" maps to the final_answer node (not END directly — we need to write the answer)
graph_builder.add_conditional_edges(
    "supervisor",
    lambda state: state["next_agent"],
    {
        "news_agent":  "news_agent",
        "price_agent": "price_agent",
        "risk_agent":  "risk_agent",
        "FINISH":      "final_answer",   # explicit map needed: "FINISH" ≠ any node name
    }
)

# All agents loop BACK to supervisor after completing — the supervisor re-evaluates
# This is the key structural difference from 4_subgraphs.py where flow was one-way
graph_builder.add_edge("news_agent",   "supervisor")
graph_builder.add_edge("price_agent",  "supervisor")
graph_builder.add_edge("risk_agent",   "supervisor")

# Final answer → END
graph_builder.add_edge("final_answer", END)

graph = graph_builder.compile()


# =================================================================================
# STEP 7: Run it
# =================================================================================
def research(question: str):
    print("\n" + "=" * 60)
    print("  MULTI-AGENT FINANCIAL RESEARCH SYSTEM")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print("-" * 60)
    print("Watch the supervisor re-evaluate after EACH agent reports!\n")

    result = graph.invoke({
        "question":      question,
        "agent_outputs": [],
        "called_agents": [],
        "next_agent":    "",
        "final_answer":  ""
    })

    print(f"\n{'=' * 60}")
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["final_answer"])
    print(f"\n[AGENTS CALLED]: {len(result['agent_outputs'])} specialist(s) contributed")


def main():
    print("\n" + "=" * 60)
    print("  LANGGRAPH — MULTI-AGENT SYSTEM DEMO")
    print("  (Supervisor re-evaluates after every agent)")
    print("=" * 60)
    print("\nSupervisor REASONS about what's missing — order decided at runtime:\n")
    print("  → 'What is the latest news on Tesla?'")
    print("     Supervisor thinks: only news needed → news_agent → FINISH\n")
    print("  → 'What is Apple stock price and PE ratio?'")
    print("     Supervisor thinks: only price needed → price_agent → FINISH\n")
    print("  → 'Should I invest in Microsoft stock?'")
    print("     Supervisor thinks: need news+price first, then risk")
    print("     → news_agent → price_agent → risk_agent → FINISH\n")
    print("  → 'Is Tesla overvalued given recent news?'")
    print("     Supervisor decides order itself based on reasoning\n")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            research(question)
        except Exception as e:
            print(f"\nError: {e}\n")
        print()


if __name__ == "__main__":
    main()
