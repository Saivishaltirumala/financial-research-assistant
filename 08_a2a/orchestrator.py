"""
===================================================================================
A2A ORCHESTRATOR — LangGraph Supervisor over HTTP Agent Network
===================================================================================

What this file does:
    Runs the SAME supervisor logic as 02_langgraph/multi_agent.py — but instead
    of calling Python functions directly (news_agent_node(), price_agent_node()),
    it sends HTTP POST requests to independent agent servers over the network.

    The internal logic (TypedDict state, supervisor LLM, conditional edges, loop)
    is identical to multi_agent.py. Only the agent nodes change:
        BEFORE (multi_agent.py):   news_agent_node(state) → calls Python function
        AFTER  (this file):        news_agent_node(state) → httpx.post("http://localhost:8001/tasks")

The A2A protocol has TWO completely separate concerns:

    CONCERN 1 — SUPERVISOR LLM decides WHO:
        Reads question + agent reports accumulated so far
        Returns a string: "news_agent", "price_agent", "risk_agent", or "FINISH"
        This is pure reasoning — no knowledge of URLs or ports

    CONCERN 2 — AGENT_REGISTRY resolves WHERE:
        Maps agent name → URL
        Populated at startup by fetching /.well-known/agent.json from each agent server
        Used only when routing the HTTP request

    These two are intentionally separate:
        - Supervisor LLM can't hallucinate a URL because it never sees URLs
        - AGENT_REGISTRY is deterministic lookup, not LLM reasoning

STARTUP SEQUENCE (dynamic discovery):
    1. We know agent server URLs at startup (from KNOWN_AGENT_URLS list)
    2. For each URL, fetch GET /.well-known/agent.json
    3. Extract name + description from the card
    4. Build AGENT_REGISTRY {name: url} and AGENT_DESCRIPTIONS {name: description}
    5. Inject descriptions into SUPERVISOR_PROMPT — supervisor learns who's available
    6. Any agent server that's down at startup is simply excluded (graceful degradation)

This startup discovery is what makes A2A extensible:
    Add a new agent? Just add its URL to KNOWN_AGENT_URLS.
    Orchestrator learns its capabilities automatically. No code changes needed.
===================================================================================
"""

import operator
import uuid
import sys
import os
from typing import Annotated

from typing_extensions import TypedDict
from dotenv import load_dotenv

import httpx                           # HTTP client for A2A agent calls
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import GROQ_MODEL

load_dotenv()


# =================================================================================
# STEP 1: Discover Agents via Agent Cards
# =================================================================================
# In multi_agent.py, AGENT_DESCRIPTIONS was a hardcoded Python dict.
# Here, we discover it dynamically by fetching Agent Cards at startup.
#
# KNOWN_AGENT_URLS is the ONLY thing hardcoded — the orchestrator knows WHERE to
# look, but learns WHAT the agents can do from the agents themselves.
# In a production system, these URLs would come from a service registry or
# environment variables — no changes to orchestrator code when adding agents.

KNOWN_AGENT_URLS = [
    "http://localhost:8001",   # news_agent
    "http://localhost:8002",   # price_agent
    "http://localhost:8003",   # risk_agent
]

# These dicts are populated by discover_agents() at startup
AGENT_REGISTRY     = {}   # {name: url}        — used for routing HTTP requests
AGENT_DESCRIPTIONS = {}   # {name: description} — used for supervisor LLM prompt
# "FINISH" is not a real agent — it's the supervisor's signal to stop the loop
# We add it manually after discovery (it has no server, so no card to fetch)


def discover_agents():
    """
    Fetch Agent Cards from all known agent servers.

    For each server URL:
        GET /.well-known/agent.json → {name, description, url, ...}

    Populates AGENT_REGISTRY and AGENT_DESCRIPTIONS.
    Agents that are not running are silently skipped (graceful degradation).

    This runs ONCE at startup — the orchestrator then has a stable view of
    available agents for the lifetime of the process.
    """
    print("\n[ORCHESTRATOR] Discovering agents via Agent Cards...")

    for url in KNOWN_AGENT_URLS:
        try:
            response = httpx.get(f"{url}/.well-known/agent.json", timeout=5.0)
            card = response.json()

            name = card["name"]
            description = card["description"]

            # Two separate registries for two separate concerns:
            AGENT_REGISTRY[name]     = url          # WHERE to send HTTP request
            AGENT_DESCRIPTIONS[name] = description  # WHAT the supervisor LLM sees

            print(f"  ✓ {name} @ {url}  →  \"{description}\"")

        except Exception as e:
            # Agent server is not running — skip it
            # The supervisor won't know this agent exists, so it can't route to it
            print(f"  ✗ {url} — not reachable ({type(e).__name__})")

    # FINISH is the supervisor's exit signal — not a real agent, no URL needed
    # We add its description so the supervisor knows it's always a valid option
    AGENT_DESCRIPTIONS["FINISH"] = (
        "no agent needed — question is general knowledge the supervisor can answer directly"
    )

    print(f"\n[ORCHESTRATOR] Discovery complete. Available agents: {list(AGENT_REGISTRY.keys())}\n")


# =================================================================================
# STEP 2: Shared State — identical to multi_agent.py
# =================================================================================
# The state shape is exactly the same as multi_agent.py because the supervisor
# pattern hasn't changed — only how agent nodes EXECUTE (HTTP vs Python call).

class OrchestratorState(TypedDict):
    question:      str
    agent_outputs: Annotated[list, operator.add]   # reports accumulate here
    called_agents: Annotated[list, operator.add]   # loop guard — never call twice
    next_agent:    str                             # supervisor's routing decision
    final_answer:  str


# =================================================================================
# STEP 3: Build Supervisor Prompt from discovered descriptions
# =================================================================================
# In multi_agent.py, SUPERVISOR_PROMPT was built from a hardcoded AGENT_DESCRIPTIONS dict.
# Here, it's built from the dynamically discovered AGENT_DESCRIPTIONS after discovery.
#
# We build it lazily (inside a function) so it captures agent descriptions after
# discover_agents() has run. If we built it at module load time, the dicts would be empty.

def build_supervisor_prompt() -> str:
    """Build the supervisor LLM prompt from the currently discovered agents."""
    agents_info = "\n".join([f"- {name}: {desc}" for name, desc in AGENT_DESCRIPTIONS.items()])
    return f"""You are a supervisor managing a team of financial research agents.

Your available agents and when to use them:
{agents_info}

Your job:
1. Read the user's question carefully
2. Read what agents have already reported
3. Think: what information is still MISSING to fully answer the question?
4. Pick the agent that fills the most important gap — or say FINISH

When to say FINISH immediately (without calling ANY agent):
- The question asks for a definition, explanation, or concept (e.g. "What is a stock?",
  "What is the difference between ETF and mutual fund?", "How does PE ratio work?")
- These are general knowledge questions answerable from training data — no live data needed

Rules for stock-specific questions (when you DO call agents):
- NEVER call an agent already listed under "Agents already called"
- Only hard rule: never call risk_agent unless both news_agent AND price_agent have reported
- Say FINISH as soon as the question is fully answerable from existing reports

Reply with ONLY one word: {', '.join(AGENT_DESCRIPTIONS.keys())}"""


# =================================================================================
# STEP 4: LLM instances (one per role, same pattern as multi_agent.py)
# =================================================================================
# Identical to multi_agent.py — the agents themselves each have their own LLM,
# but the orchestrator still needs its own supervisor LLM for routing decisions.

supervisor_llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.1)
# Low temperature → consistent routing decisions, no creative routing


# =================================================================================
# STEP 5: Supervisor Node — identical logic to multi_agent.py
# =================================================================================

def supervisor_node(state: OrchestratorState) -> dict:
    """
    SUPERVISOR AGENT — Decides which agent to call next.

    This function is IDENTICAL in logic to supervisor_node() in multi_agent.py.
    The only difference: the SUPERVISOR_PROMPT is built dynamically from
    discovered agent descriptions rather than from a hardcoded dict.

    The supervisor sees: question + called agents + accumulated reports
    It returns: one agent name string (or "FINISH")
    It has NO KNOWLEDGE of ports, URLs, or HTTP — that's AGENT_REGISTRY's job.
    """
    called = state.get("called_agents", [])
    reports_so_far = "\n\n".join(state["agent_outputs"]) if state["agent_outputs"] else "No reports yet."

    print(f"\n[SUPERVISOR] Re-evaluating... ({len(called)} agents called: {called})")

    # Loop guard — if all registered agents have been called, force FINISH
    if AGENT_REGISTRY and set(AGENT_REGISTRY.keys()).issubset(set(called)):
        print(f"[SUPERVISOR] All agents have reported → forcing FINISH")
        return {"next_agent": "FINISH"}

    prompt = build_supervisor_prompt()
    response = supervisor_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"""User Question: {state['question']}

Agents already called (DO NOT call these again): {called if called else 'None'}

Agent Reports So Far:
{reports_so_far}

What should happen next? Reply with ONLY one word.""")
    ])

    decision = response.content.strip().lower()

    # Normalize LLM output to a valid registered agent name
    next_agent = "FINISH"  # default if no match
    for name in AGENT_REGISTRY:
        if name.lower() in decision and name not in called:
            next_agent = name
            break

    print(f"[SUPERVISOR] Decision: → {next_agent}")
    return {"next_agent": next_agent}


# =================================================================================
# STEP 6: Agent Nodes — HTTP calls replace direct Python function calls
# =================================================================================
# In multi_agent.py:
#     def news_agent_node(state): search + llm → return report
# Here:
#     def news_agent_node(state): httpx.post("http://localhost:8001/tasks") → return report
#
# The SUPERVISOR PATTERN (loop structure, state accumulation) is unchanged.
# Only the implementation of each node changes — Python → HTTP.
#
# The three agent functions below are intentionally parallel in structure.
# They could be collapsed into one generic `call_agent(name, state)` function,
# but being explicit makes the code easier to read for learning purposes.

def _call_agent_server(agent_name: str, question: str, context: list[str]) -> str:
    """
    Helper: POST /tasks to an agent server, return the report string.
    Centralized so news/price/risk nodes all use the same HTTP logic.

    agent_name : key in AGENT_REGISTRY → resolves to URL
    question   : the user's original question
    context    : previous reports (empty for news/price, populated for risk)
    """
    url = AGENT_REGISTRY[agent_name]
    task_id = str(uuid.uuid4())

    payload = {
        "task_id":  task_id,
        "question": question,
        "context":  context,
    }

    print(f"[ORCHESTRATOR] → POST {url}/tasks  (task_id={task_id[:8]}...)")

    # Timeout of 60s — LLM calls inside agents can take 10-30 seconds
    response = httpx.post(f"{url}/tasks", json=payload, timeout=60.0)
    response.raise_for_status()

    data = response.json()

    if data["status"] == "failed":
        # Agent returned a failure response — include the error in the report
        # so the supervisor knows an attempt was made and can decide what to do
        print(f"[ORCHESTRATOR] Agent {agent_name} reported failure")
    else:
        print(f"[ORCHESTRATOR] ← Received report from {agent_name}")

    return data["report"]


def news_agent_node(state: OrchestratorState) -> dict:
    """
    Calls the news agent HTTP server.
    No context needed — news agent searches independently.
    """
    print(f"\n[NEWS AGENT NODE] Dispatching HTTP task to news agent server...")

    report = _call_agent_server(
        agent_name="news_agent",
        question=state["question"],
        context=[],         # news agent doesn't need prior reports
    )

    return {"agent_outputs": [report], "called_agents": ["news_agent"]}


def price_agent_node(state: OrchestratorState) -> dict:
    """
    Calls the price agent HTTP server.
    No context needed — price agent searches independently.
    """
    print(f"\n[PRICE AGENT NODE] Dispatching HTTP task to price agent server...")

    report = _call_agent_server(
        agent_name="price_agent",
        question=state["question"],
        context=[],         # price agent doesn't need prior reports
    )

    return {"agent_outputs": [report], "called_agents": ["price_agent"]}


def risk_agent_node(state: OrchestratorState) -> dict:
    """
    Calls the risk agent HTTP server.

    KEY DIFFERENCE from news/price: we pass all accumulated agent_outputs as context.
    This is A2A's answer to shared state — the orchestrator explicitly forwards
    what the risk agent needs to do its job.

    risk_agent receives context = ["[NEWS AGENT REPORT]...", "[PRICE AGENT REPORT]..."]
    and synthesizes from those reports rather than searching again.
    """
    print(f"\n[RISK AGENT NODE] Dispatching HTTP task to risk agent server (with context)...")

    # Pass ALL prior reports as context — risk agent reads them to synthesize
    context = state.get("agent_outputs", [])
    print(f"[RISK AGENT NODE] Forwarding {len(context)} prior report(s) as context")

    report = _call_agent_server(
        agent_name="risk_agent",
        question=state["question"],
        context=context,    # THIS is what distinguishes risk_agent from news/price
    )

    return {"agent_outputs": [report], "called_agents": ["risk_agent"]}


# =================================================================================
# STEP 7: Final Answer Node — identical to multi_agent.py
# =================================================================================

def write_final_answer(state: OrchestratorState) -> dict:
    """
    Called when supervisor decides FINISH.
    Synthesizes all agent reports into one final answer for the user.
    If no agents were called (general knowledge question), answers directly.
    """
    called = state.get("called_agents", [])

    if not called:
        print(f"\n[SUPERVISOR] General knowledge question — answering directly (no agents needed)...")
        response = supervisor_llm.invoke([
            SystemMessage(content="You are a knowledgeable financial supervisor. "
                                  "Answer the user's question clearly and concisely from your own knowledge. "
                                  "No need to search — this is a general knowledge / concept question. "
                                  "Keep it under 200 words."),
            HumanMessage(content=f"Question: {state['question']}")
        ])
    else:
        print(f"\n[SUPERVISOR] All needed agents done. Synthesizing final answer...")
        all_reports = "\n\n".join(state["agent_outputs"])
        response = supervisor_llm.invoke([
            SystemMessage(content="You are a financial supervisor. Synthesize all agent reports "
                                  "into a clear, concise final answer for the user. "
                                  "Be direct and structured. Keep it under 200 words."),
            HumanMessage(content=f"User Question: {state['question']}\n\nAgent Reports:\n{all_reports}")
        ])

    return {"final_answer": response.content}


# =================================================================================
# STEP 8: Build the LangGraph — identical structure to multi_agent.py
# =================================================================================
# The graph SHAPE is the same as multi_agent.py (supervisor hub, agent spokes, loop back).
# But the nodes now dispatch to HTTP servers instead of calling Python functions.
#
# This is the key insight: A2A changes the IMPLEMENTATION of nodes, not the STRUCTURE
# of the graph. The supervisor pattern (hub-and-spoke with loop-back) works regardless
# of whether agents are Python functions or remote HTTP services.

def build_graph():
    """
    Build the LangGraph after agent discovery is complete.

    We build the graph lazily (inside a function) rather than at module top level
    so that AGENT_REGISTRY is populated before we reference agent names in the graph.
    """
    builder = StateGraph(OrchestratorState)

    # Nodes — supervisor hub + one node per discovered agent
    builder.add_node("supervisor",   supervisor_node)
    builder.add_node("final_answer", write_final_answer)

    # Agent nodes — only add nodes for agents that are actually running
    agent_node_map = {
        "news_agent":  news_agent_node,
        "price_agent": price_agent_node,
        "risk_agent":  risk_agent_node,
    }
    conditional_targets = {"FINISH": "final_answer"}

    for name, fn in agent_node_map.items():
        if name in AGENT_REGISTRY:
            builder.add_node(name, fn)
            builder.add_edge(name, "supervisor")    # all agents loop BACK to supervisor
            conditional_targets[name] = name

    # Edges
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next_agent"],          # route based on supervisor's decision
        conditional_targets
    )
    builder.add_edge("final_answer", END)

    return builder.compile()


# =================================================================================
# STEP 9: Run it
# =================================================================================

def research(question: str, graph):
    print("\n" + "=" * 65)
    print("  A2A MULTI-AGENT FINANCIAL RESEARCH SYSTEM")
    print("=" * 65)
    print(f"\nQuestion: {question}")
    print("-" * 65)
    print("Agents are independent HTTP services. Watch HTTP dispatch logs!\n")

    result = graph.invoke({
        "question":      question,
        "agent_outputs": [],
        "called_agents": [],
        "next_agent":    "",
        "final_answer":  "",
    })

    print(f"\n{'=' * 65}")
    print("FINAL ANSWER:")
    print("=" * 65)
    print(result["final_answer"])
    agents_called = result.get("called_agents", [])
    if agents_called:
        print(f"\n[A2A AGENTS CALLED]: {agents_called}")
    else:
        print(f"\n[A2A AGENTS CALLED]: none — supervisor answered from general knowledge")


def main():
    # Phase 1: Discover agents by fetching their Agent Cards over HTTP
    # This populates AGENT_REGISTRY and AGENT_DESCRIPTIONS before the graph runs
    discover_agents()

    if not AGENT_REGISTRY:
        print("\n[ERROR] No agent servers are reachable.")
        print("Start them first with:  python run_all.py")
        print("Or individually:")
        print("  python agents/news_agent.py")
        print("  python agents/price_agent.py")
        print("  python agents/risk_agent.py")
        return

    # Phase 2: Build the LangGraph now that agent descriptions are known
    graph = build_graph()

    print("\n" + "=" * 65)
    print("  A2A ORCHESTRATOR — READY")
    print("=" * 65)
    print("\nThis orchestrator calls REAL HTTP servers, not Python functions.")
    print("Each agent runs in its own process and speaks the A2A protocol.\n")
    print("Try:")
    print("  → 'What is a stock?'          (general knowledge — no HTTP calls)")
    print("  → 'Latest news on Tesla?'      (news agent via HTTP)")
    print("  → 'Should I invest in Apple?'  (news + price + risk agents via HTTP)")
    print("\nType 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            research(question, graph)
        except Exception as e:
            print(f"\nError: {e}\n")
        print()


if __name__ == "__main__":
    main()
