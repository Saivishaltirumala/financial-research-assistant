"""
===================================================================================
FILE 6: SAME MULTI-AGENT SYSTEM — NOW WITH CREWAI
===================================================================================

WHAT IS CREWAI?
    CrewAI is a high-level framework for building multi-agent systems using a
    simple, declarative API. Instead of building graphs with nodes and edges
    (LangGraph), you describe:
        - WHO the agents are  (role, goal, backstory)
        - WHAT they need to do (Tasks with descriptions)
        - HOW they work together (Crew with a Process)

    CrewAI handles the orchestration, memory, tool routing, and agent looping
    internally — you don't write any of that plumbing yourself.

THE SAME SYSTEM — TWO DIFFERENT APPROACHES:
    5_multi_agent.py (LangGraph)  → you build the graph manually:
        - Define State (TypedDict)
        - Write each node as a Python function
        - Wire edges: supervisor → agent → supervisor → ... → END
        - Write the supervisor logic yourself
        - Handle called_agents tracking yourself
        - Handle general knowledge shortcut yourself

    6_crewai_approach.py (CrewAI) → you describe the crew declaratively:
        - Define Agents with role/goal/backstory (plain English)
        - Define Tasks with description/expected_output
        - Hand it to Crew — CrewAI runs everything

CREWAI'S TWO EXECUTION MODES:
    Process.sequential   → Tasks run in a FIXED ORDER, one after another
                           (like an assembly line — news → price → risk)
                           Simple, predictable, no manager needed

    Process.hierarchical → A MANAGER AGENT decides which worker agents to call
                           and in what order, just like LangGraph's supervisor.
                           Closest equivalent to 5_multi_agent.py

THIS FILE SHOWS BOTH, starting with sequential (simpler to understand),
then hierarchical (true dynamic multi-agent).

===================================================================================
CREWAI CORE CONCEPTS (before reading the code):
===================================================================================

┌─────────────────────────────────────────────────────────────────────────┐
│  AGENT                                                                  │
│  ──────                                                                 │
│  An autonomous AI worker with:                                          │
│  • role      → "Financial News Analyst"                                 │
│  • goal      → what it's trying to accomplish                           │
│  • backstory → context/personality that shapes how it responds          │
│  • tools     → what external tools it can call (search, APIs, etc.)     │
│  • llm       → which LLM powers this agent                              │
│                                                                         │
│  ↳ LangGraph equivalent: a node function + its system prompt + LLM      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  TASK                                                                   │
│  ──────                                                                 │
│  A specific unit of work assigned to an agent:                          │
│  • description     → what needs to be done (can reference context vars) │
│  • expected_output → what a good result looks like                      │
│  • agent           → which agent is responsible for this task           │
│  • context         → list of OTHER tasks whose output this task needs   │
│                      (this is how agents share information!)            │
│                                                                         │
│  ↳ LangGraph equivalent: what happens inside a node function            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  CREW                                                                   │
│  ──────                                                                 │
│  The orchestrator that runs the whole system:                           │
│  • agents    → list of all agents in the team                           │
│  • tasks     → list of all tasks to be executed                         │
│  • process   → Process.sequential or Process.hierarchical               │
│  • verbose   → show internal reasoning steps in console                 │
│                                                                         │
│  crew.kickoff(inputs={"question": "..."})  ← starts the whole system    │
│                                                                         │
│  ↳ LangGraph equivalent: graph.invoke({...})                            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  TOOL (in CrewAI)                                                       │
│  ─────────────────                                                      │
│  Defined with @tool decorator from crewai.tools.                        │
│  The function name + docstring is what the agent reads to decide        │
│  when and how to call the tool — same concept as bind_tools() in        │
│  LangGraph (tool docstrings → LLM context).                             │
└─────────────────────────────────────────────────────────────────────────┘

===================================================================================
ADVANTAGES OF CREWAI vs LANGGRAPH:
===================================================================================

✅  LESS CODE — no State TypedDict, no graph builder, no add_node, no add_edge
    LangGraph 5_multi_agent.py: ~300 lines
    CrewAI 6_crewai_approach.py: ~150 lines for the same system

✅  READABLE AGENT DEFINITIONS — role/goal/backstory in plain English makes
    agent behavior self-documenting. Non-engineers can read and adjust agents.

✅  BUILT-IN TASK CONTEXT PASSING — `context=[task1, task2]` on a Task
    automatically feeds prior task outputs to the next agent. No shared state
    dict or Annotated[list, operator.add] needed.

✅  BUILT-IN MEMORY (optional) — Crew supports short-term, long-term, and
    entity memory out of the box. No MemorySaver or checkpointer wiring.

✅  FASTER PROTOTYPING — Define agents and tasks, call kickoff(). Great for
    demos, internal tools, or when you need something running quickly.

✅  HIERARCHICAL PROCESS BUILT-IN — a manager agent (like supervisor) is
    created automatically when Process.hierarchical is used. You don't write
    the supervisor logic yourself — CrewAI injects a manager LLM that reads
    agent roles/goals and delegates tasks dynamically.

===================================================================================
DISADVANTAGES OF CREWAI vs LANGGRAPH:
===================================================================================

❌  LESS CONTROL OVER FLOW — CrewAI manages the loop internally. You CANNOT
    intercept mid-execution, add custom routing logic, or inspect intermediate
    state the way you can in LangGraph.

❌  NO HUMAN-IN-THE-LOOP — CrewAI has no equivalent of LangGraph's interrupt()
    + Command(resume=). You cannot pause mid-task, ask the user a question,
    and resume from the exact frozen point. LangGraph is the only framework
    that natively supports this.

❌  NO FINE-GRAINED STATE CONTROL — LangGraph's TypedDict State gives you
    exact control over what every agent can read and write. In CrewAI, task
    outputs flow through context= — you cannot prevent an agent from receiving
    or leaking data the way you can with isolated subgraph state in LangGraph.

❌  DEBUGGING IS HARDER — LangGraph's graph.stream() gives event-by-event
    visibility (you see every node firing). CrewAI's verbose=True is verbose
    but less structured — harder to instrument for production logging.

❌  FIXED PROCESS MODES — you pick sequential or hierarchical. LangGraph lets
    you mix: some parts sequential, some parallel (fan-out), some with loops,
    some with interrupts — all in one graph.

❌  HIERARCHICAL MANAGER IS A BLACK BOX — the auto-generated manager in
    Process.hierarchical uses a built-in prompt you cannot fully control.
    In LangGraph, you write the supervisor prompt yourself (SUPERVISOR_PROMPT
    in 5_multi_agent.py) — complete ownership of reasoning logic.

❌  TOOL EXECUTION IS IMPLICIT — in LangGraph, you see exactly when tools
    are called (ToolNode fires). In CrewAI, tool calls happen inside the
    agent loop invisibly unless verbose=True.

===================================================================================
WHEN TO USE CREWAI vs LANGGRAPH:
===================================================================================

Use CREWAI when:
    • You need a working multi-agent system quickly (prototyping, demos)
    • Agent roles are stable and clearly defined (no dynamic routing needed)
    • Sequential workflows — report generation, research pipelines, content creation
    • Team is non-technical — role/goal/backstory is readable by product/business
    • You don't need Human-in-the-Loop or mid-execution pauses
    • Simple hierarchical delegation is enough (no custom supervisor logic)

Use LANGGRAPH when:
    • You need Human-in-the-Loop (pause, ask user, resume)
    • You need precise control over execution flow (custom routing, loops, retries)
    • You need parallel execution (fan-out/fan-in for speed)
    • You need subgraph isolation (agents must NOT see each other's internal state)
    • Production systems where observability and debugging matter
    • Custom supervisor logic (your SUPERVISOR_PROMPT must encode business rules)
    • Mixed workflow patterns in one graph (some sequential, some parallel, some conditional)

SUMMARY:
    CrewAI = "Tell me WHO and WHAT, I'll figure out HOW" (declarative, fast)
    LangGraph = "You control WHO, WHAT, and HOW — completely" (imperative, precise)

===================================================================================
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults

load_dotenv()


# =================================================================================
# STEP 1: Set up LLM — CrewAI uses its own LLM wrapper (backed by LiteLLM)
# =================================================================================
# CrewAI's LLM() wraps LiteLLM under the hood, which means it supports
# 100+ providers (OpenAI, Anthropic, Groq, Ollama, etc.) via unified API.
# Format: "provider/model-name"
# In LangGraph we used: ChatGroq(model_name="llama-3.1-8b-instant")
# In CrewAI we use:     LLM(model="groq/llama-3.1-8b-instant")

groq_api_key = os.getenv("GROQ_API_KEY")

# One shared LLM for all agents in this demo.
# In production you could give each agent a different model:
# manager_llm = LLM(model="groq/llama-3.3-70b-versatile")  ← smarter for orchestration
# worker_llm  = LLM(model="groq/llama-3.1-8b-instant")     ← faster/cheaper for tasks
llm = LLM(
    model="groq/llama-3.1-8b-instant",
    api_key=groq_api_key,
    temperature=0.3
)


# =================================================================================
# STEP 2: Define the Tool — same DuckDuckGo search, CrewAI's @tool decorator
# =================================================================================
# In LangGraph (5_multi_agent.py): we used DuckDuckGoSearchResults directly
# In CrewAI: we wrap it with @tool so CrewAI agents can discover and call it.
#
# KEY POINT: Just like in LangGraph's bind_tools(), the @tool docstring IS what
# the agent reads to decide when and how to call this tool.
# A good docstring = agent uses the tool correctly.
# A vague docstring = agent calls it at wrong times or with wrong inputs.

_search = DuckDuckGoSearchResults(name="web_search", num_results=2)

@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the web for financial news, stock prices, PE ratios, market cap,
    and other real-time financial data. Use this tool when you need current
    information that may not be in your training data.
    Input: a search query string (e.g., 'Tesla stock price today PE ratio')
    """
    return _search.invoke(query)


# =================================================================================
# STEP 3: Define AGENTS — declarative, plain-English descriptions
# =================================================================================
# This is the biggest difference from LangGraph.
# In LangGraph (5_multi_agent.py): agents were Python functions with system prompts
#   def news_agent_node(state): ...  + SystemMessage(content="You are a news specialist...")
#
# In CrewAI: agents are OBJECTS described in plain English:
#   role      → job title ("Financial News Analyst")
#   goal      → what success looks like for this agent
#   backstory → expertise and personality context (shapes LLM responses)
#   tools     → list of tools this agent can use
#   llm       → which LLM powers this agent
#
# The role+goal+backstory combination is injected automatically into the agent's
# system prompt by CrewAI. You never write the raw system prompt yourself.

news_agent = Agent(
    role="Financial News Analyst",
    goal=(
        "Search for the latest financial news, announcements, and events about "
        "the stock in question. Provide a structured report with key headlines, "
        "overall sentiment, and any important events."
    ),
    backstory=(
        "You are a specialist in financial journalism with 10 years of experience "
        "tracking market-moving news. You know how to identify signal from noise "
        "in financial media, and you always structure your findings clearly so "
        "other analysts on your team can build on your work."
    ),
    tools=[web_search],   # This agent CAN search the web
    llm=llm,
    verbose=False,        # Suppress per-agent internal verbosity for cleaner output
)

price_agent = Agent(
    role="Stock Price & Metrics Specialist",
    goal=(
        "Search for current stock price, P/E ratio, market cap, 52-week range, "
        "and recent percentage change. Provide a structured metrics report that "
        "other analysts can reference."
    ),
    backstory=(
        "You are a quantitative analyst specializing in stock valuation metrics. "
        "You are precise with numbers and always note when data may be delayed "
        "or unavailable. Your reports are used by risk analysts to make "
        "investment recommendations."
    ),
    tools=[web_search],   # This agent CAN search the web
    llm=llm,
    verbose=False,
)

risk_agent = Agent(
    role="Investment Risk Assessor",
    goal=(
        "Synthesize the news report and price metrics report provided by your "
        "teammates to produce a balanced risk assessment with a clear recommendation."
    ),
    backstory=(
        "You are a senior investment risk analyst with expertise in combining "
        "qualitative news signals with quantitative price data. You never search "
        "the web yourself — you rely entirely on your team's findings and apply "
        "your analytical judgment to assess risk. You always include a disclaimer."
    ),
    # IMPORTANT: risk_agent has NO tools — it reads team outputs only.
    # In LangGraph (5_multi_agent.py), risk_agent_node also had no search call —
    # it read from state["agent_outputs"]. Here, that sharing happens via Task.context.
    tools=[],
    llm=llm,
    verbose=False,
)


# =================================================================================
# STEP 4: Define TASKS — assign work to agents
# =================================================================================
# Each Task has:
#   description     → what the agent needs to do (use {question} as a template var)
#   expected_output → what a good output looks like (shapes the agent's response)
#   agent           → who is responsible
#   context         → list of PRIOR tasks whose outputs this task should read
#
# The `context` field is how CrewAI agents share information between tasks —
# equivalent to reading from shared agent_outputs in LangGraph's state.
#
# NOTE: The {question} placeholder in description is filled at kickoff() time:
#   crew.kickoff(inputs={"question": "Should I invest in Tesla?"})

news_task = Task(
    description=(
        "Research the latest financial news about: {question}\n\n"
        "Search the web for recent news, announcements, earnings reports, "
        "leadership changes, product launches, or any market-moving events.\n\n"
        "Format your report as:\n"
        "NEWS REPORT:\n"
        "- Key headlines: (2-3 bullet points)\n"
        "- Overall news sentiment: (Positive / Negative / Neutral)\n"
        "- Important events: (earnings, launches, scandals, partnerships)\n"
        "- Source note: (brief note on data recency)"
    ),
    expected_output=(
        "A structured NEWS REPORT with headlines, sentiment classification, "
        "and key events — 150 words or less."
    ),
    agent=news_agent,
    # No context needed — this is the first task, starts fresh
)

price_task = Task(
    description=(
        "Research the current stock price and financial metrics for: {question}\n\n"
        "Search the web for current price, P/E ratio, market cap, 52-week high/low, "
        "and recent percentage change.\n\n"
        "Format your report as:\n"
        "PRICE REPORT:\n"
        "- Current Price: (value or N/A)\n"
        "- P/E Ratio: (value or N/A)\n"
        "- Market Cap: (value or N/A)\n"
        "- 52-week High/Low: (value or N/A)\n"
        "- Recent % Change: (value or N/A)\n"
        "- Valuation note: (Overvalued / Undervalued / Fair based on P/E)"
    ),
    expected_output=(
        "A structured PRICE REPORT with current metrics and a brief valuation note "
        "— 120 words or less."
    ),
    agent=price_agent,
    # No context needed — runs independently in parallel conceptually
)

risk_task = Task(
    description=(
        "Using the news report and price report provided by your teammates, "
        "assess the investment risk for: {question}\n\n"
        "Do NOT search the web. Use ONLY the context provided (news + price reports).\n\n"
        "Format your assessment as:\n"
        "RISK ASSESSMENT:\n"
        "- Risk Level: (Low / Medium / High)\n"
        "- Key risk factors: (2-3 bullet points from the news + metrics)\n"
        "- Key positive factors: (2-3 bullet points)\n"
        "- Recommendation: (what type of investor this suits)\n"
        "- Disclaimer: past performance does not guarantee future results"
    ),
    expected_output=(
        "A structured RISK ASSESSMENT with risk level, factors, recommendation, "
        "and disclaimer — 200 words or less."
    ),
    agent=risk_agent,
    # context = outputs of news_task and price_task are passed to risk_agent automatically
    # This is how CrewAI handles agent collaboration — no shared state dict needed
    # Equivalent to: risk_agent_node reading state["agent_outputs"] in LangGraph
    context=[news_task, price_task],
)


# =================================================================================
# STEP 5a: SEQUENTIAL CREW — fixed order, simple, predictable
# =================================================================================
# Process.sequential: tasks execute in the ORDER you put them in the list.
# news_task → price_task → risk_task, always in that order.
#
# WHEN TO USE: when your workflow is always the same regardless of the question.
# LIMITATION: even for simple questions ("What is the latest news?"), all 3 tasks
# still run — no ability to skip tasks the way LangGraph's supervisor can.
# (This is one of LangGraph's key advantages over CrewAI sequential mode.)

sequential_crew = Crew(
    agents=[news_agent, price_agent, risk_agent],
    tasks=[news_task, price_task, risk_task],
    process=Process.sequential,   # Fixed order: news → price → risk
    verbose=True,                 # Show task execution in console
)


# =================================================================================
# STEP 5b: HIERARCHICAL CREW — manager decides who does what (dynamic)
# =================================================================================
# Process.hierarchical: CrewAI automatically creates a MANAGER AGENT that:
#   - reads all worker agent roles/goals/backstories
#   - decides which agent to delegate each task to
#   - re-evaluates after each agent completes (like LangGraph's supervisor)
#
# You provide manager_llm — the LLM that powers the manager's decisions.
# The manager's system prompt is generated by CrewAI (you don't write it).
#
# CLOSEST EQUIVALENT TO 5_multi_agent.py's supervisor pattern.
# KEY DIFFERENCE: in LangGraph you wrote SUPERVISOR_PROMPT yourself,
# controlled called_agents tracking, and added general knowledge shortcut.
# Here CrewAI's manager is a black box — you trust it to orchestrate correctly.

hierarchical_crew = Crew(
    agents=[news_agent, price_agent, risk_agent],
    tasks=[news_task, price_task, risk_task],
    process=Process.hierarchical,  # Manager decides delegation dynamically
    manager_llm=llm,               # LLM that powers the auto-generated manager agent
    verbose=True,
)


# =================================================================================
# STEP 6: Run it
# =================================================================================

def run_sequential(question: str):
    """Run with fixed sequential flow: news → price → risk (always all 3)."""
    print("\n" + "=" * 60)
    print("  CREWAI — SEQUENTIAL PROCESS")
    print("  (Fixed order: news → price → risk, always all 3 tasks)")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")

    result = sequential_crew.kickoff(inputs={"question": question})

    print("\n" + "=" * 60)
    print("FINAL ANSWER (Sequential):")
    print("=" * 60)
    print(result.raw)


def run_hierarchical(question: str):
    """Run with hierarchical process: manager dynamically delegates to agents."""
    print("\n" + "=" * 60)
    print("  CREWAI — HIERARCHICAL PROCESS")
    print("  (Manager decides who to call and in what order)")
    print("=" * 60)
    print(f"\nQuestion: {question}\n")

    result = hierarchical_crew.kickoff(inputs={"question": question})

    print("\n" + "=" * 60)
    print("FINAL ANSWER (Hierarchical):")
    print("=" * 60)
    print(result.raw)


def main():
    print("\n" + "=" * 60)
    print("  CREWAI — MULTI-AGENT FINANCIAL RESEARCH")
    print("=" * 60)
    print("\nCrewAI abstracts away graph building — you define WHO and WHAT,")
    print("CrewAI handles HOW they collaborate.\n")
    print("Mode options:")
    print("  [1] Sequential  — news → price → risk (always, fixed order)")
    print("  [2] Hierarchical — manager delegates dynamically (like LangGraph supervisor)")
    print()

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
