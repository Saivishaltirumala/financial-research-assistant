"""
===================================================================================
FILE 2: LANGGRAPH APPROACH (Agentic AI) — THE SOLUTION
===================================================================================

This file builds the SAME Financial Research Assistant, but with LangGraph.
It solves ALL the problems from file 1:

SOLUTION 1 — CONDITIONAL ROUTING (Replaces fixed pipelines):
    The LLM DECIDES whether to search the web or answer directly.
    "What is a stock?" → answers from knowledge (no wasteful search)
    "Latest Tesla news?" → decides to search, then answers with fresh data

SOLUTION 2 — LOOPS (Iterative Refinement):
    If the LLM calls a tool, the result comes back and the LLM can:
    - Decide the answer is good enough → respond to user
    - Decide it needs more info → search again with better terms
    This is like a human analyst who keeps digging until satisfied.

SOLUTION 3 — BUILT-IN STATE (Memory):
    LangGraph maintains a "state" object that persists across the conversation.
    The message history is automatically tracked, so follow-up questions work.

SOLUTION 4 — TOOL AUTONOMY (The LLM is the decision-maker):
    Tools are "bound" to the LLM. The LLM decides WHEN to call them.
    This is the key difference: the AI is an AGENT, not a fixed pipeline.

SOLUTION 5 — HUMAN-IN-THE-LOOP via Graph Interrupt:
    The LLM can STOP the graph, ask the user a clarifying question, and RESUME
    from the exact point it paused. This is truly agentic behavior:
    - "Show me stock performance" → LLM recognizes ambiguity → asks "Which stock?"
    - User replies "Tesla" → graph resumes → LLM searches for Tesla → answers
    The LLM DECIDES when to ask — we don't hardcode it. It has an `ask_user` tool
    that it can choose to call whenever it thinks the query is unclear.
    This is IMPOSSIBLE in LangChain — a chain runs to completion or fails,
    it cannot pause mid-execution and wait for user input.

HOW IT WORKS — THE GRAPH:
    LangGraph models the workflow as a GRAPH with nodes and edges:

    ┌─────────┐     has tool calls?      ┌───────────┐
    │  Agent  │ ──── YES ──────────────→ │   Tools   │
    │  (LLM)  │                          │(Search or │
    │         │ ◄──── results ───────── │ Ask User) │
    │         │                          └───────────┘
    │         │ ──── NO (final answer) ─→  END
    └─────────┘

    When the LLM calls `ask_user`, the Tools node triggers an INTERRUPT:
    - Graph FREEZES, state is saved to checkpointer
    - Control returns to our Python code, which shows the question to user
    - User replies, we call Command(resume=user_reply)
    - Graph RESUMES from the frozen point with the user's answer
===================================================================================
"""

import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.tools import ddg_search
from shared.config import GROQ_MODEL
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# =================================================================================
# STEP 1: Define the STATE — This is what LangGraph tracks across the conversation
# =================================================================================
# In LangGraph, "State" is a TypedDict that defines what data flows through the graph.
# The key insight: `add_messages` is an ANNOTATION that tells LangGraph to APPEND
# new messages instead of replacing them. This gives us conversation memory for free!
class State(TypedDict):
    # The `add_messages` annotation means:
    # - New messages are appended to the list (not replaced)
    # - This automatically maintains conversation history
    # - Follow-up questions work because the LLM sees all previous messages
    # Compare this to LangChain where each call was independent and stateless!
    messages: Annotated[list, add_messages]


# =================================================================================
# STEP 2: Set up Tools and LLM with Tool Binding
# =================================================================================
# The critical difference from LangChain:
# In LangChain, WE decided when to use the search tool (always).
# In LangGraph, the LLM decides when to use tools — it's an autonomous agent.

# DuckDuckGo search — same tool, but now the LLM controls when to use it
# We explicitly set name="web_search" because smaller LLMs (like Llama 3.1 8B)
# sometimes hallucinate tool names — e.g., calling "brave_search" instead of
# the default "duckduckgo_results_json". A simple, generic name like "web_search"
# prevents this confusion. This is a common gotcha with tool-calling on smaller models.
search_tool = ddg_search


# =================================================================================
# THE ask_user TOOL — This is the Human-in-the-Loop feature (SOLUTION 5)
# =================================================================================
# This tool lets the LLM ASK THE USER for clarification when a query is ambiguous.
#
# HOW IT WORKS:
# 1. The LLM has two tools: web_search and ask_user
# 2. For clear queries ("Tesla stock price"), it uses web_search
# 3. For ambiguous queries ("show me stock performance"), it calls ask_user
# 4. Inside ask_user, `interrupt()` FREEZES the entire graph
# 5. The frozen state is saved to the checkpointer (MemorySaver)
# 6. Control returns to our Python code in main()
# 7. We show the question to the user and get their reply
# 8. We call Command(resume=reply) which UNFREEZES the graph
# 9. The interrupt() call returns with the user's reply
# 10. The tool returns the reply to the LLM, which continues processing
#
# KEY INSIGHT: The LLM DECIDES when to ask — we don't hardcode it.
# This is truly agentic: the AI recognizes it's missing information
# and actively asks for clarification, just like a human assistant would.
#
# WHY LANGCHAIN CAN'T DO THIS:
# A LangChain chain is a straight pipeline — it either runs to completion or fails.
# There's no concept of "pause here, ask the user, resume from where I stopped."
# You'd have to break the chain into pieces and manually stitch them together,
# losing all the benefits of the chain abstraction.
@tool
def ask_user(question: str) -> str:
    """Ask the user a clarifying question when the query is ambiguous or missing details.
    Use this when you need more information to provide an accurate answer.
    For example: which specific stock, what time period, what metric they want, etc."""
    # interrupt() FREEZES the graph here and returns `question` to our Python code.
    # When we later call Command(resume="user's reply"), the interrupt() call
    # RETURNS with that reply, and execution continues from this exact line.
    user_reply = interrupt(question)
    return user_reply


# List of tools the agent can use
# The LLM sees both tools and autonomously decides which to use:
#   - web_search: for queries needing current data
#   - ask_user: for ambiguous queries needing clarification
tools = [search_tool, ask_user]

# Initialize the LLM
llm = ChatGroq(
    model_name=GROQ_MODEL,
    temperature=0.3,
)

# KEY STEP: "Bind" tools to the LLM
# This tells the LLM about available tools and their schemas.
# The LLM can now CHOOSE to call these tools when it thinks it needs to.
# If the LLM decides it doesn't need a tool, it just responds directly.
# This is TOOL AUTONOMY — the LLM is the decision-maker, not the code.
llm_with_tools = llm.bind_tools(tools)


# =================================================================================
# STEP 3: Define the SYSTEM PROMPT — The agent's personality and instructions
# =================================================================================
# The system prompt is crucial for controlling LLM behavior with search results.
# Without explicit instructions to "present the data you find", smaller models
# like Llama 3.1 8B tend to say "I'm unable to verify" even when the search
# results contain the exact data the user asked for. The model is being overly
# cautious. The prompt below fixes this by telling the LLM to ALWAYS present
# data found in search results, along with a source disclaimer.
SYSTEM_PROMPT = """You are an expert Financial Research Assistant specialized in stock market analysis.

Your capabilities:
- Analyze stock prices, market trends, and financial news
- Search the internet for the latest financial information when needed
- Ask the user clarifying questions when their query is ambiguous
- Provide balanced analysis (never give direct buy/sell recommendations)

Guidelines:
- For FACTUAL questions you already know (like "what is a P/E ratio?"), answer directly WITHOUT searching
- For CURRENT data (prices, news, recent events), USE the web_search tool to get fresh information
- When the user's query is AMBIGUOUS or missing key details, USE the ask_user tool to ask for clarification
  Examples of when to use ask_user:
    - "Show me stock performance" → ask which stock they mean
    - "Compare these stocks" → ask which stocks to compare
    - "What's the best investment?" → ask about their risk tolerance, timeframe, etc.
- When analyzing stocks, always mention that past performance doesn't guarantee future results
- Be specific with data and numbers when available
- If you need more information, search again with refined terms

IMPORTANT — How to handle search results:
- When search results contain prices, profits, or any numerical data, ALWAYS present that data to the user
- DO NOT say "I'm unable to verify" or "I cannot confirm" if the search results contain relevant information
- Instead, present the data and add a disclaimer like "Based on search results as of [date], this may not reflect real-time data"
- Extract and clearly state specific numbers (stock price, revenue, profit, P/E ratio, etc.) from the search results
- If search results are truly empty or irrelevant, only then say you couldn't find the information

You have access to web_search and ask_user tools. Use them wisely."""


# =================================================================================
# STEP 4: Define GRAPH NODES — These are the "functions" that run at each step
# =================================================================================

def agent_node(state: State):
    """
    The AGENT node — this is where the LLM thinks and makes decisions.

    This is fundamentally different from LangChain:
    - In LangChain: the LLM just generates text from a fixed prompt
    - In LangGraph: the LLM DECIDES what to do next:
        → search the web? (calls web_search)
        → ask the user for clarification? (calls ask_user → triggers interrupt)
        → answer directly? (just returns text)

    The LLM receives the full message history (thanks to State) and decides.
    The `tools_condition` function (used in edges below) checks what it decided.
    """
    # Prepend the system prompt to give the agent its personality
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    # The LLM processes all messages and decides what to do
    # If it needs info → it generates a tool_call for web_search
    # If it needs clarity → it generates a tool_call for ask_user
    # If it can answer → it generates a regular text response
    response = llm_with_tools.invoke(messages)

    # Return the response to be added to state (add_messages appends it)
    return {"messages": [response]}


# The ToolNode automatically executes whatever tool the LLM decided to call.
# It takes the tool call from the LLM's response, runs the tool, and returns results.
# If the LLM called ask_user, the ToolNode will hit the interrupt() inside it,
# which FREEZES the graph and returns control to our Python code.
# This is plug-and-play: add more tools to the list and the agent can use them all.
tool_node = ToolNode(tools=tools)


# =================================================================================
# STEP 5: Build the GRAPH — Connect nodes with edges (including conditional edges!)
# =================================================================================
# This is where LangGraph truly shines. We define a GRAPH, not a chain.
# The graph has CONDITIONAL EDGES that let the agent loop back for more info.

# Create the graph builder with our State schema
graph_builder = StateGraph(State)

# Add nodes (the "processing stations" in our graph)
graph_builder.add_node("agent", agent_node)   # The LLM decision-maker
graph_builder.add_node("tools", tool_node)     # The tool executor

# Add edges (the "connections" between nodes)

# START → agent: Every conversation begins with the agent thinking
graph_builder.add_edge(START, "agent")

# agent → tools OR end: This is the CONDITIONAL EDGE — the magic of LangGraph!
# `tools_condition` is a built-in LangGraph function that checks the agent's response:
#   - If agent generated a tool_call  → route to "tools" node (execute the search)
#   - If agent generated a text reply → route to END (the LLM has its final answer)
#
# IMPORTANT: Notice we did NOT write a separate `add_edge("agent", END)`.
# That's because tools_condition ALREADY includes the route to END internally.
# It handles BOTH paths (tools and END) in one conditional edge.
# Writing a separate add_edge("agent", END) would be redundant and would conflict.
#
# This is IMPOSSIBLE in standard LangChain chains!
graph_builder.add_conditional_edges("agent", tools_condition)

# tools → agent: After tools run, ALWAYS go back to the agent
# This creates the LOOP: agent thinks → uses tool → gets results → thinks again
# The agent might decide to search again, or finally give an answer.
# Standard LangChain is a straight line and CANNOT loop like this.
graph_builder.add_edge("tools", "agent")

# =================================================================================
# COMPILE WITH CHECKPOINTER — Required for interrupt() to work
# =================================================================================
# MemorySaver is an in-memory checkpointer that saves graph state.
# Without it, interrupt() would fail because there's nowhere to save the frozen state.
#
# When interrupt() is called inside ask_user:
#   1. MemorySaver saves the entire graph state (all messages, where we paused)
#   2. Graph execution stops and returns to our Python code
#   3. When we call Command(resume=reply), MemorySaver loads the saved state
#   4. Graph continues from the exact point it froze
#
# MemorySaver also gives us conversation memory across turns for FREE.
# Each invocation with the same thread_id shares state, so we no longer need
# the manual conversation_history list. The checkpointer handles it all.
#
# In production, you'd use a persistent checkpointer (e.g., SqliteSaver, PostgresSaver)
# so conversations survive server restarts.
checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)


# =================================================================================
# STEP 6: Interactive Chat with MEMORY and HUMAN-IN-THE-LOOP
# =================================================================================
def main():
    """
    Run the LangGraph financial assistant.

    Try the SAME queries as file 1 and see the difference:

    1. "What is a stock?"
       → The agent answers directly WITHOUT searching! (It decided it doesn't need to)

    2. "What's the latest news on Tesla stock?"
       → The agent DECIDES to search, gets results, then answers.

    3. "Based on that news, should I buy Tesla?"
       → WORKS! The agent remembers the Tesla conversation (checkpointer has history)

    4. "Compare Apple and Microsoft stock performance"
       → The agent can search for Apple, then search for Microsoft, then compare.

    5. "Show me stock performance" (AMBIGUOUS!)
       → The agent calls ask_user → graph FREEZES → asks "Which stock?"
       → You reply → graph RESUMES → agent searches and answers.
       → This is the Human-in-the-Loop feature in action!
    """

    print("\n" + "=" * 60)
    print("  FINANCIAL RESEARCH ASSISTANT — LANGGRAPH (AGENTIC AI)")
    print("  (Notice how it DECIDES when to search vs answer directly)")
    print("=" * 60)
    print("\nType 'quit' to exit.\n")
    print("TIP: Try ambiguous queries like 'show me stock performance'\n"
          "     to see the Human-in-the-Loop interrupt feature!\n")

    # ==========================================================================
    # WHY DO WE NEED conversation_history WHEN WE ALREADY HAVE State?
    # ==========================================================================
    # SHORT ANSWER: We don't anymore! With the checkpointer (MemorySaver),
    # state persists across invocations using the same thread_id.
    #
    # Previously (without checkpointer):
    #   - State only lived WITHIN a single graph.stream() call (inner loop)
    #   - We needed a manual conversation_history list for the outer loop
    #
    # Now (with checkpointer):
    #   - MemorySaver saves state after each invocation
    #   - Same thread_id = same conversation = automatic memory
    #   - No manual list needed!
    #
    # The thread_id in config below is like a "session ID" — all messages
    # with the same thread_id are part of the same conversation.
    # ==========================================================================

    # Config with thread_id — all messages in this session share the same thread
    config = {"configurable": {"thread_id": "financial-session-1"}}

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break
        if not question:
            continue

        print(f"\n{'='*60}")

        try:
            # =================================================================
            # HOW graph.stream() WORKS vs graph.invoke():
            # =================================================================
            # graph.invoke() → runs the ENTIRE graph silently, returns only
            #   the final result. You see nothing in between: input → black box → output.
            #
            # graph.stream() → yields events ONE NODE AT A TIME as the graph executes.
            #   Each time a node finishes processing, it emits its output BEFORE
            #   the next node starts. This lets us intercept and log each step.
            #
            # For a query like "Tata Steel stock price", the stream yields:
            #   Event 1: {"agent": {messages: [AIMessage with tool_call]}}  → Agent decided to search
            #   Event 2: {"tools": {messages: [ToolMessage with results]}}  → Tool executed search
            #   Event 3: {"agent": {messages: [AIMessage with text]}}       → Agent gave final answer
            #
            # We are NOT interrupting the graph. The graph PAUSES NATURALLY after
            # each node and hands us the output (like a Python generator/iterator).
            # We log it, and then the for-loop continues, triggering the next node.
            # If the agent loops (searches twice), we simply get more events:
            #   Event 1: agent (tool_call) → Event 2: tools → Event 3: agent (tool_call again)
            #   → Event 4: tools → Event 5: agent (final answer)
            # =================================================================
            events = graph.stream(
                {"messages": [HumanMessage(content=question)]},
                config=config,
            )
            _process_events(events, config)

        except Exception as e:
            print(f"\nError: {e}")

        print(f"{'='*60}\n")


def _process_events(events, config):
    """
    Process streamed events from the graph.

    This function handles TWO scenarios:
    1. Normal flow: agent searches → answers (no interrupt)
    2. Interrupt flow: agent calls ask_user → graph freezes → we ask user → resume

    When an interrupt happens:
    - The stream ends early (graph is frozen)
    - We detect the interrupt via graph.get_state()
    - We show the question to the user and get their reply
    - We call Command(resume=reply) to unfreeze the graph
    - We recursively process the new events from the resumed graph
    """
    for event in events:
        # Each event is a dict: {node_name: node_output}
        # We iterate to find which node just finished
        for node_name, node_output in event.items():
            if node_name == "agent":
                msg = node_output["messages"][-1]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        if tc["name"] == "ask_user":
                            # The agent DECIDED to ask the user (interrupt incoming!)
                            print(f"[AGENT DECISION] ❓ Asking user: {tc['args'].get('question', '')}")
                        else:
                            # The agent DECIDED to use a tool (conditional routing!)
                            print(f"[AGENT DECISION] 🔍 Searching: {tc['args']}")
                elif msg.content:
                    # The agent DECIDED it has enough info to answer
                    print(f"\nAssistant: {msg.content}")
            elif node_name == "tools":
                print(f"[TOOL RESULT] Got results, agent is analyzing...")

    # ==========================================================================
    # HANDLE INTERRUPT — This is the Human-in-the-Loop magic
    # ==========================================================================
    # After the stream ends, check if the graph is paused (interrupted).
    # If ask_user was called, the graph froze mid-execution and is waiting
    # for us to provide a reply via Command(resume=...).
    #
    # graph.get_state() tells us if there are pending interrupts.
    # If there are, we:
    #   1. Extract the question the LLM wants to ask
    #   2. Show it to the user and get their reply
    #   3. Resume the graph with Command(resume=reply)
    #   4. Process the new events (the graph continues from where it froze)
    # ==========================================================================
    state = graph.get_state(config)
    if state.tasks and any(hasattr(t, 'interrupts') and t.interrupts for t in state.tasks):
        # Graph is paused — the LLM called ask_user and hit interrupt()
        for task in state.tasks:
            if hasattr(task, 'interrupts') and task.interrupts:
                for intr in task.interrupts:
                    # Show the LLM's question to the user
                    clarification = input(f"\n💬 Agent asks: {intr.value}\nYou: ").strip()

                    # Resume the graph from the frozen point with the user's reply
                    # Command(resume=...) unfreezes interrupt() inside ask_user,
                    # which returns the user's reply as the tool result.
                    # The graph then continues: tool result → agent → answer
                    resumed_events = graph.stream(
                        Command(resume=clarification),
                        config=config,
                    )
                    _process_events(resumed_events, config)


if __name__ == "__main__":
    main()
