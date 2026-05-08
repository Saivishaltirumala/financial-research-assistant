# Financial Research Assistant — LangChain vs LangGraph

## What This Project Demonstrates

This project builds the **same stock market chatbot twice** — once with standard LangChain (GenAI) and once with LangGraph (Agentic AI) — to expose why LangGraph was created and how it solves LangChain's fundamental limitations.

The assistant uses **DuckDuckGo Search** to pull real-time stock news and prices, and **Groq's LLM API** (Llama 3.1) for generating responses.

---

## The Core Problem with Standard LangChain (`1_langchain_approach.py`)

Standard LangChain pipelines are **linear chains**: `Input → Step1 → Step2 → Output`. This creates five critical limitations:

### Problem 1 — No Decision-Making
The chain **always executes every step**, regardless of whether it's needed. Ask "What is a stock?" and it still searches the web — wasteful and slow. The LLM has zero control over the pipeline flow.

### Problem 2 — No Loops / Retry Logic
If the search results are poor, the chain **cannot go back and search again** with better terms. It's a one-shot pipeline. A human analyst would refine their search — LangChain can't.

### Problem 3 — Memory Causes Prompt Explosion
LangChain's `ConversationBufferMemory` dumps the **entire conversation history into every prompt**. Watch the token count grow in the logs:

```
Turn 1:  ~284 tokens  (just the question)
Turn 3:  ~887 tokens  (3 exchanges accumulated)
Turn 10: ~3000+ tokens (approaching context limit)
Turn 15: CRASH — exceeds model's 8K context window
```

### Problem 4 — No Tool Autonomy
The developer **hardcodes** when tools are called. The LLM cannot decide "I need to search for this" vs "I already know this." Even `bind_tools()` only generates intent — it doesn't execute tools or handle the agent loop.

### Problem 5 — No Human-in-the-Loop
The chain **cannot pause mid-execution** to ask the user for clarification. If the query is ambiguous ("show me stock performance"), it guesses or fails. It cannot stop and ask "Which stock?"

---

## How LangGraph Solves Everything (`2_langgraph_approach.py`)

LangGraph replaces the linear chain with a **graph** — nodes connected by conditional edges, enabling loops, decisions, and interrupts.

### Solution 1 — Conditional Routing (fixes Problem 1)
The LLM **decides** whether to search the web or answer directly. "What is a P/E ratio?" gets answered instantly without a wasteful web search. "Tesla stock price today?" triggers a search because the LLM knows it needs current data.

### Solution 2 — Loops / Iterative Refinement (fixes Problem 2)
The graph has a cycle: `Agent → Tools → Agent`. After getting search results, the agent can decide to **search again** with refined terms, or give the final answer. It keeps looping until satisfied — just like a human analyst.

### Solution 3 — Checkpointer-based State (fixes Problem 3)
Instead of dumping all history into the prompt, LangGraph uses a **MemorySaver checkpointer** that persists state across turns using a `thread_id`. Conversation memory is built into the graph — no manual list, no prompt explosion.

### Solution 4 — Tool Autonomy via bind_tools (fixes Problem 4)
Tools are **bound to the LLM**. The LLM autonomously decides which tool to call (or none at all). The `ToolNode` automatically executes the chosen tool and feeds results back. No hardcoding.

### Solution 5 — Human-in-the-Loop via Graph Interrupt (fixes Problem 5)
The LLM has an `ask_user` tool. When it recognizes an ambiguous query, it **calls `ask_user`**, which triggers `interrupt()` — the graph **freezes**, asks the user for clarification, and **resumes from the exact frozen point** with the user's reply. This is impossible in LangChain.

---

## Graph Architecture

```
                      ┌─────────────┐
                      │  __start__  │
                      └──────┬──────┘
                             │
                             ▼
                ┌────────────────────────┐
                │      AGENT (LLM)       │
                │                        │
                │  Decides what to do:   │
                │  • search the web?     │
                │  • ask user for info?  │
                │  • answer directly?    │
                └────┬──────────┬────────┘
                     │          │
         has tool_calls?    no tool_calls?
            YES │               │ NO (final answer)
                │               │
                ▼               ▼
        ┌──────────────┐   ┌──────────┐
        │    TOOLS     │   │ __end__  │
        │              │   └──────────┘
        │ ┌──────────┐ │
        │ │web_search│ │──── search results ────┐
        │ └──────────┘ │                        │
        │              │                        │
        │ ┌──────────┐ │                        │
        │ │ ask_user │ │                        │
        │ └────┬─────┘ │                        │
        └──────┼───────┘                        │
               │                                │
          INTERRUPT()                           │
          Graph freezes                         │
               │                                │
               ▼                                │
        ┌─────────────┐                         │
        │    USER      │                        │
        │             │                         │
        │  Replies to │                         │
        │  question   │                         │
        └──────┬──────┘                         │
               │                                │
          Command(resume)                       │
          Graph unfreezes                       │
               │                                │
               └──────────┐                     │
                          │                     │
                          ▼                     │
               ┌─────────────────────┐          │
               │  Tool result goes   │◄─────────┘
               │  back to AGENT      │
               │  (LOOP continues)   │
               └─────────┬───────────┘
                         │
                         ▼
                ┌────────────────────────┐
                │      AGENT (LLM)       │
                │                        │
                │  Now has more info.    │
                │  Decides again:        │
                │  • search more?        │
                │  • answer now? → END   │
                └────────────────────────┘
```

### Legend
- **Solid arrow** — Normal edge (always follows this path)
- **Conditional edge** — LLM decides which path to take
- **INTERRUPT** — Graph freezes, waits for user reply
- **RESUME** — Graph unfreezes, continues from frozen point
- **LOOP** — Tools → Agent → Tools → Agent (repeats until final answer)

---

## Parallel Node Execution (`3_parallel_nodes.py`)

### The Problem with Sequential Multi-Stock Research
When comparing two stocks sequentially, every search waits for the previous one to finish:
```
Search Apple     → 2.5s wait
Search Microsoft → 2.2s wait
Total            → 4.7s
```

### The Parallel Solution
LangGraph supports **fan-out** — one node connects to multiple nodes that fire simultaneously:
```
Search Apple    ─┐
                 ├── both fire at the same time
Search Microsoft─┘
Total           → 2.5s  (just the slower one, not the sum)
```

### How It Works — Fan-out / Fan-in Pattern

```
                      ┌─────────────┐
                      │  __start__  │
                      └──────┬──────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │   parse_question   │
                  │ (extract 2 stocks) │
                  └────┬──────────┬───┘
                       │          │
               FAN-OUT: both fire simultaneously
                       │          │
                       ▼          ▼
             ┌──────────┐    ┌──────────┐
             │ search_  │    │ search_  │
             │ stock_1  │    │ stock_2  │
             └────┬─────┘    └────┬─────┘
                  │               │
              FAN-IN: waits for BOTH to complete
                       │          │
                       └────┬─────┘
                            ▼
                  ┌──────────────────────┐
                  │  generate_comparison │
                  │  (has both results)  │
                  └──────────┬───────────┘
                             │
                      ┌──────▼──────┐
                      │   __end__   │
                      └─────────────┘
```

### Key Concepts Unique to Parallel Execution

| Concept | What it does |
|---------|-------------|
| `add_edge(node, node_a)` + `add_edge(node, node_b)` | Fan-out — fires both nodes simultaneously |
| `Annotated[list, operator.add]` | Safely merges results from parallel nodes without overwriting |
| Automatic fan-in | LangGraph waits for ALL branches before the next node — no explicit barrier needed |

### Real Timing Output
```
[NODE: search_stock_1] ⚡ Starting search for 'Apple' (PARALLEL)
[NODE: search_stock_2] ⚡ Starting search for 'Microsoft' (PARALLEL)  ← fired together
[NODE: search_stock_2] ✅ Done in 2.2s
[NODE: search_stock_1] ✅ Done in 2.5s
Total wall-clock time: 3.5s   ← not 4.7s (sequential sum)
```

---

## Subgraphs (`4_subgraphs.py`)

### What is a Subgraph?
A subgraph is a **complete LangGraph graph** (its own State, nodes, edges) used as a **single node** inside a parent graph. Think of it like functions in programming — modular, reusable, isolated.

### Use Case — Query Router
The parent graph classifies the query and routes to a specialized subgraph:
- **News query** → News Subgraph (search → sentiment → summary)
- **Price query** → Price Subgraph (search → extract metrics → report)

### Graph Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  PARENT GRAPH                                                │
│                                                              │
│   ┌──────────┐     ┌──────────────────┐    ┌─────────────┐  │
│   │ __start__│────►│ classify_query   │    │   __end__   │  │
│   └──────────┘     │ (news or price?) │    └──────▲──────┘  │
│                    └────────┬─────────┘           │         │
│                             │                     │         │
│               query_type?   │                     │         │
│                             │                     │         │
│               "news" ───────┼──►┌────────────────┐│         │
│                             │   │  NEWS SUBGRAPH ├┘         │
│                             │   │  (own 3 nodes) │          │
│                             │   └────────────────┘          │
│               "price"───────┼──►┌────────────────┐          │
│                             │   │ PRICE SUBGRAPH ├──────────┘
│                             │   │  (own 3 nodes) │          │
│                             │   └────────────────┘          │
└──────────────────────────────────────────────────────────────┘

Inside each subgraph (their own isolated graphs):

NEWS SUBGRAPH:                    PRICE SUBGRAPH:
start                             start
  │                                 │
  ▼                                 ▼
search_news                       search_price
  │                                 │
  ▼                                 ▼
analyze_sentiment                 extract_metrics
  │                                 │
  ▼                                 ▼
write_news_summary                write_price_report
  │                                 │
  ▼                                 ▼
end                               end
```

### How State Flows Between Parent and Subgraph

```
Parent State:   { question, query_type, final_answer }
                      ↓                      ↑
              (overlapping key)       (overlapping key)
                      ↓                      ↑
Subgraph State: { question,  <internal keys...>,  final_answer }
                              ↑
                   ISOLATED — parent never sees these
```

Only **overlapping keys** are the communication bridge. Internal keys (`raw_news`, `sentiment`, `raw_prices`, `key_metrics`) stay isolated inside the subgraph and never appear in the parent state.

### How Conditional Routing Works
```python
parent_builder.add_node("news",  news_subgraph)   # node named "news"
parent_builder.add_node("price", price_subgraph)  # node named "price"
parent_builder.add_conditional_edges("classify_query", route_to_subgraph)
```
LangGraph auto-builds `{"news": "news", "price": "price"}` by scanning which nodes `classify_query` connects to. The routing function's return value is matched against this map. Node names **must match** the return values, or you pass an explicit map as the 3rd argument:
```python
add_conditional_edges("classify_query", route_fn, {"news": "news_subgraph_node", ...})
```

### Key Concepts

| Concept | What it does |
|---------|-------------|
| `compiled_subgraph` as a node | `add_node("news", news_subgraph)` — entire subgraph = single node in parent |
| Overlapping state keys | Only shared keys flow between parent ↔ subgraph |
| State isolation | Internal keys stay inside subgraph, never pollute parent |
| Routing by return value | Function returns node name string → LangGraph routes to that node |

### Real Output
```
[PARENT GRAPH] Classified as: 'news' → routing to news subgraph

  [NEWS SUBGRAPH] Node 1/3: Searching news...
  [NEWS SUBGRAPH] Node 2/3: Analyzing sentiment... → Neutral
  [NEWS SUBGRAPH] Node 3/3: Writing final summary...

[NOTE] Parent state has: ['question', 'query_type', 'final_answer']
[NOTE] Subgraph-internal keys (raw_news, sentiment) are NOT in parent state!
```

---

## Multi-Agent System (`5_multi_agent.py`)

### Why This Is Different from Subgraphs

| | `4_subgraphs.py` — Modular Pipeline | `5_multi_agent.py` — True Multi-Agent |
|---|---|---|
| Routing | Decided ONCE upfront by `classify_query` | Supervisor RE-EVALUATES after every agent |
| Agent awareness | Subgraphs are isolated — don't see each other | All agents share state — risk agent reads news + price reports |
| LLM per node | One shared LLM | Each agent has its own LLM + system prompt |
| Flow | Fixed sequence inside subgraph | Dynamic — supervisor decides order at runtime |
| General knowledge | Always runs subgraph nodes | Supervisor answers directly, zero agents called |

### The Supervisor Pattern

Every agent reports **back to the supervisor** after completing. The supervisor sees the full picture and decides who to call next. This loop continues until the supervisor says `FINISH`.

```
START
  │
  ▼
SUPERVISOR (LLM) ◄──────────────────────────────────────┐
  │                                                       │
  │  Decides who to call next:                           │
  │                                                       │
  ├─► NEWS AGENT (own LLM + search) ────────────────────►│
  │      "Here's what I found about news..."             │
  │                                                       │
  ├─► PRICE AGENT (own LLM + search) ───────────────────►│
  │      "Here's the price data I found..."              │
  │                                                       │
  ├─► RISK AGENT (own LLM, reads team's output) ────────►│
  │      "Based on news + price, here's my risk view..." │
  │                                                       │
  └─► FINISH ──► FINAL ANSWER NODE ──► END               │
                                                          │
          ◄──── Supervisor sees ALL outputs, re-evaluates┘
```

### General Knowledge Shortcut

A key improvement: the supervisor can say `FINISH` **before calling any agent** when the question is general knowledge — definitions, concepts, how things work. No live data is needed for these.

```
"What is a stock?"
  │
  ▼
SUPERVISOR (first evaluation)
  → "This is general knowledge. I can answer directly."
  → Decision: FINISH  (zero agents called)
  │
  ▼
FINAL ANSWER NODE
  → Supervisor answers from its own training knowledge
  → [AGENTS CALLED]: none

"Should I invest in Tesla?"
  │
  ▼
SUPERVISOR → news_agent → SUPERVISOR → price_agent → SUPERVISOR → risk_agent → SUPERVISOR → FINISH
```

### Key Multi-Agent Properties Demonstrated

| Property | How It Works |
|---|---|
| **Supervisor re-evaluates** | After each agent, supervisor runs again and reads all accumulated reports |
| **Own LLM per agent** | `supervisor_llm`, `news_llm`, `price_llm`, `risk_llm` — each with its own system prompt |
| **Shared memory** | `agent_outputs: Annotated[list, operator.add]` — every agent appends, all agents can read |
| **Dynamic ordering** | Supervisor picks the next agent at runtime based on what's still missing |
| **Agent collaboration** | `risk_agent` reads `news_agent` + `price_agent` reports before assessing risk |
| **Loop guard** | `called_agents` list passed to supervisor — it never calls the same agent twice |
| **General knowledge path** | Supervisor answers directly when no live data is needed — zero agent calls wasted |

### AGENT_DESCRIPTIONS — Single Source of Truth

```python
AGENT_DESCRIPTIONS = {
    "news_agent":  "searches for latest news, announcements, and events about a stock",
    "price_agent": "searches for stock price, PE ratio, market cap, and financial metrics",
    "risk_agent":  "assesses investment risk — requires BOTH news AND price data to work properly",
    "FINISH":      "no agent needed — question is general knowledge the supervisor can answer directly",
}
```

This dict is injected dynamically into the supervisor prompt. Adding a new agent = one entry here, prompt updates automatically. No duplication between docstrings and prompts.

### Real Output Comparison

**General knowledge question:**
```
[SUPERVISOR] Re-evaluating... (0 agents called so far: [])
[SUPERVISOR] Decision: → FINISH
[SUPERVISOR] General knowledge question — answering directly (no agents needed)...
[AGENTS CALLED]: none — supervisor answered directly from general knowledge
```

**Stock research question:**
```
[SUPERVISOR] Re-evaluating... (0 agents called so far: [])
[SUPERVISOR] Decision: → news_agent
[NEWS AGENT] Activated — researching news independently...
[SUPERVISOR] Re-evaluating... (1 agents called so far: ['news_agent'])
[SUPERVISOR] Decision: → price_agent
[PRICE AGENT] Activated — researching price data independently...
[SUPERVISOR] Re-evaluating... (2 agents called so far: ['news_agent', 'price_agent'])
[SUPERVISOR] Decision: → risk_agent
[RISK AGENT] Activated — reading team reports and assessing risk...
[SUPERVISOR] Re-evaluating... (3 agents called so far: [...])
[SUPERVISOR] Decision: → FINISH
[AGENTS CALLED]: ['news_agent', 'price_agent', 'risk_agent']
```

---

## Project Structure

```
financial-research-assistant/
├── 1_langchain_approach.py   # Standard LangChain — demonstrates all 5 limitations
├── 2_langgraph_approach.py   # LangGraph — solves all 5 problems with Human-in-the-Loop
├── 3_parallel_nodes.py       # LangGraph — parallel node execution for multi-stock research
├── 4_subgraphs.py            # LangGraph — subgraphs for modular, isolated query routing
├── 5_multi_agent.py          # LangGraph — true multi-agent system with supervisor pattern
├── graph_diagram.png         # Visual graph architecture diagram
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes .env, venv, __pycache__
└── README.md                 # This file
```

## Setup & Run

```bash
# Clone the repo
git clone https://github.com/Saivishaltirumala/financial-research-assistant.git
cd financial-research-assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API keys
echo 'GROQ_API_KEY="your_groq_api_key"' > .env

# Run LangChain version (see the problems)
python 1_langchain_approach.py

# Run LangGraph version (see the solutions)
python 2_langgraph_approach.py

# Run parallel nodes demo
python 3_parallel_nodes.py

# Run subgraphs demo
python 4_subgraphs.py

# Run multi-agent system demo
python 5_multi_agent.py
```

## Try These Queries

### On LangChain (`1_langchain_approach.py`) — See the Problems

| # | Query | What to Notice |
|---|-------|----------------|
| 1 | "What is a stock?" | Searches the web unnecessarily — it already knows this |
| 2 | "What's the latest news on Tesla?" | Works, but watch token count grow |
| 3 | "Based on that, is Tesla a good buy?" | Memory works, but prompt is getting bloated |
| 4-10 | Keep asking... | Watch `Estimated prompt size` climb toward crash |

### On LangGraph (`2_langgraph_approach.py`) — See the Solutions

| # | Query | What to Notice |
|---|-------|----------------|
| 1 | "What is a P/E ratio?" | Answers directly — no wasteful search! |
| 2 | "What's the latest news on Tesla?" | Decides to search, gets fresh data |
| 3 | "Based on that, is Tesla a good buy?" | Remembers Tesla context (checkpointer memory) |
| 4 | "Show me stock performance" | Asks "Which stock?" via interrupt — Human-in-the-Loop! |
| 5 | Keep asking... | No prompt explosion — state managed by checkpointer |

### On Parallel Nodes (`3_parallel_nodes.py`) — See the Speedup

| # | Query | What to Notice |
|---|-------|----------------|
| 1 | "Compare Apple and Microsoft stocks" | Both searches fire simultaneously in logs |
| 2 | "Compare Tesla and Toyota" | Watch timing — total ≈ slower search, not the sum |
| 3 | "Compare Infosys and TCS" | `operator.add` merges both results into one list |

### On Subgraphs (`4_subgraphs.py`) — See Modular Routing

| # | Query | What to Notice |
|---|-------|----------------|
| 1 | "What is the latest news on Tesla?" | Routes to news subgraph — 3 internal nodes run |
| 2 | "What is Apple's current PE ratio?" | Routes to price subgraph — different internal flow |
| 3 | Check `[NOTE]` in output | Parent state has only 3 keys — internal keys are isolated |

### On Multi-Agent System (`5_multi_agent.py`) — See True Agent Collaboration

| # | Query | What to Notice |
|---|-------|----------------|
| 1 | "What is a stock?" | Supervisor says FINISH immediately — zero agents called, answers directly |
| 2 | "What is the difference between ETF and mutual fund?" | Same — general knowledge path, no live data wasted |
| 3 | "What is the latest news on Tesla?" | Supervisor calls only news_agent → FINISH |
| 4 | "What is Apple's PE ratio?" | Supervisor calls only price_agent → FINISH |
| 5 | "Should I invest in Microsoft stock?" | Supervisor calls news → price → risk → FINISH (order at runtime) |
| 6 | Watch `[SUPERVISOR] Re-evaluating...` logs | See supervisor re-evaluate after EVERY agent |
| 7 | Watch `[AGENTS CALLED]:` in output | Zero agents for general knowledge; specialists for stock questions |

## Tech Stack

- **LLM**: Llama 3.1 8B via Groq (free tier)
- **Search**: DuckDuckGo (no API key needed)
- **Frameworks**: LangChain (problem demo) + LangGraph (solution)
- **Language**: Python 3.x
