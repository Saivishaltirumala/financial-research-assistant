# 02 — LangGraph: The Solution

Four progressively complex examples showing how LangGraph solves every LangChain limitation — and goes further with parallel execution, subgraphs, and true multi-agent collaboration.

## Files (run in order)

| File | Concept | What it demonstrates |
|---|---|---|
| `basic_agent.py` | Graph + HitL | Conditional routing, loops, MemorySaver, `interrupt()` / `Command(resume=)` |
| `parallel_nodes.py` | Fan-out / Fan-in | Two searches fire simultaneously; `operator.add` merges results safely |
| `subgraphs.py` | Modular graphs | Compiled subgraphs used as nodes; overlapping state keys as communication bridge |
| `multi_agent.py` | Supervisor pattern | Supervisor re-evaluates after every agent; general-knowledge shortcut |

## How to Run

```bash
pip install -r requirements.txt

python basic_agent.py       # Start here — covers core LangGraph concepts
python parallel_nodes.py    # See parallel speedup with timing logs
python subgraphs.py         # See modular routing with state isolation
python multi_agent.py       # See full supervisor-based multi-agent system
```

## Key Concepts

**State** — `TypedDict` that all nodes read/write. `Annotated[list, add_messages]` auto-appends instead of overwriting.

**Conditional edges** — `tools_condition` routes to tool execution or END based on LLM's decision.

**`interrupt()` + `Command(resume=)`** — graph freezes at a point, waits for user input, resumes from the exact frozen state. Unique to LangGraph.

**`operator.add`** — safely merges writes from parallel nodes without overwriting.

**Subgraphs** — compiled `StateGraph` used as a single node in a parent graph. Only overlapping state keys flow between parent ↔ subgraph.

**Supervisor pattern** — supervisor LLM re-evaluates after every agent. `called_agents` list prevents loops. General-knowledge questions get `FINISH` immediately.

## Suggested Queries

```
basic_agent.py:    "What is a P/E ratio?"         ← answers directly, no search
                   "Latest Tesla news?"             ← decides to search
                   "Show me stock performance"      ← triggers interrupt, asks "Which stock?"

parallel_nodes.py: "Compare Apple and Microsoft"   ← watch both searches fire simultaneously

subgraphs.py:      "Latest news on Tesla?"          ← routes to news subgraph
                   "Apple PE ratio?"                ← routes to price subgraph

multi_agent.py:    "What is a stock?"               ← FINISH immediately, no agents called
                   "Should I invest in Tesla?"      ← news → price → risk → FINISH
```
