# 03 — CrewAI: Declarative Multi-Agent

Same 3-agent system (news → price → risk), built with CrewAI's plain-English declarative API. ~150 lines vs ~300 lines for the equivalent LangGraph implementation.

## How to Run

```bash
pip install -r requirements.txt
python main.py
# Choose mode: [1] Sequential  [2] Hierarchical
```

## Two Modes

| Mode | How it works | LangGraph equivalent |
|---|---|---|
| Sequential | Tasks run in fixed order: news → price → risk | Hardcoded sequential edges |
| Hierarchical | Manager LLM delegates dynamically | Supervisor + conditional edges |

## Key Concepts

**Agent** — `Agent(role, goal, backstory, tools, llm)`. Plain-English description; CrewAI builds the system prompt for you.

**Task** — `Task(description, expected_output, agent)`. The `{question}` placeholder is filled at `kickoff()`.

**`context=[task1, task2]`** — feeds prior task outputs to the next agent automatically. No shared state dict needed. Equivalent to `state["agent_outputs"]` in LangGraph.

**`crew.kickoff(inputs={"question": "..."})`** — starts the whole system synchronously.

## vs LangGraph

| | CrewAI | LangGraph |
|---|---|---|
| Code | ~150 lines | ~300 lines |
| Agent definition | Plain English (role/goal/backstory) | Python function + SystemMessage |
| Shared memory | `context=[]` on Task | `Annotated[list, operator.add]` |
| Human-in-the-Loop | ❌ Not supported | ✅ `interrupt()` + `Command(resume=)` |
| General-knowledge skip | ❌ All tasks always run | ✅ Supervisor says FINISH immediately |
| Async | ❌ Synchronous | ❌ Synchronous |

## Suggested Queries

```
"What is the latest news on Tesla?"    ← all 3 tasks run (no skip — CrewAI limitation)
"Should I invest in Apple?"            ← news → price → risk, risk reads both via context=
```
