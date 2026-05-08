# 04 — AutoGen: Conversational Multi-Agent

Same 3-agent system, built with AutoGen's conversational model. Agents **talk to each other** until done. No task lists. No graph edges.

## How to Run

```bash
pip install -r requirements.txt
python main.py
# Choose mode: [1] RoundRobin  [2] Selector
```

## Two Modes

| Mode | How it works | LangGraph equivalent |
|---|---|---|
| RoundRobin | Agents take turns in fixed order | Sequential edges |
| Selector | Selector LLM picks who speaks next | Supervisor + conditional edges |

## Key Insight — Conversation = Shared Memory

Other frameworks need explicit wiring for agents to share information:
- **LangGraph**: `agent_outputs: Annotated[list, operator.add]` in State
- **CrewAI**: `context=[news_task, price_task]` on Task

In **AutoGen**, `risk_agent` just reads the messages above it in the conversation. Every agent sees the full chat history automatically — no wiring needed.

```
USER        → "Should I invest in Tesla?"
NEWS_AGENT  → [searches] → NEWS REPORT: ...
PRICE_AGENT → [searches] → PRICE REPORT: ...
RISK_AGENT  → [reads conversation] → RISK ASSESSMENT: ... TERMINATE
```

## Key Concepts

**`AssistantAgent`** — LLM agent with `name`, `system_message`, `tools`, `model_client`.

**`FunctionTool`** — wraps a plain Python function as a callable tool.

**`RoundRobinGroupChat`** — fixed turn order, `termination_condition` stops when done.

**`SelectorGroupChat`** — `selector_prompt` tells the LLM how to pick who speaks next.

**`TextMentionTermination("TERMINATE")`** — stops when `risk_agent` says "TERMINATE".

**`asyncio.run(team.run(...))`** — AutoGen 0.4+ is fully async; requires `asyncio.run()`.

## vs LangGraph / CrewAI

| | AutoGen | CrewAI | LangGraph |
|---|---|---|---|
| Shared memory | Conversation (automatic) | `context=[]` | TypedDict State |
| Human-in-the-Loop | ✅ Natural (UserProxyAgent) | ❌ | ✅ interrupt() |
| Code execution | ✅ Built-in | ❌ | ❌ |
| Async required | ✅ Yes | ❌ | ❌ |
| Structured output | ❌ Parse messages | ✅ expected_output | ✅ Typed state |

## Suggested Queries

```
"What is the latest news on Tesla?"    ← watch agents take turns
"Should I invest in Apple?"            ← risk_agent reads news+price from conversation
```
