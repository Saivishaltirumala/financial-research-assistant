# 08 — A2A Protocol: Agent-to-Agent over HTTP

Same 3-agent system (news → price → risk), but each agent runs as an **independent HTTP microservice**. The orchestrator calls them over the network, not as Python functions.

## What is the A2A Protocol?

A2A (Agent-to-Agent) is Google's open protocol for agents in different processes, languages, and organizations to communicate. Think of it as the HTTP standard for multi-agent systems — any agent that speaks A2A can call any other.

| Concept | REST API | A2A Protocol |
|---|---|---|
| Service discovery | OpenAPI spec | Agent Card (`/.well-known/agent.json`) |
| Request | `POST /resource` | `POST /tasks` |
| Payload schema | Your schema | `{task_id, question, context}` |
| Response schema | Your schema | `{task_id, status, agent, report}` |

Compare to MCP (Anthropic's protocol):

| | MCP | A2A |
|---|---|---|
| **Connects** | Agent ↔ Tool | Agent ↔ Agent |
| **Direction** | Always agent calls tool | Peer-to-peer |
| **Examples** | File system, databases, APIs | Orchestrator calling specialist agents |
| **Designed by** | Anthropic | Google |

MCP and A2A are **complementary** — an agent can use MCP to call tools AND be called by an orchestrator via A2A.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │           ORCHESTRATOR (port: main)       │
                    │                                          │
                    │  LangGraph Supervisor Loop               │
                    │  ┌─────────────┐                         │
                    │  │ SUPERVISOR  │ (LLM decides WHO)       │
                    │  │    LLM      │                         │
                    │  └──────┬──────┘                         │
                    │         │ next_agent = "news_agent"      │
                    │         ▼                                │
                    │  AGENT_REGISTRY["news_agent"]            │
                    │  → "http://localhost:8001"  (WHERE)      │
                    └────────────────┬─────────────────────────┘
                                     │  httpx.post /tasks
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
┌──────────────┐            ┌──────────────┐            ┌──────────────┐
│  NEWS AGENT  │            │ PRICE AGENT  │            │  RISK AGENT  │
│  :8001       │            │  :8002       │            │  :8003       │
│              │            │              │            │              │
│ GET  /.well- │            │ GET  /.well- │            │ GET  /.well- │
│   known/     │            │   known/     │            │   known/     │
│   agent.json │            │   agent.json │            │   agent.json │
│              │            │              │            │              │
│ POST /tasks  │            │ POST /tasks  │            │ POST /tasks  │
│ → searches   │            │ → searches   │            │ → reads      │
│   web+LLM    │            │   web+LLM    │            │   context    │
└──────────────┘            └──────────────┘            └──────────────┘
```

**Two separate concerns — never confused:**

| | Supervisor LLM | AGENT_REGISTRY |
|---|---|---|
| **Decides** | WHO to call next | WHERE to send the HTTP request |
| **Input** | Question + accumulated reports | Agent name string |
| **Output** | "news_agent" / "risk_agent" / "FINISH" | URL string |
| **Mechanism** | LLM reasoning | Dictionary lookup |
| **Knows URLs?** | Never | Yes |

---

## Agent Card (A2A Discovery)

Every agent exposes `GET /.well-known/agent.json`:

```json
{
  "name":        "news_agent",
  "description": "searches for latest news, announcements, and events about a stock",
  "version":     "1.0.0",
  "url":         "http://localhost:8001",
  "endpoints":   { "tasks": "POST /tasks" }
}
```

The orchestrator fetches this **once at startup** to learn:
- `name` → key in `AGENT_REGISTRY` dict
- `description` → injected into supervisor LLM prompt (so LLM knows when to call each agent)
- `url` → where to send tasks

**Adding a new agent = add its URL to `KNOWN_AGENT_URLS`.** The orchestrator learns its capabilities automatically. No orchestrator code changes needed.

---

## How A2A Solves Shared State

Other frameworks use different mechanisms for agents to share information:

| Framework | Shared state mechanism |
|---|---|
| LangGraph | `TypedDict State` — in-process dict passed between nodes |
| CrewAI | `context=[]` on Task — framework forwards outputs |
| AutoGen | Conversation history — all agents see all prior messages |
| **A2A** | `context` field in `TaskRequest` — orchestrator explicitly forwards |

For risk_agent, the orchestrator passes all accumulated reports in the request:

```python
# orchestrator.py — risk_agent_node()
context = state.get("agent_outputs", [])   # ["NEWS REPORT...", "PRICE REPORT..."]
httpx.post("http://localhost:8003/tasks", json={
    "task_id":  "...",
    "question": question,
    "context":  context,    # ← risk agent reads these, no search needed
})
```

---

## How to Run

**Option A — single command (recommended):**

```bash
pip install -r requirements.txt
python run_all.py
```

`run_all.py` starts all 3 agent servers as background subprocesses, waits for them to be ready, then starts the orchestrator interactively. Press `Ctrl+C` to stop everything.

**Option B — 4 separate terminals (shows each agent's logs):**

```bash
# Terminal 1
python agents/news_agent.py

# Terminal 2
python agents/price_agent.py

# Terminal 3
python agents/risk_agent.py

# Terminal 4
python orchestrator.py
```

---

## File Structure

```
08_a2a/
├── agents/
│   ├── news_agent.py    # FastAPI server on :8001 — search + news LLM
│   ├── price_agent.py   # FastAPI server on :8002 — search + price LLM
│   └── risk_agent.py    # FastAPI server on :8003 — reads context, synthesizes risk
├── orchestrator.py      # LangGraph supervisor + httpx calls + Agent Card discovery
├── run_all.py           # Convenience script: starts all servers + orchestrator
├── requirements.txt
└── README.md
```

---

## vs Other Modules

| | multi_agent.py | 08_a2a |
|---|---|---|
| Agent location | Same Python process | Separate HTTP servers |
| Agent calls | Python function calls | `httpx.post()` |
| Agent discovery | Hardcoded `AGENT_DESCRIPTIONS` dict | Agent Cards at `/.well-known/agent.json` |
| Shared state | LangGraph `TypedDict State` | `context` field in `TaskRequest` |
| Language freedom | Python only | Any language that speaks HTTP |
| Fault isolation | One crash kills all agents | Each agent fails independently |
| Deployability | Single process | Independent containers/VMs |
| Complexity | Lower (no HTTP) | Higher (servers + discovery) |

**When to choose A2A over LangGraph multi-agent:**
- Agents are written in different languages
- Agents are maintained by different teams
- Agents need independent deployment and scaling
- Agents will be reused across multiple orchestrators

**When to stick with LangGraph multi-agent:**
- Everything is Python, same team, same repo
- Low latency is critical (no HTTP overhead)
- Simplicity matters more than distribution
