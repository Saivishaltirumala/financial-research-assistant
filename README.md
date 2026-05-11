# Financial Research Assistant

> Building the same AI stock research assistant **4 different ways** — to understand when to use LangChain, LangGraph, CrewAI, and AutoGen.

Each module uses **DuckDuckGo** for real-time search and **Groq (Llama 3.1 8B)** as the LLM.

---

## Learning Path

```
01_langchain/    → Standard LangChain — see the 5 problems with linear pipelines
      ↓
02_langgraph/    → LangGraph — graph-based solution (4 progressive examples)
      ↓
03_crewai/       → CrewAI — same result, declarative API, half the code
      ↓
04_autogen/      → AutoGen — same result, conversational agents
      ↓
08_a2a/          → A2A Protocol — same agents, now independent HTTP microservices
```

| Module | Framework | Key Concept | Files |
|---|---|---|---|
| [`01_langchain`](./01_langchain/) | LangChain | Linear pipeline limitations | `main.py` |
| [`02_langgraph`](./02_langgraph/) | LangGraph | Graphs, HitL, parallel, multi-agent | `basic_agent.py` · `parallel_nodes.py` · `subgraphs.py` · `multi_agent.py` |
| [`03_crewai`](./03_crewai/) | CrewAI | Declarative agents and tasks | `main.py` |
| [`04_autogen`](./04_autogen/) | AutoGen | Conversational agents | `main.py` |
| [`08_a2a`](./08_a2a/) | A2A Protocol | Agent-to-Agent over HTTP | `agents/` · `orchestrator.py` · `run_all.py` |

---

## Quick Start

```bash
git clone https://github.com/Saivishaltirumala/financial-research-assistant.git
cd financial-research-assistant

python -m venv venv && source venv/bin/activate

cp .env.example .env          # add your GROQ_API_KEY

# Run any module independently
pip install -r 02_langgraph/requirements.txt
python 02_langgraph/multi_agent.py

# Run the A2A module (starts 3 HTTP servers + orchestrator)
pip install -r 08_a2a/requirements.txt
python 08_a2a/run_all.py
```

---

## Framework Comparison

| | LangChain | LangGraph | CrewAI | AutoGen | A2A |
|---|---|---|---|---|---|
| **Mental model** | Chain | Graph | Crew | Group chat | HTTP microservices |
| **Routing** | Fixed pipeline | Conditional edges | Process mode | Selector LLM | Supervisor LLM + registry |
| **Shared memory** | ConversationBuffer (grows unbounded) | TypedDict State | `context=[]` on Task | Conversation history | `context` in TaskRequest |
| **Agent discovery** | N/A | Hardcoded nodes | Hardcoded agents | Hardcoded agents | Agent Cards (`/.well-known/`) |
| **Human-in-the-Loop** | ❌ | ✅ `interrupt()` + resume | ❌ | ✅ UserProxyAgent | ✅ Natural (HTTP boundary) |
| **Parallel execution** | ❌ | ✅ Fan-out / fan-in | ❌ | ❌ | ✅ (concurrent HTTP) |
| **Code execution** | ❌ | ❌ | ❌ | ✅ Built-in | ❌ |
| **Async required** | ❌ | ❌ | ❌ | ✅ | ❌ (sync HTTP) |
| **Language freedom** | Python only | Python only | Python only | Python only | Any language |
| **Lines of code** | ~100 | ~300 | ~150 | ~150 | ~400 (3 servers + orch) |

**One-line summary of each:**
- **LangChain** → fixed pipeline, you control every step
- **LangGraph** → you draw the graph, full control over flow
- **CrewAI** → describe WHO and WHAT, CrewAI figures out HOW
- **AutoGen** → put agents in a room and let them talk it out
- **A2A** → agents are independent HTTP services, orchestrator discovers and calls them at runtime

---

## Shared Utilities

```
shared/
├── tools.py    # DuckDuckGo search — one implementation, imported by all modules
└── config.py   # Groq model name, API key, base URL — change model in one place
```

---

## Setup

```bash
# Copy and fill in your API key
cp .env.example .env

# Each module has its own requirements — install only what you need
pip install -r 01_langchain/requirements.txt   # LangChain only
pip install -r 02_langgraph/requirements.txt   # LangGraph only
pip install -r 03_crewai/requirements.txt      # CrewAI only
pip install -r 04_autogen/requirements.txt     # AutoGen only
pip install -r 08_a2a/requirements.txt         # A2A (FastAPI + httpx + LangGraph)
```

---

## Tech Stack

- **LLM** — Llama 3.1 8B via [Groq](https://console.groq.com) (free tier)
- **Search** — DuckDuckGo (no API key needed)
- **Frameworks** — LangChain · LangGraph · CrewAI · AutoGen
- **Language** — Python 3.10+
