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
```

| Module | Framework | Key Concept | Files |
|---|---|---|---|
| [`01_langchain`](./01_langchain/) | LangChain | Linear pipeline limitations | `main.py` |
| [`02_langgraph`](./02_langgraph/) | LangGraph | Graphs, HitL, parallel, multi-agent | `basic_agent.py` · `parallel_nodes.py` · `subgraphs.py` · `multi_agent.py` |
| [`03_crewai`](./03_crewai/) | CrewAI | Declarative agents and tasks | `main.py` |
| [`04_autogen`](./04_autogen/) | AutoGen | Conversational agents | `main.py` |

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
```

---

## Framework Comparison

| | LangChain | LangGraph | CrewAI | AutoGen |
|---|---|---|---|---|
| **Mental model** | Chain | Graph | Crew | Group chat |
| **Routing** | Fixed pipeline | Conditional edges | Process mode | Selector LLM |
| **Shared memory** | ConversationBuffer (grows unbounded) | TypedDict State | `context=[]` on Task | Conversation history |
| **Human-in-the-Loop** | ❌ | ✅ `interrupt()` + resume | ❌ | ✅ UserProxyAgent |
| **Parallel execution** | ❌ | ✅ Fan-out / fan-in | ❌ | ❌ |
| **Code execution** | ❌ | ❌ | ❌ | ✅ Built-in |
| **Async required** | ❌ | ❌ | ❌ | ✅ |
| **Lines of code** | ~100 | ~300 | ~150 | ~150 |

**One-line summary of each:**
- **LangChain** → fixed pipeline, you control every step
- **LangGraph** → you draw the graph, full control over flow
- **CrewAI** → describe WHO and WHAT, CrewAI figures out HOW
- **AutoGen** → put agents in a room and let them talk it out

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
```

---

## Tech Stack

- **LLM** — Llama 3.1 8B via [Groq](https://console.groq.com) (free tier)
- **Search** — DuckDuckGo (no API key needed)
- **Frameworks** — LangChain · LangGraph · CrewAI · AutoGen
- **Language** — Python 3.10+
