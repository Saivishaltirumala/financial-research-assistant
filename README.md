# Financial Research Assistant — Why LangChain Falls Short for Agentic AI

## What This Project Demonstrates

This project builds a **stock market chatbot** using standard LangChain to expose the fundamental limitations of linear chain-based architectures. It serves as a practical case study for understanding **why LangGraph (Agentic AI) was needed** as an evolution beyond traditional LangChain (GenAI).

The assistant uses **DuckDuckGo Search** to pull real-time stock news and prices, and **Groq's LLM API** (Llama 3.1) for generating responses.

## The Core Problem with Standard LangChain

Standard LangChain pipelines are **linear chains**: `Input → Step1 → Step2 → Output`. This creates four critical limitations when building real-world AI assistants:

### Problem 1 — No Decision-Making
The chain **always executes every step**, regardless of whether it's needed. Ask "What is a stock?" and it still searches the web — wasteful and slow. The LLM has zero control over the pipeline flow.

### Problem 2 — No Loops / Retry Logic
If the search results are poor, the chain **cannot go back and search again** with better terms. It's a one-shot pipeline. A human analyst would refine their search — LangChain can't.

### Problem 3 — Memory Causes Prompt Explosion
LangChain's `ConversationBufferMemory` (used in this project) dumps the **entire conversation history into every prompt**. Watch the token count grow in the logs:

```
Turn 1:  ~284 tokens  (just the question)
Turn 3:  ~887 tokens  (3 exchanges accumulated)
Turn 10: ~3000+ tokens (approaching context limit)
Turn 15: 💥 CRASH — exceeds model's 8K context window
```

### Problem 4 — No Tool Autonomy
The developer **hardcodes** when tools are called. The LLM cannot decide "I need to search for this" vs "I already know this." Even `bind_tools()` only generates intent — it doesn't execute tools or handle the agent loop.

### Problem 5 — bind_tools() Exists But Isn't Enough
LangChain does support `llm.bind_tools()`, but it only makes the LLM **express intent** to call a tool. You still have to manually execute the tool, feed results back, and call the LLM again — essentially re-inventing the orchestration that LangGraph provides natively.

## Project Structure

```
7-financial-research-assistant/
├── 1_langchain_approach.py   # Standard LangChain — demonstrates all limitations
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

# Run
python 1_langchain_approach.py
```

## Try These Queries (In Order)

| # | Query | What to Notice |
|---|-------|----------------|
| 1 | "What is a stock?" | Searches the web unnecessarily — it already knows this |
| 2 | "What's the latest news on Tesla?" | Works fine, but watch token count grow |
| 3 | "Based on that, is Tesla a good buy?" | Memory works, but prompt is getting bloated |
| 4-10 | Keep asking... | Watch `⚠️ Estimated prompt size` climb toward crash |

## Tech Stack

- **LLM**: Llama 3.1 8B via Groq (free tier)
- **Search**: DuckDuckGo (no API key needed)
- **Framework**: LangChain with ConversationBufferMemory
- **Language**: Python 3.x

## What's Next

A **LangGraph version** of this same assistant will be added to show how graph-based architecture solves all these problems with conditional routing, loops, tool autonomy, and proper state management.
