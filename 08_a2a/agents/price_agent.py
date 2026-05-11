"""
===================================================================================
A2A AGENT SERVER — PRICE AGENT  (runs on http://localhost:8002)
===================================================================================

Same A2A protocol as news_agent.py (same two endpoints).
Only the port, description, and LLM system prompt differ.

Key design point — why separate processes for each agent?
    In multi_agent.py, all agents share a single Python process and import from
    each other. That's fine for a demo but in production you want isolation:

    - FAULT ISOLATION: if price_agent crashes, news_agent keeps running
    - INDEPENDENT SCALING: price lookups are slow → scale price_agent to 3 replicas
      without touching news_agent
    - INDEPENDENT DEPLOYMENT: update the price LLM prompt without restarting everything
    - LANGUAGE FREEDOM: this is Python/FastAPI but another team could write a
      Node.js price agent that speaks the same protocol

This file is intentionally parallel in structure to news_agent.py so the
difference between agents is obvious at a glance.
===================================================================================
"""

import sys
import os

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.tools import ddg_search
from shared.config import GROQ_MODEL

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="Price Agent", version="1.0.0")

llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.3)


# =================================================================================
# A2A DATA MODELS
# =================================================================================

class TaskRequest(BaseModel):
    task_id: str
    question: str
    context: list[str] = []    # price_agent doesn't use prior context either


class TaskResponse(BaseModel):
    task_id: str
    status: str
    agent: str
    report: str


# =================================================================================
# ENDPOINT 1: Agent Card
# =================================================================================
# description here is what the orchestrator reads at startup and injects into the
# supervisor LLM prompt — so the LLM learns "price_agent looks up stock prices".
# Change this description and the supervisor's routing behavior changes automatically.

@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name":        "price_agent",
        "description": "searches for stock price, PE ratio, market cap, and financial metrics",
        "version":     "1.0.0",
        "url":         "http://localhost:8002",
        "endpoints": {
            "tasks": "POST /tasks"
        },
        "capabilities": {
            "input":  ["question", "context"],
            "output": ["report"]
        }
    }


# =================================================================================
# ENDPOINT 2: POST /tasks
# =================================================================================

@app.post("/tasks", response_model=TaskResponse)
def run_task(req: TaskRequest):
    """
    Searches for price and valuation data, returns a structured price report.
    Identical flow to news_agent but with a different search query and system prompt.
    """
    print(f"\n[PRICE AGENT] Received task {req.task_id}: {req.question}")

    try:
        # Search query targets financial metrics — different from news search
        search_results = ddg_search.invoke(f"{req.question} stock price PE ratio valuation")
        print(f"[PRICE AGENT] Search done. Generating report...")

        response = llm.invoke([
            SystemMessage(content="""You are a financial metrics specialist.
Analyze the search results and write a concise structured report.
Format:
PRICE REPORT:
- Current Price: (value or N/A)
- P/E Ratio: (value or N/A)
- Market Cap: (value or N/A)
- Recent % Change: (value or N/A)
- Valuation assessment: (Overvalued/Undervalued/Fair)"""),
            HumanMessage(content=f"Question: {req.question}\n\nSearch Results: {search_results}")
        ])

        report = f"[PRICE AGENT REPORT]\n{response.content}"
        print(f"[PRICE AGENT] Report ready. Sending back to orchestrator.")

        return TaskResponse(
            task_id=req.task_id,
            status="completed",
            agent="price_agent",
            report=report
        )

    except Exception as e:
        print(f"[PRICE AGENT] Error: {e}")
        return TaskResponse(
            task_id=req.task_id,
            status="failed",
            agent="price_agent",
            report=f"[PRICE AGENT ERROR] {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
