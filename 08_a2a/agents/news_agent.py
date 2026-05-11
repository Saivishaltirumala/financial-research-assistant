"""
===================================================================================
A2A AGENT SERVER — NEWS AGENT  (runs on http://localhost:8001)
===================================================================================

What this file is:
    An independent HTTP microservice that follows the A2A (Agent-to-Agent) protocol.
    It exposes two endpoints that ANY orchestrator (LangGraph, CrewAI, raw Python,
    a different company's system) can call without knowing how this agent works internally.

Why this matters over plain Python functions (like in multi_agent.py):
    In multi_agent.py, news_agent_node() is a Python function directly imported and
    called by the orchestrator. That means:
      - Orchestrator and agent MUST be in the same process
      - Orchestrator MUST be written in Python
      - You can't independently deploy, scale, or update the news agent
      - You can't reuse this agent in another project without copy-pasting code

    With A2A, this agent is a completely independent service:
      - Runs in its own process (can be on a different machine or cloud)
      - Speaks HTTP — any language, any framework can call it
      - Can be versioned, restarted, or replaced without touching the orchestrator
      - The orchestrator discovers its capabilities at runtime via the Agent Card

A2A PROTOCOL IMPLEMENTATION:
    GET  /.well-known/agent.json  → Agent Card (who am I, what can I do, how to reach me)
    POST /tasks                   → Submit a task (run my search + LLM, return the report)

HOW THE ORCHESTRATOR LEARNS ABOUT THIS AGENT:
    1. Orchestrator fetches GET /.well-known/agent.json
    2. Reads `description` → injects into supervisor LLM prompt
    3. Reads `name` + `url` → stores in AGENT_REGISTRY for routing
    The orchestrator never needs to know this file exists — it only knows the agent URL.
===================================================================================
"""

import sys
import os
import uuid

from fastapi import FastAPI
from pydantic import BaseModel

# ---------- LLM + search setup (same shared utilities as multi_agent.py) ----------
# We insert the parent directory so we can import from shared/ even though
# this file lives inside 08_a2a/agents/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.tools import ddg_search, search_web
from shared.config import GROQ_MODEL

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ---------- FastAPI app ----------
app = FastAPI(title="News Agent", version="1.0.0")

# One LLM instance for this agent — isolated from other agents (different process)
llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.3)


# =================================================================================
# A2A DATA MODELS — Pydantic enforces the schema for both request and response
# =================================================================================

class TaskRequest(BaseModel):
    """
    What the orchestrator sends when it wants this agent to do work.

    task_id : unique ID so the orchestrator can correlate responses
              (important when tasks are async — not used for sync here but good practice)
    question: the user's original question
    context : previous agent reports the orchestrator passes for context
              (news_agent ignores this — it only searches, reads nothing from prior agents)
    """
    task_id: str
    question: str
    context: list[str] = []     # news_agent doesn't need prior context, but accepts it


class TaskResponse(BaseModel):
    """
    What this agent sends back after completing the task.

    status: "completed" or "failed" — orchestrator checks this before reading report
    agent : which agent produced this (useful when orchestrator logs multiple responses)
    report: the actual content written to shared state in the orchestrator
    """
    task_id: str
    status: str
    agent: str
    report: str


# =================================================================================
# ENDPOINT 1: Agent Card — the A2A "business card"
# =================================================================================
# WHY /.well-known/?
#   The .well-known/ path is a web standard (RFC 5785) for machine-readable metadata.
#   Examples: /.well-known/openid-configuration, /.well-known/apple-app-site-association
#   A2A adopts this convention so orchestrators know exactly where to look.
#
# The orchestrator calls this ONCE at startup to learn:
#   - name        → key in AGENT_REGISTRY dict
#   - description → injected into supervisor LLM prompt (so LLM knows when to call us)
#   - url         → where to send POST /tasks requests
#
# This is SELF-DESCRIBING: the agent announces its own capabilities. No hardcoded
# descriptions in the orchestrator needed.

@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name":        "news_agent",
        "description": "searches for latest news, announcements, and events about a stock",
        "version":     "1.0.0",
        "url":         "http://localhost:8001",
        "endpoints": {
            "tasks": "POST /tasks"
        },
        "capabilities": {
            "input":  ["question", "context"],
            "output": ["report"]
        }
    }


# =================================================================================
# ENDPOINT 2: POST /tasks — do the actual work
# =================================================================================
# This is synchronous (returns result immediately) for simplicity.
# A production A2A implementation would return task_id immediately and let the
# orchestrator poll GET /tasks/{task_id} for the result (async pattern).
# We keep it sync here to stay focused on the agent collaboration concept.

@app.post("/tasks", response_model=TaskResponse)
def run_task(req: TaskRequest):
    """
    Receive a task from the orchestrator, run search + LLM, return the report.

    Flow:
        1. Search DuckDuckGo for news about the question
        2. Feed search results to LLM with a structured news reporter system prompt
        3. Return the LLM's report in a TaskResponse
    """
    print(f"\n[NEWS AGENT] Received task {req.task_id}: {req.question}")

    try:
        # Step 1: Search — same query pattern as news_agent_node() in multi_agent.py
        search_results = ddg_search.invoke(f"{req.question} stock news latest")
        print(f"[NEWS AGENT] Search done. Generating report...")

        # Step 2: LLM formats the raw search results into a structured report
        response = llm.invoke([
            SystemMessage(content="""You are a financial news specialist.
Analyze the search results and write a concise structured report.
Format:
NEWS REPORT:
- Key headlines: (2-3 bullet points)
- Overall news sentiment: (Positive/Negative/Neutral)
- Important events: (any earnings, launches, scandals, partnerships)"""),
            HumanMessage(content=f"Question: {req.question}\n\nSearch Results: {search_results}")
        ])

        report = f"[NEWS AGENT REPORT]\n{response.content}"
        print(f"[NEWS AGENT] Report ready. Sending back to orchestrator.")

        return TaskResponse(
            task_id=req.task_id,
            status="completed",
            agent="news_agent",
            report=report
        )

    except Exception as e:
        # Return a failed status so orchestrator can decide what to do
        # (in our orchestrator, a failed report still gets added to agent_outputs
        #  so the supervisor knows an attempt was made)
        print(f"[NEWS AGENT] Error: {e}")
        return TaskResponse(
            task_id=req.task_id,
            status="failed",
            agent="news_agent",
            report=f"[NEWS AGENT ERROR] {str(e)}"
        )


# =================================================================================
# Run as standalone server
# =================================================================================
# Normally started by run_all.py via subprocess, but you can also run directly:
#   python agents/news_agent.py

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
