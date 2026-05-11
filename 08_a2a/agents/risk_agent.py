"""
===================================================================================
A2A AGENT SERVER — RISK AGENT  (runs on http://localhost:8003)
===================================================================================

This agent is the most important one for demonstrating A2A agent COLLABORATION.

In multi_agent.py, risk_agent_node() reads agent_outputs directly from shared
LangGraph state — it can do that because all agents live in the same Python process.

In A2A, there IS no shared state dict. Each agent is an isolated HTTP service.
So how does risk_agent get the news + price reports?

ANSWER: The orchestrator passes them in the TaskRequest.context field.

The flow for A2A agent collaboration:
    1. Orchestrator calls news_agent → gets NEWS REPORT
    2. Orchestrator calls price_agent → gets PRICE REPORT
    3. Orchestrator builds context = [NEWS REPORT, PRICE REPORT]
    4. Orchestrator calls risk_agent with context=["NEWS REPORT...", "PRICE REPORT..."]
    5. risk_agent reads context field → synthesizes risk assessment

This is A2A's answer to "shared state":
    - LangGraph solves it with TypedDict State (in-process dict passed between nodes)
    - CrewAI solves it with context=[] on Task (framework-managed passing)
    - AutoGen solves it with conversation history (every agent sees all prior messages)
    - A2A solves it with context in TaskRequest (orchestrator explicitly passes what's needed)

There's NO SEARCH in risk_agent — same design decision as multi_agent.py.
The agent reads its teammates' reports and synthesizes, rather than searching again.
This is intentional: searching for "Tesla risk" would give generic info. Reading
the specific NEWS + PRICE reports found by the other agents gives better synthesis.
===================================================================================
"""

import sys
import os

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from shared.config import GROQ_MODEL

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

app = FastAPI(title="Risk Agent", version="1.0.0")

# Lower temperature than news/price — risk assessments should be measured and consistent
llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.2)


# =================================================================================
# A2A DATA MODELS
# =================================================================================

class TaskRequest(BaseModel):
    task_id: str
    question: str
    # context is REQUIRED for risk_agent — it contains the news + price reports
    # The orchestrator always populates this before calling risk_agent.
    # Default [] so the API doesn't break if called directly without context,
    # but the response will note that reports are missing.
    context: list[str] = []


class TaskResponse(BaseModel):
    task_id: str
    status: str
    agent: str
    report: str


# =================================================================================
# ENDPOINT 1: Agent Card
# =================================================================================
# The description says "requires BOTH news AND price data" — this is intentional.
# The orchestrator's supervisor LLM reads this description and learns the constraint:
# don't call risk_agent until news_agent AND price_agent have reported.
#
# This is A2A's self-describing capability: the agent announces its own preconditions.
# The orchestrator doesn't need hardcoded "call risk last" rules — it learns from the card.

@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "name":        "risk_agent",
        "description": "assesses investment risk — requires BOTH news AND price data to work properly",
        "version":     "1.0.0",
        "url":         "http://localhost:8003",
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
    Reads the news + price reports from req.context, synthesizes a risk assessment.

    Key difference from news/price agents:
        - Those agents SEARCH (go to the web to get data)
        - This agent SYNTHESIZES (reads what other agents found, reasons about it)

    This mirrors the real-world split between data-gathering and analysis agents.
    In a real financial system, you might have:
        - 10 data agents (price, news, earnings, sentiment, SEC filings...)
        - 1 synthesis agent that reads all their outputs and makes recommendations
    """
    print(f"\n[RISK AGENT] Received task {req.task_id}: {req.question}")
    print(f"[RISK AGENT] Context contains {len(req.context)} prior report(s)")

    try:
        if not req.context:
            # Called without context — can still try but quality will be low
            # The supervisor SHOULD prevent this, but we handle it gracefully
            team_reports = "No prior agent reports provided. Risk assessment will be limited."
            print(f"[RISK AGENT] WARNING: No context received. Supervisor may have called out of order.")
        else:
            # This is the A2A equivalent of reading shared LangGraph state
            # The orchestrator serialized the news + price reports into this list
            team_reports = "\n\n".join(req.context)
            print(f"[RISK AGENT] Reading team reports and synthesizing risk assessment...")

        response = llm.invoke([
            SystemMessage(content="""You are a financial risk assessment specialist.
Based on the news and price reports from your team members, assess the investment risk.
Format:
RISK ASSESSMENT:
- Risk Level: (Low/Medium/High)
- Key risk factors: (2-3 bullet points)
- Key positive factors: (2-3 bullet points)
- Overall recommendation: (what type of investor this suits)
- Disclaimer: (always include past performance disclaimer)"""),
            HumanMessage(content=f"""Original Question: {req.question}

Team Reports:
{team_reports}

Based on the above team reports, assess the investment risk.""")
        ])

        report = f"[RISK AGENT REPORT]\n{response.content}"
        print(f"[RISK AGENT] Risk assessment complete. Sending back to orchestrator.")

        return TaskResponse(
            task_id=req.task_id,
            status="completed",
            agent="risk_agent",
            report=report
        )

    except Exception as e:
        print(f"[RISK AGENT] Error: {e}")
        return TaskResponse(
            task_id=req.task_id,
            status="failed",
            agent="risk_agent",
            report=f"[RISK AGENT ERROR] {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
