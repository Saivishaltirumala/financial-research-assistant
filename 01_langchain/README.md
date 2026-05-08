# 01 — LangChain: The Problem

Standard LangChain builds linear pipelines. This module demonstrates the **5 fundamental limitations** that motivated LangGraph's creation — by running them intentionally.

## The 5 Problems

| # | Problem | What you'll see |
|---|---|---|
| 1 | No decision-making | "What is a stock?" still triggers a web search |
| 2 | No loops / retry | One-shot pipeline — bad results can't be refined |
| 3 | Prompt explosion | Token count printed each turn — watch it grow toward crash |
| 4 | No tool autonomy | Developer hardcodes `search_tool.invoke()` — LLM has no say |
| 5 | No Human-in-the-Loop | Chain runs to completion or fails; cannot pause and ask |

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Suggested Query Order

Run these in sequence to see each problem in action:

1. `"What is a stock?"` — unnecessary search (Problem 1)
2. `"Latest Tesla news?"` — watch token count (Problem 3)
3. `"Based on that, should I buy?"` — memory works but prompt is bloated
4. Keep asking — watch token count climb until it crashes

## What's Next

→ [`02_langgraph/basic_agent.py`](../02_langgraph/) solves all 5 problems.
