"""
===================================================================================
FILE 4: SUBGRAPHS IN LANGGRAPH
===================================================================================

USE CASE: Query Router — Dispatch to specialized subgraphs based on intent

WHAT IS A SUBGRAPH?
    A subgraph is a COMPLETE LangGraph graph (its own State, nodes, edges)
    that is used as a single NODE inside a parent graph.

    Think of it like functions in programming:
    - Without subgraphs: one giant graph with all nodes mixed together
    - With subgraphs:    modular graphs, each responsible for one task,
                         plugged into a parent that orchestrates them

WHY SUBGRAPHS?
    1. MODULARITY: Each subgraph encapsulates its own logic independently
    2. REUSABILITY: A subgraph can be plugged into multiple parent graphs
    3. CLARITY: Complex workflows become readable — parent graph is high-level,
                subgraphs handle the detail
    4. ISOLATION: Each subgraph has its own internal state keys that don't
                  leak into the parent graph

HOW STATE FLOWS BETWEEN PARENT AND SUBGRAPH:
    The parent and subgraph communicate ONLY through OVERLAPPING STATE KEYS.

    Parent State:   { question, query_type, final_answer }
                          ↓              ↑
                    (overlapping)   (overlapping)
                          ↓              ↑
    Subgraph State: { question, <internal keys...>, final_answer }

    - `question`     : parent → subgraph  (subgraph receives the user's question)
    - `final_answer` : subgraph → parent  (subgraph's output flows back up)
    - Internal keys  : STAY INSIDE the subgraph, never visible to parent

THIS FILE'S GRAPH ARCHITECTURE:

    ┌──────────────────────────────────────────────────────────────┐
    │  PARENT GRAPH                                                │
    │                                                              │
    │   ┌─────────┐     ┌──────────────────┐    ┌─────────────┐  │
    │   │ __start__│────►│ classify_query   │    │   __end__   │  │
    │   └─────────┘     │ (news or price?)  │    └──────▲──────┘  │
    │                   └────────┬──────────┘           │         │
    │                            │                      │         │
    │              query_type?   │                      │         │
    │                            │                      │         │
    │              "news" ───────┼──►┌─────────────────┐│         │
    │                            │   │  NEWS SUBGRAPH  ├┘         │
    │                            │   │  (own 3 nodes)  │          │
    │                            │   └─────────────────┘          │
    │              "price"───────┼──►┌─────────────────┐          │
    │                            │   │  PRICE SUBGRAPH ├──────────┘
    │                            │   │  (own 3 nodes)  │          │
    │                            │   └─────────────────┘          │
    └──────────────────────────────────────────────────────────────┘

    Inside each subgraph (their own isolated graphs):

    NEWS SUBGRAPH:                    PRICE SUBGRAPH:
    start                             start
      │                                 │
      ▼                                 ▼
    search_news                       search_price
      │                                 │
      ▼                                 ▼
    analyze_sentiment                 extract_metrics
      │                                 │
      ▼                                 ▼
    write_news_summary                write_price_report
      │                                 │
      ▼                                 ▼
    end                               end
===================================================================================
"""

import operator
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.tools import ddg_search
from shared.config import GROQ_MODEL
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END

load_dotenv()

# Setup (covered in files 1-3, skipping comments)
llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.3)
search_tool = ddg_search


# =================================================================================
# STEP 1: Define ALL States — Parent + Both Subgraphs
# =================================================================================
# KEY INSIGHT: Overlapping keys = communication channel between parent and subgraph
# Non-overlapping keys = isolated, internal to that graph only

class MainState(TypedDict):
    """
    Parent graph's state. Only contains high-level fields.
    Internal details of news/price research are NOT here — they live
    inside each subgraph's state and never pollute the parent.
    """
    question:     str   # OVERLAPS with both subgraphs → flows DOWN into them
    query_type:   str   # Parent-only → routing decision, subgraphs don't see this
    final_answer: str   # OVERLAPS with both subgraphs → flows UP from them


class NewsSubgraphState(TypedDict):
    """
    News subgraph's own isolated state.
    `question` and `final_answer` overlap with MainState — that's the bridge.
    `raw_news` and `sentiment` are internal — parent never sees these.
    """
    question:     str   # Received FROM parent (overlapping key)
    raw_news:     str   # Internal only — raw DuckDuckGo results
    sentiment:    str   # Internal only — positive / negative / neutral
    final_answer: str   # Written BACK to parent (overlapping key)


class PriceSubgraphState(TypedDict):
    """
    Price subgraph's own isolated state.
    Same bridge pattern: `question` in, `final_answer` out.
    `raw_prices` and `key_metrics` are internal — parent never sees these.
    """
    question:     str   # Received FROM parent (overlapping key)
    raw_prices:   str   # Internal only — raw search results
    key_metrics:  str   # Internal only — extracted price/PE/volume data
    final_answer: str   # Written BACK to parent (overlapping key)


# =================================================================================
# STEP 2: Build the NEWS SUBGRAPH
# =================================================================================
# This is a complete, standalone graph with its own 3 nodes.
# It knows nothing about the parent graph — fully self-contained.

def search_news(state: NewsSubgraphState) -> dict:
    """
    SUBGRAPH NODE 1: Search for financial news about the stock.
    Stores raw results in `raw_news` — an internal key the parent never sees.
    """
    print(f"\n  [NEWS SUBGRAPH] Node 1/3: Searching news for: '{state['question']}'")
    results = search_tool.invoke(f"{state['question']} stock news latest")
    print(f"  [NEWS SUBGRAPH] Got {len(str(results))} chars of news data")
    return {"raw_news": str(results)}


def analyze_sentiment(state: NewsSubgraphState) -> dict:
    """
    SUBGRAPH NODE 2: Determine if the news sentiment is positive/negative/neutral.
    Uses `raw_news` (set by previous node) as input.
    This internal processing is invisible to the parent graph.
    """
    print(f"  [NEWS SUBGRAPH] Node 2/3: Analyzing sentiment...")
    response = llm.invoke([
        SystemMessage(content="Analyze the sentiment of this financial news. "
                              "Reply with exactly ONE word: Positive, Negative, or Neutral."),
        HumanMessage(content=state["raw_news"])
    ])
    sentiment = response.content.strip()
    print(f"  [NEWS SUBGRAPH] Sentiment detected: {sentiment}")
    return {"sentiment": sentiment}


def write_news_summary(state: NewsSubgraphState) -> dict:
    """
    SUBGRAPH NODE 3: Write the final news summary with sentiment context.
    This node writes to `final_answer` — the overlapping key that will
    flow back up to the parent graph when this subgraph completes.
    """
    print(f"  [NEWS SUBGRAPH] Node 3/3: Writing final summary...")
    response = llm.invoke([
        SystemMessage(content=f"""You are a financial news analyst.
Summarize the news and mention the overall sentiment is: {state['sentiment']}
Be concise — 3-4 bullet points max.
Add a disclaimer that this is based on recent news and may not reflect real-time data."""),
        HumanMessage(content=f"Question: {state['question']}\n\nNews Data:\n{state['raw_news']}")
    ])
    # Writing to `final_answer` — this overlapping key flows back to parent
    return {"final_answer": response.content}


# Assemble the News Subgraph
news_graph_builder = StateGraph(NewsSubgraphState)
news_graph_builder.add_node("search_news",       search_news)
news_graph_builder.add_node("analyze_sentiment",  analyze_sentiment)
news_graph_builder.add_node("write_news_summary", write_news_summary)

# Sequential flow inside the subgraph
news_graph_builder.add_edge(START,              "search_news")
news_graph_builder.add_edge("search_news",       "analyze_sentiment")
news_graph_builder.add_edge("analyze_sentiment", "write_news_summary")
news_graph_builder.add_edge("write_news_summary", END)

# Compile the subgraph — this turns it into a runnable that can be used as a NODE
news_subgraph = news_graph_builder.compile()


# =================================================================================
# STEP 3: Build the PRICE SUBGRAPH
# =================================================================================
# Completely independent from the news subgraph.
# Same interface (question in, final_answer out) but different internal logic.

def search_price(state: PriceSubgraphState) -> dict:
    """
    SUBGRAPH NODE 1: Search for price and financial metrics data.
    """
    print(f"\n  [PRICE SUBGRAPH] Node 1/3: Searching price data for: '{state['question']}'")
    results = search_tool.invoke(f"{state['question']} stock price PE ratio market cap")
    print(f"  [PRICE SUBGRAPH] Got {len(str(results))} chars of price data")
    return {"raw_prices": str(results)}


def extract_metrics(state: PriceSubgraphState) -> dict:
    """
    SUBGRAPH NODE 2: Extract specific metrics from the raw search results.
    Pulls out price, P/E ratio, market cap — isolates the numbers from the noise.
    """
    print(f"  [PRICE SUBGRAPH] Node 2/3: Extracting key metrics...")
    response = llm.invoke([
        SystemMessage(content="""Extract these specific metrics from the data (use N/A if not found):
- Current Price
- P/E Ratio
- Market Cap
- 52-week High/Low
- Recent % Change
Reply in a clean bullet-point list only."""),
        HumanMessage(content=state["raw_prices"])
    ])
    print(f"  [PRICE SUBGRAPH] Metrics extracted successfully")
    return {"key_metrics": response.content}


def write_price_report(state: PriceSubgraphState) -> dict:
    """
    SUBGRAPH NODE 3: Write the final price analysis report.
    Uses `key_metrics` (extracted by previous node) to generate a clean report.
    Writes to `final_answer` — the overlapping key that flows back to the parent.
    """
    print(f"  [PRICE SUBGRAPH] Node 3/3: Writing final price report...")
    response = llm.invoke([
        SystemMessage(content="""You are a financial analyst. Write a brief price analysis report.
Include the extracted metrics and what they suggest about the stock's valuation.
Always add a disclaimer: past performance does not guarantee future results."""),
        HumanMessage(content=f"Question: {state['question']}\n\nKey Metrics:\n{state['key_metrics']}")
    ])
    # Writing to `final_answer` — flows back to parent
    return {"final_answer": response.content}


# Assemble the Price Subgraph
price_graph_builder = StateGraph(PriceSubgraphState)
price_graph_builder.add_node("search_price",    search_price)
price_graph_builder.add_node("extract_metrics", extract_metrics)
price_graph_builder.add_node("write_price_report", write_price_report)

price_graph_builder.add_edge(START,             "search_price")
price_graph_builder.add_edge("search_price",    "extract_metrics")
price_graph_builder.add_edge("extract_metrics", "write_price_report")
price_graph_builder.add_edge("write_price_report", END)

# Compile the subgraph into a runnable
price_subgraph = price_graph_builder.compile()


# =================================================================================
# STEP 4: Build the PARENT GRAPH
# =================================================================================
# The parent graph is high-level — it only decides WHERE to route.
# It delegates all detailed work to the subgraphs.

def classify_query(state: MainState) -> dict:
    """
    PARENT NODE: Classify the user's question as 'news' or 'price'.
    This is the only real node in the parent graph — everything else
    is delegated to subgraphs.
    """
    print(f"\n[PARENT GRAPH] Classifying query: '{state['question']}'")
    response = llm.invoke([
        SystemMessage(content="""Classify this financial question into exactly ONE category:
- 'news'  : if asking about news, events, announcements, sentiment, what happened
- 'price' : if asking about stock price, PE ratio, valuation, metrics, performance numbers
Reply with ONLY the word: news or price"""),
        HumanMessage(content=state["question"])
    ])
    query_type = response.content.strip().lower()
    # Ensure it's one of our two valid types
    query_type = "news" if "news" in query_type else "price"
    print(f"[PARENT GRAPH] Classified as: '{query_type}' → routing to {query_type} subgraph")
    return {"query_type": query_type}


def route_to_subgraph(state: MainState) -> str:
    """
    ROUTING FUNCTION: Determines which subgraph to invoke next.

    HOW THE ROUTING ACTUALLY WORKS:
    When you call add_conditional_edges("classify_query", route_to_subgraph),
    LangGraph scans the graph for all nodes that "classify_query" has outgoing
    edges to — in our case "news" and "price" — and builds this internal map:
        { "news": "news", "price": "price" }
    (return value string → node name to jump to)

    So when this function returns "news", LangGraph looks up "news" in that map
    and routes execution to the node named "news" (which IS the news subgraph).

    This works because we DELIBERATELY named our nodes "news" and "price" to
    match exactly what this function returns. If node names were different
    (e.g., "news_analysis_subgraph"), you'd need an explicit map as 3rd arg:
        add_conditional_edges(
            "classify_query",
            route_to_subgraph,
            {
                "news":  "news_analysis_subgraph",   # return value → node name
                "price": "price_analysis_subgraph"
            }
        )
    Without that explicit map, LangGraph would fail to find the node and crash.
    """
    return state["query_type"]  # Returns "news" or "price" → matched to node name


# Assemble the Parent Graph
parent_builder = StateGraph(MainState)

# Add the classify node (a regular Python function node)
parent_builder.add_node("classify_query", classify_query)

# Add the SUBGRAPHS as NODES in the parent graph.
# This is the core of subgraphs: compiled subgraphs become nodes!
# The parent treats each entire subgraph (3 nodes, 2 edges) as a SINGLE node.
# Node names are "news" and "price" — must match what route_to_subgraph returns.
parent_builder.add_node("news",  news_subgraph)    # The entire news subgraph = one node
parent_builder.add_node("price", price_subgraph)   # The entire price subgraph = one node

# Sequential edge: START → classify
parent_builder.add_edge(START, "classify_query")

# Conditional edge: classify → news subgraph OR price subgraph
# LangGraph auto-builds { "news": "news", "price": "price" } because those are
# the only two nodes connected from "classify_query". The routing function's
# return value is matched against this map to find the next node to execute.
parent_builder.add_conditional_edges("classify_query", route_to_subgraph)

# Both subgraph nodes lead to END after completing
parent_builder.add_edge("news",  END)
parent_builder.add_edge("price", END)

# Compile the parent graph
parent_graph = parent_builder.compile()


# =================================================================================
# STEP 5: Run it
# =================================================================================
def ask(question: str):
    print("\n" + "=" * 60)
    print(f"  SUBGRAPH DEMO — LANGGRAPH")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print("-" * 60)

    result = parent_graph.invoke({
        "question":     question,
        "query_type":   "",
        "final_answer": ""
    })

    print(f"\n{'=' * 60}")
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["final_answer"])
    print(f"\n[NOTE] Parent state has: {list(result.keys())}")
    print(f"[NOTE] Subgraph-internal keys (raw_news, sentiment, raw_prices,")
    print(f"       key_metrics) are NOT in parent state — they stayed isolated!")


def main():
    print("\n" + "=" * 60)
    print("  LANGGRAPH — SUBGRAPHS DEMO")
    print("  (Modular, specialized subgraphs for news vs price queries)")
    print("=" * 60)
    print("\nType 'quit' to exit.\n")
    print("Try:")
    print("  News queries  → 'What is the latest news on Tesla?'")
    print("  Price queries → 'What is the current price of Apple stock?'\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            ask(question)
        except Exception as e:
            print(f"\nError: {e}\n")
        print()


if __name__ == "__main__":
    main()
