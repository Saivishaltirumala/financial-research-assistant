"""
===================================================================================
FILE 3: PARALLEL NODE EXECUTION IN LANGGRAPH
===================================================================================

USE CASE: "Compare two stocks simultaneously"

The Problem with Sequential Search (what file 2 does for comparisons):
    User: "Compare Apple and Microsoft stocks"
    Step 1: Search Apple   → wait 2 seconds
    Step 2: Search Microsoft → wait 2 seconds
    Total: ~4 seconds (one after the other)

The Parallel Solution (what this file demonstrates):
    User: "Compare Apple and Microsoft stocks"
    Step 1: Search Apple   ─┐
                             ├─ BOTH fire at the same time → wait ~2 seconds
    Step 2: Search Microsoft─┘
    Total: ~2 seconds (simultaneously)

HOW PARALLEL NODES WORK IN LANGGRAPH:
    LangGraph supports "fan-out" — one node can connect to MULTIPLE nodes.
    When multiple nodes receive control at the same time, LangGraph runs them
    in parallel (using Python's async/threading under the hood).
    It then "fans-in" — waits for ALL parallel nodes to finish before the
    next node (aggregator) starts.

    The Graph looks like this:

                        ┌─────────────┐
                        │   __start__ │
                        └──────┬──────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   parse_question    │
                    │ (extract 2 stocks)  │
                    └────┬──────────┬─────┘
                         │          │
                    FAN-OUT: both fire simultaneously
                         │          │
                         ▼          ▼
               ┌──────────┐    ┌──────────┐
               │ search_  │    │ search_  │
               │ stock_1  │    │ stock_2  │
               │ (Apple)  │    │(Microsoft│
               └────┬─────┘    └────┬─────┘
                    │               │
                    FAN-IN: waits for BOTH to complete
                         │          │
                         └────┬─────┘
                              ▼
                    ┌─────────────────────┐
                    │  generate_comparison│
                    │  (has both results) │
                    └──────────┬──────────┘
                               │
                        ┌──────▼──────┐
                        │   __end__   │
                        └─────────────┘

KEY CONCEPTS UNIQUE TO THIS FILE (skip what was covered in files 1 & 2):
    1. Fan-out edges: add_edge(node, [node_a, node_b]) — fires both simultaneously
    2. Annotated with operator.add — merges results from parallel nodes into one list
    3. Fan-in is automatic — LangGraph waits for all branches before next node
    4. Time measurement — so you can SEE the parallel speedup vs sequential
===================================================================================
"""

import time
import operator
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, START, END

load_dotenv()


# =================================================================================
# STEP 1: Define STATE for Parallel Execution
# =================================================================================
# This state is different from files 1 & 2. Instead of a messages list,
# we have dedicated fields for each stock and a special field for parallel results.

class StockComparisonState(TypedDict):
    # Input: the user's original question
    question: str

    # Extracted stock names (set by parse_question node)
    stock_1: str
    stock_2: str

    # THE KEY TO PARALLEL EXECUTION:
    # `Annotated[list, operator.add]` means:
    # - When MULTIPLE parallel nodes write to `search_results` at the same time,
    #   LangGraph does NOT overwrite — it MERGES them using operator.add (list concat)
    # - search_stock_1 returns ["Apple results..."]
    # - search_stock_2 returns ["Microsoft results..."]
    # - LangGraph merges: ["Apple results...", "Microsoft results..."]
    # - generate_comparison node receives the combined list
    #
    # Without this annotation, whichever parallel node finishes last would
    # OVERWRITE the other's results — we'd lose one stock's data!
    search_results: Annotated[list, operator.add]

    # Final output: the comparison generated from both results
    final_answer: str


# =================================================================================
# STEP 2: Initialize LLM and Search Tool
# =================================================================================
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.3)

# Same tool name trick from file 2 — prevent Llama from hallucinating tool names
search_tool = DuckDuckGoSearchResults(name="web_search", num_results=2)


# =================================================================================
# STEP 3: Define the NODES
# =================================================================================

def parse_question(state: StockComparisonState) -> dict:
    """
    NODE 1: Extract the two stock names from the user's question.

    This runs FIRST (sequential) — we need to know which stocks to search
    before we can fan-out to parallel searches.

    We ask the LLM to extract exactly two stock names so we can pass each
    to its own dedicated parallel search node.
    """
    print(f"\n[NODE: parse_question] Extracting stock names from question...")

    response = llm.invoke([
        SystemMessage(content="""Extract exactly two stock/company names from the user's question.
Reply with ONLY the two names separated by a comma. Example: 'Apple, Microsoft'
No extra words, no explanation."""),
        HumanMessage(content=state["question"])
    ])

    # Parse "Apple, Microsoft" → ["Apple", "Microsoft"]
    parts = response.content.strip().split(",")
    stock_1 = parts[0].strip() if len(parts) > 0 else "Unknown"
    stock_2 = parts[1].strip() if len(parts) > 1 else "Unknown"

    print(f"[NODE: parse_question] Identified stocks: '{stock_1}' and '{stock_2}'")
    print(f"[NODE: parse_question] Done. Fanning out to parallel search nodes...")

    return {
        "stock_1": stock_1,
        "stock_2": stock_2,
        "search_results": []  # Initialize empty — parallel nodes will add to this
    }


def search_stock_1(state: StockComparisonState) -> dict:
    """
    NODE 2a: Search for the FIRST stock — runs in PARALLEL with search_stock_2.

    This node has no knowledge of search_stock_2. It just does its job:
    search for stock_1 and return the results.

    LangGraph runs this simultaneously with search_stock_2.
    We track timing here so you can see the parallel speedup in the logs.
    """
    stock = state["stock_1"]
    print(f"\n[NODE: search_stock_1] ⚡ Starting search for '{stock}' (PARALLEL)")

    start = time.time()
    results = search_tool.invoke(f"{stock} stock price performance news 2024")
    elapsed = time.time() - start

    print(f"[NODE: search_stock_1] ✅ Done in {elapsed:.1f}s — got {len(str(results))} chars")

    # Return as a list — operator.add will concatenate this with stock_2's results
    # The key must match the state field name: "search_results"
    return {
        "search_results": [f"=== {stock} ===\n{results}"]
    }


def search_stock_2(state: StockComparisonState) -> dict:
    """
    NODE 2b: Search for the SECOND stock — runs in PARALLEL with search_stock_1.

    Identical structure to search_stock_1, just for the other stock.
    Both nodes write to `search_results` simultaneously.
    LangGraph's operator.add annotation merges both writes safely.
    """
    stock = state["stock_2"]
    print(f"\n[NODE: search_stock_2] ⚡ Starting search for '{stock}' (PARALLEL)")

    start = time.time()
    results = search_tool.invoke(f"{stock} stock price performance news 2024")
    elapsed = time.time() - start

    print(f"[NODE: search_stock_2] ✅ Done in {elapsed:.1f}s — got {len(str(results))} chars")

    return {
        "search_results": [f"=== {stock} ===\n{results}"]
    }


def generate_comparison(state: StockComparisonState) -> dict:
    """
    NODE 3: Generate the final comparison — runs AFTER both parallel nodes finish.

    This is the "fan-in" node. LangGraph automatically waits for BOTH
    search_stock_1 and search_stock_2 to complete before this node starts.

    By the time this node runs, `state["search_results"]` contains a merged
    list with results from BOTH stocks (thanks to operator.add).
    We combine them into one string and ask the LLM to compare.
    """
    print(f"\n[NODE: generate_comparison] Both searches complete. Generating comparison...")
    print(f"[NODE: generate_comparison] Received {len(state['search_results'])} result blocks from parallel nodes")

    # Combine both stock results into one context string
    combined_results = "\n\n".join(state["search_results"])

    response = llm.invoke([
        SystemMessage(content=f"""You are a financial analyst. Compare the two stocks based on the search results below.
Structure your response as:
1. {state['stock_1']}: key metrics and recent performance
2. {state['stock_2']}: key metrics and recent performance
3. Side-by-side comparison
4. Disclaimer (past performance disclaimer)

Search Results:
{combined_results}"""),
        HumanMessage(content=state["question"])
    ])

    return {"final_answer": response.content}


# =================================================================================
# STEP 4: Build the Graph with FAN-OUT and FAN-IN
# =================================================================================
graph_builder = StateGraph(StockComparisonState)

# Add all nodes
graph_builder.add_node("parse_question", parse_question)
graph_builder.add_node("search_stock_1", search_stock_1)    # Parallel branch A
graph_builder.add_node("search_stock_2", search_stock_2)    # Parallel branch B
graph_builder.add_node("generate_comparison", generate_comparison)

# Sequential edge: START → parse_question
graph_builder.add_edge(START, "parse_question")

# FAN-OUT: parse_question → [search_stock_1, search_stock_2] simultaneously
# This single line is all it takes to make two nodes run in parallel!
# LangGraph sees multiple outgoing edges from one node and fires them at the same time.
graph_builder.add_edge("parse_question", "search_stock_1")
graph_builder.add_edge("parse_question", "search_stock_2")

# FAN-IN: both parallel nodes → generate_comparison
# LangGraph automatically waits for ALL incoming edges to complete before
# running generate_comparison. No explicit "join" or "barrier" needed.
graph_builder.add_edge("search_stock_1", "generate_comparison")
graph_builder.add_edge("search_stock_2", "generate_comparison")

# Final edge to END
graph_builder.add_edge("generate_comparison", END)

graph = graph_builder.compile()


# =================================================================================
# STEP 5: Run with timing to show parallel speedup
# =================================================================================
def compare_stocks(question: str):
    """
    Run the parallel stock comparison and show timing.

    The timing output will show:
    - Both search nodes starting at (almost) the same time
    - Each taking ~2 seconds independently
    - Total wall-clock time ≈ max(search_1_time, search_2_time)
      NOT search_1_time + search_2_time (sequential would be the sum)
    """
    print("\n" + "=" * 60)
    print("  PARALLEL STOCK COMPARISON — LANGGRAPH")
    print("  (Watch both searches fire simultaneously)")
    print("=" * 60)
    print(f"\nQuestion: {question}")
    print("-" * 60)

    total_start = time.time()

    result = graph.invoke({
        "question": question,
        "search_results": [],
        "stock_1": "",
        "stock_2": "",
        "final_answer": ""
    })

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 60}")
    print(f"COMPARISON RESULT:")
    print(f"{'=' * 60}")
    print(result["final_answer"])
    print(f"\n[TIMING] Total wall-clock time: {total_elapsed:.1f}s")
    print(f"[TIMING] Sequential would have taken ~2x the individual search times")
    print(f"[TIMING] Parallel saves ~50% time on multi-stock research!")


def main():
    print("\n" + "=" * 60)
    print("  LANGGRAPH — PARALLEL NODE EXECUTION DEMO")
    print("=" * 60)
    print("\nThis demo shows parallel node execution for stock comparison.")
    print("Notice in the logs that BOTH searches start almost simultaneously!\n")
    print("Type 'quit' to exit.\n")

    while True:
        print("Enter a comparison question (e.g. 'Compare Apple and Microsoft stocks')")
        question = input("You: ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            compare_stocks(question)
        except Exception as e:
            print(f"\nError: {e}\n")

        print()


if __name__ == "__main__":
    main()
