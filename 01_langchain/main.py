"""
===================================================================================
01_langchain — Standard LangChain Approach (The Problem)
===================================================================================

Demonstrates a Financial Research Assistant with standard LangChain to expose
5 fundamental limitations that motivated LangGraph's creation.

PROBLEM 1 — NO DECISION-MAKING:
    Linear pipeline: Input → Search → Answer. Always. Every time.
    "What is a stock?" doesn't need a web search — but the chain does it anyway.

PROBLEM 2 — NO LOOPS / RETRY:
    One-shot pipeline. If search results are poor, the chain cannot search again
    with better terms. A human analyst would refine the search — LangChain can't.

PROBLEM 3 — MEMORY CAUSES PROMPT EXPLOSION:
    ConversationBufferMemory dumps the ENTIRE history into every prompt:
      Turn 1:  ~500 tokens
      Turn 5:  ~2500 tokens
      Turn 15: CRASH — exceeds 8K context window

PROBLEM 4 — NO TOOL AUTONOMY:
    The developer hardcodes when tools are called. The LLM has no say.
    Even bind_tools() only expresses intent — it doesn't execute tools or loop.

PROBLEM 5 — NO HUMAN-IN-THE-LOOP:
    The chain cannot pause mid-execution to ask for clarification.
    It either completes or fails — no freeze/resume capability.

Run this, then run 02_langgraph/basic_agent.py and compare the difference.
===================================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from shared.tools import ddg_search
from shared.config import GROQ_MODEL, GROQ_API_KEY

load_dotenv()


# ---------------------------------------------------------------------------------
# LLM + Search Tool
# ---------------------------------------------------------------------------------
llm = ChatGroq(model_name=GROQ_MODEL, temperature=0.3)

# NOTE ON bind_tools():
# LangChain supports llm.bind_tools([tool]) which lets the LLM EXPRESS INTENT
# to call a tool. But bind_tools() does NOT execute the tool or loop back with
# results — you'd have to write that orchestration yourself, which is exactly
# what LangGraph provides natively. So we call the search tool directly here.
search_tool = ddg_search


# ---------------------------------------------------------------------------------
# Memory — the "bolted on" approach
# ---------------------------------------------------------------------------------
# InMemoryChatMessageHistory stores ALL messages. Every turn, the entire history
# is dumped into the prompt. Watch the token count grow in the logs.
chat_history = InMemoryChatMessageHistory()


# ---------------------------------------------------------------------------------
# Fixed Chain — ALWAYS follows Search → Answer, no decision-making
# ---------------------------------------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Financial Research Assistant.
Use the following search results to answer the user's question.
If results are not relevant, answer from your own knowledge.

Search Results:
{search_results}
"""),
    # This placeholder expands to the ENTIRE conversation history every turn.
    # Turn 1: empty. Turn 10: 20 full messages. Turn 15: CRASH.
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

chain = prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------------
# Run function
# ---------------------------------------------------------------------------------
def ask(question: str) -> str:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # ALWAYS searches — no decision-making (PROBLEM 1)
    print("\n[LANGCHAIN] Searching the web (always happens, no choice)...")
    search_results = search_tool.invoke(question)

    # Estimate prompt size to show the growth problem
    history_text   = str(chat_history.messages)
    est_tokens     = (len(history_text) + len(str(search_results)) + len(question)) // 4
    num_msgs       = len(chat_history.messages)

    print(f"[LANGCHAIN] History: {num_msgs} messages  |  Estimated prompt: ~{est_tokens} tokens")
    if est_tokens > 5000:
        print("[LANGCHAIN] 🚨 WARNING: Prompt dangerously large — crash imminent!")

    # One-shot generation, no loops (PROBLEM 2)
    answer = chain.invoke({
        "search_results": search_results,
        "chat_history":   chat_history.messages,
        "question":       question,
    })

    # Save to memory — making the NEXT prompt even bigger
    chat_history.add_message(HumanMessage(content=question))
    chat_history.add_message(AIMessage(content=answer))
    print("[LANGCHAIN] Saved to memory. Next prompt will be larger.")

    return answer


# ---------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("  LANGCHAIN — THE PROBLEM DEMO")
    print("  Watch the token count grow each turn until it crashes")
    print("=" * 60)
    print("\nType 'quit' to exit.\n")
    print("Suggested queries (run in order to see problems):")
    print("  1. 'What is a stock?'                  ← unnecessary search")
    print("  2. 'Latest Tesla news?'                 ← watch token count grow")
    print("  3. 'Based on that, should I buy Tesla?' ← memory works, but bloated")
    print("  4. Keep asking...                       ← watch it approach the crash\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break
        if not question:
            continue
        try:
            answer = ask(question)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"\n🚨 CRASH: {e}")
            print("This is the long-prompt problem in action.")
            print("→ See 02_langgraph/basic_agent.py for the solution.\n")


if __name__ == "__main__":
    main()
