"""
===================================================================================
FILE 1: STANDARD LANGCHAIN APPROACH (GenAI) — THE PROBLEM
===================================================================================

This file demonstrates a Financial Research Assistant built with standard LangChain.
It works, but has CRITICAL LIMITATIONS that become obvious when you use it:

PROBLEM 1 — NO DECISION-MAKING (No Conditional Logic):
    LangChain chains are LINEAR pipelines: Input → Step1 → Step2 → Output.
    The LLM cannot DECIDE whether to search the web or answer from memory.
    Every query goes through the SAME fixed pipeline, even if the answer is obvious.
    Example: "What is a stock?" doesn't need a web search, but the chain does it anyway.

PROBLEM 2 — NO LOOPS (No Iterative Refinement):
    If the search results are poor or incomplete, the chain cannot go back and retry.
    It's a one-shot pipeline: you get ONE chance, and if the result is bad, tough luck.
    A human analyst would say "let me search again with different terms" — LangChain can't.

PROBLEM 3 — MEMORY IS BOLTED ON, NOT BUILT IN:
    LangChain offers ConversationBufferMemory to add memory, but it comes with a
    CRITICAL FLAW: it dumps the ENTIRE conversation history into every prompt.
    After 10-15 exchanges, the prompt becomes enormous:
      - System prompt + search results + ALL previous messages + new question
    This causes: token limit errors, slower responses, higher API costs,
    and eventually crashes when the context window is exceeded.
    LangGraph solves this with proper state management (see file 2).

PROBLEM 4 — NO TOOL AUTONOMY:
    The LLM doesn't decide WHEN to use tools. YOU hardcode the pipeline.
    In a real assistant, the AI should decide: "I need to search for this" vs "I already know this."

PROBLEM 5 — LONG PROMPT / CONTEXT EXPLOSION:
    With ConversationBufferMemory, every turn adds to the prompt:
      Turn 1:  system + search_results + question1 + answer1                     (~500 tokens)
      Turn 5:  system + search_results + question1-5 + answer1-5                 (~2500 tokens)
      Turn 15: system + search_results + question1-15 + answer1-15               (~7500 tokens)
    The search results are also appended every turn (even duplicate searches).
    Eventually the prompt EXCEEDS the model's context window and the app CRASHES.
    You can see the token count growing in the logs when you run this file.

These problems are exactly why LangGraph was created — see file 2 for the solution.
===================================================================================
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# ---------------------------------------------------------------------------------
# STEP 1: Initialize the LLM and Search Tool
# ---------------------------------------------------------------------------------
# We use Groq's free API with Llama model for fast inference
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.3,  # Low temperature for factual financial responses
)

# DuckDuckGo search tool — free, no API key needed
search_tool = DuckDuckGoSearchResults(num_results=3)

# ---------------------------------------------------------------------------------
# NOTE ON bind_tools():
# LangChain DOES support `llm.bind_tools([search_tool])` which lets the LLM
# generate tool call metadata (e.g., "I want to call search with query X").
# HOWEVER, bind_tools() only makes the LLM EXPRESS INTENT to use a tool —
# it does NOT automatically execute the tool or feed results back to the LLM.
# You would still need to manually:
#   1. Check if the LLM's response contains a tool call
#   2. Execute the tool yourself in Python code
#   3. Feed the tool result back to the LLM
#   4. Call the LLM again to generate the final answer
# This is essentially re-inventing the Agent loop that LangGraph provides natively.
# So even with bind_tools(), standard LangChain lacks the ORCHESTRATION layer
# (conditional routing, loops, state management) that makes tools truly autonomous.
# That's why we call the search tool directly here — with or without bind_tools(),
# the control flow burden is on the developer, not the framework.
# ---------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------
# STEP 2: Set up ConversationBufferMemory — the "bolted on" memory solution
# ---------------------------------------------------------------------------------
# InMemoryChatMessageHistory acts like ConversationBufferMemory — stores ALL messages.
# This "works" for short conversations but creates a GROWING PROMPT problem:
# - Every turn, the entire chat history is prepended to the new prompt
# - Search results from EVERY turn are also included (duplicates and all)
# - The prompt size grows linearly with conversation length
# - Eventually it EXCEEDS the model's context window (e.g., 8K tokens for Llama 3.1 8B)
# - At that point, the app either crashes or silently truncates important context
#
# LangGraph's State + add_messages handles this more elegantly because:
# - State is managed separately from the prompt
# - You can add summarization or trimming strategies
# - The graph can decide what context is relevant (not dump everything)
chat_history = InMemoryChatMessageHistory()


# ---------------------------------------------------------------------------------
# STEP 3: Build a FIXED Chain with Memory (This is the problem!)
# ---------------------------------------------------------------------------------
# Notice how this chain ALWAYS follows the same path:
#   User Question → Search Web → Generate Answer
# There is NO way for the LLM to skip the search or do a second search.
#
# Now with memory added, the prompt template includes a MessagesPlaceholder
# that expands to ALL previous messages. Watch the token count grow!

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Financial Research Assistant specialized in stock market analysis.
You help users understand stocks, market trends, and financial news.

Use the following search results to answer the user's question.
If the search results are not relevant, answer from your own knowledge but mention
that the information might not be current.

Search Results:
{search_results}
"""),
    # This placeholder EXPANDS to the ENTIRE conversation history every single turn.
    # Turn 1: empty. Turn 5: all 10 messages (5 human + 5 AI). Turn 20: 40 messages!
    # Each message includes the full text of questions AND answers AND search results.
    # THIS is the long prompt problem — unbounded growth with no trimming.
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

# The chain: prompt (with expanding history) → LLM → parse output
chain = prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------------
# STEP 4: The "run" function — notice the prompt grows every turn
# ---------------------------------------------------------------------------------
def ask_financial_assistant(question: str) -> str:
    """
    This function shows TWO FUNDAMENTAL LIMITATIONS of standard LangChain:

    1. FIXED PIPELINE: Every question goes through the exact same steps
    2. GROWING PROMPT: With ConversationBufferMemory, the prompt gets bigger every turn

    The prompt size problem:
    - Turn 1:  ~500 tokens (system + search + question)
    - Turn 5:  ~2500 tokens (system + search + 10 previous messages + question)
    - Turn 10: ~5000 tokens (system + search + 20 previous messages + question)
    - Turn 15: CRASH! Exceeds model's context window

    You'll see the token count printed in the logs — watch it climb!
    """

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # ALWAYS searches, even when unnecessary (PROBLEM 1: no decision-making)
    print("\n[LANGCHAIN] Step 1: Searching the web (ALWAYS happens, no choice)...")
    search_results = search_tool.invoke(question)
    print(f"[LANGCHAIN] Search returned {len(str(search_results))} characters of results")

    # Load ALL previous conversation history from memory
    # This is the LONG PROMPT problem — history grows unbounded!
    history_messages = chat_history.messages
    num_history_messages = len(history_messages)

    # Estimate the prompt size to show the growth problem
    history_text = str(history_messages)
    estimated_tokens = (len(history_text) + len(str(search_results)) + len(question)) // 4
    print(f"[LANGCHAIN] Step 2: Loading conversation history...")
    print(f"[LANGCHAIN] ⚠️  History contains {num_history_messages} messages")
    print(f"[LANGCHAIN] ⚠️  Estimated prompt size: ~{estimated_tokens} tokens")

    if estimated_tokens > 5000:
        print(f"[LANGCHAIN] 🚨 WARNING: Prompt is getting dangerously large!")
        print(f"[LANGCHAIN] 🚨 Model context window is ~8K tokens. Crash imminent!")

    # ONE-SHOT generation with the bloated prompt (PROBLEM 2: no loops)
    print("[LANGCHAIN] Step 3: Generating answer (ONE chance, no retry)...")
    answer = chain.invoke({
        "search_results": search_results,
        "chat_history": history_messages,
        "question": question
    })

    # Save this exchange to memory — making the next prompt even BIGGER
    # Every question + answer + search results accumulate forever
    chat_history.add_message(HumanMessage(content=question))
    chat_history.add_message(AIMessage(content=answer))
    print(f"[LANGCHAIN] Step 4: Saved to memory. Next prompt will be even LARGER.")

    return answer


# ---------------------------------------------------------------------------------
# STEP 5: Interactive Chat Loop
# ---------------------------------------------------------------------------------
def main():
    """
    Run the standard LangChain financial assistant WITH memory.

    Try these queries IN ORDER to see both the memory AND the long prompt problem:

    1. "What is a stock?"
       → Searches the web unnecessarily. Watch the token count.

    2. "What's the latest news on Tesla stock?"
       → Searches again. Token count grows (now includes Q1 + A1 in history).

    3. "Based on that news, should I buy Tesla?"
       → Memory works now! It remembers Tesla. But token count keeps growing.

    4. "Tell me about Apple stock"
       → Token count even bigger. History now has 3 full exchanges.

    5. "Compare Apple and Microsoft"
       → Still can only do ONE search. And prompt is huge now.

    6-15. Keep asking questions...
       → Watch the token count climb toward the crash point!
       → Around turn 10-15, the prompt will exceed the model's context window.

    Compare this to file 2 (LangGraph) where:
    - The LLM decides when to search (no wasteful searches)
    - State management is built-in (not bolted on)
    - The graph structure allows loops and conditional logic
    """

    print("\n" + "=" * 60)
    print("  FINANCIAL RESEARCH ASSISTANT — STANDARD LANGCHAIN")
    print("  (Now with ConversationBufferMemory — watch the prompt GROW)")
    print("=" * 60)
    print("\nType 'quit' to exit.\n")
    print("TIP: Keep chatting and watch the token count grow each turn!\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ["quit", "q", "exit"]:
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            answer = ask_financial_assistant(question)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"\n🚨 Error (likely context window exceeded): {e}\n")
            print("This is the LONG PROMPT problem in action!")
            print("ConversationBufferMemory kept growing the prompt until it crashed.")
            print("LangGraph's state management avoids this — see file 2.\n")


if __name__ == "__main__":
    main()
