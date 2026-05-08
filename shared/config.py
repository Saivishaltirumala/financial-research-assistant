"""
shared/config.py — Groq API constants shared across all modules.

Each framework initialises its LLM differently:
    LangGraph / LangChain → ChatGroq(model_name=GROQ_MODEL, ...)
    CrewAI                → LLM(model=f"groq/{GROQ_MODEL}", api_key=GROQ_API_KEY, ...)
    AutoGen               → OpenAIChatCompletionClient(model=GROQ_MODEL,
                                base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY, ...)

Keeping the model name and endpoint in one place means switching models
(e.g., upgrading to llama-3.3-70b) is a one-line change here, not
a search-and-replace across 7 files.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL    = "llama-3.1-8b-instant"
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"   # OpenAI-compatible endpoint
