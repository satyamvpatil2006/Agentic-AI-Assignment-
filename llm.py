"""Provider factory for chat models."""
from __future__ import annotations
import os
from typing import Any


def build_llm(provider: str, model_name: str, api_key: str = "", temperature: float = 0.2) -> Any:
    """Create an OpenAI, Ollama, or Anthropic LangChain chat model."""
    provider = provider.lower()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        key = api_key.strip() or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OpenAI API key is missing. Enter it in the sidebar or set OPENAI_API_KEY.")
        return ChatOpenAI(model=model_name or "gpt-4o-mini", api_key=key, temperature=temperature, max_retries=1)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model_name or "llama3.1", temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        key = api_key.strip() or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("Anthropic API key is missing. Enter it in the sidebar or set ANTHROPIC_API_KEY.")
        return ChatAnthropic(model=model_name or "claude-sonnet-4-5", api_key=key, temperature=temperature, max_retries=1)
    raise ValueError(f"Unsupported provider: {provider}")
