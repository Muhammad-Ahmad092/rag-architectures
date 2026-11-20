"""Chat models: Groq (this folder's default) or OpenAI (fair-mode eval generator)."""

from __future__ import annotations

from .config import ConfigError, Settings, import_extra, require


def get_llm(settings: Settings, provider: str = "groq", model: str | None = None):
    if provider == "groq":
        from langchain_groq import ChatGroq

        key = require(settings.groq_api_key, "GROQ_API_KEY")
        return ChatGroq(model=model or settings.llm_model, api_key=key, temperature=0)
    if provider == "openai":
        key = require(settings.openai_api_key, "OPENAI_API_KEY")
        mod = import_extra("langchain_openai", "eval")
        return mod.ChatOpenAI(model=model or settings.eval_llm, api_key=key, temperature=0)
    raise ConfigError(f"Unknown LLM provider '{provider}'")


def model_name(llm) -> str:
    return str(getattr(llm, "model_name", None) or getattr(llm, "model", "?"))
