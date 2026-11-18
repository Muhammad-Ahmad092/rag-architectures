"""LlamaIndex LLMs: Gemini (this folder's default) and OpenAI (fair-mode evaluation)."""

from __future__ import annotations

from typing import Literal

from .config import Settings, import_extra, require


def get_llm(
    settings: Settings,
    provider: Literal["gemini", "openai"] = "gemini",
    model: str | None = None,
):
    if provider == "gemini":
        key = require(settings.gemini_api_key, "GEMINI_API_KEY")
        mod = import_extra("llama_index.llms.google_genai", "gemini")
        return mod.GoogleGenAI(model=model or settings.llm_model, api_key=key, temperature=0)
    if provider == "openai":
        key = require(settings.openai_api_key, "OPENAI_API_KEY")
        mod = import_extra("llama_index.llms.openai", "eval")
        return mod.OpenAI(model=model or settings.eval_llm, api_key=key, temperature=0)
    raise ValueError(f"unknown LLM provider {provider!r}")


def model_name(llm) -> str:
    return getattr(llm, "model", None) or getattr(llm.metadata, "model_name", "unknown")
