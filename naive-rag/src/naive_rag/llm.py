"""Generator LLM. This folder's default provider is OpenAI."""

from __future__ import annotations

from typing import Protocol

from .config import Settings, import_extra, require


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAILLM:
    def __init__(self, settings: Settings, model: str | None = None):
        key = require(settings.openai_api_key, "OPENAI_API_KEY")
        self.model = model or settings.llm_model
        self._client = import_extra("openai", "openai").OpenAI(api_key=key)

    def complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


def get_llm(settings: Settings, model: str | None = None) -> LLM:
    return OpenAILLM(settings, model)
