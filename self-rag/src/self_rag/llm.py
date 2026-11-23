"""LLM access for the Self-RAG graph.

Two small classes share one interface, ``complete(system, user) -> str`` and
``judge(system, user, schema) -> schema``, so the graph does not care who answers:

* ``ClaudeLLM``: this folder's model, through the official ``anthropic`` SDK. Current
  Claude models run adaptive thinking and reject sampling parameters, so no
  ``temperature`` is sent; depth is set with ``output_config.effort``. Graders use
  JSON-schema structured output (``output_config.format``), not forced tool use, which
  current models reject. Models that support it get the server-side refusal fallback.
* ``OpenAILLM``: the common fair-mode eval generator.
"""

from __future__ import annotations

import copy
from typing import Protocol, TypeVar

from pydantic import BaseModel

from .config import ConfigError, Settings, import_extra, require

T = TypeVar("T", bound=BaseModel)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models with server-side refusal fallback ("default" routing); Haiku 5.5 has none.
FALLBACK_MODELS = {
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
    "claude-fable-5-2",
}


class LLMError(Exception):
    """The model declined or returned something unusable."""


class LLM(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...

    def judge(self, system: str, user: str, schema: type[T]) -> T: ...


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema in the strict shape structured outputs expect."""
    schema = copy.deepcopy(model.model_json_schema())
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    schema["additionalProperties"] = False
    schema["required"] = list(schema.get("properties", {}))
    return schema


class ClaudeLLM:
    def __init__(
        self,
        model: str,
        effort: str = "medium",
        api_key: str | None = None,
        max_tokens: int = 16000,  # thinking is always on and counts toward max_tokens
        client=None,
    ):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self.client = client

    def _create(self, system: str, user: str, output_config: dict) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        if self.model in FALLBACK_MODELS:
            kwargs |= {"betas": [FALLBACK_BETA], "fallbacks": "default"}
        response = self.client.beta.messages.create(**kwargs)
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise LLMError(f"Claude declined this request (category: {category}).")
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def complete(self, system: str, user: str) -> str:
        return self._create(system, user, {"effort": self.effort})

    def judge(self, system: str, user: str, schema: type[T]) -> T:
        fmt = {"type": "json_schema", "schema": strict_schema(schema)}
        text = self._create(system, user, {"effort": self.effort, "format": fmt})
        try:
            return schema.model_validate_json(text)
        except ValueError as exc:
            raise LLMError(f"Grader returned invalid JSON: {text[:200]}") from exc


class OpenAILLM:
    """Fair-mode eval generator (needs --extra eval or --extra openai)."""

    def __init__(self, model: str, api_key: str):
        mod = import_extra("langchain_openai", "eval")
        self.model = model
        self._chat = mod.ChatOpenAI(model=model, api_key=api_key, temperature=0)

    def complete(self, system: str, user: str) -> str:
        return str(self._chat.invoke([("system", system), ("human", user)]).content).strip()

    def judge(self, system: str, user: str, schema: type[T]) -> T:
        return self._chat.with_structured_output(schema).invoke(
            [("system", system), ("human", user)]
        )


def get_llms(settings: Settings) -> tuple[ClaudeLLM, ClaudeLLM]:
    """(generator, grader). Empty GRADER_MODEL reuses LLM_MODEL at GRADER_EFFORT."""
    key = require(settings.anthropic_api_key, "ANTHROPIC_API_KEY")
    gen = ClaudeLLM(settings.llm_model, settings.llm_effort, api_key=key)
    grader_model = settings.grader_model or settings.llm_model
    grader = ClaudeLLM(grader_model, settings.grader_effort, api_key=key, max_tokens=4000)
    return gen, grader


def get_openai_llm(settings: Settings, model: str) -> OpenAILLM:
    key = require(settings.openai_api_key, "OPENAI_API_KEY")
    if not model:
        raise ConfigError("Set EVAL_LLM in .env")
    return OpenAILLM(model, key)
