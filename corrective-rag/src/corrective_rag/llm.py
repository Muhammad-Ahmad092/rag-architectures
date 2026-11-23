"""OpenAI chat model behind a two-method interface the graph uses.

``complete(system, user) -> str`` and ``judge(system, user, schema) -> schema`` (structured
output via ``with_structured_output``), so tests can swap in a scripted fake.
"""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from .config import Settings, require

T = TypeVar("T", bound=BaseModel)


class OpenAILLM:
    def __init__(self, model: str, api_key: str, chat=None):
        if chat is None:
            from langchain_openai import ChatOpenAI

            chat = ChatOpenAI(model=model, api_key=api_key, temperature=0)
        self.model = model
        self._chat = chat

    def complete(self, system: str, user: str) -> str:
        return str(self._chat.invoke([("system", system), ("human", user)]).content).strip()

    def judge(self, system: str, user: str, schema: type[T]) -> T:
        return self._chat.with_structured_output(schema).invoke(
            [("system", system), ("human", user)]
        )


def get_llm(settings: Settings, model: str | None = None) -> OpenAILLM:
    key = require(settings.openai_api_key, "OPENAI_API_KEY")
    return OpenAILLM(model or settings.llm_model, key)
