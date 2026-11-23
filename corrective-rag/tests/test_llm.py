from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from corrective_rag.config import ConfigError, get_settings
from corrective_rag.llm import OpenAILLM, get_llm
from corrective_rag.web import TavilyWeb, WebSearchError, get_web


class Score(BaseModel):
    score: float
    reason: str


class FakeChat:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=" hello ")

    def with_structured_output(self, schema):
        chat = self

        class Bound:
            def invoke(self, messages):
                chat.calls.append(("structured", schema, messages))
                return schema(score=0.8, reason="r")

        return Bound()


def test_openai_llm_complete_and_judge():
    chat = FakeChat()
    llm = OpenAILLM("gpt-4o-mini", "k", chat=chat)
    assert llm.complete("sys", "user") == "hello"
    assert chat.calls[0] == [("system", "sys"), ("human", "user")]
    assert llm.judge("sys", "user", Score) == Score(score=0.8, reason="r")


def test_get_llm_requires_key():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_llm(get_settings(_env_file=None, openai_api_key=None))
    llm = get_llm(get_settings(_env_file=None, openai_api_key="k"))
    assert llm.model == "gpt-4o-mini"


def test_tavily_results_become_web_documents():
    client = SimpleNamespace(
        search=lambda query, max_results: {
            "results": [
                {"title": "T", "url": "https://a.example", "content": "Alpha.", "score": 0.9},
                {"title": "Empty", "url": "https://b.example", "content": "  "},
            ]
        }
    )
    docs = TavilyWeb("k", 3, client=client).search("q")
    assert len(docs) == 1  # empty snippets dropped
    d = docs[0]
    assert d.page_content == "Alpha."
    assert d.metadata["source"] == "https://a.example" and d.metadata["web"] is True
    assert d.metadata["web_score"] == 0.9


def test_tavily_errors_are_wrapped():
    def boom(query, max_results):
        raise RuntimeError("401 invalid key")

    with pytest.raises(WebSearchError, match="invalid key"):
        TavilyWeb("k", client=SimpleNamespace(search=boom)).search("q")


def test_get_web_none_without_key_or_when_off():
    assert get_web(get_settings(_env_file=None, tavily_api_key=None)) is None
    assert get_web(get_settings(_env_file=None, tavily_api_key="k", web_search=False)) is None
    web = get_web(get_settings(_env_file=None, tavily_api_key="k", web_results=5))
    assert isinstance(web, TavilyWeb) and web.max_results == 5
