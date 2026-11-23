from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from advanced_rag import pipeline
from advanced_rag.config import ConfigError
from advanced_rag.models import Chunk, RAGResult, ScoredChunk, Step

APP = str(Path(__file__).parent.parent / "app.py")


class FakeStore:
    def count(self):
        return 7


class FakeRAG:
    def __init__(self, settings, **kw):
        self.settings = settings
        self.embed_spec = type("E", (), {"name": "fake:emb", "dim": 3})()
        self.store = FakeStore()
        self.corpus = [object()] * 7
        self.reranker_name = "local (no COHERE_API_KEY)"

    def refresh_corpus(self):
        pass

    def answer(self, question):
        chunk = Chunk(id="a::0", text="ctx", source="a.txt", metadata={})
        return RAGResult(
            answer=f"echo: {question}",
            sources=[ScoredChunk(chunk=chunk, score=0.9)],
            steps=[Step("rewrite", [question, "alt query"]), Step("check", [])],
            verdict="revised",
        )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setattr(pipeline, "AdvancedRAG", FakeRAG)


def test_app_renders_queries_verdict_and_reranker():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    text = " ".join(m.value for m in at.sidebar.markdown)
    assert "claude-opus-5-5" in text and "local (no COHERE_API_KEY)" in text and "RRF" in text
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert any("“alt query”" in c.value for c in at.caption)
    assert any("removed in a revision" in i.value for i in at.info)
    assert any("echo: hi" in m.value for m in at.markdown)


def test_config_error_shown_not_raised(monkeypatch):
    def boom(settings, **kw):
        raise ConfigError("Set ANTHROPIC_API_KEY in .env")

    monkeypatch.setattr(pipeline, "AdvancedRAG", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("ANTHROPIC_API_KEY" in e.value for e in at.error)
