from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from simple_rag import pipeline
from simple_rag.config import ConfigError
from simple_rag.models import Chunk, RAGResult, ScoredChunk, Step

APP = str(Path(__file__).parent.parent / "app.py")


class FakeStore:
    def count(self):
        return 7


class FakeRAG:
    def __init__(self, settings, **kw):
        self.settings = settings
        self.embed_spec = type("E", (), {"name": "fake:emb", "dim": 3})()
        self.store = FakeStore()

    def answer(self, question):
        chunk = Chunk(id="a.txt::0", text="ctx", source="a.txt", metadata={})
        return RAGResult(
            answer=f"echo: {question}",
            sources=[ScoredChunk(chunk=chunk, score=0.9)],
            steps=[Step("retrieve", [("a.txt::0", 0.9)]), Step("prompt", "Context: ...")],
        )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")


def _sidebar_text(at):
    return " ".join(m.value for m in at.sidebar.markdown) + " ".join(
        c.value for c in at.sidebar.caption
    )


def test_app_renders_and_answers(monkeypatch):
    monkeypatch.setattr(pipeline, "SimpleRAG", FakeRAG)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "chroma" in _sidebar_text(at)
    assert "7" in _sidebar_text(at)
    assert "SentenceSplitter" in _sidebar_text(at)
    assert "gemini" in _sidebar_text(at)
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert any("echo: hi" in m.value for m in at.markdown)
    labels = [e.label for e in at.expander]
    assert "Under the hood" in labels
    assert any(lbl.startswith("Sources") for lbl in labels)
    assert any("Source 1" in m.value for m in at.markdown)


def test_config_error_shown_not_raised(monkeypatch):
    def boom(settings, **kw):
        raise ConfigError("Set GEMINI_API_KEY in .env")

    monkeypatch.setattr(pipeline, "SimpleRAG", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("GEMINI_API_KEY" in e.value for e in at.error)
