from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from corrective_rag import pipeline
from corrective_rag.config import ConfigError
from corrective_rag.models import Chunk, RAGResult, ScoredChunk, Step

APP = str(Path(__file__).parent.parent / "app.py")
ACTION = {"value": "ambiguous"}


class FakeStore:
    def count(self):
        return 7


class FakeRAG:
    def __init__(self, settings, **kw):
        self.settings = settings
        self.embed_spec = type("E", (), {"name": "fake:emb", "dim": 3})()
        self.store = FakeStore()
        self.web = None

    def answer(self, question):
        internal = Chunk(id="a::0", text="ctx", source="a.txt", metadata={})
        web = Chunk(
            id="web::u",
            text="web ctx",
            source="https://x.example",
            metadata={"web": True, "title": "X page"},
        )
        return RAGResult(
            answer=f"echo: {question}",
            sources=[ScoredChunk(chunk=internal, score=0.5), ScoredChunk(chunk=web, score=0.9)],
            steps=[Step("evaluate", {"action": ACTION["value"]})],
            verdict=ACTION["value"],
        )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    ACTION["value"] = "ambiguous"
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setattr(pipeline, "CorrectiveRAG", FakeRAG)


def _sidebar_text(at):
    return " ".join(m.value for m in at.sidebar.markdown)


def test_app_renders_action_and_marks_web_sources():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    text = _sidebar_text(at)
    assert "gpt-4o-mini" in text and "TAVILY_API_KEY" in text and "0.7" in text
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert any("Ambiguous" in i.value for i in at.info)
    assert any("🌐 [X page](https://x.example)" in m.value for m in at.markdown)
    assert any("📄 `a.txt`" in m.value for m in at.markdown)


def test_incorrect_is_a_warning():
    ACTION["value"] = "incorrect"
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hi").run()
    assert any("answered from the web" in w.value for w in at.warning)


def test_config_error_shown_not_raised(monkeypatch):
    def boom(settings, **kw):
        raise ConfigError("Set OPENAI_API_KEY in .env")

    monkeypatch.setattr(pipeline, "CorrectiveRAG", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("OPENAI_API_KEY" in e.value for e in at.error)
