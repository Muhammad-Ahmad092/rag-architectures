from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from simple_rag_memory import pipeline
from simple_rag_memory.config import ConfigError
from simple_rag_memory.memory import SessionMemory
from simple_rag_memory.models import Chunk, RAGResult, ScoredChunk, Step

APP = str(Path(__file__).parent.parent / "app.py")
CALLS: list[dict] = []


class FakeStore:
    def count(self):
        return 7


class FakeRAG:
    def __init__(self, settings, **kw):
        self.settings = settings
        self.embed_spec = type("E", (), {"name": "fake:emb", "dim": 3})()
        self.store = FakeStore()
        self.memory = SessionMemory(settings.memory_turns)

    def answer(self, question, session_id="default", rewrite=True):
        CALLS.append({"question": question, "session_id": session_id, "rewrite": rewrite})
        follow_up = self.memory.turns(session_id) > 0
        self.memory.add_turn(session_id, question, f"echo: {question}")
        chunk = Chunk(id="a.txt::0", text="ctx", source="a.txt", metadata={})
        return RAGResult(
            answer=f"echo: {question}",
            sources=[ScoredChunk(chunk=chunk, score=0.9)],
            steps=[Step("rewrite", {}), Step("prompt", "Context: ...")],
            search_query=f"standalone {question}" if follow_up else question,
        )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    CALLS.clear()
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")


def _sidebar_text(at):
    return " ".join(m.value for m in at.sidebar.markdown) + " ".join(
        c.value for c in at.sidebar.caption
    )


def test_app_renders_and_answers_with_memory(monkeypatch):
    monkeypatch.setattr(pipeline, "MemoryRAG", FakeRAG)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    text = _sidebar_text(at)
    assert "chroma" in text and "7" in text and "groq" in text and "Memory" in text
    at.chat_input[0].set_value("first").run()
    at.chat_input[0].set_value("second").run()
    assert not at.exception
    assert any("echo: second" in m.value for m in at.markdown)
    assert "This chat:** 2 turns" in _sidebar_text(at)  # counted after this turn is stored
    assert CALLS[0]["session_id"] == CALLS[1]["session_id"]  # same browser session
    assert all(c["rewrite"] for c in CALLS)
    # follow-up shows what was actually searched for
    assert any("standalone second" in c.value for c in at.caption)
    labels = [e.label for e in at.expander]
    assert "Under the hood" in labels and any(lbl.startswith("Sources") for lbl in labels)


def test_rewrite_toggle_and_clear(monkeypatch):
    monkeypatch.setattr(pipeline, "MemoryRAG", FakeRAG)
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("first").run()
    at.sidebar.toggle[0].set_value(False).run()
    at.chat_input[0].set_value("second").run()
    assert CALLS[-1]["rewrite"] is False
    clear = next(b for b in at.sidebar.button if b.label == "Clear chat")
    clear.click().run()
    assert not at.exception
    assert at.session_state.rag.memory.turns(at.session_state.session_id) == 0
    assert at.session_state.messages == []


def test_config_error_shown_not_raised(monkeypatch):
    def boom(settings, **kw):
        raise ConfigError("Set GROQ_API_KEY in .env")

    monkeypatch.setattr(pipeline, "MemoryRAG", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("GROQ_API_KEY" in e.value for e in at.error)
