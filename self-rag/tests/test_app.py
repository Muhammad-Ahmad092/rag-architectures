from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from self_rag import pipeline
from self_rag.config import ConfigError
from self_rag.llm import LLMError
from self_rag.models import Chunk, RAGResult, ScoredChunk, Step

APP = str(Path(__file__).parent.parent / "app.py")
VERDICT = {"value": "verified"}


class FakeStore:
    def count(self):
        return 7


class FakeRAG:
    def __init__(self, settings, **kw):
        self.settings = settings
        self.embed_spec = type("E", (), {"name": "fake:emb", "dim": 3})()
        self.store = FakeStore()

    def answer(self, question):
        if VERDICT["value"] == "refusal":
            raise LLMError("Claude declined this request (category: cyber).")
        chunk = Chunk(id="a.txt::0", text="ctx", source="a.txt", metadata={})
        steps = [
            Step("retrieve", {}),
            Step("rewrite", {}),
            Step("retrieve", {}),
            Step("generate", {}),
            Step("check_support", {}),
        ]
        return RAGResult(
            answer=f"echo: {question}",
            sources=[ScoredChunk(chunk=chunk, score=0.9)],
            steps=steps,
            verdict=VERDICT["value"],
        )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    VERDICT["value"] = "verified"
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setattr(pipeline, "SelfRAG", FakeRAG)


def _sidebar_text(at):
    return " ".join(m.value for m in at.sidebar.markdown) + " ".join(
        c.value for c in at.sidebar.caption
    )


def test_app_renders_answer_verdict_and_loop():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    text = _sidebar_text(at)
    assert "claude-opus-5-5" in text and "Grader" in text and "chroma" in text and "7" in text
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert any("echo: hi" in m.value for m in at.markdown)
    assert any("Verified" in s.value for s in at.success)
    assert any("2 retrieval(s) · 1 rewrite(s) · 1 answer attempt(s)" in c.value for c in at.caption)
    labels = [e.label for e in at.expander]
    assert "Under the hood" in labels and any(lbl.startswith("Relevant sources") for lbl in labels)


def test_unsupported_verdict_is_a_warning():
    VERDICT["value"] = "unsupported"
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hi").run()
    assert any("Not fully supported" in w.value for w in at.warning)


def test_refusal_shown_not_raised():
    VERDICT["value"] = "refusal"
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert any("declined" in e.value for e in at.error)


def test_config_error_shown_not_raised(monkeypatch):
    def boom(settings, **kw):
        raise ConfigError("Set ANTHROPIC_API_KEY in .env")

    monkeypatch.setattr(pipeline, "SelfRAG", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("ANTHROPIC_API_KEY" in e.value for e in at.error)
