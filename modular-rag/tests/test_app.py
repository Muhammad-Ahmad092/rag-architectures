from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modular_rag import pipeline
from modular_rag.config import ConfigError
from modular_rag.models import Chunk, RAGResult, ScoredChunk, Step
from modular_rag.modules import Built, Modules

APP = str(Path(__file__).parent.parent / "app.py")
ASKED: list[Modules] = []


class FakeStore:
    def count(self):
        return 7


class FakeRAG:
    def __init__(self, settings, **kw):
        self.settings = settings
        self.embedder = type("E", (), {"name": "fake:emb", "dim": 3})()
        self.store = FakeStore()
        self.corpus = [object()] * 7

    def default_modules(self):
        return Modules("hybrid", "none", "mistral")

    def pipeline_for(self, modules):
        return Built(
            pipeline=None,
            inputs={"bm25_retriever": {"query": "query"}},
            edges=[("bm25_retriever.documents", "ranker.documents")],
            stages=[],
        )

    def refresh_corpus(self):
        pass

    def answer(self, question, modules=None):
        ASKED.append(modules)
        chunk = Chunk(id="a::0", text="ctx", source="a.txt", metadata={})
        return RAGResult(
            answer=f"echo: {question}",
            sources=[ScoredChunk(chunk=chunk, score=0.9)],
            steps=[Step("modules", {}), Step("pipeline", "flowchart LR")],
        )


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    ASKED.clear()
    monkeypatch.setenv("VECTOR_STORE", "chroma")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "huggingface")
    monkeypatch.setattr(pipeline, "ModularRAG", FakeRAG)


def test_modules_are_picked_in_the_sidebar():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert [r.label for r in at.sidebar.radio] == ["Retriever", "Ranker", "Generator"]
    assert at.sidebar.radio[0].value == "hybrid"
    at.sidebar.radio[0].set_value("bm25").run()
    at.sidebar.radio[1].set_value("lost-in-the-middle").run()
    at.chat_input[0].set_value("hi").run()
    assert not at.exception
    assert ASKED[-1] == Modules("bm25", "lost-in-the-middle", "mistral")
    assert any("bm25 → lost-in-the-middle → mistral" in c.value for c in at.caption)
    assert any("echo: hi" in m.value for m in at.markdown)


def test_config_error_shown_not_raised(monkeypatch):
    def boom(settings, **kw):
        raise ConfigError("Set MISTRAL_API_KEY in .env")

    monkeypatch.setattr(pipeline, "ModularRAG", boom)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("MISTRAL_API_KEY" in e.value for e in at.error)
