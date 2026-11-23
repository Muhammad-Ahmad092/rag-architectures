"""Offline fakes shared by pipeline / ingest / app tests."""

import hashlib
import math
import os
from pathlib import Path

import pytest
from haystack import component
from haystack.dataclasses import ChatMessage

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: similar words -> similar vectors."""

    name = "fake:bow-32"
    dim = 32

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in text.lower().split():
            h = int(hashlib.md5(word.strip(".,:;!?").encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@component
class FakeGenerator:
    """Haystack chat-generator stand-in: records the messages, replies citing [1]."""

    def __init__(self, name="fake"):
        self.name = name
        self.calls = []

    @component.output_types(replies=list[ChatMessage])
    def run(self, messages: list[ChatMessage]):
        self.calls.append(messages)
        return {"replies": [ChatMessage.from_assistant(f"{self.name} answer [1]")]}


class FakeCrossEncoder:
    """Scores a (query, passage) pair by shared words, like a (very) small cross-encoder."""

    def predict(self, pairs):
        return [len(set(q.lower().split()) & set(p.lower().split())) for q, p in pairs]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def settings(tmp_path):
    from modular_rag.config import get_settings

    return get_settings(
        _env_file=None,
        vector_store="faiss",
        index_dir=tmp_path / ".index",
        data_dir=FIXTURES,
        top_k=2,
    )


@pytest.fixture
def faiss_store(settings, fake_embedder):
    from modular_rag.vectorstore import get_vector_store

    return get_vector_store(settings, fake_embedder.name, fake_embedder.dim)
