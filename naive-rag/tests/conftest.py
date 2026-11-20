"""Offline fakes shared by pipeline / ingest / app tests."""

import hashlib
import math
import os
from pathlib import Path

import pytest

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


class FakeLLM:
    model = "fake-llm"

    def __init__(self):
        self.calls = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return "fake answer [1]"


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def settings(tmp_path):
    from naive_rag.config import get_settings

    return get_settings(
        _env_file=None,
        vector_store="faiss",
        index_dir=tmp_path / ".index",
        data_dir=FIXTURES,
        top_k=2,
    )


@pytest.fixture
def faiss_store(settings, fake_embedder):
    from naive_rag.vectorstore import get_vector_store

    return get_vector_store(settings, fake_embedder.name, fake_embedder.dim)
