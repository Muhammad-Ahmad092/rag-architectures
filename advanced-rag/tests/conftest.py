"""Offline fakes shared by the tests (no API keys, no model downloads)."""

import hashlib
import math
import os
from pathlib import Path

import pytest
from llama_index.core.embeddings import BaseEmbedding

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

FIXTURES = Path(__file__).parent / "fixtures"
DIM = 32


def bow(text: str, dim: int = DIM) -> list[float]:
    """Bag-of-words hashing vector: texts sharing words get similar vectors."""
    v = [0.0] * dim
    for word in text.lower().split():
        h = int(hashlib.md5(word.strip(".,:;!?()[]").encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class BowEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> list[float]:
        return bow(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return bow(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return bow(query)


class FakeLLM:
    """Scripted stand-in for ClaudeLLM (``complete`` / ``judge``).

    * ``Queries`` judge -> ``queries`` (extra search queries);
    * ``Support`` judge -> next of ``supported`` (then True);
    * ``complete``: revise prompts (system mentions "revise") -> ``revised``; else ``answer``.
    """

    def __init__(
        self,
        model="fake-llm",
        queries=None,
        supported=None,
        answer="answer [1]",
        revised="revised answer [1]",
    ):
        self.model = model
        self.queries = queries if queries is not None else ["Orion speed km/h"]
        self.supported = list(supported or [])
        self.answer, self.revised = answer, revised
        self.calls: list[tuple[str, str, str]] = []

    def complete(self, system, user):
        kind = "revise" if "revise" in system.lower() else "generate"
        self.calls.append((kind, system, user))
        return self.revised if kind == "revise" else self.answer

    def judge(self, system, user, schema):
        self.calls.append((schema.__name__, system, user))
        if schema.__name__ == "Queries":
            return schema(queries=list(self.queries))
        ok = self.supported.pop(0) if self.supported else True
        return schema(supported=ok, unsupported_claims="" if ok else "the made-up claim")

    def kinds(self):
        return [k for k, _, _ in self.calls]


class FakeReranker:
    """Reverses the fused order and keeps top_n, so tests can see the reranker acted."""

    def __init__(self, top_n):
        self.top_n, self.calls = top_n, []

    def postprocess_nodes(self, nodes, query_str=None):
        self.calls.append(query_str)
        return list(reversed(nodes))[: self.top_n]


class EmbedSpecLike:
    def __init__(self):
        self.model = BowEmbedding()
        self.name = "fake:bow-32"
        self.dim = DIM


@pytest.fixture
def embed_spec():
    return EmbedSpecLike()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def settings(tmp_path):
    from advanced_rag.config import get_settings

    return get_settings(
        _env_file=None,
        vector_store="faiss",
        index_dir=tmp_path / ".index",
        data_dir=FIXTURES,
        top_k=2,
    )
