"""Offline fakes shared by the tests (no API keys, no model downloads)."""

import hashlib
import math
import os
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

FIXTURES = Path(__file__).parent / "fixtures"
DIM = 32


def bow(text: str, dim: int = DIM) -> list[float]:
    """Bag-of-words hashing vector: texts sharing words get similar vectors."""
    v = [0.0] * dim
    for word in text.lower().split():
        h = int(hashlib.md5(word.strip(".,:;!?()[]\"'").encode()).hexdigest(), 16)
        v[h % dim] += 1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class BowEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [bow(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return bow(text)


class FakeLLM:
    """Scripted stand-in for ClaudeLLM.

    ``complete``: rewrite prompts (system mentions "search query") return the next item of
    ``rewrites``; others return the next item of ``answers`` (the last one repeats).
    ``judge``: ``verdicts[schema_name]`` is a list of bools consumed in order (then True),
    or a callable ``user_text -> bool``; unspecified graders always say yes.
    """

    def __init__(self, model="fake-llm", answers=None, rewrites=None, verdicts=None):
        self.model = model
        self.answers = list(answers or ["answer [1]"])
        self.rewrites = list(rewrites or ["rewritten query"])
        self.verdicts = dict(verdicts or {})
        self.calls: list[tuple[str, str, str]] = []  # (kind, system, user)

    def complete(self, system, user):
        if "search query" in system.lower():
            self.calls.append(("rewrite", system, user))
            return self.rewrites.pop(0) if len(self.rewrites) > 1 else self.rewrites[0]
        self.calls.append(("generate", system, user))
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    def judge(self, system, user, schema):
        name = schema.__name__
        self.calls.append((name, system, user))
        v = self.verdicts.get(name, True)
        if callable(v):
            value = v(user)
        elif isinstance(v, list):
            value = v.pop(0) if v else True
        else:
            value = bool(v)
        fields = {}
        for field, info in schema.model_fields.items():
            fields[field] = value if info.annotation is bool else f"fake {field}"
        return schema(**fields)

    def kinds(self):
        return [k for k, _, _ in self.calls]


class EmbedSpecLike:
    def __init__(self):
        self.model = BowEmbeddings()
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
    from self_rag.config import get_settings

    return get_settings(
        _env_file=None,
        vector_store="faiss",
        index_dir=tmp_path / ".index",
        data_dir=FIXTURES,
        top_k=2,
    )
