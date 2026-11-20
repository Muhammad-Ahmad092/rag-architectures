"""Offline fakes shared by the tests (no API keys, no model downloads)."""

import hashlib
import math
import os
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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


class FakeChat(BaseChatModel):
    """Records every message list. Rewrite prompts get ``rewrite_reply``; others ``answer``."""

    calls: list[list[BaseMessage]] = []
    rewrite_reply: str = "standalone question"
    answer: str = "answer [1]"
    model_name: str = "fake-chat"

    @property
    def _llm_type(self) -> str:
        return "fake-chat"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        self.calls.append(list(messages))
        is_rewrite = "standalone question" in str(messages[0].content).lower()
        text = self.rewrite_reply if is_rewrite else self.answer
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def rewrite_calls(self):
        return [c for c in self.calls if "standalone question" in str(c[0].content).lower()]

    def answer_calls(self):
        return [c for c in self.calls if "standalone question" not in str(c[0].content).lower()]


class EmbedSpecLike:
    def __init__(self):
        self.model = BowEmbeddings()
        self.name = "fake:bow-32"
        self.dim = DIM


@pytest.fixture
def embed_spec():
    return EmbedSpecLike()


@pytest.fixture
def fake_chat():
    return FakeChat(calls=[])


@pytest.fixture
def settings(tmp_path):
    from simple_rag_memory.config import get_settings

    return get_settings(
        _env_file=None,
        vector_store="faiss",
        index_dir=tmp_path / ".index",
        data_dir=FIXTURES,
        top_k=2,
        memory_turns=2,
    )
