"""Offline fakes shared by the tests (no API keys, no model downloads)."""

import hashlib
import math
import os
from pathlib import Path
from typing import Any

import pytest
from llama_index.core.base.llms.types import CompletionResponse, LLMMetadata
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import CustomLLM
from llama_index.core.llms.callbacks import llm_completion_callback

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


class FakeLLM(CustomLLM):
    """Records every prompt; always answers citing source 1."""

    prompts: list[str] = []
    model: str = "fake-llm"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name=self.model, context_window=32000, num_output=256)

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        self.prompts.append(prompt)
        return CompletionResponse(text="answer [1]")

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        yield self.complete(prompt)


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
    return FakeLLM(prompts=[])


@pytest.fixture
def settings(tmp_path):
    from simple_rag.config import get_settings

    return get_settings(
        _env_file=None,
        vector_store="faiss",
        index_dir=tmp_path / ".index",
        data_dir=FIXTURES,
        top_k=2,
    )
