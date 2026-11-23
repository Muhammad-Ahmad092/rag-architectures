"""Embedding factory tests: LlamaIndex integration classes are replaced by recorders."""

import importlib
from types import SimpleNamespace

import pytest

from advanced_rag import embeddings as emb
from advanced_rag.config import ConfigError, get_settings

KEYS = dict(openai_api_key="ko", gemini_api_key="kg", cohere_api_key="kc", voyage_api_key="kv")


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _module(cls_name):
    return SimpleNamespace(**{cls_name: type(cls_name, (Recorder,), {})})


FAKES = {
    "llama_index.embeddings.openai": _module("OpenAIEmbedding"),
    "llama_index.embeddings.google_genai": _module("GoogleGenAIEmbedding"),
    "llama_index.embeddings.cohere": _module("CohereEmbedding"),
    "llama_index.embeddings.voyageai": _module("VoyageEmbedding"),
    "llama_index.embeddings.huggingface": _module("HuggingFaceEmbedding"),
}


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(emb, "import_extra", lambda module, extra: FAKES[module])


@pytest.mark.parametrize(
    "provider,cls,model,key",
    [
        ("openai", "OpenAIEmbedding", "text-embedding-3-small", "ko"),
        ("gemini", "GoogleGenAIEmbedding", "gemini-embedding-001", "kg"),
        ("cohere", "CohereEmbedding", "embed-english-v3.0", "kc"),
        ("voyage", "VoyageEmbedding", "voyage-3.5-lite", "kv"),
        ("huggingface", "HuggingFaceEmbedding", "BAAI/bge-small-en-v1.5", None),
    ],
)
def test_factory_selects_each_provider(fakes, provider, cls, model, key):
    spec = emb.get_embed_model(get_settings(_env_file=None, embedding_provider=provider, **KEYS))
    assert type(spec.model).__name__ == cls
    assert model in spec.model.kwargs.values()
    if key:
        assert key in spec.model.kwargs.values()
    assert spec.name == f"{provider}:{model}"


def test_known_dims_without_network(fakes):
    s = get_settings(_env_file=None, embedding_provider="gemini", **KEYS)
    spec = emb.get_embed_model(s)
    assert spec.dim == 768
    assert spec.model.kwargs["embedding_config"] == {"output_dimensionality": 768}


def test_empty_embedding_model_uses_default(fakes):
    s = get_settings(_env_file=None, embedding_provider="cohere", embedding_model="", **KEYS)
    assert emb.get_embed_model(s).name == "cohere:embed-english-v3.0"


def test_missing_key_raises_config_error(fakes):
    s = get_settings(_env_file=None, embedding_provider="gemini", gemini_api_key=None)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        emb.get_embed_model(s)


def test_missing_extra_raises_config_error(monkeypatch):
    real = importlib.import_module

    def fake_import(name, *a, **kw):
        if name == "llama_index.embeddings.voyageai":
            raise ImportError("missing")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    s = get_settings(_env_file=None, embedding_provider="voyage", voyage_api_key="k")
    with pytest.raises(ConfigError, match="--extra voyage"):
        emb.get_embed_model(s)


@pytest.mark.slow
def test_huggingface_real_dim():
    spec = emb.get_embed_model(get_settings(_env_file=None, embedding_provider="huggingface"))
    assert spec.dim == 384
    assert len(spec.model.get_text_embedding("hi")) == 384
