"""Embedding factory tests: LangChain integration classes are replaced by recorders."""

import importlib
from types import SimpleNamespace

import pytest

from simple_rag_memory import embeddings as emb
from simple_rag_memory.config import ConfigError, get_settings

KEYS = dict(openai_api_key="ko", gemini_api_key="kg", cohere_api_key="kc", voyage_api_key="kv")


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _module(cls_name):
    return SimpleNamespace(**{cls_name: type(cls_name, (Recorder,), {})})


FAKES = {
    "langchain_openai": _module("OpenAIEmbeddings"),
    "langchain_google_genai": _module("GoogleGenerativeAIEmbeddings"),
    "langchain_cohere": _module("CohereEmbeddings"),
    "langchain_voyageai": _module("VoyageAIEmbeddings"),
    "langchain_huggingface": _module("HuggingFaceEmbeddings"),
}


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(emb, "import_extra", lambda module, extra: FAKES[module])


@pytest.mark.parametrize(
    "provider,cls,model,key",
    [
        ("openai", "OpenAIEmbeddings", "text-embedding-3-small", "ko"),
        ("gemini", "GoogleGenerativeAIEmbeddings", "gemini-embedding-001", "kg"),
        ("cohere", "CohereEmbeddings", "embed-english-v3.0", "kc"),
        ("voyage", "VoyageAIEmbeddings", "voyage-3.5-lite", "kv"),
        ("huggingface", "HuggingFaceEmbeddings", "BAAI/bge-small-en-v1.5", None),
    ],
)
def test_factory_selects_each_provider(fakes, provider, cls, model, key):
    spec = emb.get_embeddings(get_settings(_env_file=None, embedding_provider=provider, **KEYS))
    assert type(spec.model).__name__ == cls
    assert model in spec.model.kwargs.values()
    if key:
        assert key in spec.model.kwargs.values()
    assert spec.name == f"{provider}:{model}"


def test_known_dims_without_network(fakes):
    spec = emb.get_embeddings(get_settings(_env_file=None, embedding_provider="gemini", **KEYS))
    assert spec.dim == 768
    assert spec.model.kwargs["output_dimensionality"] == 768


def test_huggingface_normalizes(fakes):
    spec = emb.get_embeddings(get_settings(_env_file=None, embedding_provider="huggingface"))
    assert spec.model.kwargs["encode_kwargs"] == {"normalize_embeddings": True}
    assert spec.dim == 384


def test_empty_embedding_model_uses_default(fakes):
    s = get_settings(_env_file=None, embedding_provider="cohere", embedding_model="", **KEYS)
    assert emb.get_embeddings(s).name == "cohere:embed-english-v3.0"


def test_unknown_model_dim_is_probed(fakes, monkeypatch):
    class Probe(Recorder):
        def embed_query(self, text):
            return [0.0] * 5

    monkeypatch.setitem(FAKES, "langchain_openai", SimpleNamespace(OpenAIEmbeddings=Probe))
    s = get_settings(_env_file=None, embedding_provider="openai", embedding_model="custom", **KEYS)
    assert emb.get_embeddings(s).dim == 5


def test_missing_key_raises_config_error(fakes):
    s = get_settings(_env_file=None, embedding_provider="gemini", gemini_api_key=None)
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        emb.get_embeddings(s)


def test_missing_extra_raises_config_error(monkeypatch):
    real = importlib.import_module

    def fake_import(name, *a, **kw):
        if name == "langchain_voyageai":
            raise ImportError("missing")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    s = get_settings(_env_file=None, embedding_provider="voyage", voyage_api_key="k")
    with pytest.raises(ConfigError, match="--extra voyage"):
        emb.get_embeddings(s)


@pytest.mark.slow
def test_huggingface_real_dim():
    spec = emb.get_embeddings(get_settings(_env_file=None, embedding_provider="huggingface"))
    assert spec.dim == 384
    assert len(spec.model.embed_query("hi")) == 384
