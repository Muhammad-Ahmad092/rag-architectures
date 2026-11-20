"""Embedding factory tests. SDKs are replaced by fakes at the import boundary (no network)."""

from types import SimpleNamespace

import pytest

from naive_rag import embeddings as emb
from naive_rag.config import ConfigError, get_settings

DIM = 8


def _vec(text: str) -> list[float]:
    return [float(len(text))] + [0.0] * (DIM - 1)


class _FakeOpenAI:
    def __init__(self, api_key):
        self.embeddings = SimpleNamespace(
            create=lambda model, input: SimpleNamespace(
                data=[SimpleNamespace(embedding=_vec(t)) for t in input]
            )
        )


class _FakeGenaiClient:
    def __init__(self, api_key):
        self.models = SimpleNamespace(
            embed_content=lambda model, contents, config=None: SimpleNamespace(
                embeddings=[SimpleNamespace(values=_vec(t)) for t in contents]
            )
        )


class _FakeCohere:
    def __init__(self, api_key):
        pass

    def embed(self, texts, model, input_type, embedding_types):
        assert input_type in ("search_document", "search_query")
        return SimpleNamespace(embeddings=SimpleNamespace(float_=[_vec(t) for t in texts]))


class _FakeVoyage:
    def __init__(self, api_key):
        pass

    def embed(self, texts, model, input_type):
        return SimpleNamespace(embeddings=[_vec(t) for t in texts])


class _FakeST:
    def __init__(self, model):
        pass

    def get_sentence_embedding_dimension(self):
        return DIM

    def encode(self, texts, normalize_embeddings=True):
        return [_vec(t) for t in texts]


FAKE_MODULES = {
    "openai": SimpleNamespace(OpenAI=_FakeOpenAI),
    "google.genai": SimpleNamespace(Client=_FakeGenaiClient),
    "google.genai.types": SimpleNamespace(EmbedContentConfig=lambda **kw: kw),
    "cohere": SimpleNamespace(ClientV2=_FakeCohere),
    "voyageai": SimpleNamespace(Client=_FakeVoyage),
    "sentence_transformers": SimpleNamespace(SentenceTransformer=_FakeST),
}

KEYS = {
    "openai_api_key": "k",
    "gemini_api_key": "k",
    "cohere_api_key": "k",
    "voyage_api_key": "k",
}


@pytest.fixture
def fake_sdks(monkeypatch):
    monkeypatch.setattr(emb, "import_extra", lambda module, extra: FAKE_MODULES[module])


@pytest.mark.parametrize(
    "provider,cls",
    [
        ("openai", "OpenAIEmbedder"),
        ("gemini", "GeminiEmbedder"),
        ("cohere", "CohereEmbedder"),
        ("voyage", "VoyageEmbedder"),
        ("huggingface", "HuggingFaceEmbedder"),
    ],
)
def test_factory_selects_each_provider(fake_sdks, provider, cls):
    s = get_settings(_env_file=None, embedding_provider=provider, **KEYS)
    e = emb.get_embedder(s)
    assert type(e).__name__ == cls
    assert len(e.embed(["a", "bb"])) == 2
    assert len(e.embed_query("abc")) == DIM
    assert e.name.startswith(f"{provider}:")


def test_empty_embedding_model_uses_provider_default(fake_sdks):
    s = get_settings(_env_file=None, embedding_provider="openai", embedding_model="", **KEYS)
    assert emb.get_embedder(s).name == "openai:text-embedding-3-small"


def test_batches_large_inputs(fake_sdks, monkeypatch):
    calls = []

    class Counting(_FakeVoyage):
        def embed(self, texts, model, input_type):
            calls.append(len(texts))
            return super().embed(texts, model, input_type)

    FAKE = dict(FAKE_MODULES, voyageai=SimpleNamespace(Client=Counting))
    monkeypatch.setattr(emb, "import_extra", lambda module, extra: FAKE[module])
    e = emb.get_embedder(get_settings(_env_file=None, embedding_provider="voyage", **KEYS))
    assert len(e.embed(["x"] * 200)) == 200
    assert calls == [96, 96, 8]


def test_missing_key_raises_config_error(fake_sdks):
    s = get_settings(_env_file=None, embedding_provider="openai", openai_api_key=None)
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        emb.get_embedder(s)


def test_missing_extra_raises_config_error(monkeypatch):
    import importlib

    real = importlib.import_module

    def fake_import(name, *a, **kw):
        if name == "voyageai":
            raise ImportError("no voyageai")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    s = get_settings(_env_file=None, embedding_provider="voyage", voyage_api_key="k")
    with pytest.raises(ConfigError, match="--extra voyage"):
        emb.get_embedder(s)


@pytest.mark.slow
def test_huggingface_real_dim():
    e = emb.get_embedder(get_settings(_env_file=None, embedding_provider="huggingface"))
    assert e.dim == 384
    assert len(e.embed_query("hello")) == 384
