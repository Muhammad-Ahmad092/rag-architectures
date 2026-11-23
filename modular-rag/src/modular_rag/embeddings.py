"""Pluggable embedding providers selected by ``EMBEDDING_PROVIDER``.

SDKs are imported lazily so only the selected provider's package must be installed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from .config import ConfigError, Settings, import_extra, require

BATCH = 96  # Cohere's per-call max; used for every API provider

DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "cohere": "embed-english-v3.0",
    "voyage": "voyage-3.5-lite",
    "huggingface": "BAAI/bge-small-en-v1.5",
}

KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "gemini-embedding-001": 768,  # we request 768 via output_dimensionality
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3.5": 1024,
}

GEMINI_DIM = 768


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _batches(texts: list[str]) -> Iterator[list[str]]:
    for i in range(0, len(texts), BATCH):
        yield texts[i : i + BATCH]


class _Base:
    provider = ""

    def __init__(self, model: str):
        self.model = model
        self.name = f"{self.provider}:{model}"
        self._dim: int | None = KNOWN_DIMS.get(model)

    @property
    def dim(self) -> int:
        if self._dim is None:  # unknown model: probe once
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for batch in _batches(texts):
            out.extend(self._embed(batch, query=False))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], query=True)[0]

    def _embed(self, texts: list[str], query: bool) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(_Base):
    provider = "openai"

    def __init__(self, settings: Settings, model: str):
        key = require(settings.openai_api_key, "OPENAI_API_KEY")
        super().__init__(model)
        self._client = import_extra("openai", "openai").OpenAI(api_key=key)

    def _embed(self, texts, query):
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class GeminiEmbedder(_Base):
    provider = "gemini"

    def __init__(self, settings: Settings, model: str):
        key = require(settings.gemini_api_key, "GEMINI_API_KEY")
        super().__init__(model)
        self._client = import_extra("google.genai", "gemini").Client(api_key=key)
        self._types = import_extra("google.genai.types", "gemini")

    def _embed(self, texts, query):
        config = self._types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT",
            output_dimensionality=GEMINI_DIM,
        )
        resp = self._client.models.embed_content(model=self.model, contents=texts, config=config)
        return [list(e.values) for e in resp.embeddings]


class CohereEmbedder(_Base):
    provider = "cohere"

    def __init__(self, settings: Settings, model: str):
        key = require(settings.cohere_api_key, "COHERE_API_KEY")
        super().__init__(model)
        self._client = import_extra("cohere", "cohere").ClientV2(api_key=key)

    def _embed(self, texts, query):
        resp = self._client.embed(
            texts=texts,
            model=self.model,
            input_type="search_query" if query else "search_document",
            embedding_types=["float"],
        )
        return [list(v) for v in resp.embeddings.float_]


class VoyageEmbedder(_Base):
    provider = "voyage"

    def __init__(self, settings: Settings, model: str):
        key = require(settings.voyage_api_key, "VOYAGE_API_KEY")
        super().__init__(model)
        self._client = import_extra("voyageai", "voyage").Client(api_key=key)

    def _embed(self, texts, query):
        resp = self._client.embed(
            texts, model=self.model, input_type="query" if query else "document"
        )
        return [list(v) for v in resp.embeddings]


class HuggingFaceEmbedder(_Base):
    """Local sentence-transformers model. Free, no key, downloads once."""

    provider = "huggingface"

    def __init__(self, settings: Settings, model: str):
        super().__init__(model)
        st = import_extra("sentence_transformers", "huggingface")
        self._model = st.SentenceTransformer(model)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension  # sentence-transformers < 5
        )
        self._dim = int(get_dim())

    def _embed(self, texts, query):
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, v)) for v in vectors]


_PROVIDERS = {
    "openai": OpenAIEmbedder,
    "gemini": GeminiEmbedder,
    "cohere": CohereEmbedder,
    "voyage": VoyageEmbedder,
    "huggingface": HuggingFaceEmbedder,
}


def get_embedder(settings: Settings) -> Embedder:
    provider = settings.embedding_provider
    if provider not in _PROVIDERS:
        raise ConfigError(f"Unknown EMBEDDING_PROVIDER '{provider}'")
    model = settings.embedding_model or DEFAULT_MODELS[provider]
    return _PROVIDERS[provider](settings, model)
