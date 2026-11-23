"""LangChain embedding models for all 5 providers, selected by ``EMBEDDING_PROVIDER``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ConfigError, Settings, import_extra, require

DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "cohere": "embed-english-v3.0",
    "voyage": "voyage-3.5-lite",
    "huggingface": "BAAI/bge-small-en-v1.5",
}

GEMINI_DIM = 768
KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "gemini-embedding-001": GEMINI_DIM,
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3.5": 1024,
    "BAAI/bge-small-en-v1.5": 384,
}


@dataclass
class EmbedSpec:
    """A LangChain ``Embeddings`` plus the identity used to name its index."""

    model: Any
    name: str  # "<provider>:<model>"
    dim: int


def _build(provider: str, model: str, s: Settings):
    if provider == "openai":
        key = require(s.openai_api_key, "OPENAI_API_KEY")
        return import_extra("langchain_openai", "openai").OpenAIEmbeddings(model=model, api_key=key)
    if provider == "gemini":
        key = require(s.gemini_api_key, "GEMINI_API_KEY")
        mod = import_extra("langchain_google_genai", "gemini")
        return mod.GoogleGenerativeAIEmbeddings(
            model=model, api_key=key, output_dimensionality=GEMINI_DIM
        )
    if provider == "cohere":
        key = require(s.cohere_api_key, "COHERE_API_KEY")
        mod = import_extra("langchain_cohere", "cohere")
        return mod.CohereEmbeddings(model=model, cohere_api_key=key)
    if provider == "voyage":
        key = require(s.voyage_api_key, "VOYAGE_API_KEY")
        mod = import_extra("langchain_voyageai", "voyage")
        return mod.VoyageAIEmbeddings(model=model, api_key=key)
    if provider == "huggingface":
        mod = import_extra("langchain_huggingface", "huggingface")
        return mod.HuggingFaceEmbeddings(
            model_name=model, encode_kwargs={"normalize_embeddings": True}
        )
    raise ConfigError(f"Unknown EMBEDDING_PROVIDER '{provider}'")


def get_embeddings(settings: Settings) -> EmbedSpec:
    provider = settings.embedding_provider
    if provider not in DEFAULT_MODELS:
        raise ConfigError(f"Unknown EMBEDDING_PROVIDER '{provider}'")
    model_name = settings.embedding_model or DEFAULT_MODELS[provider]
    model = _build(provider, model_name, settings)
    dim = KNOWN_DIMS.get(model_name)
    if dim is None:  # unknown model: probe once
        dim = len(model.embed_query("dimension probe"))
    return EmbedSpec(model=model, name=f"{provider}:{model_name}", dim=dim)
