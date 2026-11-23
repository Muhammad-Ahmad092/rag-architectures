"""LlamaIndex embedding models for all 5 providers, selected by ``EMBEDDING_PROVIDER``."""

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
    """A LlamaIndex ``BaseEmbedding`` plus the identity used to name its index."""

    model: Any
    name: str  # "<provider>:<model>"
    dim: int


def _build(provider: str, model: str, s: Settings):
    if provider == "openai":
        key = require(s.openai_api_key, "OPENAI_API_KEY")
        mod = import_extra("llama_index.embeddings.openai", "openai")
        return mod.OpenAIEmbedding(model=model, api_key=key)
    if provider == "gemini":
        key = require(s.gemini_api_key, "GEMINI_API_KEY")
        mod = import_extra("llama_index.embeddings.google_genai", "gemini")
        return mod.GoogleGenAIEmbedding(
            model_name=model, api_key=key, embedding_config={"output_dimensionality": GEMINI_DIM}
        )
    if provider == "cohere":
        key = require(s.cohere_api_key, "COHERE_API_KEY")
        mod = import_extra("llama_index.embeddings.cohere", "cohere")
        return mod.CohereEmbedding(model_name=model, api_key=key)
    if provider == "voyage":
        key = require(s.voyage_api_key, "VOYAGE_API_KEY")
        mod = import_extra("llama_index.embeddings.voyageai", "voyage")
        return mod.VoyageEmbedding(model_name=model, voyage_api_key=key)
    if provider == "huggingface":
        mod = import_extra("llama_index.embeddings.huggingface", "huggingface")
        return mod.HuggingFaceEmbedding(model_name=model, normalize=True)
    raise ConfigError(f"Unknown EMBEDDING_PROVIDER '{provider}'")


def get_embed_model(settings: Settings) -> EmbedSpec:
    provider = settings.embedding_provider
    if provider not in DEFAULT_MODELS:
        raise ConfigError(f"Unknown EMBEDDING_PROVIDER '{provider}'")
    model_name = settings.embedding_model or DEFAULT_MODELS[provider]
    model = _build(provider, model_name, settings)
    dim = KNOWN_DIMS.get(model_name)
    if dim is None:  # local or unknown model: probe once
        dim = len(model.get_text_embedding("dimension probe"))
    return EmbedSpec(model=model, name=f"{provider}:{model_name}", dim=dim)
