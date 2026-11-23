"""Settings loaded from environment / ``.env``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["openai", "gemini", "cohere", "voyage", "huggingface"]
VectorStoreName = Literal[
    "chroma", "faiss", "qdrant-local", "lancedb", "pinecone", "qdrant-cloud", "weaviate-cloud"
]


class ConfigError(Exception):
    """Raised for user-fixable configuration problems (missing key, missing extra, ...)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (this folder: Mistral). OpenAI is a swappable generator module + fair-mode eval.
    mistral_api_key: str | None = None
    llm_model: str = "mistral-small-latest"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Default modules (each can be swapped per question in the UI)
    retriever: Literal["vector", "bm25", "hybrid"] = "hybrid"
    ranker: Literal["none", "cross-encoder", "lost-in-the-middle"] = "none"
    generator: Literal["mistral", "openai"] = "mistral"
    candidate_k: int = Field(8, ge=1, le=50)  # retrieved before ranking; TOP_K kept after
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Embeddings
    embedding_provider: EmbeddingProvider = "huggingface"
    embedding_model: str | None = None  # None -> provider default (see embeddings.py)
    gemini_api_key: str | None = None
    cohere_api_key: str | None = None
    voyage_api_key: str | None = None

    # Vector store
    vector_store: VectorStoreName = "chroma"
    index_dir: Path = Path(".index")
    pinecone_api_key: str | None = None
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    weaviate_url: str | None = None
    weaviate_api_key: str | None = None

    # Data / retrieval
    data_dir: Path = Path("../sample_data")
    eval_questions: Path = Path("../sample_data/eval/questions.jsonl")
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Evaluation (fair mode): common generator + separate judge
    eval_llm: str = "gpt-4o-mini"
    anthropic_api_key: str | None = None
    eval_judge_model: str = "claude-sonnet-5-5"


def get_settings(**overrides) -> Settings:
    return Settings(**overrides)


def require(value: str | None, env_var: str) -> str:
    """Return ``value`` or raise a ConfigError naming the env var to set."""
    if not value:
        raise ConfigError(f"Set {env_var} in .env")
    return value


def import_extra(module: str, extra: str):
    """Import an optional dependency or raise a ConfigError naming the uv extra."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ConfigError(
            f"Missing optional dependency '{module}'. Install with: uv sync --extra {extra}"
        ) from exc
