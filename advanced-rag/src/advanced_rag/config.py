"""Settings loaded from environment / ``.env``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]
EmbeddingProvider = Literal["openai", "gemini", "cohere", "voyage", "huggingface"]
VectorStoreName = Literal[
    "chroma", "faiss", "qdrant-local", "lancedb", "pinecone", "qdrant-cloud", "weaviate-cloud"
]


class ConfigError(Exception):
    """Raised for user-fixable configuration problems (missing key, missing extra, ...)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (this folder: Anthropic Claude via the official SDK; also the eval judge)
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-5-5"
    llm_effort: Effort = "medium"
    # Rewriter + answer checker. Empty = LLM_MODEL; e.g. claude-haiku-5-5
    grader_model: str | None = None
    grader_effort: Effort = "low"
    # Other providers: openai embeddings + fair-mode eval generator; gemini embeddings
    openai_api_key: str | None = None
    gemini_api_key: str | None = None

    # Advanced pipeline
    num_queries: int = Field(3, ge=1, le=6)  # rewritten search queries (incl. the original)
    candidate_k: int = Field(8, ge=1, le=50)  # per retriever per query, before fusion
    rrf_k: int = Field(60, ge=1)  # reciprocal rank fusion constant
    reranker: Literal["cohere", "local", "none"] = "cohere"  # cohere w/o key -> local
    rerank_model: str = "rerank-v3.5"  # Cohere model
    local_rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    check_answer: bool = True  # verify support; revise once if claims are unsupported

    # Embeddings
    embedding_provider: EmbeddingProvider = "huggingface"
    embedding_model: str | None = None  # None -> provider default (see embeddings.py)
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
    chunk_size: int = 512  # tokens, LlamaIndex SentenceSplitter
    chunk_overlap: int = 50

    # Evaluation (fair mode): common generator + separate judge
    eval_llm: str = "gpt-4o-mini"
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
