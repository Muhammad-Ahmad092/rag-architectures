"""Settings loaded from environment / ``.env``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["openai", "gemini", "cohere", "voyage", "huggingface"]
VectorStoreName = Literal[
    "chroma", "faiss", "qdrant-local", "lancedb", "pinecone", "qdrant-cloud", "weaviate-cloud"
]


class ConfigError(Exception):
    """Raised for user-fixable configuration problems (missing key, missing extra, ...)."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM (this folder: OpenAI, also used for the retrieval evaluator and openai embeddings)
    openai_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"

    # CRAG retrieval evaluator: per-passage relevance score in [0, 1]
    upper_threshold: float = Field(0.7, ge=0.0, le=1.0)  # any passage >= -> CORRECT
    lower_threshold: float = Field(0.3, ge=0.0, le=1.0)  # all passages < -> INCORRECT
    refine_knowledge: bool = True  # keep only the question-relevant sentences of kept passages

    # Web search fallback (Tavily)
    tavily_api_key: str | None = None
    web_search: bool = True  # off, or no key -> answer from internal passages only
    web_results: int = Field(3, ge=1, le=10)

    # Other providers / eval judge
    anthropic_api_key: str | None = None
    gemini_api_key: str | None = None

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
    chunk_size: int = 1000  # characters, RecursiveCharacterTextSplitter
    chunk_overlap: int = 150

    # Evaluation (fair mode): common generator + separate judge
    eval_llm: str = "gpt-4o-mini"
    eval_judge_model: str = "claude-sonnet-5-5"

    @model_validator(mode="after")
    def _thresholds_ordered(self):
        if self.lower_threshold > self.upper_threshold:
            raise ValueError("LOWER_THRESHOLD must be <= UPPER_THRESHOLD")
        return self


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
