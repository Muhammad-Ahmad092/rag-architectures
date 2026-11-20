import pytest
from pydantic import ValidationError

from simple_rag_memory.config import get_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "VECTOR_STORE",
        "EMBEDDING_PROVIDER",
        "TOP_K",
        "LLM_MODEL",
        "REWRITE_MODEL",
        "MEMORY_TURNS",
        "CHUNK_SIZE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_need_only_groq_key():
    s = get_settings(_env_file=None)
    assert s.llm_model == "llama-3.3-70b-versatile"
    assert s.rewrite_model is None  # falls back to LLM_MODEL
    assert s.vector_store == "chroma"
    assert s.embedding_provider == "huggingface"
    assert (s.chunk_size, s.chunk_overlap, s.top_k, s.memory_turns) == (1000, 150, 4, 6)


def test_invalid_store_rejected(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "mongo")
    with pytest.raises(ValidationError):
        get_settings(_env_file=None)


def test_env_override(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "faiss")
    monkeypatch.setenv("MEMORY_TURNS", "2")
    s = get_settings(_env_file=None)
    assert s.vector_store == "faiss" and s.memory_turns == 2


def test_memory_turns_must_be_positive():
    with pytest.raises(ValidationError):
        get_settings(_env_file=None, memory_turns=0)
