import pytest
from pydantic import ValidationError

from simple_rag.config import get_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("VECTOR_STORE", "EMBEDDING_PROVIDER", "TOP_K", "LLM_MODEL", "CHUNK_SIZE"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_need_only_gemini_key():
    s = get_settings(_env_file=None)
    assert s.llm_model == "gemini-flash-latest"
    assert s.vector_store == "chroma"
    assert s.embedding_provider == "huggingface"
    assert (s.chunk_size, s.chunk_overlap, s.citation_chunk_size, s.top_k) == (512, 50, 256, 4)


def test_invalid_store_rejected(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "mongo")
    with pytest.raises(ValidationError):
        get_settings(_env_file=None)


def test_env_override(monkeypatch):
    monkeypatch.setenv("VECTOR_STORE", "faiss")
    assert get_settings(_env_file=None).vector_store == "faiss"
