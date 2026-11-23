import pytest
from pydantic import ValidationError

from advanced_rag.config import get_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("VECTOR_STORE", "EMBEDDING_PROVIDER", "LLM_MODEL", "RERANKER", "NUM_QUERIES"):
        monkeypatch.delenv(var, raising=False)


def test_defaults_need_only_anthropic_key():
    s = get_settings(_env_file=None)
    assert (s.llm_model, s.llm_effort, s.grader_effort) == ("claude-opus-5-5", "medium", "low")
    assert (s.num_queries, s.candidate_k, s.rrf_k, s.top_k) == (3, 8, 60, 4)
    assert (s.reranker, s.rerank_model, s.check_answer) == ("cohere", "rerank-v3.5", True)
    assert s.vector_store == "chroma" and s.embedding_provider == "huggingface"


def test_invalid_reranker_rejected():
    with pytest.raises(ValidationError):
        get_settings(_env_file=None, reranker="magic")
