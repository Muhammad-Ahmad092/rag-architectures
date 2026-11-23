import pytest
from pydantic import ValidationError

from self_rag.config import get_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "VECTOR_STORE",
        "EMBEDDING_PROVIDER",
        "TOP_K",
        "LLM_MODEL",
        "GRADER_MODEL",
        "MAX_REWRITES",
        "MAX_GENERATIONS",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_need_only_anthropic_key():
    s = get_settings(_env_file=None)
    assert (s.llm_model, s.llm_effort) == ("claude-opus-5-5", "medium")
    assert (s.grader_model, s.grader_effort) == (None, "low")
    assert (s.max_rewrites, s.max_generations) == (2, 2)
    assert s.vector_store == "chroma" and s.embedding_provider == "huggingface"


def test_invalid_effort_rejected():
    with pytest.raises(ValidationError):
        get_settings(_env_file=None, llm_effort="turbo")


def test_env_override(monkeypatch):
    monkeypatch.setenv("GRADER_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("MAX_REWRITES", "0")
    s = get_settings(_env_file=None)
    assert s.grader_model == "claude-haiku-5-5" and s.max_rewrites == 0
