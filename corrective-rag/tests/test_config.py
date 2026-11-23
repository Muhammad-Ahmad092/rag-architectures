import pytest
from pydantic import ValidationError

from corrective_rag.config import get_settings


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "VECTOR_STORE",
        "EMBEDDING_PROVIDER",
        "LLM_MODEL",
        "UPPER_THRESHOLD",
        "LOWER_THRESHOLD",
        "WEB_SEARCH",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_defaults_need_only_openai_key():
    s = get_settings(_env_file=None)
    assert s.llm_model == "gpt-4o-mini"
    assert (s.upper_threshold, s.lower_threshold) == (0.7, 0.3)
    assert s.refine_knowledge and s.web_search and s.web_results == 3
    assert s.vector_store == "chroma" and s.embedding_provider == "huggingface"


def test_thresholds_must_be_ordered():
    with pytest.raises(ValidationError, match="LOWER_THRESHOLD"):
        get_settings(_env_file=None, upper_threshold=0.2, lower_threshold=0.5)


def test_env_override(monkeypatch):
    monkeypatch.setenv("WEB_SEARCH", "false")
    monkeypatch.setenv("UPPER_THRESHOLD", "0.8")
    s = get_settings(_env_file=None)
    assert s.web_search is False and s.upper_threshold == 0.8
