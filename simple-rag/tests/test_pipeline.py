import re

import pytest
from conftest import BowEmbedding

from simple_rag.config import ConfigError, get_settings
from simple_rag.ingest import build_nodes, ingest
from simple_rag.llm import get_llm
from simple_rag.pipeline import EMPTY_INDEX_MSG, SimpleRAG
from simple_rag.vectorstore import get_vector_store


@pytest.fixture
def store(settings, embed_spec):
    return get_vector_store(settings, embed_spec.name, embed_spec.dim)


@pytest.fixture
def rag(settings, embed_spec, store, fake_llm):
    ingest(settings, embed_spec=embed_spec, store=store)
    return SimpleRAG(settings, embed_spec=embed_spec, store=store, llm=fake_llm)


def test_answer_returns_citations_in_engine_order(rag, fake_llm):
    result = rag.answer("Orion rover top speed")
    assert result.answer == "answer [1]"
    prompt = fake_llm.prompts[-1]
    for n, src in enumerate(result.sources, start=1):
        assert f"Source {n}:" in prompt
        assert src.chunk.text.strip()[:40] in prompt
    first = prompt.index("Source 1:")
    assert result.sources[0].chunk.text.strip()[:40] in prompt[first : first + 400]


def test_steps_order(rag):
    names = [s.name for s in rag.answer("battery").steps]
    assert names == ["retrieve", "cite_split", "prompt", "generate"]


def test_top_k_respected(rag, settings):
    result = rag.answer("Orion")
    retrieved = next(s.detail for s in result.steps if s.name == "retrieve")
    assert len(retrieved) == settings.top_k


def test_empty_index_message_no_llm_call(settings, embed_spec, store, fake_llm):
    result = SimpleRAG(settings, embed_spec=embed_spec, store=store, llm=fake_llm).answer("hi")
    assert result.answer == EMPTY_INDEX_MSG
    assert result.sources == [] and result.steps == []
    assert fake_llm.prompts == []


def test_sentence_splitter_does_not_cut_words(settings):
    long_doc = ("The Orion rover crosses dunes quickly. " * 120).strip()
    nodes = build_nodes([("long.txt", long_doc)], chunk_size=64, chunk_overlap=8)
    assert len(nodes) > 1
    for node in nodes:
        text = node.get_content()
        assert re.match(r"^\w", text) and re.search(r"[\w.]$", text)
        assert node.metadata["source"] == "long.txt"


def test_ingest_report_and_count(settings, embed_spec, store):
    report = ingest(settings, embed_spec=embed_spec, store=store)
    n = store.count()
    assert n > 0
    assert "broken.pdf" in dict(report.skipped)
    ingest(settings, embed_spec=embed_spec, store=store, reset=True)
    assert store.count() == n


def test_ingest_failure_keeps_existing_index(settings, embed_spec, store):
    ingest(settings, embed_spec=embed_spec, store=store)
    n = store.count()

    class Failing(BowEmbedding):
        def _get_text_embedding(self, text):
            raise RuntimeError("API down")

        def _get_text_embeddings(self, texts):
            raise RuntimeError("API down")

    embed_spec.model = Failing()
    with pytest.raises(RuntimeError):
        ingest(settings, embed_spec=embed_spec, store=store, reset=True)
    assert store.count() == n


def test_ingest_missing_data_dir_raises_config_error(settings, embed_spec, store, tmp_path):
    bad = settings.model_copy(update={"data_dir": tmp_path / "nope"})
    with pytest.raises(ConfigError, match="DATA_DIR"):
        ingest(bad, embed_spec=embed_spec, store=store)


@pytest.mark.parametrize(
    "provider,var", [("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY")]
)
def test_llm_missing_key_raises_config_error(provider, var):
    s = get_settings(_env_file=None, gemini_api_key=None, openai_api_key=None)
    with pytest.raises(ConfigError, match=var):
        get_llm(s, provider=provider)
