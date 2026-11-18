import pytest

from naive_rag.config import ConfigError, get_settings
from naive_rag.ingest import ingest
from naive_rag.llm import get_llm
from naive_rag.models import Chunk
from naive_rag.pipeline import EMPTY_INDEX_MSG, SYSTEM_PROMPT, NaiveRAG


@pytest.fixture
def rag(settings, fake_embedder, faiss_store, fake_llm):
    texts = [
        "The Orion rover top speed is 3 m/s.",
        "Orion battery lasts 6 hours.",
        "Support is open 24/7.",
        "The rover weighs 12 kg.",
    ]
    chunks = [
        Chunk(id=f"doc{i}.txt::0", text=t, source=f"doc{i}.txt", metadata={"chunk": 0})
        for i, t in enumerate(texts)
    ]
    faiss_store.add(chunks, fake_embedder.embed(texts))
    return NaiveRAG(settings, embedder=fake_embedder, store=faiss_store, llm=fake_llm)


def test_answer_uses_top_k_context(rag):
    result = rag.answer("What is the rover top speed?")
    prompt = next(s.detail for s in result.steps if s.name == "prompt")
    assert "[1]" in prompt and "[2]" in prompt and "[3]" not in prompt
    assert len(result.sources) == 2
    assert result.answer == "fake answer [1]"


def test_steps_order(rag):
    names = [s.name for s in rag.answer("battery?").steps]
    assert names == ["embed_query", "retrieve", "prompt", "generate"]


def test_context_includes_source_names(rag):
    prompt = next(s.detail for s in rag.answer("rover speed").steps if s.name == "prompt")
    assert "(doc" in prompt and ".txt)" in prompt
    assert "Question: rover speed" in prompt


def test_empty_index_message_no_llm_call(settings, fake_embedder, faiss_store, fake_llm):
    rag = NaiveRAG(settings, embedder=fake_embedder, store=faiss_store, llm=fake_llm)
    result = rag.answer("anything")
    assert result.answer == EMPTY_INDEX_MSG
    assert result.sources == [] and result.steps == []
    assert fake_llm.calls == []


def test_system_prompt_has_refusal_rule():
    assert "I don't know based on the provided documents." in SYSTEM_PROMPT
    assert "[n]" in SYSTEM_PROMPT


def test_no_rerank_or_filter(rag, fake_embedder, faiss_store):
    q = "Orion rover"
    expected = faiss_store.search(fake_embedder.embed_query(q), k=2)
    got = rag.answer(q).sources
    assert [s.chunk.id for s in got] == [s.chunk.id for s in expected]
    assert [s.score for s in got] == pytest.approx([s.score for s in expected])


def test_llm_missing_key_raises_config_error():
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_llm(get_settings(_env_file=None, openai_api_key=None))


def test_ingest_report_and_count(settings, fake_embedder, faiss_store):
    report = ingest(settings, embedder=fake_embedder, store=faiss_store)
    n = faiss_store.count()
    assert n > 0
    assert "broken.pdf" in dict(report.skipped)
    ingest(settings, embedder=fake_embedder, store=faiss_store, reset=True)
    assert faiss_store.count() == n


def test_ingest_without_reset_appends(settings, fake_embedder, faiss_store, tmp_path):
    ingest(settings, embedder=fake_embedder, store=faiss_store)
    n = faiss_store.count()
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "new.txt").write_text("brand new fact", encoding="utf-8")
    ingest(
        settings.model_copy(update={"data_dir": extra}),
        embedder=fake_embedder,
        store=faiss_store,
        reset=False,
    )
    assert faiss_store.count() == n + 1


def test_ingest_missing_data_dir_raises_config_error(
    settings, fake_embedder, faiss_store, tmp_path
):
    bad = settings.model_copy(update={"data_dir": tmp_path / "nope"})
    with pytest.raises(ConfigError, match="DATA_DIR"):
        ingest(bad, embedder=fake_embedder, store=faiss_store)


def test_ingest_failure_keeps_existing_index(settings, fake_embedder, faiss_store):
    ingest(settings, embedder=fake_embedder, store=faiss_store)
    n = faiss_store.count()

    class Failing(type(fake_embedder)):
        def embed(self, texts):
            raise RuntimeError("API down")

    with pytest.raises(RuntimeError):
        ingest(settings, embedder=Failing(), store=faiss_store, reset=True)
    assert faiss_store.count() == n
