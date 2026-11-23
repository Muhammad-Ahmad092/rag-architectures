import pytest
from conftest import BowEmbedding, FakeLLM, FakeReranker
from llama_index.core.schema import NodeWithScore, TextNode

from advanced_rag.config import ConfigError
from advanced_rag.corpus import corpus_path, load_corpus
from advanced_rag.ingest import build_nodes, ingest
from advanced_rag.pipeline import EMPTY_INDEX_MSG, AdvancedRAG, rrf
from advanced_rag.vectorstore import get_vector_store


@pytest.fixture
def store(settings, embed_spec):
    s = get_vector_store(settings, embed_spec.name, embed_spec.dim)
    ingest(settings, embed_spec=embed_spec, store=s)
    return s


def _rag(settings, embed_spec, store, llm, reranker="fake", **over):
    s = settings.model_copy(update=over) if over else settings
    rr = FakeReranker(s.top_k) if reranker == "fake" else reranker
    return AdvancedRAG(s, embed_spec=embed_spec, store=store, llm=llm, grader=llm, reranker=rr)


def _step(result, name):
    return next(s.detail for s in result.steps if s.name == name)


def _nws(node_id, score=1.0):
    return NodeWithScore(node=TextNode(id_=node_id, text=node_id), score=score)


def test_rrf_rewards_agreement_across_lists():
    a = [_nws("x"), _nws("y"), _nws("z")]
    b = [_nws("y"), _nws("w")]
    fused = rrf([a, b], k=60)
    assert [n.node.node_id for n in fused] == ["y", "x", "w", "z"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_handles_empty_lists():
    assert rrf([[], []], k=60) == []


def test_full_pipeline_steps_and_verdict(settings, embed_spec, store, fake_llm):
    result = _rag(settings, embed_spec, store, fake_llm).answer("Orion rover top speed")
    names = [s.name for s in result.steps]
    assert names == ["rewrite", "retrieve", "fuse", "rerank", "generate", "check"]
    assert result.verdict == "verified"
    assert fake_llm.kinds() == ["Queries", "generate", "Support"]


def test_rewrite_keeps_original_first_and_caps(settings, embed_spec, store):
    llm = FakeLLM(queries=["Orion rover top speed", "q2", "q3", "q4", "q5"])
    result = _rag(settings, embed_spec, store, llm, num_queries=3).answer("Orion rover top speed")
    assert _step(result, "rewrite") == ["Orion rover top speed", "q2", "q3"]


def test_retrieves_vector_and_bm25_for_every_query(settings, embed_spec, store, fake_llm):
    result = _rag(settings, embed_spec, store, fake_llm).answer("Orion rover top speed")
    lists = _step(result, "retrieve")
    assert [(r["query"], r["retriever"]) for r in lists] == [
        ("Orion rover top speed", "vector"),
        ("Orion rover top speed", "bm25"),
        ("Orion speed km/h", "vector"),
        ("Orion speed km/h", "bm25"),
    ]


def test_bm25_brings_in_exact_keyword_matches(settings, embed_spec, store):
    llm = FakeLLM(queries=[])
    result = _rag(settings, embed_spec, store, llm, reranker=None).answer("24/7")
    bm25 = next(r for r in _step(result, "retrieve") if r["retriever"] == "bm25")
    assert bm25["hits"][0][0] == "g.html"


def test_reranker_sees_fused_candidates_and_cuts_to_top_k(settings, embed_spec, store, fake_llm):
    rr = FakeReranker(settings.top_k)
    rag = AdvancedRAG(
        settings, embed_spec=embed_spec, store=store, llm=fake_llm, grader=fake_llm, reranker=rr
    )
    result = rag.answer("Orion rover top speed")
    assert rr.calls == ["Orion rover top speed"]  # reranks against the ORIGINAL question
    fused = _step(result, "fuse")
    assert [s.chunk.id for s in result.sources] == [fid for fid, _ in reversed(fused)][
        : settings.top_k
    ]


def test_no_reranker_keeps_fused_order(settings, embed_spec, store, fake_llm):
    result = _rag(settings, embed_spec, store, fake_llm, reranker=None).answer("Orion")
    fused_ids = [fid for fid, _ in _step(result, "fuse")]
    assert [s.chunk.id for s in result.sources] == fused_ids[: settings.top_k]
    assert _step(result, "rerank") == {"reranker": "none"}


def test_unsupported_answer_is_revised_once(settings, embed_spec, store):
    llm = FakeLLM(supported=[False, True])
    result = _rag(settings, embed_spec, store, llm).answer("Orion rover top speed")
    assert result.verdict == "revised" and result.answer == "revised answer [1]"
    revise_prompt = next(u for k, _, u in llm.calls if k == "revise")
    assert "the made-up claim" in revise_prompt
    assert llm.kinds().count("Support") == 2


def test_still_unsupported_after_revision_is_flagged(settings, embed_spec, store):
    llm = FakeLLM(supported=[False, False])
    result = _rag(settings, embed_spec, store, llm).answer("Orion rover top speed")
    assert result.verdict == "unsupported" and llm.kinds().count("revise") == 1


def test_check_disabled(settings, embed_spec, store, fake_llm):
    result = _rag(settings, embed_spec, store, fake_llm, check_answer=False).answer("Orion")
    assert result.verdict == "unchecked" and "Support" not in fake_llm.kinds()


def test_reranker_selection(settings, embed_spec, store, fake_llm, monkeypatch):
    from advanced_rag import pipeline as p

    made = {}
    monkeypatch.setattr(p, "_cohere_reranker", lambda s: made.setdefault("cohere", object()))
    monkeypatch.setattr(p, "_local_reranker", lambda s: made.setdefault("local", object()))
    with_key = settings.model_copy(update={"cohere_api_key": "k"})
    assert (
        AdvancedRAG(with_key, embed_spec=embed_spec, store=store, llm=fake_llm).reranker_name
        == "cohere"
    )
    no_key = settings.model_copy(update={"cohere_api_key": None})
    rag = AdvancedRAG(no_key, embed_spec=embed_spec, store=store, llm=fake_llm)
    assert rag.reranker_name == "local (no COHERE_API_KEY)"
    off = settings.model_copy(update={"reranker": "none"})
    assert (
        AdvancedRAG(off, embed_spec=embed_spec, store=store, llm=fake_llm).reranker_name == "none"
    )


def test_missing_corpus_falls_back_to_vector_only(settings, embed_spec, store, fake_llm):
    corpus_path(settings).unlink()
    result = _rag(settings, embed_spec, store, fake_llm).answer("Orion")
    assert {r["retriever"] for r in _step(result, "retrieve")} == {"vector"}
    assert result.sources


def test_ingest_writes_corpus(settings, embed_spec, store):
    nodes = load_corpus(corpus_path(settings))
    assert len(nodes) == store.count() > 0
    assert all(n.metadata.get("source") for n in nodes)


def test_empty_index_message_no_llm_call(settings, embed_spec, tmp_path, fake_llm):
    s = settings.model_copy(update={"index_dir": tmp_path / "empty"})
    empty = get_vector_store(s, embed_spec.name, embed_spec.dim)
    rag = AdvancedRAG(
        s, embed_spec=embed_spec, store=empty, llm=fake_llm, grader=fake_llm, reranker=None
    )
    assert rag.answer("hi").answer == EMPTY_INDEX_MSG and fake_llm.calls == []


def test_sentence_chunks():
    nodes = build_nodes([("long.txt", "The Orion rover crosses dunes. " * 120)], 64, 8)
    assert len(nodes) > 1 and all(n.metadata["source"] == "long.txt" for n in nodes)


def test_ingest_failure_keeps_existing_index(settings, embed_spec, store):
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
