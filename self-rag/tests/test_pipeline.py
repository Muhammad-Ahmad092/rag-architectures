import pytest
from conftest import BowEmbeddings, FakeLLM

from self_rag.config import ConfigError
from self_rag.ingest import build_documents, ingest
from self_rag.pipeline import EMPTY_INDEX_MSG, NO_ANSWER, SelfRAG
from self_rag.vectorstore import get_vector_store


@pytest.fixture
def store(settings, embed_spec):
    s = get_vector_store(settings, embed_spec)
    ingest(settings, embed_spec=embed_spec, store=s)
    return s


def _rag(settings, embed_spec, store, llm, grader=None, **over):
    s = settings.model_copy(update=over) if over else settings
    return SelfRAG(s, embed_spec=embed_spec, store=store, llm=llm, grader=grader or llm)


def _names(result):
    return [s.name for s in result.steps]


def test_happy_path_is_verified(settings, embed_spec, store, fake_llm):
    result = _rag(settings, embed_spec, store, fake_llm).answer("Orion rover top speed")
    assert result.verdict == "verified"
    assert result.answer == "answer [1]"
    assert _names(result) == [
        "retrieve",
        "grade_documents",
        "generate",
        "check_support",
        "check_useful",
    ]
    assert fake_llm.kinds() == ["Relevance", "Relevance", "generate", "Support", "Usefulness"]


def test_irrelevant_docs_trigger_rewrite_and_reretrieve(settings, embed_spec, store):
    new_query = "Orion support is open 24/7"
    llm = FakeLLM(verdicts={"Relevance": [False, False, True, True]}, rewrites=[new_query])
    result = _rag(settings, embed_spec, store, llm).answer("when can I call?")
    assert result.verdict == "verified"
    assert llm.kinds().count("rewrite") == 1
    retrieves = [s.detail for s in result.steps if s.name == "retrieve"]
    assert [r["query"] for r in retrieves] == ["when can I call?", new_query]
    assert retrieves[1]["hits"][0][0] == "g.html"  # rewritten query finds the support page


def test_gives_up_after_max_rewrites_without_generating(settings, embed_spec, store):
    llm = FakeLLM(verdicts={"Relevance": lambda user: False})
    result = _rag(settings, embed_spec, store, llm).answer("stock ticker?")
    assert result.answer == NO_ANSWER
    assert result.verdict == "no relevant documents"
    assert llm.kinds().count("rewrite") == settings.max_rewrites == 2
    assert "generate" not in llm.kinds()
    assert result.sources == []


def test_max_rewrites_zero_gives_up_immediately(settings, embed_spec, store):
    llm = FakeLLM(verdicts={"Relevance": lambda user: False})
    result = _rag(settings, embed_spec, store, llm, max_rewrites=0).answer("x")
    assert result.verdict == "no relevant documents" and "rewrite" not in llm.kinds()


def test_unsupported_answer_is_regenerated_with_feedback(settings, embed_spec, store):
    llm = FakeLLM(answers=["made up [1]", "grounded [1]"], verdicts={"Support": [False, True]})
    result = _rag(settings, embed_spec, store, llm).answer("Orion rover top speed")
    assert result.answer == "grounded [1]" and result.verdict == "verified"
    gens = [u for k, _, u in llm.calls if k == "generate"]
    assert len(gens) == 2
    assert "fake unsupported_claims" in gens[1]  # grader feedback reaches the retry
    assert "made up [1]" in gens[1]


def test_still_unsupported_after_max_generations_is_flagged(settings, embed_spec, store):
    llm = FakeLLM(answers=["guess"], verdicts={"Support": lambda user: False})
    result = _rag(settings, embed_spec, store, llm).answer("Orion rover top speed")
    assert result.verdict == "unsupported"
    assert llm.kinds().count("generate") == settings.max_generations == 2
    assert "Usefulness" not in llm.kinds()


def test_not_useful_answer_triggers_rewrite(settings, embed_spec, store):
    llm = FakeLLM(verdicts={"Usefulness": [False, True]})
    result = _rag(settings, embed_spec, store, llm).answer("Orion rover top speed")
    assert result.verdict == "verified"
    assert llm.kinds().count("rewrite") == 1
    assert llm.kinds().count("generate") == 2


def test_not_useful_after_max_rewrites_is_flagged(settings, embed_spec, store):
    llm = FakeLLM(verdicts={"Usefulness": lambda user: False})
    result = _rag(settings, embed_spec, store, llm, max_rewrites=1).answer("Orion")
    assert result.verdict == "not useful"


def test_only_relevant_chunks_are_context_and_sources(settings, embed_spec, store):
    llm = FakeLLM(verdicts={"Relevance": lambda user: "support is open" in user})
    rag = _rag(settings, embed_spec, store, llm, top_k=4)
    result = rag.answer("Orion support open")
    assert [s.chunk.source for s in result.sources] == ["g.html"]
    prompt = next(u for k, _, u in llm.calls if k == "generate")
    assert "[1] (source: g.html)" in prompt and "[2]" not in prompt
    grade = next(s.detail for s in result.steps if s.name == "grade_documents")
    assert sum(1 for g in grade if g["relevant"]) == 1 and len(grade) == 4


def test_separate_grader_does_the_grading(settings, embed_spec, store):
    gen, grader = FakeLLM(model="big"), FakeLLM(model="small")
    _rag(settings, embed_spec, store, gen, grader=grader).answer("Orion")
    assert set(gen.kinds()) == {"generate"}
    assert set(grader.kinds()) == {"Relevance", "Support", "Usefulness"}


def test_graph_has_the_self_rag_nodes(settings, embed_spec, store, fake_llm):
    rag = _rag(settings, embed_spec, store, fake_llm)
    nodes = set(rag.graph.get_graph().nodes)
    expected = {
        "retrieve",
        "grade_documents",
        "rewrite",
        "generate",
        "check_support",
        "check_useful",
        "give_up",
    }
    assert expected <= nodes


def test_empty_index_message_no_llm_call(settings, embed_spec, tmp_path, fake_llm):
    s = settings.model_copy(update={"index_dir": tmp_path / "empty"})
    empty = get_vector_store(s, embed_spec)
    result = SelfRAG(s, embed_spec=embed_spec, store=empty, llm=fake_llm, grader=fake_llm).answer(
        "hi"
    )
    assert result.answer == EMPTY_INDEX_MSG and result.steps == []
    assert fake_llm.calls == []


def test_chunks_have_source_and_ids():
    docs = build_documents([("long.txt", "word " * 600)], chunk_size=200, chunk_overlap=20)
    assert [d.metadata["chunk_id"] for d in docs[:2]] == ["long.txt::0", "long.txt::1"]


def test_ingest_failure_keeps_existing_index(settings, embed_spec, store):
    n = store.count()

    class Failing(BowEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("API down")

    embed_spec.model = Failing()
    with pytest.raises(RuntimeError):
        ingest(settings, embed_spec=embed_spec, store=store, reset=True)
    assert store.count() == n


def test_ingest_missing_data_dir_raises_config_error(settings, embed_spec, store, tmp_path):
    bad = settings.model_copy(update={"data_dir": tmp_path / "nope"})
    with pytest.raises(ConfigError, match="DATA_DIR"):
        ingest(bad, embed_spec=embed_spec, store=store)
