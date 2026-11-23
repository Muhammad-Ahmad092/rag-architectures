import pytest
from conftest import BowEmbeddings, FakeLLM
from langchain_core.documents import Document

from corrective_rag.config import ConfigError
from corrective_rag.ingest import build_documents, ingest
from corrective_rag.pipeline import EMPTY_INDEX_MSG, NO_ANSWER, CorrectiveRAG, decide_action
from corrective_rag.vectorstore import get_vector_store
from corrective_rag.web import WebSearchError


class FakeWeb:
    def __init__(self, docs=None, error=None):
        self.queries, self.error = [], error
        self.docs = (
            docs
            if docs is not None
            else [
                Document(
                    page_content="Nimbus Robotics is privately held and has no stock ticker.",
                    metadata={
                        "source": "https://news.example/nimbus",
                        "web": True,
                        "chunk_id": "web::https://news.example/nimbus",
                        "title": "Nimbus",
                    },
                )
            ]
        )

    def search(self, query):
        self.queries.append(query)
        if self.error:
            raise WebSearchError(self.error)
        return list(self.docs)


@pytest.fixture
def store(settings, embed_spec):
    s = get_vector_store(settings, embed_spec)
    ingest(settings, embed_spec=embed_spec, store=s)
    return s


def _rag(settings, embed_spec, store, llm, web="default", **over):
    s = settings.model_copy(update=over) if over else settings
    web = FakeWeb() if web == "default" else web
    return CorrectiveRAG(s, embed_spec=embed_spec, store=store, llm=llm, web=web)


def _step(result, name):
    return next(s.detail for s in result.steps if s.name == name)


@pytest.mark.parametrize(
    "scores,action",
    [
        ([0.9, 0.1], "correct"),
        ([0.2, 0.1], "incorrect"),
        ([0.5, 0.2], "ambiguous"),
        ([], "incorrect"),
    ],
)
def test_decide_action_thresholds(scores, action):
    assert decide_action(scores, upper=0.7, lower=0.3) == action


def test_correct_uses_internal_only_no_web(settings, embed_spec, store):
    web, llm = FakeWeb(), FakeLLM(verdicts={"PassageScore": [0.9, 0.8]})
    result = _rag(settings, embed_spec, store, llm, web=web).answer("Orion rover top speed")
    assert result.verdict == "correct"
    assert web.queries == []
    assert [s.name for s in result.steps] == ["retrieve", "evaluate", "refine", "generate"]
    assert all(not s.chunk.metadata.get("web") for s in result.sources)


def test_incorrect_discards_internal_and_searches_web(settings, embed_spec, store):
    web = FakeWeb()
    llm = FakeLLM(verdicts={"PassageScore": [0.1, 0.0]}, rewrites=["Nimbus Robotics stock ticker"])
    result = _rag(settings, embed_spec, store, llm, web=web).answer("What's the stock ticker?")
    assert result.verdict == "incorrect"
    assert web.queries == ["Nimbus Robotics stock ticker"]  # rewritten for the web
    assert [s.name for s in result.steps] == [
        "retrieve",
        "evaluate",
        "web_rewrite",
        "web_search",
        "generate",
    ]
    assert [s.chunk.source for s in result.sources] == ["https://news.example/nimbus"]
    prompt = next(u for k, _, u in llm.calls if k == "generate")
    assert "(web: https://news.example/nimbus)" in prompt
    assert "refine" not in llm.kinds()  # nothing internal survived to refine


def test_ambiguous_combines_refined_internal_with_web(settings, embed_spec, store):
    web = FakeWeb()
    llm = FakeLLM(verdicts={"PassageScore": [0.5, 0.1]})
    result = _rag(settings, embed_spec, store, llm, web=web).answer("Orion rover top speed")
    assert result.verdict == "ambiguous"
    assert len(web.queries) == 1
    kinds = [s.chunk.metadata.get("web", False) for s in result.sources]
    assert kinds == [False, True]  # one internal passage kept (score 0.5), plus web


def test_refinement_strips_irrelevant_sentences(settings, embed_spec, store):
    llm = FakeLLM(refine=lambda user: "Orion rover top speed is 12 km/h.")
    result = _rag(settings, embed_spec, store, llm).answer("Orion rover top speed")
    assert all(s.chunk.text == "Orion rover top speed is 12 km/h." for s in result.sources)
    refine = _step(result, "refine")
    assert refine["enabled"] is True and len(refine["kept"]) == settings.top_k


def test_refinement_dropping_everything_falls_back(settings, embed_spec, store):
    llm = FakeLLM(refine=lambda user: "NONE", verdicts={"PassageScore": [0.9, 0.9]})
    web = FakeWeb()
    result = _rag(settings, embed_spec, store, llm, web=web).answer("Orion rover top speed")
    assert result.verdict == "correct"
    assert len(web.queries) == 1  # nothing usable left after refinement -> web fallback
    assert result.sources and all(s.chunk.metadata.get("web") for s in result.sources)


def test_refinement_off_keeps_passages_verbatim(settings, embed_spec, store):
    llm = FakeLLM()
    result = _rag(settings, embed_spec, store, llm, refine_knowledge=False).answer("Orion")
    assert "refine" not in llm.kinds()
    assert _step(result, "refine")["enabled"] is False


def test_no_web_and_incorrect_says_i_dont_know_without_generating(settings, embed_spec, store):
    llm = FakeLLM(verdicts={"PassageScore": lambda user: 0.0})
    result = _rag(settings, embed_spec, store, llm, web=None).answer("stock ticker?")
    assert result.answer == NO_ANSWER
    assert _step(result, "web_search") == {"skipped": "web search off or TAVILY_API_KEY not set"}
    assert "generate" not in llm.kinds()


def test_web_error_degrades_gracefully(settings, embed_spec, store):
    llm = FakeLLM(verdicts={"PassageScore": [0.5, 0.1]})
    web = FakeWeb(error="InvalidAPIKeyError: bad key")
    result = _rag(settings, embed_spec, store, llm, web=web).answer("Orion rover top speed")
    assert "bad key" in _step(result, "web_search")["error"]
    assert result.sources and not any(s.chunk.metadata.get("web") for s in result.sources)


def test_evaluate_records_score_per_passage(settings, embed_spec, store):
    llm = FakeLLM(verdicts={"PassageScore": [0.9, 0.2]})
    result = _rag(settings, embed_spec, store, llm).answer("Orion")
    ev = _step(result, "evaluate")
    assert [p["score"] for p in ev["passages"]] == [0.9, 0.2]
    assert ev["action"] == "correct" and ev["thresholds"] == {"upper": 0.7, "lower": 0.3}


def test_scores_are_clamped(settings, embed_spec, store):
    llm = FakeLLM(verdicts={"PassageScore": [7.0, -2.0]})
    ev = _step(_rag(settings, embed_spec, store, llm).answer("Orion"), "evaluate")
    assert [p["score"] for p in ev["passages"]] == [1.0, 0.0]


def test_graph_has_crag_nodes(settings, embed_spec, store):
    nodes = set(_rag(settings, embed_spec, store, FakeLLM()).graph.get_graph().nodes)
    assert {"retrieve", "evaluate", "refine", "web_rewrite", "web_search", "generate"} <= nodes


def test_empty_index_message_no_llm_call(settings, embed_spec, tmp_path):
    s = settings.model_copy(update={"index_dir": tmp_path / "empty"})
    llm = FakeLLM()
    result = CorrectiveRAG(
        s, embed_spec=embed_spec, store=get_vector_store(s, embed_spec), llm=llm, web=None
    ).answer("hi")
    assert result.answer == EMPTY_INDEX_MSG and llm.calls == []


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
