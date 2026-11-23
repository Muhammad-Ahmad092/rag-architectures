import pytest
from conftest import FakeCrossEncoder, FakeGenerator

from modular_rag.config import ConfigError
from modular_rag.corpus import corpus_path, load_corpus
from modular_rag.ingest import ingest
from modular_rag.modules import CrossEncoderRanker, Modules, make_generator, make_ranker
from modular_rag.pipeline import EMPTY_INDEX_MSG, ModularRAG


class Factories:
    """Generator/ranker factories that record which modules were requested."""

    def __init__(self):
        self.generators: dict[str, FakeGenerator] = {}

    def generator(self, name, settings):
        self.generators[name] = FakeGenerator(name)
        return self.generators[name]

    def ranker(self, name, settings):
        if name == "cross-encoder":
            return CrossEncoderRanker("fake", settings.top_k, model_obj=FakeCrossEncoder())
        return make_ranker(name, settings)


@pytest.fixture
def factories():
    return Factories()


@pytest.fixture
def rag(settings, fake_embedder, faiss_store, factories):
    s = settings.model_copy(update={"candidate_k": 6})
    ingest(s, embedder=fake_embedder, store=faiss_store)
    return ModularRAG(
        s,
        embedder=fake_embedder,
        store=faiss_store,
        generator_factory=factories.generator,
        ranker_factory=factories.ranker,
    )


def _stage(result, name):
    return next(s.detail for s in result.steps if s.name == name)


def _names(result):
    return [s.name for s in result.steps]


@pytest.mark.parametrize(
    "retriever,stages",
    [
        ("vector", ["vector_retriever", "ranker"]),
        ("bm25", ["bm25_retriever", "ranker"]),
        ("hybrid", ["vector_retriever", "bm25_retriever", "joiner", "ranker"]),
    ],
)
def test_each_retriever_module_runs(rag, retriever, stages):
    result = rag.answer("Orion support open", Modules(retriever=retriever, ranker="none"))
    assert _names(result) == ["modules", "pipeline", *stages, "prompt", "generate"]
    assert result.answer == "mistral answer [1]"
    assert len(result.sources) == rag.settings.top_k


def test_bm25_matches_exact_keywords(rag):
    result = rag.answer("24/7", Modules(retriever="bm25", ranker="none"))
    assert result.sources[0].chunk.source == "g.html"


def test_hybrid_fuses_and_deduplicates(rag):
    result = rag.answer("Orion rover", Modules(retriever="hybrid", ranker="none"))
    joined = _stage(result, "joiner")
    ids = [chunk_id for chunk_id, _ in joined]
    assert len(ids) == len(set(ids))  # same chunk from both retrievers counted once
    vec = {c for c, _ in _stage(result, "vector_retriever")}
    kw = {c for c, _ in _stage(result, "bm25_retriever")}
    assert set(ids) <= vec | kw and len(set(ids)) >= max(len(vec), len(kw))


def test_cross_encoder_reorders_candidates(rag):
    q = "Orion support is open 24/7"
    plain = rag.answer(q, Modules(retriever="vector", ranker="none"))
    ranked = rag.answer(q, Modules(retriever="vector", ranker="cross-encoder"))
    assert ranked.sources[0].chunk.source == "g.html"
    scores = [s.score for s in ranked.sources]
    assert scores == sorted(scores, reverse=True)  # cross-encoder scores, best first
    assert [s.chunk.id for s in plain.sources] != [s.chunk.id for s in ranked.sources] or (
        plain.sources[0].chunk.source == "g.html"
    )


def test_lost_in_the_middle_keeps_top_k(rag):
    result = rag.answer("Orion", Modules(retriever="hybrid", ranker="lost-in-the-middle"))
    assert len(result.sources) == rag.settings.top_k


def test_generator_module_is_swappable(rag, factories):
    rag.answer("Orion", Modules(generator="openai"))
    assert "openai" in factories.generators
    assert factories.generators["openai"].calls


def test_prompt_numbers_passages_with_sources(rag, factories):
    result = rag.answer("Orion support open", Modules(retriever="vector"))
    prompt = _stage(result, "prompt")
    for i, s in enumerate(result.sources, start=1):
        assert f"[{i}] ({s.chunk.source})" in prompt
    assert "Question: Orion support open" in prompt


def test_pipelines_are_cached_per_module_choice(rag):
    a = rag.pipeline_for(Modules("vector", "none", "mistral"))
    assert rag.pipeline_for(Modules("vector", "none", "mistral")) is a
    assert rag.pipeline_for(Modules("bm25", "none", "mistral")) is not a
    rag.refresh_corpus()
    assert rag.pipeline_for(Modules("vector", "none", "mistral")) is not a


def test_mermaid_shows_the_assembled_graph(rag):
    diagram = _stage(rag.answer("Orion", Modules(retriever="hybrid")), "pipeline")
    assert diagram.startswith("flowchart LR")
    assert "vector_retriever -->|documents| joiner" in diagram
    assert "bm25_retriever -->|documents| joiner" in diagram
    assert "prompt_builder -->|prompt| generator" in diagram


def test_dot_graph_for_the_ui(rag):
    from modular_rag.modules import dot

    built = rag.pipeline_for(Modules("hybrid", "cross-encoder", "mistral"))
    graph = dot(built.edges, built.inputs)
    assert graph.startswith("digraph G") and graph.rstrip().endswith("}")
    assert 'question -> ranker [label="query"];' in graph  # cross-encoder also reads the query
    assert 'joiner -> ranker [label="documents"];' in graph


def test_default_modules_come_from_settings(rag):
    assert rag.default_modules() == Modules("hybrid", "none", "mistral")
    assert _stage(rag.answer("Orion"), "modules") == {
        "retriever": "hybrid",
        "ranker": "none",
        "generator": "mistral",
    }


def test_invalid_module_rejected():
    with pytest.raises(ValueError, match="teleport"):
        Modules(retriever="teleport")


def test_keyword_retrieval_without_corpus_explains(settings, fake_embedder, faiss_store, factories):
    ingest(settings, embedder=fake_embedder, store=faiss_store)
    corpus_path(settings).unlink()  # e.g. indexed by an older version
    rag = ModularRAG(
        settings, embedder=fake_embedder, store=faiss_store, generator_factory=factories.generator
    )
    with pytest.raises(ConfigError, match="Index DATA_DIR"):
        rag.answer("Orion", Modules(retriever="bm25"))


def test_ingest_writes_corpus_and_appends_without_duplicates(settings, fake_embedder, faiss_store):
    ingest(settings, embedder=fake_embedder, store=faiss_store)
    n = len(load_corpus(corpus_path(settings)))
    assert n == faiss_store.count() > 0
    ingest(settings, embedder=fake_embedder, store=faiss_store, reset=False)
    assert len(load_corpus(corpus_path(settings))) == n


def test_empty_index_message_no_generator_call(settings, fake_embedder, faiss_store, factories):
    rag = ModularRAG(
        settings, embedder=fake_embedder, store=faiss_store, generator_factory=factories.generator
    )
    assert rag.answer("hi").answer == EMPTY_INDEX_MSG
    assert factories.generators == {}


@pytest.mark.parametrize("name,var", [("mistral", "MISTRAL_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_generator_needs_its_key(settings, name, var):
    s = settings.model_copy(update={"mistral_api_key": None, "openai_api_key": None})
    with pytest.raises(ConfigError, match=var):
        make_generator(name, s)


def test_real_generators_build_with_keys(settings):
    s = settings.model_copy(update={"mistral_api_key": "k", "openai_api_key": "k"})
    assert type(make_generator("mistral", s)).__name__ == "MistralChatGenerator"
    assert type(make_generator("openai", s)).__name__ == "OpenAIChatGenerator"
