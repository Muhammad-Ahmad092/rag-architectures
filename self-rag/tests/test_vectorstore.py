"""Vector store handles over LangChain integrations.

Local stores run for real; qdrant-cloud uses qdrant's in-memory mode; Pinecone and Weaviate
are checked for wiring with injected fake clients and recorder classes.
"""

from types import SimpleNamespace

import pytest
from conftest import DIM, BowEmbeddings, EmbedSpecLike, bow
from langchain_core.documents import Document

from self_rag import vectorstore as vs
from self_rag.config import ConfigError, get_settings

LOCAL = ["chroma", "faiss", "qdrant-local", "lancedb"]
TEXTS = ["apple fruit is red", "banana fruit is yellow", "the engine is loud"]


def _docs():
    return [
        Document(page_content=t, metadata={"source": f"s{i}.txt", "chunk_id": f"s{i}.txt::0"})
        for i, t in enumerate(TEXTS)
    ]


def _settings(tmp_path, store, **kw):
    return get_settings(_env_file=None, vector_store=store, index_dir=tmp_path / ".index", **kw)


def _spec(name="fake:bow", dim=DIM):
    spec = EmbedSpecLike()
    spec.name, spec.dim = name, dim
    return spec


@pytest.mark.parametrize("store", LOCAL)
def test_local_roundtrip(tmp_path, store):
    h = vs.get_vector_store(_settings(tmp_path, store), _spec())
    assert h.count() == 0
    h.add(_docs())
    hits = h.search("apple red", k=2)
    assert hits[0][0].page_content == "apple fruit is red"
    assert hits[0][0].metadata["source"] == "s0.txt"
    assert hits[0][1] >= hits[1][1]  # relevance: higher is better
    assert all(0.0 <= score <= 1.0 for _, score in hits)  # FAISS/LanceDB score fixes
    assert h.count() == 3
    h.reset()
    assert h.count() == 0  # HydeRAG answers EMPTY_INDEX_MSG before querying an empty store
    h.add(_docs()[:1])  # usable again after reset
    assert h.count() == 1
    assert h.search("apple red", k=1)[0][0].page_content == "apple fruit is red"
    h.close()


@pytest.mark.parametrize("store", LOCAL)
def test_append_keeps_existing(tmp_path, store):
    h = vs.get_vector_store(_settings(tmp_path, store), _spec())
    h.add(_docs()[:2])
    h.add(_docs()[2:])  # e.g. uploaded files indexed without reset
    assert h.count() == 3
    h.close()


@pytest.mark.parametrize("store", LOCAL)
def test_precomputed_vectors_skip_embedding(tmp_path, store):
    spec = _spec()
    h = vs.get_vector_store(_settings(tmp_path, store), spec)
    docs = _docs()
    h.add(docs, vectors=[bow(d.page_content) for d in docs])
    assert spec.model.calls == 0  # ingest embeds once, before reset; add must not re-embed
    assert h.count() == 3
    h.close()


@pytest.mark.parametrize("store", LOCAL)
def test_local_persists_and_retrieves_after_reopen(tmp_path, store):
    h = vs.get_vector_store(_settings(tmp_path, store), _spec())
    h.add(_docs())
    h.close()
    again = vs.get_vector_store(_settings(tmp_path, store), _spec())
    assert again.count() == 3
    assert again.search("engine loud", k=1)[0][0].page_content == "the engine is loud"
    again.close()


@pytest.mark.parametrize("store", LOCAL)
def test_dim_mismatch_raises(tmp_path, store):
    h = vs.get_vector_store(_settings(tmp_path, store), _spec())
    h.add(_docs())
    h.close()
    with pytest.raises(ConfigError, match="re-index"):
        vs.get_vector_store(_settings(tmp_path, store), _spec(dim=1536))


def test_switching_embedder_uses_separate_index(tmp_path):
    a = vs.get_vector_store(_settings(tmp_path, "chroma"), _spec("huggingface:BAAI/bge-small"))
    a.add(_docs())
    b = vs.get_vector_store(_settings(tmp_path, "chroma"), _spec("openai:text-embedding", 5))
    assert b.count() == 0 and a.count() == 3


def test_index_name_is_safe():
    long = vs.index_name("huggingface:BAAI/bge-small-en-v1.5")
    assert long.startswith("self-rag-huggingface-baai") and len(long) <= 45
    assert long == vs.index_name("huggingface:BAAI/bge-small-en-v1.5")  # deterministic
    assert long != vs.index_name("huggingface:BAAI/bge-small-en-v1.6")  # hash keeps them apart
    assert vs.index_name("fake:bow") == "self-rag-fake-bow"
    assert len(vs.index_name("x" * 200)) <= 45


def test_qdrant_cloud_roundtrip(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient

    monkeypatch.setattr(vs, "_connect_qdrant_cloud", lambda s: QdrantClient(":memory:"))
    s = _settings(tmp_path, "qdrant-cloud", qdrant_url="https://x", qdrant_api_key="k")
    h = vs.get_vector_store(s, _spec())
    h.add(_docs())
    assert h.search("banana yellow", k=1)[0][0].page_content == "banana fruit is yellow"
    assert h.count() == 3
    h.reset()
    assert h.count() == 0


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_pinecone_wiring(tmp_path, monkeypatch):
    created = {}
    index = SimpleNamespace(describe_index_stats=lambda: SimpleNamespace(total_vector_count=0))
    pc = SimpleNamespace(
        has_index=lambda name: False,
        create_index=lambda **kw: created.update(kw),
        Index=lambda name: index,
    )
    monkeypatch.setattr(vs, "_connect_pinecone", lambda s: pc)
    monkeypatch.setattr(vs, "_pinecone_vector_store_cls", lambda: Recorder)
    h = vs.get_vector_store(_settings(tmp_path, "pinecone", pinecone_api_key="k"), _spec())
    assert created["name"] == "self-rag-fake-bow" and created["dimension"] == DIM
    assert h.vector_store.kwargs["index"] is index
    assert isinstance(h.vector_store.kwargs["embedding"], vs.PrecomputedEmbeddings)
    assert h.count() == 0


def test_weaviate_wiring(tmp_path, monkeypatch):
    made = {}
    collections = SimpleNamespace(
        exists=lambda name: False,
        create=lambda name, description, **kw: made.update(name=name, description=description),
    )
    client = SimpleNamespace(collections=collections, close=lambda: None)
    monkeypatch.setattr(vs, "_connect_weaviate", lambda s: client)
    monkeypatch.setattr(vs, "_weaviate_vector_store_cls", lambda: Recorder)
    s = _settings(tmp_path, "weaviate-cloud", weaviate_url="https://x", weaviate_api_key="k")
    h = vs.get_vector_store(s, _spec())
    assert made["name"] == "SelfRag_fake_bow"
    assert '"dim": 32' in made["description"]
    kw = h.vector_store.kwargs
    assert kw["client"] is client and kw["index_name"] == "SelfRag_fake_bow"
    assert kw["text_key"] == "text"


@pytest.mark.parametrize(
    "store,var",
    [
        ("pinecone", "PINECONE_API_KEY"),
        ("qdrant-cloud", "QDRANT_URL"),
        ("weaviate-cloud", "WEAVIATE_URL"),
    ],
)
def test_cloud_missing_key(tmp_path, store, var):
    with pytest.raises(ConfigError, match=var):
        vs.get_vector_store(_settings(tmp_path, store), _spec())


@pytest.mark.parametrize("store", LOCAL)
def test_search_by_vector(tmp_path, store):
    h = vs.get_vector_store(_settings(tmp_path, store), _spec())
    h.add(_docs())
    # a vector that matches "banana" even though the query text is unrelated
    hits = h.search_by_vector(bow("banana fruit is yellow"), k=2)
    assert hits[0][0].page_content == "banana fruit is yellow"
    assert hits[0][1] >= hits[1][1] and all(0.0 <= s <= 1.0 for _, s in hits)
    # the override is one-shot: a normal text search embeds its own query again
    assert h.search("engine loud", k=1)[0][0].page_content == "the engine is loud"
    h.close()


def test_search_by_vector_clears_override_on_error(tmp_path):
    h = vs.get_vector_store(_settings(tmp_path, "faiss"), _spec())
    h.add(_docs())

    def boom(query, k):
        raise RuntimeError("store down")

    h.search = boom
    with pytest.raises(RuntimeError):
        h.search_by_vector(bow("banana"), k=1)
    assert h.embeddings.embed_query("q") == bow("q")


def test_precomputed_embeddings_falls_back_to_model():
    base = BowEmbeddings()
    pe = vs.PrecomputedEmbeddings(base)
    pe.prime(["known"], [[1.0] * DIM])
    out = pe.embed_documents(["known", "unknown"])
    assert out[0] == [1.0] * DIM and out[1] == bow("unknown")
    assert base.calls == 1  # only the unknown text was embedded
    assert pe.embed_query("q") == bow("q")
