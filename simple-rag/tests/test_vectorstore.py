"""Vector store handles over LlamaIndex integrations.

Local stores run for real; qdrant-cloud uses qdrant's in-memory mode; Pinecone and Weaviate
are checked for wiring with injected fake clients and recorder classes.
"""

import uuid
from types import SimpleNamespace

import pytest
from conftest import DIM, BowEmbedding, bow
from llama_index.core.schema import TextNode

from simple_rag import vectorstore as vs
from simple_rag.config import ConfigError, get_settings

LOCAL = ["chroma", "faiss", "qdrant-local", "lancedb"]
TEXTS = ["apple fruit is red", "banana fruit is yellow", "the engine is loud"]


def _nodes():
    return [
        TextNode(
            id_=str(uuid.uuid5(uuid.NAMESPACE_URL, f"n{i}")),
            text=t,
            metadata={"source": f"s{i}.txt"},
            embedding=bow(t),
        )
        for i, t in enumerate(TEXTS)
    ]


def _settings(tmp_path, store, **kw):
    return get_settings(_env_file=None, vector_store=store, index_dir=tmp_path / ".index", **kw)


def _top(handle, query, k=2):
    retriever = handle.index(BowEmbedding()).as_retriever(similarity_top_k=k)
    return retriever.retrieve(query)


@pytest.mark.parametrize("store", LOCAL)
def test_local_roundtrip(tmp_path, store):
    h = vs.get_vector_store(_settings(tmp_path, store), "fake:bow", DIM)
    assert h.count() == 0
    h.insert(_nodes(), BowEmbedding())
    hits = _top(h, "apple red")
    assert hits[0].node.text == "apple fruit is red"
    assert hits[0].node.metadata["source"] == "s0.txt"
    assert h.count() == 3
    h.reset()
    assert h.count() == 0  # SimpleRAG answers EMPTY_INDEX_MSG before querying an empty store
    h.insert(_nodes()[:1], BowEmbedding())  # usable again after reset
    assert h.count() == 1
    assert _top(h, "apple red")[0].node.text == "apple fruit is red"
    h.close()


@pytest.mark.parametrize("store", LOCAL)
def test_local_persists_and_retrieves_after_reopen(tmp_path, store):
    h = vs.get_vector_store(_settings(tmp_path, store), "fake:bow", DIM)
    h.insert(_nodes(), BowEmbedding())
    h.close()
    again = vs.get_vector_store(_settings(tmp_path, store), "fake:bow", DIM)
    assert again.count() == 3
    assert _top(again, "engine loud")[0].node.text == "the engine is loud"
    again.close()


@pytest.mark.parametrize("store", LOCAL)
def test_dim_mismatch_raises(tmp_path, store):
    h = vs.get_vector_store(_settings(tmp_path, store), "fake:bow", DIM)
    h.insert(_nodes(), BowEmbedding())
    h.close()
    with pytest.raises(ConfigError, match="re-index"):
        vs.get_vector_store(_settings(tmp_path, store), "fake:bow", 1536)


def test_switching_embedder_uses_separate_index(tmp_path):
    a = vs.get_vector_store(
        _settings(tmp_path, "chroma"), "huggingface:BAAI/bge-small-en-v1.5", DIM
    )
    a.insert(_nodes(), BowEmbedding())
    b = vs.get_vector_store(_settings(tmp_path, "chroma"), "openai:text-embedding-3-small", 5)
    assert b.count() == 0 and a.count() == 3


def test_index_name_is_safe():
    assert vs.index_name("huggingface:BAAI/bge-small-en-v1.5") == (
        "simple-rag-huggingface-baai-bge-small-en-v1-5"
    )
    assert len(vs.index_name("x" * 200)) <= 45


def test_qdrant_cloud_roundtrip(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient

    monkeypatch.setattr(vs, "_connect_qdrant_cloud", lambda s: QdrantClient(":memory:"))
    s = _settings(tmp_path, "qdrant-cloud", qdrant_url="https://x", qdrant_api_key="k")
    h = vs.get_vector_store(s, "fake:bow", DIM)
    h.insert(_nodes(), BowEmbedding())
    assert _top(h, "banana yellow")[0].node.text == "banana fruit is yellow"
    assert h.count() == 3
    h.reset()
    assert h.count() == 0


class Recorder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_pinecone_wiring(tmp_path, monkeypatch):
    created, index = (
        {},
        SimpleNamespace(describe_index_stats=lambda: SimpleNamespace(total_vector_count=0)),
    )
    pc = SimpleNamespace(
        has_index=lambda name: False,
        create_index=lambda **kw: created.update(kw),
        Index=lambda name: index,
    )
    monkeypatch.setattr(vs, "_connect_pinecone", lambda s: pc)
    monkeypatch.setattr(vs, "_pinecone_vector_store_cls", lambda: Recorder)
    h = vs.get_vector_store(_settings(tmp_path, "pinecone", pinecone_api_key="k"), "fake:bow", DIM)
    assert created["name"] == "simple-rag-fake-bow" and created["dimension"] == DIM
    assert h.vector_store.kwargs["pinecone_index"] is index
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
    h = vs.get_vector_store(s, "fake:bow", DIM)
    assert made["name"] == "SimpleRag_fake_bow"
    assert '"dim": 32' in made["description"]
    assert h.vector_store.kwargs == {"weaviate_client": client, "index_name": "SimpleRag_fake_bow"}


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
        vs.get_vector_store(_settings(tmp_path, store), "fake:bow", DIM)
