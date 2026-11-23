"""Vector store tests.

Local stores run for real in tmp_path. Cloud stores are exercised through in-memory fakes
injected at their ``_connect`` boundary (qdrant-cloud uses qdrant's real in-memory mode).
"""

from types import SimpleNamespace

import pytest

from modular_rag import vectorstore as vs
from modular_rag.config import ConfigError, get_settings
from modular_rag.models import Chunk

DIM = 3
VECS = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
CHUNKS = [
    Chunk(
        id=f"c{i}",
        text=f"text {i}",
        source=f"s{i}.txt",
        metadata={"source": f"s{i}.txt", "chunk": i},
    )
    for i in range(3)
]

LOCAL = ["chroma", "faiss", "qdrant-local", "lancedb"]


def _settings(tmp_path, store, **kw):
    return get_settings(_env_file=None, vector_store=store, index_dir=tmp_path / ".index", **kw)


@pytest.mark.parametrize("store", LOCAL)
def test_local_store_roundtrip(tmp_path, store):
    s = vs.get_vector_store(_settings(tmp_path, store), "fake:model", DIM)
    s.add(CHUNKS, VECS)
    hits = s.search([1.0, 0.1, 0.0], k=2)
    assert hits[0].chunk.id == "c0"
    assert hits[0].chunk.text == "text 0" and hits[0].chunk.source == "s0.txt"
    assert hits[0].score >= hits[1].score
    assert 0.9 < hits[0].score <= 1.0001
    assert s.count() == 3
    s.reset()
    assert s.count() == 0
    assert s.search([1.0, 0.0, 0.0], k=2) == []
    s.close()


@pytest.mark.parametrize("store", LOCAL)
def test_local_store_persists(tmp_path, store):
    s = vs.get_vector_store(_settings(tmp_path, store), "fake:model", DIM)
    s.add(CHUNKS, VECS)
    s.close()
    again = vs.get_vector_store(_settings(tmp_path, store), "fake:model", DIM)
    assert again.count() == 3
    again.close()


@pytest.mark.parametrize("store", LOCAL)
def test_search_k_larger_than_count(tmp_path, store):
    s = vs.get_vector_store(_settings(tmp_path, store), "fake:model", DIM)
    s.add(CHUNKS[:1], VECS[:1])
    assert len(s.search([1.0, 0.0, 0.0], k=10)) == 1
    s.close()


@pytest.mark.parametrize("store", LOCAL)
def test_dim_mismatch_raises(tmp_path, store):
    s = vs.get_vector_store(_settings(tmp_path, store), "fake:model", DIM)
    s.add(CHUNKS, VECS)
    s.close()
    with pytest.raises(ConfigError, match="re-index"):
        vs.get_vector_store(_settings(tmp_path, store), "fake:model", 1536)


def test_switching_embedder_uses_separate_index(tmp_path):
    a = vs.get_vector_store(
        _settings(tmp_path, "chroma"), "huggingface:BAAI/bge-small-en-v1.5", DIM
    )
    a.add(CHUNKS, VECS)
    b = vs.get_vector_store(_settings(tmp_path, "chroma"), "openai:text-embedding-3-small", 5)
    assert b.count() == 0
    assert a.count() == 3


def test_index_name_is_safe_for_every_backend():
    name = vs.index_name("huggingface:BAAI/bge-small-en-v1.5")
    # longer than Pinecone's 45-char limit with this prefix -> truncated + stable hash
    assert name.startswith("modular-rag-huggingface-baai") and len(name) <= 45
    assert name == vs.index_name("huggingface:BAAI/bge-small-en-v1.5")
    assert name != vs.index_name("huggingface:BAAI/bge-small-en-v1.6")
    assert len(vs.index_name("x" * 200)) <= 45


# ----------------------------------------------------------------------------- cloud fakes


class FakePineconeIndex:
    def __init__(self):
        self.rows = {}

    def upsert(self, vectors, **kw):
        for v in vectors:
            self.rows[v["id"]] = v

    def query(self, vector, top_k, include_metadata, **kw):
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True))

        ranked = sorted(self.rows.values(), key=lambda r: -dot(r["values"], vector))[:top_k]
        return SimpleNamespace(
            matches=[
                SimpleNamespace(id=r["id"], score=dot(r["values"], vector), metadata=r["metadata"])
                for r in ranked
            ]
        )

    def describe_index_stats(self, **kw):
        return SimpleNamespace(total_vector_count=len(self.rows))

    def delete(self, delete_all=False, **kw):
        self.rows.clear()


class FakePinecone:
    def __init__(self):
        self.indexes = {}

    def has_index(self, name):
        return name in self.indexes

    def create_index(self, name, dimension, metric, spec):
        self.indexes[name] = (dimension, FakePineconeIndex())

    def describe_index(self, name):
        return SimpleNamespace(dimension=self.indexes[name][0])

    def Index(self, name):  # noqa: N802 - mirrors the SDK
        return self.indexes[name][1]


class FakeWeaviateCollection:
    def __init__(self, description):
        self.description = description
        self.objects = []
        outer = self
        self.config = SimpleNamespace(get=lambda: SimpleNamespace(description=outer.description))
        self.data = SimpleNamespace(insert_many=lambda objs: outer.objects.extend(objs))
        self.aggregate = SimpleNamespace(
            over_all=lambda total_count: SimpleNamespace(total_count=len(outer.objects))
        )
        self.query = SimpleNamespace(near_vector=self._near_vector)

    def _near_vector(self, near_vector, limit, return_metadata):
        def dot(a, b):
            return sum(x * y for x, y in zip(a, b, strict=True))

        ranked = sorted(self.objects, key=lambda o: -dot(o.vector, near_vector))[:limit]
        return SimpleNamespace(
            objects=[
                SimpleNamespace(
                    properties=o.properties,
                    metadata=SimpleNamespace(distance=1 - dot(o.vector, near_vector)),
                )
                for o in ranked
            ]
        )


class FakeWeaviate:
    def __init__(self):
        self.cols = {}
        outer = self
        self.collections = SimpleNamespace(
            exists=lambda name: name in outer.cols,
            create=lambda name, description, **kw: outer.cols.setdefault(
                name, FakeWeaviateCollection(description)
            ),
            get=lambda name: outer.cols[name],
            delete=lambda name: outer.cols.pop(name, None),
        )

    def close(self):
        pass


@pytest.fixture
def cloud_fakes(monkeypatch):
    from qdrant_client import QdrantClient

    pc, wv = FakePinecone(), FakeWeaviate()
    monkeypatch.setattr(vs.PineconeStore, "_connect", lambda self, settings: pc)
    monkeypatch.setattr(vs.WeaviateStore, "_connect", lambda self, settings: wv)
    monkeypatch.setattr(vs.QdrantStore, "_connect", lambda self, settings: QdrantClient(":memory:"))
    return pc, wv


CLOUD_KEYS = dict(
    pinecone_api_key="k",
    qdrant_url="https://x",
    qdrant_api_key="k",
    weaviate_url="https://x",
    weaviate_api_key="k",
)


@pytest.mark.parametrize("store", ["pinecone", "qdrant-cloud", "weaviate-cloud"])
def test_cloud_store_calls_client(tmp_path, cloud_fakes, store):
    s = vs.get_vector_store(_settings(tmp_path, store, **CLOUD_KEYS), "fake:model", DIM)
    s.add(CHUNKS, VECS)
    hits = s.search([0.0, 1.0, 0.0], k=2)
    assert hits[0].chunk.id == "c1"
    assert hits[0].chunk.source == "s1.txt"
    assert s.count() == 3
    s.reset()
    assert s.count() == 0


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
        vs.get_vector_store(_settings(tmp_path, store), "fake:model", DIM)


def test_chroma_adds_in_batches(tmp_path, monkeypatch):
    s = vs.get_vector_store(_settings(tmp_path, "chroma"), "fake:model", DIM)
    sizes = []
    real = s._col.upsert
    monkeypatch.setattr(s._col, "upsert", lambda **kw: (sizes.append(len(kw["ids"])), real(**kw)))
    monkeypatch.setattr(s, "_max_batch", 2)
    s.add(CHUNKS, VECS)
    assert sizes == [2, 1]
    assert s.count() == 3
