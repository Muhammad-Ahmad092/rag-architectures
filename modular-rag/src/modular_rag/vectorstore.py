"""Seven pluggable vector stores behind one small protocol, selected by ``VECTOR_STORE``.

Every index is named after the embedding model (``index_name``), so switching
``EMBEDDING_PROVIDER`` never mixes vectors from different models. Each store also remembers
the dimension it was built with and refuses to open with a different one.
All scores are cosine similarity (higher is better).
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Protocol

from .config import ConfigError, Settings, import_extra, require
from .models import Chunk, ScoredChunk

PREFIX = "modular-rag"
MAX_NAME = 45  # Pinecone's index-name limit is the strictest


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int) -> list[ScoredChunk]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


def index_name(embedder_name: str) -> str:
    """Backend-safe name: lowercase letters, digits and hyphens, at most 45 characters."""
    slug = re.sub(r"[^a-z0-9]+", "-", embedder_name.lower()).strip("-")
    name = f"{PREFIX}-{slug}"
    if len(name) > MAX_NAME:
        digest = hashlib.sha1(slug.encode()).hexdigest()[:8]
        name = f"{name[: MAX_NAME - 9].rstrip('-')}-{digest}"
    return name


def _mismatch(found_dim: int, dim: int, embedder: str) -> ConfigError:
    return ConfigError(
        f"Index for '{embedder}' was built with dim={found_dim} "
        f"but the embedder returns dim={dim}. "
        "Click 'Index DATA_DIR' to re-index, or change EMBEDDING_PROVIDER / EMBEDDING_MODEL."
    )


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _chunk_from(payload: dict) -> Chunk:
    return Chunk(
        id=payload["chunk_id"],
        text=payload["text"],
        source=payload["source"],
        metadata={"source": payload["source"], "chunk": int(payload.get("chunk", 0))},
    )


def _payload(chunk: Chunk) -> dict:
    return {
        "chunk_id": chunk.id,
        "text": chunk.text,
        "source": chunk.source,
        "chunk": int(chunk.metadata.get("chunk", 0)),
    }


class _Base:
    def __init__(self, settings: Settings, embedder_name: str, dim: int):
        self.embedder_name = embedder_name
        self.dim = dim
        self.name = index_name(embedder_name)

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------- local


class ChromaStore(_Base):
    def __init__(self, settings, embedder_name, dim):
        super().__init__(settings, embedder_name, dim)
        import chromadb

        path = Path(settings.index_dir) / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        existing = {c.name if hasattr(c, "name") else c for c in self._client.list_collections()}
        if self.name in existing:
            self._col = self._client.get_collection(self.name)
            found = int(self._col.metadata.get("dim", dim))
            if found != dim:
                raise _mismatch(found, dim, embedder_name)
        else:
            self._col = self._create()
        self._max_batch = int(self._client.get_max_batch_size())

    def _create(self):
        return self._client.create_collection(
            self.name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedder": self.embedder_name, "dim": self.dim},
        )

    def add(self, chunks, vectors):
        for i in range(0, len(chunks), self._max_batch):  # Chroma rejects oversized batches
            batch = chunks[i : i + self._max_batch]
            self._col.upsert(
                ids=[c.id for c in batch],
                embeddings=vectors[i : i + self._max_batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {"source": c.source, "chunk": int(c.metadata.get("chunk", 0))} for c in batch
                ],
            )

    def search(self, vector, k):
        n = self.count()
        if n == 0:
            return []
        r = self._col.query(query_embeddings=[vector], n_results=min(k, n))
        out = []
        for cid, doc, meta, dist in zip(
            r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0], strict=True
        ):
            chunk = Chunk(id=cid, text=doc, source=meta["source"], metadata=dict(meta))
            out.append(ScoredChunk(chunk=chunk, score=1.0 - float(dist)))
        return out

    def count(self):
        return self._col.count()

    def reset(self):
        self._client.delete_collection(self.name)
        self._col = self._create()


class FaissStore(_Base):
    """Exact inner-product search over L2-normalized vectors (= cosine)."""

    def __init__(self, settings, embedder_name, dim):
        super().__init__(settings, embedder_name, dim)
        self._faiss = import_extra("faiss", "faiss")
        import numpy as np

        self._np = np
        self._dir = Path(settings.index_dir) / "faiss" / self.name
        self._dir.mkdir(parents=True, exist_ok=True)
        meta_path = self._dir / "meta.json"
        if meta_path.exists():
            found = json.loads(meta_path.read_text(encoding="utf-8"))["dim"]
            if found != dim:
                raise _mismatch(found, dim, embedder_name)
            raw = np.fromfile(self._dir / "index.faiss", dtype="uint8")
            self._index = self._faiss.deserialize_index(raw)
            self._chunks = [
                Chunk(**c)
                for c in json.loads((self._dir / "chunks.json").read_text(encoding="utf-8"))
            ]
        else:
            self._index = self._faiss.IndexFlatIP(dim)
            self._chunks: list[Chunk] = []
            self._save()

    def _normalize(self, vectors):
        arr = self._np.asarray(vectors, dtype="float32")
        norms = self._np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def _save(self):
        self._faiss.serialize_index(self._index).tofile(self._dir / "index.faiss")
        (self._dir / "chunks.json").write_text(
            json.dumps([c.__dict__ for c in self._chunks]), encoding="utf-8"
        )
        (self._dir / "meta.json").write_text(
            json.dumps({"embedder": self.embedder_name, "dim": self.dim}), encoding="utf-8"
        )

    def add(self, chunks, vectors):
        if not chunks:
            return
        self._index.add(self._normalize(vectors))
        self._chunks.extend(chunks)
        self._save()

    def search(self, vector, k):
        if self._index.ntotal == 0:
            return []
        scores, ids = self._index.search(self._normalize([vector]), min(k, self._index.ntotal))
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(s))
            for s, i in zip(scores[0], ids[0], strict=True)
            if i != -1
        ]

    def count(self):
        return int(self._index.ntotal)

    def reset(self):
        self._index = self._faiss.IndexFlatIP(self.dim)
        self._chunks = []
        self._save()


class LanceDBStore(_Base):
    def __init__(self, settings, embedder_name, dim):
        super().__init__(settings, embedder_name, dim)
        lancedb = import_extra("lancedb", "lancedb")
        self._pa = import_extra("pyarrow", "lancedb")
        path = Path(settings.index_dir) / "lancedb"
        path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(path))
        self._meta = path / f"{self.name}.meta.json"
        if self.name in self._tables():
            found = (
                json.loads(self._meta.read_text(encoding="utf-8"))["dim"]
                if self._meta.exists()
                else dim
            )
            if found != dim:
                raise _mismatch(found, dim, embedder_name)
            self._table = self._db.open_table(self.name)
        else:
            self._table = self._create()

    def _tables(self) -> list[str]:
        resp = self._db.list_tables()
        return list(getattr(resp, "tables", resp))

    def _create(self):
        pa = self._pa
        schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("source", pa.string()),
                pa.field("chunk", pa.int64()),
                pa.field("vector", pa.list_(pa.float32(), self.dim)),
            ]
        )
        self._meta.write_text(
            json.dumps({"embedder": self.embedder_name, "dim": self.dim}), encoding="utf-8"
        )
        return self._db.create_table(self.name, schema=schema, mode="overwrite")

    def add(self, chunks, vectors):
        if not chunks:
            return
        rows = [dict(_payload(c), vector=v) for c, v in zip(chunks, vectors, strict=True)]
        self._table.add(rows)

    def search(self, vector, k):
        if self.count() == 0:
            return []
        rows = self._table.search(vector).distance_type("cosine").limit(k).to_list()
        return [ScoredChunk(chunk=_chunk_from(r), score=1.0 - float(r["_distance"])) for r in rows]

    def count(self):
        return int(self._table.count_rows())

    def reset(self):
        self._db.drop_table(self.name)
        self._table = self._create()


# ---------------------------------------------------------------- qdrant (local + cloud)


class QdrantStore(_Base):
    def __init__(self, settings, embedder_name, dim, cloud: bool = False):
        super().__init__(settings, embedder_name, dim)
        if cloud:
            require(settings.qdrant_url, "QDRANT_URL")
            require(settings.qdrant_api_key, "QDRANT_API_KEY")
        self._cloud = cloud
        self._models = import_extra("qdrant_client.models", "qdrant")
        self._client = self._connect(settings)
        if self._client.collection_exists(self.name):
            found = self._client.get_collection(self.name).config.params.vectors.size
            if found != dim:
                raise _mismatch(found, dim, embedder_name)
        else:
            self._create()

    def _connect(self, settings):
        qc = import_extra("qdrant_client", "qdrant")
        if self._cloud:
            return qc.QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        path = Path(settings.index_dir) / "qdrant"
        path.mkdir(parents=True, exist_ok=True)
        return qc.QdrantClient(path=str(path))

    def _create(self):
        m = self._models
        self._client.create_collection(
            self.name, vectors_config=m.VectorParams(size=self.dim, distance=m.Distance.COSINE)
        )

    def add(self, chunks, vectors):
        if not chunks:
            return
        points = [
            self._models.PointStruct(id=_point_id(c.id), vector=v, payload=_payload(c))
            for c, v in zip(chunks, vectors, strict=True)
        ]
        self._client.upsert(self.name, points=points, wait=True)

    def search(self, vector, k):
        res = self._client.query_points(self.name, query=vector, limit=k, with_payload=True)
        return [ScoredChunk(chunk=_chunk_from(p.payload), score=float(p.score)) for p in res.points]

    def count(self):
        return int(self._client.count(self.name, exact=True).count)

    def reset(self):
        # Delete all points rather than the collection: qdrant's local file mode on Windows
        # does not purge storage on delete_collection, so a re-created collection reloads it.
        m = self._models
        self._client.delete(
            self.name, points_selector=m.FilterSelector(filter=m.Filter()), wait=True
        )

    def close(self):
        self._client.close()


# ----------------------------------------------------------------------------- cloud


class PineconeStore(_Base):
    def __init__(self, settings, embedder_name, dim):
        super().__init__(settings, embedder_name, dim)
        require(settings.pinecone_api_key, "PINECONE_API_KEY")
        self._pc = self._connect(settings)
        if self._pc.has_index(self.name):
            found = int(self._pc.describe_index(self.name).dimension)
            if found != dim:
                raise _mismatch(found, dim, embedder_name)
        else:
            spec = import_extra("pinecone", "pinecone").ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            )
            self._pc.create_index(name=self.name, dimension=dim, metric="cosine", spec=spec)
        self._index = self._pc.Index(self.name)

    def _connect(self, settings):
        return import_extra("pinecone", "pinecone").Pinecone(api_key=settings.pinecone_api_key)

    def add(self, chunks, vectors):
        rows = [
            {"id": c.id, "values": v, "metadata": _payload(c)}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(rows), 100):
            self._index.upsert(vectors=rows[i : i + 100])

    def search(self, vector, k):
        res = self._index.query(vector=vector, top_k=k, include_metadata=True)
        return [
            ScoredChunk(chunk=_chunk_from(m.metadata), score=float(m.score)) for m in res.matches
        ]

    def count(self):
        return int(self._index.describe_index_stats().total_vector_count)

    def reset(self):
        try:
            self._index.delete(delete_all=True)
        except Exception:  # noqa: BLE001 - Pinecone errors when the namespace is already empty
            pass


class WeaviateStore(_Base):
    def __init__(self, settings, embedder_name, dim):
        super().__init__(settings, embedder_name, dim)
        require(settings.weaviate_url, "WEAVIATE_URL")
        require(settings.weaviate_api_key, "WEAVIATE_API_KEY")
        # Weaviate class names must start upper-case and use no hyphens
        self.name = "ModularRag_" + self.name.removeprefix(f"{PREFIX}-").replace("-", "_")
        self._client = self._connect(settings)
        if self._client.collections.exists(self.name):
            desc = self._client.collections.get(self.name).config.get().description or "{}"
            found = int(json.loads(desc).get("dim", dim))
            if found != dim:
                raise _mismatch(found, dim, embedder_name)
        else:
            self._create()
        self._col = self._client.collections.get(self.name)

    def _connect(self, settings):
        weaviate = import_extra("weaviate", "weaviate")
        auth = import_extra("weaviate.classes.init", "weaviate").Auth
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.weaviate_url,
            auth_credentials=auth.api_key(settings.weaviate_api_key),
        )

    def _create(self):
        cfg = import_extra("weaviate.classes.config", "weaviate")
        self._client.collections.create(
            self.name,
            description=json.dumps({"embedder": self.embedder_name, "dim": self.dim}),
            vector_config=cfg.Configure.Vectors.self_provided(),
            properties=[
                cfg.Property(name="chunk_id", data_type=cfg.DataType.TEXT),
                cfg.Property(name="text", data_type=cfg.DataType.TEXT),
                cfg.Property(name="source", data_type=cfg.DataType.TEXT),
                cfg.Property(name="chunk", data_type=cfg.DataType.INT),
            ],
        )

    def add(self, chunks, vectors):
        if not chunks:
            return
        data_object = import_extra("weaviate.classes.data", "weaviate").DataObject
        self._col.data.insert_many(
            [
                data_object(properties=_payload(c), vector=v, uuid=_point_id(c.id))
                for c, v in zip(chunks, vectors, strict=True)
            ]
        )

    def search(self, vector, k):
        mq = import_extra("weaviate.classes.query", "weaviate").MetadataQuery(distance=True)
        res = self._col.query.near_vector(near_vector=vector, limit=k, return_metadata=mq)
        return [
            ScoredChunk(chunk=_chunk_from(o.properties), score=1.0 - float(o.metadata.distance))
            for o in res.objects
        ]

    def count(self):
        return int(self._col.aggregate.over_all(total_count=True).total_count)

    def reset(self):
        self._client.collections.delete(self.name)
        self._create()
        self._col = self._client.collections.get(self.name)

    def close(self):
        self._client.close()


def get_vector_store(settings: Settings, embedder_name: str, dim: int) -> VectorStore:
    store = settings.vector_store
    if store == "chroma":
        return ChromaStore(settings, embedder_name, dim)
    if store == "faiss":
        return FaissStore(settings, embedder_name, dim)
    if store == "lancedb":
        return LanceDBStore(settings, embedder_name, dim)
    if store == "qdrant-local":
        return QdrantStore(settings, embedder_name, dim, cloud=False)
    if store == "qdrant-cloud":
        return QdrantStore(settings, embedder_name, dim, cloud=True)
    if store == "pinecone":
        return PineconeStore(settings, embedder_name, dim)
    if store == "weaviate-cloud":
        return WeaviateStore(settings, embedder_name, dim)
    raise ConfigError(f"Unknown VECTOR_STORE '{store}'")
