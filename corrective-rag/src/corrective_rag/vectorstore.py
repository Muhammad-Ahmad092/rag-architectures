"""All 7 vector stores as LangChain integrations, wrapped in small handles.

A handle owns one LangChain ``VectorStore`` plus what LangChain leaves to the caller:
counting, resetting, persistence, and a guard against opening an index with the wrong
embedding dimension. Indexes are named after the embedding model (``index_name``) so
switching ``EMBEDDING_PROVIDER`` never mixes vectors.

Every store is built over a ``PrecomputedEmbeddings`` wrapper: ingest embeds everything
*before* resetting the index (so an embedding-API failure leaves the old index intact),
then hands the vectors to ``add`` and the store reuses them instead of embedding again.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .config import ConfigError, Settings, import_extra, require

PREFIX = "corrective-rag"
MAX_NAME = 45  # Pinecone's index-name limit is the strictest


def index_name(embed_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", embed_name.lower()).strip("-")
    name = f"{PREFIX}-{slug}"
    if len(name) > MAX_NAME:
        digest = hashlib.sha1(slug.encode()).hexdigest()[:8]
        name = f"{name[: MAX_NAME - 9].rstrip('-')}-{digest}"
    return name


def _mismatch(found: int, dim: int, embed_name: str) -> ConfigError:
    return ConfigError(
        f"Index for '{embed_name}' was built with dim={found} "
        f"but the embedder returns dim={dim}. "
        "Click 'Index DATA_DIR' to re-index, or change EMBEDDING_PROVIDER / EMBEDDING_MODEL."
    )


def _read_meta(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_meta(path: Path, embed_name: str, dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"embedder": embed_name, "dim": dim}), encoding="utf-8")


class PrecomputedEmbeddings(Embeddings):
    """Serves vectors primed by ingest; anything not primed goes to the real model.

    ``query_override`` makes the next query embedding a given vector: that is how HyDE's
    ready-made query vector (e.g. a HyDE average) is searched on every store through the same
    (tested) relevance-score path as a text query.
    """

    def __init__(self, base: Embeddings):
        self.base = base
        self._cache: dict[str, list[float]] = {}
        self.query_override: list[float] | None = None

    def prime(self, texts: list[str], vectors: list[list[float]]) -> None:
        self._cache.update(zip(texts, vectors, strict=True))

    def clear(self) -> None:
        self._cache.clear()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            self._cache.update(zip(missing, self.base.embed_documents(missing), strict=True))
        return [self._cache[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.query_override is not None:
            return list(self.query_override)
        return self.base.embed_query(text)


def _point_id(doc: Document) -> str:
    """Stable UUID per chunk (Qdrant requires UUIDs; the others accept them)."""
    key = doc.metadata.get("chunk_id") or doc.page_content
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class StoreHandle:
    """Base handle: subclasses set ``vector_store`` and implement count/reset."""

    kind = ""

    def __init__(self, settings: Settings, spec):
        self.settings = settings
        self.embed_name = spec.name
        self.dim = spec.dim
        self.name = index_name(spec.name)
        self.embeddings = PrecomputedEmbeddings(spec.model)
        self.vector_store: Any = None

    def add(self, docs: list[Document], vectors: list[list[float]] | None = None) -> None:
        if not docs:
            return
        if vectors is not None:
            self.embeddings.prime([d.page_content for d in docs], vectors)
        try:
            self._add(docs, [_point_id(d) for d in docs])
        finally:
            self.embeddings.clear()
        self._after_add()

    def _add(self, docs: list[Document], ids: list[str]) -> None:
        self.vector_store.add_documents(docs, ids=ids)

    def _after_add(self) -> None:
        pass

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Top-k with LangChain's normalised relevance score (higher is better)."""
        return self.vector_store.similarity_search_with_relevance_scores(query, k=k)

    def search_by_vector(self, vector: list[float], k: int) -> list[tuple[Document, float]]:
        """Top-k for a ready-made query vector (e.g. HyDE), same scores as ``search``."""
        self.embeddings.query_override = vector
        try:
            return self.search("<vector query>", k=k)
        finally:
            self.embeddings.query_override = None

    def count(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------- local


class ChromaHandle(StoreHandle):
    kind = "chroma"

    def __init__(self, settings, spec):
        super().__init__(settings, spec)
        import chromadb
        from langchain_chroma import Chroma

        self._cls = Chroma
        path = Path(settings.index_dir) / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        names = {c.name if hasattr(c, "name") else c for c in self._client.list_collections()}
        if self.name in names:
            self._col = self._client.get_collection(self.name)
            found = int((self._col.metadata or {}).get("dim", self.dim))
            if found != self.dim:
                raise _mismatch(found, self.dim, self.embed_name)
        else:
            self._col = self._create()
        self._open()

    def _create(self):
        return self._client.create_collection(
            self.name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedder": self.embed_name, "dim": self.dim},
        )

    def _open(self):
        self.vector_store = self._cls(
            client=self._client,
            collection_name=self.name,
            embedding_function=self.embeddings,
            create_collection_if_not_exists=False,
        )

    def _add(self, docs, ids):
        batch = self._client.get_max_batch_size()
        for i in range(0, len(docs), batch):
            self.vector_store.add_documents(docs[i : i + batch], ids=ids[i : i + batch])

    def count(self):
        return self._col.count()

    def reset(self):
        self._client.delete_collection(self.name)
        self._col = self._create()
        self._open()


class FaissHandle(StoreHandle):
    """FAISS (inner product over normalised vectors = cosine), saved with its docstore."""

    kind = "faiss"

    def __init__(self, settings, spec):
        super().__init__(settings, spec)
        self._faiss = import_extra("faiss", "faiss")
        self._mod = import_extra("langchain_community.vectorstores.faiss", "faiss")
        self._dir = Path(settings.index_dir) / "faiss" / self.name
        meta = _read_meta(self._dir / "meta.json")
        if meta and meta["dim"] != self.dim:
            raise _mismatch(meta["dim"], self.dim, self.embed_name)
        self._open()

    def _open(self):
        strategy = self._mod.DistanceStrategy.MAX_INNER_PRODUCT
        # FAISS returns the inner product (= cosine, vectors are normalised): already a
        # relevance score. LangChain's default MAX_INNER_PRODUCT mapping inverts the ranking.
        # Clamped: float rounding gives 1.0000001 for an exact match.
        common = {
            "distance_strategy": strategy,
            "relevance_score_fn": lambda score: min(1.0, max(0.0, float(score))),
        }
        if (self._dir / "index.faiss").exists():
            # Our own file, written by _after_add: the pickle is the docstore we saved.
            self.vector_store = self._mod.FAISS.load_local(
                str(self._dir), self.embeddings, allow_dangerous_deserialization=True, **common
            )
        else:
            from langchain_community.docstore.in_memory import InMemoryDocstore

            self.vector_store = self._mod.FAISS(
                embedding_function=self.embeddings,
                index=self._faiss.IndexFlatIP(self.dim),
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
                **common,
            )

    def _after_add(self):
        self.vector_store.save_local(str(self._dir))
        _write_meta(self._dir / "meta.json", self.embed_name, self.dim)

    def count(self):
        return int(self.vector_store.index.ntotal)

    def reset(self):
        shutil.rmtree(self._dir, ignore_errors=True)
        self._open()


class LanceDBHandle(StoreHandle):
    kind = "lancedb"

    def __init__(self, settings, spec):
        super().__init__(settings, spec)
        self._lancedb = import_extra("lancedb", "lancedb")
        self._mod = import_extra("langchain_community.vectorstores.lancedb", "lancedb")
        path = Path(settings.index_dir) / "lancedb"
        path.mkdir(parents=True, exist_ok=True)
        self._db = self._lancedb.connect(str(path))
        if self._exists():
            found = self._db.open_table(self.name).schema.field("vector").type.list_size
            if found != self.dim:
                raise _mismatch(found, self.dim, self.embed_name)
        self._open()

    def _exists(self) -> bool:
        resp = self._db.list_tables()
        return self.name in list(getattr(resp, "tables", resp))

    def _open(self):
        # mode="append": LangChain's default "overwrite" would replace the table on every add
        self.vector_store = self._mod.LanceDB(
            connection=self._db,
            embedding=self.embeddings,
            table_name=self.name,
            mode="append",
            distance="cosine",
        )

    def search(self, query, k):
        # LangChain's LanceDB searches with L2 unless told otherwise, yet scores with `distance`
        return self.vector_store.similarity_search_with_relevance_scores(
            query, k=k, metrics="cosine"
        )

    def count(self):
        return int(self._db.open_table(self.name).count_rows()) if self._exists() else 0

    def reset(self):
        if self._exists():
            self._db.drop_table(self.name)
        self._open()


# ----------------------------------------------------------------------------- qdrant


def _connect_qdrant_cloud(settings: Settings):
    qc = import_extra("qdrant_client", "qdrant")
    return qc.QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


class QdrantHandle(StoreHandle):
    def __init__(self, settings, spec, cloud: bool):
        super().__init__(settings, spec)
        self.kind = "qdrant-cloud" if cloud else "qdrant-local"
        if cloud:
            require(settings.qdrant_url, "QDRANT_URL")
            require(settings.qdrant_api_key, "QDRANT_API_KEY")
        mod = import_extra("langchain_qdrant", "qdrant")
        self._models = import_extra("qdrant_client.models", "qdrant")
        if cloud:
            self._client = _connect_qdrant_cloud(settings)
        else:
            path = Path(settings.index_dir) / "qdrant"
            path.mkdir(parents=True, exist_ok=True)
            self._client = import_extra("qdrant_client", "qdrant").QdrantClient(path=str(path))
        if self._client.collection_exists(self.name):
            found = self._client.get_collection(self.name).config.params.vectors.size
            if found != self.dim:
                raise _mismatch(found, self.dim, self.embed_name)
        else:
            m = self._models
            self._client.create_collection(
                self.name, vectors_config=m.VectorParams(size=self.dim, distance=m.Distance.COSINE)
            )
        # validate_collection_config would embed a dummy text on every start (a paid API call
        # for cloud embedders); the dimension is already checked above.
        self.vector_store = mod.QdrantVectorStore(
            client=self._client,
            collection_name=self.name,
            embedding=self.embeddings,
            validate_collection_config=False,
        )

    def count(self):
        return int(self._client.count(self.name, exact=True).count)

    def reset(self):
        # Delete points, not the collection: qdrant's local mode on Windows does not purge
        # storage on delete_collection (see naive-rag).
        m = self._models
        self._client.delete(
            self.name, points_selector=m.FilterSelector(filter=m.Filter()), wait=True
        )

    def close(self):
        self._client.close()


# ----------------------------------------------------------------------------- cloud


def _connect_pinecone(settings: Settings):
    return import_extra("pinecone", "pinecone").Pinecone(api_key=settings.pinecone_api_key)


def _pinecone_vector_store_cls():
    return import_extra("langchain_pinecone", "pinecone").PineconeVectorStore


class PineconeHandle(StoreHandle):
    kind = "pinecone"

    def __init__(self, settings, spec):
        super().__init__(settings, spec)
        require(settings.pinecone_api_key, "PINECONE_API_KEY")
        pc = _connect_pinecone(settings)
        if pc.has_index(self.name):
            found = int(pc.describe_index(self.name).dimension)
            if found != self.dim:
                raise _mismatch(found, self.dim, self.embed_name)
        else:
            spec_cls = import_extra("pinecone", "pinecone").ServerlessSpec
            pc.create_index(
                name=self.name,
                dimension=self.dim,
                metric="cosine",
                spec=spec_cls(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        self._pindex = pc.Index(self.name)
        self.vector_store = _pinecone_vector_store_cls()(
            index=self._pindex, embedding=self.embeddings
        )

    def count(self):
        return int(self._pindex.describe_index_stats().total_vector_count)

    def reset(self):
        try:
            self._pindex.delete(delete_all=True)
        except Exception:  # noqa: BLE001 - Pinecone errors when the namespace is already empty
            pass


def _connect_weaviate(settings: Settings):
    weaviate = import_extra("weaviate", "weaviate")
    auth = import_extra("weaviate.classes.init", "weaviate").Auth
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.weaviate_url,
        auth_credentials=auth.api_key(settings.weaviate_api_key),
    )


def _weaviate_vector_store_cls():
    return import_extra("langchain_weaviate", "weaviate").WeaviateVectorStore


class WeaviateHandle(StoreHandle):
    kind = "weaviate-cloud"

    def __init__(self, settings, spec):
        super().__init__(settings, spec)
        require(settings.weaviate_url, "WEAVIATE_URL")
        require(settings.weaviate_api_key, "WEAVIATE_API_KEY")
        # Weaviate class names must start upper-case and contain no hyphens
        self.name = "CorrectiveRag_" + self.name.removeprefix(f"{PREFIX}-").replace("-", "_")
        self._client = _connect_weaviate(settings)
        if self._client.collections.exists(self.name):
            desc = self._client.collections.get(self.name).config.get().description or "{}"
            found = int(json.loads(desc).get("dim", self.dim))
            if found != self.dim:
                raise _mismatch(found, self.dim, self.embed_name)
        else:
            self._create()
        self._open()

    def _create(self):
        cfg = import_extra("weaviate.classes.config", "weaviate")
        self._client.collections.create(
            self.name,
            description=json.dumps({"embedder": self.embed_name, "dim": self.dim}),
            vector_config=cfg.Configure.Vectors.self_provided(),
        )

    def _open(self):
        self.vector_store = _weaviate_vector_store_cls()(
            client=self._client, index_name=self.name, text_key="text", embedding=self.embeddings
        )

    def count(self):
        col = self._client.collections.get(self.name)
        return int(col.aggregate.over_all(total_count=True).total_count)

    def reset(self):
        self._client.collections.delete(self.name)
        self._create()
        self._open()

    def close(self):
        self._client.close()


def get_vector_store(settings: Settings, spec) -> StoreHandle:
    """``spec`` is an ``EmbedSpec`` (model, name, dim)."""
    store = settings.vector_store
    if store == "chroma":
        return ChromaHandle(settings, spec)
    if store == "faiss":
        return FaissHandle(settings, spec)
    if store == "lancedb":
        return LanceDBHandle(settings, spec)
    if store in ("qdrant-local", "qdrant-cloud"):
        return QdrantHandle(settings, spec, cloud=store == "qdrant-cloud")
    if store == "pinecone":
        return PineconeHandle(settings, spec)
    if store == "weaviate-cloud":
        return WeaviateHandle(settings, spec)
    raise ConfigError(f"Unknown VECTOR_STORE '{store}'")
