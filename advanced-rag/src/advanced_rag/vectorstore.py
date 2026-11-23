"""All 7 vector stores as LlamaIndex integrations, wrapped in small handles.

A handle owns one LlamaIndex vector store plus the bits LlamaIndex leaves to the caller:
counting, resetting, persistence, and a guard against opening an index with the wrong
embedding dimension. Indexes are named after the embedding model (``index_name``) so
switching ``EMBEDDING_PROVIDER`` never mixes vectors.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

from .config import ConfigError, Settings, import_extra, require

PREFIX = "advanced-rag"
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


class StoreHandle:
    """Base handle: subclasses set ``vector_store`` and implement count/reset."""

    kind = ""

    def __init__(self, settings: Settings, embed_name: str, dim: int):
        self.settings = settings
        self.embed_name = embed_name
        self.dim = dim
        self.name = index_name(embed_name)
        self.vector_store: Any = None
        self._index: VectorStoreIndex | None = None

    def index(self, embed_model) -> VectorStoreIndex:
        if self._index is None:
            self._index = VectorStoreIndex.from_vector_store(
                self.vector_store, embed_model=embed_model
            )
        return self._index

    def insert(self, nodes, embed_model) -> None:
        if nodes:
            self.index(embed_model).insert_nodes(nodes)
            self._after_insert()

    def _after_insert(self) -> None:
        pass

    def count(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


# ----------------------------------------------------------------------------- local


class ChromaHandle(StoreHandle):
    kind = "chroma"

    def __init__(self, settings, embed_name, dim):
        super().__init__(settings, embed_name, dim)
        import chromadb
        from llama_index.vector_stores.chroma import ChromaVectorStore

        self._cls = ChromaVectorStore
        path = Path(settings.index_dir) / "chroma"
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        names = {c.name if hasattr(c, "name") else c for c in self._client.list_collections()}
        if self.name in names:
            self._col = self._client.get_collection(self.name)
            found = int((self._col.metadata or {}).get("dim", dim))
            if found != dim:
                raise _mismatch(found, dim, embed_name)
        else:
            self._col = self._create()
        self.vector_store = self._cls(chroma_collection=self._col)

    def _create(self):
        return self._client.create_collection(
            self.name,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedder": self.embed_name, "dim": self.dim},
        )

    def count(self):
        return self._col.count()

    def reset(self):
        self._client.delete_collection(self.name)
        self._col = self._create()
        self.vector_store = self._cls(chroma_collection=self._col)
        self._index = None


class FaissHandle(StoreHandle):
    """FAISS keeps no text, so LlamaIndex's docstore + index store are persisted beside it."""

    kind = "faiss"

    def __init__(self, settings, embed_name, dim):
        super().__init__(settings, embed_name, dim)
        self._faiss = import_extra("faiss", "faiss")
        self._mod = import_extra("llama_index.vector_stores.faiss", "faiss")
        self._dir = Path(settings.index_dir) / "faiss" / self.name
        meta = _read_meta(self._dir / "meta.json")
        if meta and meta["dim"] != dim:
            raise _mismatch(meta["dim"], dim, embed_name)
        self._open(fresh=not (self._dir / "index_store.json").exists())

    def _open(self, fresh: bool):
        if fresh:
            self.vector_store = self._mod.FaissVectorStore(
                faiss_index=self._faiss.IndexFlatIP(self.dim)
            )
            self._storage = StorageContext.from_defaults(vector_store=self.vector_store)
        else:
            self.vector_store = self._mod.FaissVectorStore.from_persist_dir(str(self._dir))
            self._storage = StorageContext.from_defaults(
                vector_store=self.vector_store, persist_dir=str(self._dir)
            )
        self._fresh = fresh
        self._index = None

    def index(self, embed_model):
        if self._index is None:
            if self._fresh:
                self._index = VectorStoreIndex(
                    nodes=[], storage_context=self._storage, embed_model=embed_model
                )
            else:
                self._index = load_index_from_storage(self._storage, embed_model=embed_model)
        return self._index

    def _after_insert(self):
        self._storage.persist(persist_dir=str(self._dir))
        _write_meta(self._dir / "meta.json", self.embed_name, self.dim)
        self._fresh = False

    def count(self):
        return int(self.vector_store.client.ntotal)

    def reset(self):
        shutil.rmtree(self._dir, ignore_errors=True)
        self._open(fresh=True)


class LanceDBHandle(StoreHandle):
    kind = "lancedb"

    def __init__(self, settings, embed_name, dim):
        super().__init__(settings, embed_name, dim)
        self._lancedb = import_extra("lancedb", "lancedb")
        self._mod = import_extra("llama_index.vector_stores.lancedb", "lancedb")
        self._path = Path(settings.index_dir) / "lancedb"
        self._path.mkdir(parents=True, exist_ok=True)
        self._meta = self._path / f"{self.name}.meta.json"
        meta = _read_meta(self._meta)
        if meta and meta["dim"] != dim and self._exists():
            raise _mismatch(meta["dim"], dim, embed_name)
        self._open()

    def _db(self):
        return self._lancedb.connect(str(self._path))

    def _exists(self) -> bool:
        resp = self._db().list_tables()
        return self.name in list(getattr(resp, "tables", resp))

    def _open(self):
        self.vector_store = self._mod.LanceDBVectorStore(
            uri=str(self._path), table_name=self.name, mode="overwrite"
        )
        self._index = None

    def _after_insert(self):
        _write_meta(self._meta, self.embed_name, self.dim)

    def count(self):
        return int(self._db().open_table(self.name).count_rows()) if self._exists() else 0

    def reset(self):
        if self._exists():
            self._db().drop_table(self.name)
        self._open()


# ----------------------------------------------------------------------------- qdrant


def _connect_qdrant_cloud(settings: Settings):
    qc = import_extra("qdrant_client", "qdrant")
    return qc.QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


class QdrantHandle(StoreHandle):
    def __init__(self, settings, embed_name, dim, cloud: bool):
        super().__init__(settings, embed_name, dim)
        self.kind = "qdrant-cloud" if cloud else "qdrant-local"
        if cloud:
            require(settings.qdrant_url, "QDRANT_URL")
            require(settings.qdrant_api_key, "QDRANT_API_KEY")
        mod = import_extra("llama_index.vector_stores.qdrant", "qdrant")
        self._models = import_extra("qdrant_client.models", "qdrant")
        if cloud:
            self._client = _connect_qdrant_cloud(settings)
        else:
            path = Path(settings.index_dir) / "qdrant"
            path.mkdir(parents=True, exist_ok=True)
            self._client = import_extra("qdrant_client", "qdrant").QdrantClient(path=str(path))
        if self._client.collection_exists(self.name):
            found = self._client.get_collection(self.name).config.params.vectors.size
            if found != dim:
                raise _mismatch(found, dim, embed_name)
        self.vector_store = mod.QdrantVectorStore(collection_name=self.name, client=self._client)

    def count(self):
        if not self._client.collection_exists(self.name):
            return 0
        return int(self._client.count(self.name, exact=True).count)

    def reset(self):
        # Delete points, not the collection: qdrant's local mode on Windows does not purge
        # storage on delete_collection (see naive-rag).
        if self._client.collection_exists(self.name):
            m = self._models
            self._client.delete(
                self.name, points_selector=m.FilterSelector(filter=m.Filter()), wait=True
            )
        self._index = None

    def close(self):
        self._client.close()


# ----------------------------------------------------------------------------- cloud


def _connect_pinecone(settings: Settings):
    return import_extra("pinecone", "pinecone").Pinecone(api_key=settings.pinecone_api_key)


def _pinecone_vector_store_cls():
    return import_extra("llama_index.vector_stores.pinecone", "pinecone").PineconeVectorStore


class PineconeHandle(StoreHandle):
    kind = "pinecone"

    def __init__(self, settings, embed_name, dim):
        super().__init__(settings, embed_name, dim)
        require(settings.pinecone_api_key, "PINECONE_API_KEY")
        pc = _connect_pinecone(settings)
        if pc.has_index(self.name):
            found = int(pc.describe_index(self.name).dimension)
            if found != dim:
                raise _mismatch(found, dim, embed_name)
        else:
            spec = import_extra("pinecone", "pinecone").ServerlessSpec(
                cloud=settings.pinecone_cloud, region=settings.pinecone_region
            )
            pc.create_index(name=self.name, dimension=dim, metric="cosine", spec=spec)
        self._pindex = pc.Index(self.name)
        self.vector_store = _pinecone_vector_store_cls()(pinecone_index=self._pindex)

    def count(self):
        return int(self._pindex.describe_index_stats().total_vector_count)

    def reset(self):
        try:
            self._pindex.delete(delete_all=True)
        except Exception:  # noqa: BLE001 - Pinecone errors when the namespace is already empty
            pass
        self._index = None


def _connect_weaviate(settings: Settings):
    weaviate = import_extra("weaviate", "weaviate")
    auth = import_extra("weaviate.classes.init", "weaviate").Auth
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=settings.weaviate_url,
        auth_credentials=auth.api_key(settings.weaviate_api_key),
    )


def _weaviate_vector_store_cls():
    return import_extra("llama_index.vector_stores.weaviate", "weaviate").WeaviateVectorStore


class WeaviateHandle(StoreHandle):
    kind = "weaviate-cloud"

    def __init__(self, settings, embed_name, dim):
        super().__init__(settings, embed_name, dim)
        require(settings.weaviate_url, "WEAVIATE_URL")
        require(settings.weaviate_api_key, "WEAVIATE_API_KEY")
        # Weaviate class names must start upper-case and contain no hyphens
        self.name = "AdvancedRag_" + self.name.removeprefix(f"{PREFIX}-").replace("-", "_")
        self._client = _connect_weaviate(settings)
        if self._client.collections.exists(self.name):
            desc = self._client.collections.get(self.name).config.get().description or "{}"
            found = int(json.loads(desc).get("dim", dim))
            if found != dim:
                raise _mismatch(found, dim, embed_name)
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
            weaviate_client=self._client, index_name=self.name
        )
        self._index = None

    def count(self):
        col = self._client.collections.get(self.name)
        return int(col.aggregate.over_all(total_count=True).total_count)

    def reset(self):
        self._client.collections.delete(self.name)
        self._create()
        self._open()

    def close(self):
        self._client.close()


def get_vector_store(settings: Settings, embed_name: str, dim: int) -> StoreHandle:
    store = settings.vector_store
    if store == "chroma":
        return ChromaHandle(settings, embed_name, dim)
    if store == "faiss":
        return FaissHandle(settings, embed_name, dim)
    if store == "lancedb":
        return LanceDBHandle(settings, embed_name, dim)
    if store in ("qdrant-local", "qdrant-cloud"):
        return QdrantHandle(settings, embed_name, dim, cloud=store == "qdrant-cloud")
    if store == "pinecone":
        return PineconeHandle(settings, embed_name, dim)
    if store == "weaviate-cloud":
        return WeaviateHandle(settings, embed_name, dim)
    raise ConfigError(f"Unknown VECTOR_STORE '{store}'")
