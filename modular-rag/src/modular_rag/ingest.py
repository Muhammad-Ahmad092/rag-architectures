"""Index every file in DATA_DIR.  CLI: ``uv run python -m modular_rag.ingest [--no-reset]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigError, Settings, get_settings
from .corpus import corpus_path, save_corpus
from .embeddings import Embedder, get_embedder
from .loaders import LoadReport, chunk_documents, load_documents
from .vectorstore import VectorStore, get_vector_store


def ingest(
    settings: Settings,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    reset: bool = True,
    data_dir: Path | None = None,
) -> LoadReport:
    data_dir = Path(data_dir or settings.data_dir)
    if not data_dir.is_dir():
        raise ConfigError(f"DATA_DIR '{data_dir}' does not exist. Set DATA_DIR in .env.")
    embedder = embedder or get_embedder(settings)
    store = store or get_vector_store(settings, embedder.name, embedder.dim)
    docs, report = load_documents(data_dir)
    chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
    # Embed before touching the index: if the embedding API fails, the old index survives.
    vectors = embedder.embed([c.text for c in chunks]) if chunks else []
    if reset:
        store.reset()
    if chunks:
        store.add(chunks, vectors)
    save_corpus(corpus_path(settings), chunks, append=not reset)  # text for the BM25 module
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index DATA_DIR into the configured vector store.")
    parser.add_argument("--no-reset", action="store_true", help="append instead of rebuilding")
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        embedder = get_embedder(settings)
        store = get_vector_store(settings, embedder.name, embedder.dim)
        report = ingest(settings, embedder, store, reset=not args.no_reset)
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}") from exc
    print(
        f"Indexed {len(report.loaded)} files into {settings.vector_store} "
        f"({store.count()} chunks, embedder {embedder.name})."
    )
    for path, reason in report.skipped:
        print(f"  skipped {path}: {reason}")
    store.close()


if __name__ == "__main__":
    main()
