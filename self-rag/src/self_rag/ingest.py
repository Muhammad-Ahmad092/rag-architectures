"""Index DATA_DIR.  CLI: ``uv run python -m self_rag.ingest [--no-reset]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import ConfigError, Settings, get_settings
from .embeddings import EmbedSpec, get_embeddings
from .loaders import LoadReport, load_documents
from .vectorstore import StoreHandle, get_vector_store


def build_documents(
    docs: list[tuple[str, str]], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split on paragraphs, then lines, then words: chunks rarely cut a sentence mid-way."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out: list[Document] = []
    for source, text in docs:
        for i, piece in enumerate(splitter.split_text(text)):
            out.append(
                Document(
                    page_content=piece, metadata={"source": source, "chunk_id": f"{source}::{i}"}
                )
            )
    return out


def ingest(
    settings: Settings,
    embed_spec: EmbedSpec | None = None,
    store: StoreHandle | None = None,
    reset: bool = True,
    data_dir: Path | None = None,
) -> LoadReport:
    data_dir = Path(data_dir or settings.data_dir)
    if not data_dir.is_dir():
        raise ConfigError(f"DATA_DIR '{data_dir}' does not exist. Set DATA_DIR in .env.")
    embed_spec = embed_spec or get_embeddings(settings)
    store = store or get_vector_store(settings, embed_spec)
    docs, report = load_documents(data_dir)
    chunks = build_documents(docs, settings.chunk_size, settings.chunk_overlap)
    # Embed before touching the index: if the embedding API fails, the old index survives.
    vectors = embed_spec.model.embed_documents([c.page_content for c in chunks]) if chunks else []
    if reset:
        store.reset()
    store.add(chunks, vectors)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index DATA_DIR into the configured vector store.")
    parser.add_argument("--no-reset", action="store_true", help="append instead of rebuilding")
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        spec = get_embeddings(settings)
        store = get_vector_store(settings, spec)
        report = ingest(settings, spec, store, reset=not args.no_reset)
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}") from exc
    print(
        f"Indexed {len(report.loaded)} files into {settings.vector_store} "
        f"({store.count()} chunks, embedder {spec.name})."
    )
    for path, reason in report.skipped:
        print(f"  skipped {path}: {reason}")
    store.close()


if __name__ == "__main__":
    main()
