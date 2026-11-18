"""Index DATA_DIR with LlamaIndex.  CLI: ``uv run python -m simple_rag.ingest [--no-reset]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode

from .config import ConfigError, Settings, get_settings
from .embeddings import EmbedSpec, get_embed_model
from .loaders import LoadReport, load_documents
from .vectorstore import StoreHandle, get_vector_store


def build_nodes(docs: list[tuple[str, str]], chunk_size: int, chunk_overlap: int):
    """Sentence-aware chunking: SentenceSplitter keeps sentences (and words) intact."""
    documents = [Document(text=text, metadata={"source": source}) for source, text in docs]
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(documents)


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
    embed_spec = embed_spec or get_embed_model(settings)
    store = store or get_vector_store(settings, embed_spec.name, embed_spec.dim)
    docs, report = load_documents(data_dir)
    nodes = build_nodes(docs, settings.chunk_size, settings.chunk_overlap)
    # Embed before touching the index: if the embedding API fails, the old index survives.
    if nodes:
        texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
        for node, vector in zip(
            nodes, embed_spec.model.get_text_embedding_batch(texts), strict=True
        ):
            node.embedding = vector
    if reset:
        store.reset()
    store.insert(nodes, embed_spec.model)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Index DATA_DIR into the configured vector store.")
    parser.add_argument("--no-reset", action="store_true", help="append instead of rebuilding")
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        spec = get_embed_model(settings)
        store = get_vector_store(settings, spec.name, spec.dim)
        report = ingest(settings, spec, store, reset=not args.no_reset)
    except ConfigError as exc:
        raise SystemExit(f"Config error: {exc}") from exc
    print(
        f"Indexed {len(report.loaded)} files into {settings.vector_store} "
        f"({store.count()} nodes, embedder {spec.name})."
    )
    for path, reason in report.skipped:
        print(f"  skipped {path}: {reason}")
    store.close()


if __name__ == "__main__":
    main()
