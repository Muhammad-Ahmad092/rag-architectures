"""The node corpus on disk, for the in-process BM25 retriever.

Vector stores hold embeddings; BM25 needs every node's text. Ingest writes the same nodes
it indexes to ``INDEX_DIR/corpus.jsonl`` (id, text, metadata), so hybrid retrieval works
on every vector store, cloud ones included.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from llama_index.core.schema import TextNode

from .config import Settings


def corpus_path(settings: Settings) -> Path:
    return Path(settings.index_dir) / "corpus.jsonl"


def save_corpus(path: Path, nodes, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = {n.node_id: n for n in (load_corpus(path) if append else [])}
    for n in nodes:
        rows[n.node_id] = n
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "\n".join(
            json.dumps({"id": n.node_id, "text": n.get_content(), "metadata": n.metadata})
            for n in rows.values()
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_corpus(path: Path) -> list[TextNode]:
    if not Path(path).exists():
        return []
    nodes = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            nodes.append(TextNode(id_=row["id"], text=row["text"], metadata=row["metadata"]))
    return nodes
