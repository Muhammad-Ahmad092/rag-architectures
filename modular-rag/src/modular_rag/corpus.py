"""The chunk corpus on disk, for the keyword (BM25) retriever module.

Vector stores hold embeddings, but BM25 needs the raw text of every chunk. Ingest writes
the same chunks it indexes to ``INDEX_DIR/corpus.jsonl``; the BM25 module loads them into
a Haystack ``InMemoryDocumentStore``. It works the same whichever vector store is used.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Settings
from .models import Chunk


def corpus_path(settings: Settings) -> Path:
    return Path(settings.index_dir) / "corpus.jsonl"


def save_corpus(path: Path, chunks: list[Chunk], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_corpus(path) if append else []
    seen = {c.id for c in existing}
    rows = existing + [c for c in chunks if c.id not in seen]
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        "\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in rows), encoding="utf-8"
    )
    os.replace(tmp, path)


def load_corpus(path: Path) -> list[Chunk]:
    if not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [Chunk(**json.loads(line)) for line in lines if line.strip()]
