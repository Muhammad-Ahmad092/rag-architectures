"""Plain data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass
class Step:
    name: str
    detail: Any


@dataclass
class RAGResult:
    answer: str
    sources: list[ScoredChunk]
    steps: list[Step]
    verdict: str = ""  # answer check: verified | revised | unsupported | unchecked
