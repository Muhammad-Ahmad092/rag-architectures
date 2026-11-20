"""Naive RAG: embed the question, take the top-k chunks, stuff them into one prompt, generate.

Deliberately absent (each is what a later RAG type adds): query rewriting, reranking,
relevance filtering, answer verification, memory.
"""

from __future__ import annotations

from .config import Settings
from .embeddings import Embedder, get_embedder
from .llm import LLM, get_llm
from .models import RAGResult, ScoredChunk, Step
from .vectorstore import VectorStore, get_vector_store

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided.\n"
    "Cite the passages you used with their numbers in square brackets, like [n].\n"
    'If the answer is not in the context, say "I don\'t know based on the provided documents."\n'
    "Be concise."
)

EMPTY_INDEX_MSG = (
    "Index is empty — click “Index DATA_DIR” or run: uv run python -m naive_rag.ingest"
)


def build_prompt(question: str, hits: list[ScoredChunk]) -> str:
    context = "\n\n".join(
        f"[{i}] ({h.chunk.source}) {h.chunk.text}" for i, h in enumerate(hits, start=1)
    )
    return f"Context:\n{context}\n\nQuestion: {question}"


class NaiveRAG:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: LLM | None = None,
    ):
        self.settings = settings
        self.embedder = embedder or get_embedder(settings)
        self.store = store or get_vector_store(settings, self.embedder.name, self.embedder.dim)
        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None:  # created lazily so an empty index never needs the API key
            self._llm = get_llm(self.settings)
        return self._llm

    def answer(self, question: str) -> RAGResult:
        if self.store.count() == 0:
            return RAGResult(answer=EMPTY_INDEX_MSG, sources=[], steps=[])
        steps: list[Step] = []
        vector = self.embedder.embed_query(question)
        steps.append(Step("embed_query", {"embedder": self.embedder.name, "dim": len(vector)}))
        hits = self.store.search(vector, self.settings.top_k)
        steps.append(Step("retrieve", [(h.chunk.id, round(h.score, 4)) for h in hits]))
        prompt = build_prompt(question, hits)
        steps.append(Step("prompt", prompt))
        answer = self.llm.complete(SYSTEM_PROMPT, prompt)
        steps.append(Step("generate", {"model": self.llm.model}))
        return RAGResult(answer=answer, sources=hits, steps=steps)
