"""Simple RAG with LlamaIndex: one retrieval, then an answer with numbered citations.

Over naive-rag it adds sentence-aware chunks (``SentenceSplitter``) and a
``CitationQueryEngine``, which splits each retrieved node into small "Source n" passages
so the LLM can cite exactly the passage it used. It is still a single step: no rewriting,
reranking or answer checking.
"""

from __future__ import annotations

from llama_index.core import QueryBundle
from llama_index.core.query_engine import CitationQueryEngine

from .config import Settings
from .embeddings import EmbedSpec, get_embed_model
from .llm import get_llm, model_name
from .models import Chunk, RAGResult, ScoredChunk, Step
from .vectorstore import StoreHandle, get_vector_store

EMPTY_INDEX_MSG = (
    "Index is empty — click “Index DATA_DIR” or run: uv run python -m simple_rag.ingest"
)


def _strip_label(text: str) -> str:
    """Citation nodes are prefixed 'Source n:\\n'; keep only the passage."""
    head, sep, rest = text.partition(":\n")
    return rest if sep and head.startswith("Source ") else text


class SimpleRAG:
    def __init__(
        self,
        settings: Settings,
        embed_spec: EmbedSpec | None = None,
        store: StoreHandle | None = None,
        llm=None,
    ):
        self.settings = settings
        self.embed_spec = embed_spec or get_embed_model(settings)
        self.store = store or get_vector_store(settings, self.embed_spec.name, self.embed_spec.dim)
        self._llm = llm

    @property
    def llm(self):
        if self._llm is None:  # lazy: an empty index never needs the API key
            self._llm = get_llm(self.settings)
        return self._llm

    def _engine(self) -> CitationQueryEngine:
        index = self.store.index(self.embed_spec.model)
        return CitationQueryEngine.from_args(
            index,
            llm=self.llm,
            similarity_top_k=self.settings.top_k,
            citation_chunk_size=self.settings.citation_chunk_size,
            citation_chunk_overlap=0,
        )

    def answer(self, question: str) -> RAGResult:
        if self.store.count() == 0:
            return RAGResult(answer=EMPTY_INDEX_MSG, sources=[], steps=[])
        engine = self._engine()
        bundle = QueryBundle(question)

        retrieved = engine.retrieve(bundle)
        steps = [
            Step(
                "retrieve",
                [(n.node.metadata.get("source", "?"), round(n.score or 0.0, 4)) for n in retrieved],
            )
        ]
        # Private method, used on purpose: CitationQueryEngine._query runs exactly
        # retrieve -> _create_citation_nodes -> synthesize; we call the same three steps
        # so each one can be shown in the UI without retrieving twice.
        cited = engine._create_citation_nodes(retrieved)
        steps.append(
            Step("cite_split", {"retrieved_nodes": len(retrieved), "citable_passages": len(cited)})
        )
        context = "\n".join(n.node.get_content() for n in cited)
        steps.append(Step("prompt", f"{context}\nQuery: {question}"))

        response = engine.synthesize(bundle, cited)
        steps.append(Step("generate", {"model": model_name(self.llm)}))

        sources = [
            ScoredChunk(
                chunk=Chunk(
                    id=n.node.node_id,
                    text=_strip_label(n.node.get_content()),
                    source=n.node.metadata.get("source", "?"),
                    metadata=dict(n.node.metadata),
                ),
                score=float(n.score or 0.0),
            )
            for n in cited
        ]
        return RAGResult(answer=str(response).strip(), sources=sources, steps=steps)
