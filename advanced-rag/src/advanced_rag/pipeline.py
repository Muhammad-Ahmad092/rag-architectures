"""Advanced RAG: rewrite -> hybrid retrieve -> fuse -> rerank -> generate -> check.

Each stage fixes a known failure of single-step RAG:

1. **Query rewriting**: the user's wording rarely matches the documents'. Claude (the
   grader model, cheap effort) writes extra search queries; the original is always kept.
2. **Hybrid retrieval**: dense vectors miss exact identifiers ("INC-2041", "24/7"); BM25
   misses paraphrases. For every query both run: the LlamaIndex vector retriever on the
   configured store, and an in-process ``BM25Retriever`` over the saved node corpus.
3. **Reciprocal rank fusion**: all lists merged by ``sum(1 / (RRF_K + rank))``, so passages
   that several queries/retrievers agree on rise to the top, with no score calibration.
4. **Rerank**: Cohere ``rerank-v3.5`` (or a local cross-encoder without a key) re-scores the
   fused candidates against the *original* question and keeps ``TOP_K``.
5. **Generate** with numbered passages and ``[n]`` citations (Claude).
6. **Answer check**: the grader lists unsupported claims; if any, one revision pass removes
   them, then it is re-checked. Verdict: verified | revised | unsupported | unchecked.
"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel, Field

from .config import Settings, require
from .corpus import corpus_path, load_corpus
from .embeddings import EmbedSpec, get_embed_model
from .llm import get_llms
from .models import Chunk, RAGResult, ScoredChunk, Step
from .vectorstore import StoreHandle, get_vector_store

EMPTY_INDEX_MSG = (
    "Index is empty — click “Index DATA_DIR” or run: uv run python -m advanced_rag.ingest"
)
NO_ANSWER = "I don't know based on the provided documents."


class Queries(BaseModel):
    queries: list[str] = Field(description="Alternative search queries for the question")


class Support(BaseModel):
    supported: bool = Field(description="Is every claim in the answer stated in the passages?")
    unsupported_claims: str = Field(description="Claims the passages do not support, or empty")


REWRITE_SYSTEM = (
    "Write up to {n} alternative search queries for the user's question about a company's "
    "internal documents: use the wording documents would use, expand abbreviations, split "
    "compound questions. Do not answer it."
)
ANSWER_SYSTEM = (
    "Answer the question using ONLY the numbered passages, citing them as [1], [2], ... "
    f'If they do not contain the answer, say: "{NO_ANSWER}"'
)
CHECK_SYSTEM = (
    "Check the answer against the passages. supported=true only if every factual claim is "
    "stated in them; list any claim that is not. Saying the documents lack the answer counts "
    "as supported."
)
REVISE_SYSTEM = (
    "Revise the answer so it contains only claims the numbered passages support: remove or "
    "correct the listed unsupported claims, keep the [n] citations, change nothing else."
)


def rrf(lists: list[list[NodeWithScore]], k: int = 60) -> list[NodeWithScore]:
    """Reciprocal rank fusion: score(d) = sum over lists of 1 / (k + rank_of_d)."""
    scores: dict[str, float] = {}
    nodes: dict[str, NodeWithScore] = {}
    for ranked in lists:
        for rank, n in enumerate(ranked, start=1):
            nid = n.node.node_id
            scores[nid] = scores.get(nid, 0.0) + 1.0 / (k + rank)
            nodes.setdefault(nid, n)
    order = sorted(scores, key=lambda nid: -scores[nid])
    return [NodeWithScore(node=nodes[nid].node, score=scores[nid]) for nid in order]


def _cohere_reranker(settings: Settings):
    from llama_index.postprocessor.cohere_rerank import CohereRerank

    key = require(settings.cohere_api_key, "COHERE_API_KEY")
    return CohereRerank(api_key=key, model=settings.rerank_model, top_n=settings.top_k)


def _local_reranker(settings: Settings):
    from llama_index.core.postprocessor import SentenceTransformerRerank

    return SentenceTransformerRerank(model=settings.local_rerank_model, top_n=settings.top_k)


def _numbered(nodes: list[NodeWithScore]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {n.node.metadata.get('source', '?')})\n{n.node.get_content()}"
        for i, n in enumerate(nodes, start=1)
    )


def _ids(nodes) -> list[tuple[str, float]]:
    return [(n.node.node_id, round(float(n.score or 0.0), 5)) for n in nodes]


class AdvancedRAG:
    def __init__(
        self,
        settings: Settings,
        embed_spec: EmbedSpec | None = None,
        store: StoreHandle | None = None,
        llm=None,
        grader=None,
        reranker="auto",
    ):
        self.settings = settings
        self.embed_spec = embed_spec or get_embed_model(settings)
        self.store = store or get_vector_store(settings, self.embed_spec.name, self.embed_spec.dim)
        self._llm, self._grader = llm, grader
        if reranker == "auto":
            self.reranker, self.reranker_name = self._pick_reranker()
        else:
            self.reranker = reranker
            self.reranker_name = "custom" if reranker is not None else "none"
        self.refresh_corpus()

    def _pick_reranker(self):
        s = self.settings
        if s.reranker == "none":
            return None, "none"
        if s.reranker == "cohere" and s.cohere_api_key:
            return _cohere_reranker(s), "cohere"
        label = "local" if s.reranker == "local" else "local (no COHERE_API_KEY)"
        return _local_reranker(s), label

    def refresh_corpus(self) -> None:
        from llama_index.retrievers.bm25 import BM25Retriever

        self.corpus = load_corpus(corpus_path(self.settings))
        self._bm25 = (
            BM25Retriever.from_defaults(
                nodes=self.corpus, similarity_top_k=min(self.settings.candidate_k, len(self.corpus))
            )
            if self.corpus
            else None
        )

    def _ensure_llms(self):
        if self._llm is None or self._grader is None:  # lazy: empty index needs no key
            gen, grader = get_llms(self.settings)
            self._llm, self._grader = self._llm or gen, self._grader or grader
        return self._llm, self._grader

    # ------------------------------------------------------------------ stages

    def rewrite(self, question: str) -> list[str]:
        n = self.settings.num_queries
        if n <= 1:
            return [question]
        _, grader = self._ensure_llms()
        out = grader.judge(REWRITE_SYSTEM.format(n=n - 1), f"Question: {question}", Queries)
        queries, seen = [question], {question.strip().lower()}
        for q in out.queries:
            if q.strip() and q.strip().lower() not in seen:
                seen.add(q.strip().lower())
                queries.append(q.strip())
        return queries[:n]

    def retrieve(self, queries: list[str]) -> tuple[list[list[NodeWithScore]], list[dict]]:
        vector = self.store.index(self.embed_spec.model).as_retriever(
            similarity_top_k=self.settings.candidate_k
        )
        lists, trace = [], []
        for q in queries:
            for name, retriever in (("vector", vector), ("bm25", self._bm25)):
                if retriever is None:
                    continue
                hits = retriever.retrieve(q)
                lists.append(hits)
                trace.append(
                    {
                        "query": q,
                        "retriever": name,
                        "hits": [
                            (h.node.metadata.get("source", "?"), round(float(h.score or 0), 4))
                            for h in hits
                        ],
                    }
                )
        return lists, trace

    def answer(self, question: str) -> RAGResult:
        if self.store.count() == 0:
            return RAGResult(answer=EMPTY_INDEX_MSG, sources=[], steps=[])
        s = self.settings
        steps: list[Step] = []
        queries = self.rewrite(question)
        steps.append(Step("rewrite", queries))
        lists, trace = self.retrieve(queries)
        steps.append(Step("retrieve", trace))
        fused = rrf(lists, k=s.rrf_k)[: s.candidate_k]
        steps.append(Step("fuse", _ids(fused)))
        if self.reranker is not None and fused:
            ranked = self.reranker.postprocess_nodes(fused, query_str=question)[: s.top_k]
            steps.append(Step("rerank", {"reranker": self.reranker_name, "kept": _ids(ranked)}))
        else:
            ranked = fused[: s.top_k]
            steps.append(Step("rerank", {"reranker": "none"}))

        llm, grader = self._ensure_llms()
        passages = _numbered(ranked)
        user = f"Passages:\n{passages}\n\nQuestion: {question}"
        answer = llm.complete(ANSWER_SYSTEM, user)
        steps.append(Step("generate", {"model": llm.model, "prompt": user}))

        verdict = "unchecked"
        if s.check_answer:
            check = grader.judge(
                CHECK_SYSTEM, f"Passages:\n{passages}\n\nAnswer:\n{answer}", Support
            )
            checks = [
                {"supported": check.supported, "unsupported_claims": check.unsupported_claims}
            ]
            verdict = "verified"
            if not check.supported:
                answer = llm.complete(
                    REVISE_SYSTEM,
                    f"Passages:\n{passages}\n\nAnswer:\n{answer}\n\n"
                    f"Unsupported claims: {check.unsupported_claims}",
                )
                again = grader.judge(
                    CHECK_SYSTEM, f"Passages:\n{passages}\n\nAnswer:\n{answer}", Support
                )
                checks.append(
                    {"supported": again.supported, "unsupported_claims": again.unsupported_claims}
                )
                verdict = "revised" if again.supported else "unsupported"
            steps.append(Step("check", checks))

        sources = [
            ScoredChunk(
                chunk=Chunk(
                    id=n.node.node_id,
                    text=n.node.get_content(),
                    source=n.node.metadata.get("source", "?"),
                    metadata=dict(n.node.metadata),
                ),
                score=float(n.score or 0.0),
            )
            for n in ranked
        ]
        return RAGResult(answer=answer.strip(), sources=sources, steps=steps, verdict=verdict)
