r"""Corrective RAG (CRAG): check retrieval quality, correct it, fall back to the web.

CRAG (Yan et al., 2024, "Corrective Retrieval Augmented Generation") puts a lightweight
*retrieval evaluator* between retrieval and generation. Each retrieved passage gets a
relevance score; two thresholds turn the scores into one of three actions:

* **Correct**: some passage scores >= ``UPPER_THRESHOLD``. Keep the passages, then
  *refine* them: strip each down to the sentences that bear on the question.
* **Incorrect**: every passage scores < ``LOWER_THRESHOLD``. Discard them all, rewrite the
  question as a web query, search the web (Tavily), and answer from the web results.
* **Ambiguous**: anything in between. Refine the passages that are not clearly wrong
  *and* add web results.

    retrieve -> evaluate --correct-----> refine --> generate
                    |  \--ambiguous---> refine --> web_rewrite -> web_search -> generate
                    \----incorrect-------------> web_rewrite -> web_search -> generate

If refinement leaves nothing usable, the graph also falls back to the web. With web search
off (or no ``TAVILY_API_KEY``) and nothing usable, it says it does not know.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Settings
from .embeddings import EmbedSpec, get_embeddings
from .llm import get_llm
from .models import Chunk, RAGResult, ScoredChunk, Step
from .vectorstore import StoreHandle, get_vector_store
from .web import WebSearchError, get_web

EMPTY_INDEX_MSG = (
    "Index is empty — click “Index DATA_DIR” or run: uv run python -m corrective_rag.ingest"
)
NO_ANSWER = "I don't know based on the provided documents."
NO_WEB = "web search off or TAVILY_API_KEY not set"


class PassageScore(BaseModel):
    score: float = Field(description="0.0 = irrelevant, 1.0 = directly answers the question")
    reason: str


EVALUATOR_SYSTEM = (
    "You are a retrieval evaluator. Score how useful the passage is for answering the "
    "question, from 0.0 (irrelevant or about something else) to 1.0 (contains the answer). "
    "Use 0.4-0.6 for passages that are on-topic but incomplete."
)
REFINE_SYSTEM = (
    "Extract from the passage only the sentences that help answer the question, verbatim. "
    "If no sentence helps, reply exactly NONE."
)
WEB_REWRITE_SYSTEM = (
    "Rewrite the question as a concise web search query: keep names, products and "
    "identifiers, drop conversational words. Return only the search query."
)
ANSWER_SYSTEM = (
    "Answer the question using ONLY the numbered passages, and cite them as [1], [2], ... "
    "Passages marked 'web' come from the internet; prefer internal documents when they "
    f'conflict. If the passages do not contain the answer, say: "{NO_ANSWER}"'
)


def decide_action(scores: list[float], upper: float, lower: float) -> str:
    if any(s >= upper for s in scores):
        return "correct"
    if all(s < lower for s in scores):  # also true for no passages at all
        return "incorrect"
    return "ambiguous"


def _label(doc) -> str:
    src = doc.metadata.get("source", "?")
    return f"web: {src}" if doc.metadata.get("web") else f"source: {src}"


class State(TypedDict, total=False):
    question: str
    hits: list  # [(Document, score)]
    scores: list  # evaluator score per hit
    action: str
    knowledge: list  # [(Document, score)] going into the answer
    web_query: str
    answer: str
    trace: list


class CorrectiveRAG:
    def __init__(
        self,
        settings: Settings,
        embed_spec: EmbedSpec | None = None,
        store: StoreHandle | None = None,
        llm=None,
        web="auto",
    ):
        self.settings = settings
        self.embed_spec = embed_spec or get_embeddings(settings)
        self.store = store or get_vector_store(settings, self.embed_spec)
        self._llm = llm
        self.web = get_web(settings) if web == "auto" else web
        self.graph = self._build()

    @property
    def llm(self):
        if self._llm is None:  # lazy: an empty index never needs the API key
            self._llm = get_llm(self.settings)
        return self._llm

    # ------------------------------------------------------------------ nodes

    def _retrieve(self, state: State) -> State:
        hits = self.store.search(state["question"], k=self.settings.top_k)
        detail = [(d.metadata.get("source", "?"), round(float(s), 4)) for d, s in hits]
        return {"hits": hits, "trace": [*state["trace"], Step("retrieve", detail)]}

    def _evaluate(self, state: State) -> State:
        scores, rows = [], []
        for doc, _ in state["hits"]:
            g = self.llm.judge(
                EVALUATOR_SYSTEM,
                f"Question: {state['question']}\n\nPassage:\n{doc.page_content}",
                PassageScore,
            )
            score = min(1.0, max(0.0, float(g.score)))
            scores.append(score)
            rows.append(
                {"source": doc.metadata.get("source", "?"), "score": score, "reason": g.reason}
            )
        s = self.settings
        action = decide_action(scores, s.upper_threshold, s.lower_threshold)
        detail = {
            "action": action,
            "thresholds": {"upper": s.upper_threshold, "lower": s.lower_threshold},
            "passages": rows,
        }
        return {
            "scores": scores,
            "action": action,
            "trace": [*state["trace"], Step("evaluate", detail)],
        }

    def _refine(self, state: State) -> State:
        keep = [
            (doc, score)
            for (doc, _), score in zip(state["hits"], state["scores"], strict=True)
            if score >= self.settings.lower_threshold
        ]
        knowledge, kept = [], []
        for doc, score in keep:
            text = doc.page_content
            if self.settings.refine_knowledge:
                text = self.llm.complete(
                    REFINE_SYSTEM, f"Question: {state['question']}\n\nPassage:\n{doc.page_content}"
                ).strip()
                if not text or text.upper() == "NONE":
                    continue
            refined = doc.model_copy(update={"page_content": text})
            knowledge.append((refined, score))
            kept.append(
                {
                    "source": doc.metadata.get("source", "?"),
                    "chars": [len(doc.page_content), len(text)],
                }
            )
        detail = {"enabled": self.settings.refine_knowledge, "kept": kept}
        return {"knowledge": knowledge, "trace": [*state["trace"], Step("refine", detail)]}

    def _web_rewrite(self, state: State) -> State:
        query = self.llm.complete(WEB_REWRITE_SYSTEM, state["question"]).strip().strip("\"'")
        query = query or state["question"]
        step = Step("web_rewrite", {"question": state["question"], "web_query": query})
        return {"web_query": query, "trace": [*state["trace"], step]}

    def _web_search(self, state: State) -> State:
        knowledge = list(state.get("knowledge", []))
        if self.web is None:
            return {
                "knowledge": knowledge,
                "trace": [*state["trace"], Step("web_search", {"skipped": NO_WEB})],
            }
        try:
            docs = self.web.search(state["web_query"])
        except WebSearchError as exc:
            step = Step("web_search", {"query": state["web_query"], "error": str(exc)})
            return {"knowledge": knowledge, "trace": [*state["trace"], step]}
        knowledge += [(d, float(d.metadata.get("web_score", 0.0))) for d in docs]
        step = Step(
            "web_search",
            {"query": state["web_query"], "results": [d.metadata["source"] for d in docs]},
        )
        return {"knowledge": knowledge, "trace": [*state["trace"], step]}

    def _generate(self, state: State) -> State:
        knowledge = state.get("knowledge", [])
        if not knowledge:
            step = Step("generate", {"skipped": "no usable passages"})
            return {"answer": NO_ANSWER, "trace": [*state["trace"], step]}
        context = "\n\n".join(
            f"[{i}] ({_label(d)})\n{d.page_content}" for i, (d, _) in enumerate(knowledge, start=1)
        )
        user = f"Passages:\n{context}\n\nQuestion: {state['question']}"
        answer = self.llm.complete(ANSWER_SYSTEM, user)
        step = Step("generate", {"model": self.llm.model, "prompt": user})
        return {"answer": answer, "trace": [*state["trace"], step]}

    # ------------------------------------------------------------------ routing

    def _to_web(self) -> str:
        # without a web client, skip the rewrite LLM call; web_search records why
        return "web_rewrite" if self.web is not None else "web_search"

    def _after_evaluate(self, state: State) -> str:
        return self._to_web() if state["action"] == "incorrect" else "refine"

    def _after_refine(self, state: State) -> str:
        if state["action"] == "ambiguous" or not state["knowledge"]:
            return self._to_web()
        return "generate"

    def _build(self):
        g = StateGraph(State)
        for name, fn in [
            ("retrieve", self._retrieve),
            ("evaluate", self._evaluate),
            ("refine", self._refine),
            ("web_rewrite", self._web_rewrite),
            ("web_search", self._web_search),
            ("generate", self._generate),
        ]:
            g.add_node(name, fn)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "evaluate")
        g.add_conditional_edges("evaluate", self._after_evaluate)
        g.add_conditional_edges("refine", self._after_refine)
        g.add_edge("web_rewrite", "web_search")
        g.add_edge("web_search", "generate")
        g.add_edge("generate", END)
        return g.compile()

    # ------------------------------------------------------------------ entry

    def answer(self, question: str) -> RAGResult:
        if self.store.count() == 0:
            return RAGResult(answer=EMPTY_INDEX_MSG, sources=[], steps=[])
        state = self.graph.invoke({"question": question, "trace": []})
        sources = [
            ScoredChunk(
                chunk=Chunk(
                    id=d.metadata.get("chunk_id", "?"),
                    text=d.page_content,
                    source=d.metadata.get("source", "?"),
                    metadata=dict(d.metadata),
                ),
                score=float(s),
            )
            for d, s in state.get("knowledge", [])
        ]
        return RAGResult(
            answer=state["answer"].strip(),
            sources=sources,
            steps=state["trace"],
            verdict=state["action"],
        )
