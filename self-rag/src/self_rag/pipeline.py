"""Self-RAG: the model critiques its own retrieval and its own answer, and loops.

Self-RAG (Asai et al., 2023) trains a model to emit reflection tokens: is retrieval
needed, is this passage relevant (ISREL), is the answer supported by it (ISSUP), is it
useful (ISUSE). Here an LLM grader plays those critics, as a LangGraph state machine:

    retrieve -> grade_documents --(none relevant)--> rewrite -> retrieve ...
                     |                                  ^
                (some relevant)                         | (not useful)
                     v                                  |
                 generate -> check_support --(supported)--> check_useful -> END
                     ^              |
                     +--(not supported: retry with the grader's feedback)

Loops are bounded: ``MAX_REWRITES`` query rewrites, ``MAX_GENERATIONS`` answer attempts
per retrieval. When nothing relevant is ever found, the graph says so instead of guessing.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Settings
from .embeddings import EmbedSpec, get_embeddings
from .llm import get_llms
from .models import Chunk, RAGResult, ScoredChunk, Step
from .vectorstore import StoreHandle, get_vector_store

EMPTY_INDEX_MSG = "Index is empty — click “Index DATA_DIR” or run: uv run python -m self_rag.ingest"
NO_ANSWER = "I don't know based on the provided documents."

# ----------------------------------------------------------------------------- critics


class Relevance(BaseModel):
    relevant: bool = Field(description="Does the passage help answer the question?")
    reason: str


class Support(BaseModel):
    supported: bool = Field(description="Is every claim in the answer backed by the passages?")
    unsupported_claims: str = Field(description="Claims not backed by the passages, or empty")


class Usefulness(BaseModel):
    useful: bool = Field(description="Does the answer actually address the question?")
    reason: str


RELEVANCE_SYSTEM = (
    "You grade whether a retrieved passage is relevant to a user's question. Relevant "
    "means it contains information that helps answer the question, even partially. "
    "Keyword overlap alone is not enough."
)
SUPPORT_SYSTEM = (
    "You check an answer against the passages it was written from. supported=true only "
    "if every factual claim in the answer is stated in the passages. List any claim that "
    "is not. Saying the documents do not contain the answer counts as supported."
)
USEFUL_SYSTEM = (
    "You check whether an answer addresses the user's question. useful=false if it "
    "dodges, answers a different question, or only restates the question."
)
REWRITE_SYSTEM = (
    "The search query below did not retrieve relevant passages from a company's internal "
    "documents (products, incidents, pricing, people, policies). Write one better search "
    "query for the same question: use likely document wording, expand abbreviations, "
    "drop filler. Return only the new search query."
)
ANSWER_SYSTEM = (
    "You answer questions about a company's documents. Use ONLY the numbered passages "
    "and cite them as [1], [2], ... If they do not contain the answer, say: "
    f'"{NO_ANSWER}"'
)


# ----------------------------------------------------------------------------- state


class State(TypedDict, total=False):
    question: str
    query: str
    hits: list  # [(Document, score)] from the last retrieval
    relevant: list  # [(Document, score)] that passed the relevance grader
    answer: str
    feedback: str  # grader's unsupported claims, fed into the next attempt
    rewrites: int
    generations: int  # attempts for the current retrieval
    verdict: str
    trace: list  # [Step]


def _context(relevant) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
        for i, (d, _) in enumerate(relevant, start=1)
    )


class SelfRAG:
    def __init__(
        self,
        settings: Settings,
        embed_spec: EmbedSpec | None = None,
        store: StoreHandle | None = None,
        llm=None,
        grader=None,
    ):
        self.settings = settings
        self.embed_spec = embed_spec or get_embeddings(settings)
        self.store = store or get_vector_store(settings, self.embed_spec)
        self._llm, self._grader = llm, grader
        self.graph = self._build()

    def _ensure_llms(self):
        if self._llm is None or self._grader is None:  # lazy: empty index needs no key
            gen, grader = get_llms(self.settings)
            self._llm = self._llm or gen
            self._grader = self._grader or grader
        return self._llm, self._grader

    @property
    def llm(self):
        return self._ensure_llms()[0]

    @property
    def grader(self):
        return self._ensure_llms()[1]

    # ------------------------------------------------------------------ nodes

    def _retrieve(self, state: State) -> State:
        hits = self.store.search(state["query"], k=self.settings.top_k)
        detail = {
            "query": state["query"],
            "hits": [(d.metadata.get("source", "?"), round(float(s), 4)) for d, s in hits],
        }
        return {
            "hits": hits,
            "generations": 0,
            "trace": [*state["trace"], Step("retrieve", detail)],
        }

    def _grade_documents(self, state: State) -> State:
        relevant, grades = [], []
        for doc, score in state["hits"]:
            g = self.grader.judge(
                RELEVANCE_SYSTEM,
                f"Question: {state['question']}\n\nPassage:\n{doc.page_content}",
                Relevance,
            )
            grades.append(
                {
                    "source": doc.metadata.get("source", "?"),
                    "relevant": g.relevant,
                    "reason": g.reason,
                }
            )
            if g.relevant:
                relevant.append((doc, score))
        return {"relevant": relevant, "trace": [*state["trace"], Step("grade_documents", grades)]}

    def _rewrite(self, state: State) -> State:
        query = (
            self.llm.complete(
                REWRITE_SYSTEM, f"Question: {state['question']}\nPrevious query: {state['query']}"
            )
            .strip()
            .strip("\"'")
            or state["question"]
        )
        step = Step("rewrite", {"from": state["query"], "to": query})
        return {"query": query, "rewrites": state["rewrites"] + 1, "trace": [*state["trace"], step]}

    def _generate(self, state: State) -> State:
        user = f"Passages:\n{_context(state['relevant'])}\n\nQuestion: {state['question']}"
        if state.get("feedback"):
            user += (
                f"\n\nYour previous answer was:\n{state['answer']}\n"
                f"It contained claims the passages do not support: {state['feedback']}\n"
                "Answer again using only what the passages state."
            )
        answer = self.llm.complete(ANSWER_SYSTEM, user)
        n = state["generations"] + 1
        step = Step("generate", {"attempt": n, "model": self.llm.model, "prompt": user})
        return {"answer": answer, "generations": n, "trace": [*state["trace"], step]}

    def _check_support(self, state: State) -> State:
        g = self.grader.judge(
            SUPPORT_SYSTEM,
            f"Passages:\n{_context(state['relevant'])}\n\nAnswer:\n{state['answer']}",
            Support,
        )
        step = Step(
            "check_support", {"supported": g.supported, "unsupported_claims": g.unsupported_claims}
        )
        feedback = "" if g.supported else (g.unsupported_claims or "unspecified claims")
        return {"feedback": feedback, "trace": [*state["trace"], step]}

    def _check_useful(self, state: State) -> State:
        g = self.grader.judge(
            USEFUL_SYSTEM,
            f"Question: {state['question']}\n\nAnswer:\n{state['answer']}",
            Usefulness,
        )
        step = Step("check_useful", {"useful": g.useful, "reason": g.reason})
        out: State = {"trace": [*state["trace"], step]}
        if g.useful:
            out["verdict"] = "verified"
        elif state["rewrites"] >= self.settings.max_rewrites:
            out["verdict"] = "not useful"
        return out

    def _give_up(self, state: State) -> State:
        step = Step("give_up", {"rewrites": state["rewrites"], "reason": "no relevant passages"})
        return {
            "answer": NO_ANSWER,
            "relevant": [],
            "verdict": "no relevant documents",
            "trace": [*state["trace"], step],
        }

    # ------------------------------------------------------------------ routing

    def _after_grading(self, state: State) -> str:
        if state["relevant"]:
            return "generate"
        return "rewrite" if state["rewrites"] < self.settings.max_rewrites else "give_up"

    def _after_support(self, state: State) -> str:
        if not state["feedback"]:
            return "check_useful"
        return "generate" if state["generations"] < self.settings.max_generations else "unsupported"

    def _after_useful(self, state: State) -> str:
        return END if state.get("verdict") else "rewrite"

    def _build(self):
        g = StateGraph(State)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade_documents", self._grade_documents)
        g.add_node("rewrite", self._rewrite)
        g.add_node("generate", self._generate)
        g.add_node("check_support", self._check_support)
        g.add_node("check_useful", self._check_useful)
        g.add_node("give_up", self._give_up)
        g.add_node("unsupported", lambda state: {"verdict": "unsupported"})
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade_documents")
        g.add_conditional_edges("grade_documents", self._after_grading)
        g.add_edge("rewrite", "retrieve")
        g.add_edge("generate", "check_support")
        g.add_conditional_edges("check_support", self._after_support)
        g.add_conditional_edges("check_useful", self._after_useful)
        g.add_edge("give_up", END)
        g.add_edge("unsupported", END)
        return g.compile()

    # ------------------------------------------------------------------ entry

    def answer(self, question: str) -> RAGResult:
        if self.store.count() == 0:
            return RAGResult(answer=EMPTY_INDEX_MSG, sources=[], steps=[])
        state = self.graph.invoke(
            {"question": question, "query": question, "rewrites": 0, "trace": []},
            {"recursion_limit": 100},
        )
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
            for d, s in state.get("relevant", [])
        ]
        return RAGResult(
            answer=state["answer"].strip(),
            sources=sources,
            steps=state["trace"],
            verdict=state["verdict"],
        )
