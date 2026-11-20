"""Simple RAG with memory: conversation history + history-aware query rewriting.

A follow-up such as "who wrote its postmortem?" is useless as a search query on its own.
Each turn therefore:

1. loads the session's recent history (sliding window, ``MEMORY_TURNS``);
2. if there is history, asks the LLM to *condense* the follow-up into a standalone
   question ("Who wrote the postmortem for INC-2041?");
3. retrieves top-k chunks with the standalone question;
4. answers the *original* question from the numbered context, with the history in the
   prompt so the reply reads naturally in the conversation;
5. appends the turn to the history.

Built from LangChain 1.x core primitives (prompt templates, ``MessagesPlaceholder``,
``InMemoryChatMessageHistory``, ``trim_messages``) rather than the legacy
``create_history_aware_retriever`` chains, so every step is visible and testable.
"""

from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .config import Settings
from .embeddings import EmbedSpec, get_embeddings
from .llm import get_llm, model_name
from .memory import SessionMemory
from .models import Chunk, RAGResult, ScoredChunk, Step
from .vectorstore import StoreHandle, get_vector_store

EMPTY_INDEX_MSG = (
    "Index is empty — click “Index DATA_DIR” or run: uv run python -m simple_rag_memory.ingest"
)

CONDENSE_SYSTEM = (
    "Given the chat history and a follow-up question, rewrite the follow-up as a standalone "
    "question that can be understood without the history. Replace pronouns and references "
    "(it, that incident, she, the second one) with the names they refer to. Do NOT answer "
    "the question. If it is already standalone, return it unchanged. Return only the "
    "standalone question."
)

ANSWER_SYSTEM = (
    "You answer questions about a company's documents. Use ONLY the numbered context "
    "below and cite the passages you used as [1], [2], ... If the context does not contain "
    'the answer, say: "I don\'t know based on the provided documents." The chat history '
    "only tells you what the user is referring to; facts must come from the context.\n\n"
    "Context:\n{context}"
)

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", CONDENSE_SYSTEM), MessagesPlaceholder("history"), ("human", "{question}")]
)
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [("system", ANSWER_SYSTEM), MessagesPlaceholder("history"), ("human", "{question}")]
)

_LABEL = re.compile(r"^\s*(standalone question|rewritten question|question)\s*:\s*", re.I)


def clean_rewrite(text: str, fallback: str) -> str:
    """Strip quotes / 'Standalone question:' labels; empty output falls back to the original."""
    out = _LABEL.sub("", text.strip()).strip().strip("\"'“”").strip()
    return out or fallback


class MemoryRAG:
    def __init__(
        self,
        settings: Settings,
        embed_spec: EmbedSpec | None = None,
        store: StoreHandle | None = None,
        llm=None,
        rewrite_llm=None,
        memory: SessionMemory | None = None,
    ):
        self.settings = settings
        self.embed_spec = embed_spec or get_embeddings(settings)
        self.store = store or get_vector_store(settings, self.embed_spec)
        self.memory = memory or SessionMemory(settings.memory_turns)
        self._llm = llm
        self._rewrite_llm = rewrite_llm

    @property
    def llm(self):
        if self._llm is None:  # lazy: an empty index never needs the API key
            self._llm = get_llm(self.settings)
        return self._llm

    @property
    def rewrite_llm(self):
        if self._rewrite_llm is None:
            if self.settings.rewrite_model and self._llm is None:
                self._rewrite_llm = get_llm(self.settings, model=self.settings.rewrite_model)
            else:
                self._rewrite_llm = self.llm
        return self._rewrite_llm

    def condense(self, question: str, history) -> str:
        chain = CONDENSE_PROMPT | self.rewrite_llm | StrOutputParser()
        return clean_rewrite(chain.invoke({"history": history, "question": question}), question)

    def answer(self, question: str, session_id: str = "default", rewrite: bool = True) -> RAGResult:
        if self.store.count() == 0:
            return RAGResult(answer=EMPTY_INDEX_MSG, sources=[], steps=[])

        window = self.memory.window(session_id)
        steps = [
            Step(
                "memory",
                {
                    "session": session_id,
                    "turns_in_window": len(window) // 2,
                    "turns_total": self.memory.turns(session_id),
                },
            )
        ]

        if not rewrite:
            query, detail = question, {"skipped": "disabled"}
        elif not window:
            query, detail = question, {"skipped": "no history yet"}
        else:
            query = self.condense(question, window)
            detail = {
                "original": question,
                "standalone": query,
                "model": model_name(self.rewrite_llm),
            }
        steps.append(Step("rewrite", detail))

        hits = self.store.search(query, k=self.settings.top_k)
        steps.append(
            Step("retrieve", [(d.metadata.get("source", "?"), round(float(s), 4)) for d, s in hits])
        )

        context = "\n\n".join(
            f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
            for i, (d, _) in enumerate(hits, start=1)
        )
        messages = ANSWER_PROMPT.format_messages(context=context, history=window, question=question)
        steps.append(Step("prompt", "\n\n".join(f"[{m.type}] {m.content}" for m in messages)))

        answer = StrOutputParser().invoke(self.llm.invoke(messages)).strip()
        steps.append(Step("generate", {"model": model_name(self.llm)}))

        self.memory.add_turn(session_id, question, answer)
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
            for d, s in hits
        ]
        return RAGResult(answer=answer, sources=sources, steps=steps, search_query=query)
