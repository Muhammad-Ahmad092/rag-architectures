"""Streamlit UI for Simple RAG with memory.  Run: ``uv run streamlit run app.py``"""

from __future__ import annotations

import uuid
from pathlib import Path

import streamlit as st

from simple_rag_memory import pipeline
from simple_rag_memory.config import ConfigError, get_settings
from simple_rag_memory.ingest import ingest
from simple_rag_memory.models import RAGResult

st.set_page_config(page_title="Simple RAG + Memory", page_icon="🧠", layout="wide")
settings = get_settings()
st.session_state.setdefault("session_id", uuid.uuid4().hex[:8])
st.session_state.setdefault("messages", [])


def get_rag():
    if "rag" not in st.session_state:
        st.session_state.rag = pipeline.MemoryRAG(settings)
    return st.session_state.rag


try:
    rag = get_rag()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()


def show_report(report) -> None:
    st.success(f"Indexed {len(report.loaded)} files → {rag.store.count()} chunks")
    for path, reason in report.skipped:
        st.warning(f"Skipped `{path}`: {reason}")


def render_result(result: RAGResult, question: str) -> None:
    if result.search_query and result.search_query != question:
        st.caption(f"🔎 Searched for: *{result.search_query}*")
    st.markdown(result.answer)
    if result.sources:
        with st.expander(f"Sources ({len(result.sources)})"):
            for i, s in enumerate(result.sources, start=1):
                st.markdown(f"**[{i}]** · `{s.chunk.source}` · relevance `{s.score:.3f}`")
                st.text(s.chunk.text[:600])
    if result.steps:
        with st.expander("Under the hood"):
            for step in result.steps:
                st.markdown(f"**{step.name}**")
                if isinstance(step.detail, str):
                    st.code(step.detail, language="text")
                else:
                    st.json(step.detail)


# ----------------------------------------------------------------------------- sidebar
sid = st.session_state.session_id
with st.sidebar:
    st.header("Simple RAG + Memory")
    st.caption("LangChain conversational RAG: follow-ups are rewritten using the chat history.")
    rewrite_model = settings.rewrite_model or settings.llm_model
    st.markdown(f"**LLM:** groq / `{settings.llm_model}` · **Rewriter:** `{rewrite_model}`")
    memory_slot = st.empty()  # filled at the end of the run, after this turn is stored
    rewrite = st.toggle(
        "History-aware rewriting",
        value=True,
        help="Off: follow-ups are searched as typed. Try it to see why rewriting matters.",
    )
    st.markdown(f"**Embeddings:** `{rag.embed_spec.name}`")
    st.markdown(
        f"**Chunker:** RecursiveCharacterTextSplitter "
        f"{settings.chunk_size}/{settings.chunk_overlap} chars"
    )
    st.markdown(f"**Vector store:** `{settings.vector_store}`")
    st.markdown(f"**Top-k:** {settings.top_k} · **Indexed chunks:** {rag.store.count()}")
    st.markdown(f"**DATA_DIR:** `{settings.data_dir}`")

    if st.button("Index DATA_DIR", type="primary"):
        try:
            with st.spinner("Indexing…"):
                show_report(ingest(settings, rag.embed_spec, rag.store, reset=True))
        except ConfigError as exc:
            st.error(str(exc))

    uploads = st.file_uploader("Add files", accept_multiple_files=True)
    if uploads and st.button("Index uploaded files"):
        target = Path(settings.index_dir) / "uploads"
        target.mkdir(parents=True, exist_ok=True)
        for f in uploads:
            (target / Path(f.name).name).write_bytes(f.getvalue())
        try:
            show_report(ingest(settings, rag.embed_spec, rag.store, reset=False, data_dir=target))
        except ConfigError as exc:
            st.error(str(exc))

    if st.button("Clear chat"):
        st.session_state.messages = []
        rag.memory.clear(sid)  # the conversation and the model's memory go together

# ----------------------------------------------------------------------------- chat
st.title("🧠 Simple RAG + Memory")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["content"], msg["question"])

if question := st.chat_input("Ask, then ask a follow-up (e.g. “who wrote its postmortem?”)…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                result = rag.answer(question, session_id=sid, rewrite=rewrite)
        except ConfigError as exc:
            st.error(str(exc))
        else:
            render_result(result, question)
            st.session_state.messages.append(
                {"role": "assistant", "content": result, "question": question}
            )

memory_slot.markdown(
    f"**Memory:** last {settings.memory_turns} turns · "
    f"**This chat:** {rag.memory.turns(sid)} turns (session `{sid}`)"
)
