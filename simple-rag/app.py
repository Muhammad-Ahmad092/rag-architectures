"""Streamlit UI for Simple RAG.  Run: ``uv run streamlit run app.py``"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from simple_rag import pipeline
from simple_rag.config import ConfigError, get_settings
from simple_rag.ingest import ingest
from simple_rag.models import RAGResult

st.set_page_config(page_title="Simple RAG", page_icon="📎", layout="wide")
settings = get_settings()


def get_rag():
    if "rag" not in st.session_state:
        st.session_state.rag = pipeline.SimpleRAG(settings)
    return st.session_state.rag


try:
    rag = get_rag()
except ConfigError as exc:
    st.error(str(exc))
    st.stop()


def show_report(report) -> None:
    st.success(f"Indexed {len(report.loaded)} files → {rag.store.count()} nodes")
    for path, reason in report.skipped:
        st.warning(f"Skipped `{path}`: {reason}")


def render_result(result: RAGResult) -> None:
    st.markdown(result.answer)
    if result.sources:
        with st.expander(f"Sources ({len(result.sources)})"):
            for i, s in enumerate(result.sources, start=1):
                st.markdown(f"**Source {i}** · `{s.chunk.source}` · score `{s.score:.3f}`")
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
with st.sidebar:
    st.header("Simple RAG")
    st.caption(
        "LlamaIndex single-step retrieval with sentence-aware chunks and numbered citations."
    )
    st.markdown(f"**LLM:** gemini / `{settings.llm_model}`")
    st.markdown(f"**Embeddings:** `{rag.embed_spec.name}`")
    st.markdown(
        f"**Chunker:** SentenceSplitter {settings.chunk_size}/{settings.chunk_overlap} tokens · "
        f"**Citation passages:** {settings.citation_chunk_size}"
    )
    st.markdown(f"**Vector store:** `{settings.vector_store}`")
    st.markdown(f"**Top-k:** {settings.top_k} · **Indexed nodes:** {rag.store.count()}")
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

# ----------------------------------------------------------------------------- chat
st.title("📎 Simple RAG")
st.session_state.setdefault("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["content"])

if question := st.chat_input("Ask about the documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                result = rag.answer(question)
        except ConfigError as exc:
            st.error(str(exc))
        else:
            render_result(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
