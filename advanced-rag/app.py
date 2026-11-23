"""Streamlit UI for Advanced RAG.  Run: ``uv run streamlit run app.py``"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from advanced_rag import pipeline
from advanced_rag.config import ConfigError, get_settings
from advanced_rag.ingest import ingest
from advanced_rag.llm import LLMError
from advanced_rag.models import RAGResult

st.set_page_config(page_title="Advanced RAG", page_icon="🛠️", layout="wide")
settings = get_settings()
st.session_state.setdefault("messages", [])

VERDICTS = {
    "verified": ("success", "✅ Answer checked: every claim is supported by the passages"),
    "revised": ("info", "✏️ Unsupported claims were found and removed in a revision"),
    "unsupported": (
        "warning",
        "⚠️ Still contains unsupported claims after revision: check the sources",
    ),
    "unchecked": ("caption", "Answer check off"),
}


def get_rag():
    if "rag" not in st.session_state:
        st.session_state.rag = pipeline.AdvancedRAG(settings)
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
    if result.verdict in VERDICTS:
        kind, text = VERDICTS[result.verdict]
        getattr(st, kind)(text)
    queries = next((s.detail for s in result.steps if s.name == "rewrite"), [])
    if len(queries) > 1:
        st.caption("🔎 Searched: " + " · ".join(f"“{q}”" for q in queries))
    st.markdown(result.answer)
    if result.sources:
        with st.expander(f"Sources after rerank ({len(result.sources)})"):
            for i, s in enumerate(result.sources, start=1):
                st.markdown(f"**[{i}]** · `{s.chunk.source}` · score `{s.score:.4f}`")
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
    st.header("Advanced RAG")
    st.caption("Rewrite → hybrid (vector + BM25) → RRF → rerank → generate → check.")
    grader = settings.grader_model or settings.llm_model
    st.markdown(f"**LLM:** anthropic / `{settings.llm_model}` · **grader:** `{grader}`")
    st.markdown(
        f"**Queries:** up to {settings.num_queries} · **candidates:** {settings.candidate_k} "
        f"per retriever · **RRF k:** {settings.rrf_k}"
    )
    st.markdown(f"**Reranker:** {rag.reranker_name} → top {settings.top_k}")
    st.markdown(f"**Answer check:** {'on' if settings.check_answer else 'off'}")
    st.markdown(f"**BM25 corpus:** {len(rag.corpus)} nodes · **Indexed:** {rag.store.count()}")
    st.markdown(
        f"**Embeddings:** `{rag.embed_spec.name}` · **Vector store:** `{settings.vector_store}`"
    )
    st.markdown(f"**DATA_DIR:** `{settings.data_dir}`")

    if st.button("Index DATA_DIR", type="primary"):
        try:
            with st.spinner("Indexing…"):
                show_report(ingest(settings, rag.embed_spec, rag.store, reset=True))
            rag.refresh_corpus()
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
            rag.refresh_corpus()
        except ConfigError as exc:
            st.error(str(exc))

    if st.button("Clear chat"):
        st.session_state.messages = []

# ----------------------------------------------------------------------------- chat
st.title("🛠️ Advanced RAG")

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
            with st.spinner("Rewriting, retrieving, reranking, checking…"):
                result = rag.answer(question)
        except (ConfigError, LLMError) as exc:
            st.error(str(exc))
        else:
            render_result(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
