"""Streamlit UI for Corrective RAG.  Run: ``uv run streamlit run app.py``"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from corrective_rag import pipeline
from corrective_rag.config import ConfigError, get_settings
from corrective_rag.ingest import ingest
from corrective_rag.models import RAGResult

st.set_page_config(page_title="Corrective RAG", page_icon="🩹", layout="wide")
settings = get_settings()
st.session_state.setdefault("messages", [])

ACTIONS = {
    "correct": (
        "success",
        "✅ Correct: internal documents answer this (refined to the relevant sentences)",
    ),
    "ambiguous": (
        "info",
        "🔀 Ambiguous: internal passages were partial, so web results were added",
    ),
    "incorrect": (
        "warning",
        "🌐 Incorrect: internal passages were discarded, answered from the web",
    ),
}


def get_rag():
    if "rag" not in st.session_state:
        st.session_state.rag = pipeline.CorrectiveRAG(settings)
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


def render_result(result: RAGResult) -> None:
    if result.verdict in ACTIONS:
        kind, text = ACTIONS[result.verdict]
        getattr(st, kind)(text)
    st.markdown(result.answer)
    if result.sources:
        with st.expander(f"Sources ({len(result.sources)})"):
            for i, s in enumerate(result.sources, start=1):
                if s.chunk.metadata.get("web"):
                    title = s.chunk.metadata.get("title", s.chunk.source)
                    st.markdown(f"**[{i}]** 🌐 [{title}]({s.chunk.source})")
                else:
                    st.markdown(f"**[{i}]** 📄 `{s.chunk.source}` · evaluator `{s.score:.2f}`")
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
    st.header("Corrective RAG")
    st.caption("Retrieval evaluator → correct / ambiguous / incorrect → refine or web search.")
    st.markdown(f"**LLM:** openai / `{settings.llm_model}`")
    st.markdown(
        f"**Thresholds:** correct ≥ {settings.upper_threshold} · "
        f"incorrect < {settings.lower_threshold}"
    )
    st.markdown(f"**Knowledge refinement:** {'on' if settings.refine_knowledge else 'off'}")
    if rag.web is None:
        st.markdown("**Web search:** off (set `TAVILY_API_KEY` to enable)")
    else:
        st.markdown(f"**Web search:** Tavily · {settings.web_results} results")
    st.markdown(f"**Embeddings:** `{rag.embed_spec.name}`")
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

# ----------------------------------------------------------------------------- chat
st.title("🩹 Corrective RAG")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["content"])

if question := st.chat_input("Ask about the documents (or something they don't cover)…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and evaluating…"):
                result = rag.answer(question)
        except ConfigError as exc:
            st.error(str(exc))
        else:
            render_result(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
