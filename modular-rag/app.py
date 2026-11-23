"""Streamlit UI for Modular RAG.  Run: ``uv run streamlit run app.py``"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from modular_rag import pipeline
from modular_rag.config import ConfigError, get_settings
from modular_rag.ingest import ingest
from modular_rag.models import RAGResult
from modular_rag.modules import GENERATORS, RANKERS, RETRIEVERS, Modules, dot

st.set_page_config(page_title="Modular RAG", page_icon="🧩", layout="wide")
settings = get_settings()
st.session_state.setdefault("messages", [])

HELP = {
    "vector": "dense embedding search on the configured store",
    "bm25": "keyword search (Haystack InMemoryBM25Retriever)",
    "hybrid": "vector + BM25, fused with reciprocal rank fusion",
    "none": "keep the retriever's top-k as is",
    "cross-encoder": "re-score each (query, passage) pair; downloads a small model once",
    "lost-in-the-middle": "put the best passages at the start and end of the prompt",
    "mistral": f"Mistral `{settings.llm_model}`",
    "openai": f"OpenAI `{settings.openai_model}`",
}


def get_rag():
    if "rag" not in st.session_state:
        st.session_state.rag = pipeline.ModularRAG(settings)
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


def render_result(result: RAGResult, modules: Modules | None) -> None:
    if modules is not None:
        st.caption(f"🧩 {modules.retriever} → {modules.ranker} → {modules.generator}")
    st.markdown(result.answer)
    if result.sources:
        with st.expander(f"Sources ({len(result.sources)})"):
            for i, s in enumerate(result.sources, start=1):
                st.markdown(f"**[{i}]** · `{s.chunk.source}` · score `{s.score:.3f}`")
                st.text(s.chunk.text[:600])
    if result.steps:
        with st.expander("Under the hood"):
            for step in result.steps:
                if step.name == "pipeline":
                    continue
                st.markdown(f"**{step.name}**")
                if isinstance(step.detail, str):
                    st.code(step.detail, language="text")
                else:
                    st.json(step.detail)


def pick(label: str, options: tuple, default: str) -> str:
    return st.radio(
        label,
        options,
        index=options.index(default),
        format_func=lambda o: o,
        captions=[HELP[o] for o in options],
    )


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Modular RAG")
    st.caption("Pick the modules; Haystack assembles the pipeline.")
    d = rag.default_modules()
    modules = Modules(
        retriever=pick("Retriever", RETRIEVERS, d.retriever),
        ranker=pick("Ranker", RANKERS, d.ranker),
        generator=pick("Generator", GENERATORS, d.generator),
    )
    st.markdown(
        f"**Candidates → kept:** {settings.candidate_k} → {settings.top_k} · "
        f"**BM25 corpus:** {len(rag.corpus)} chunks"
    )
    st.markdown(
        f"**Embeddings:** `{rag.embedder.name}` · **Vector store:** `{settings.vector_store}`"
    )
    st.markdown(f"**Indexed chunks:** {rag.store.count()} · **DATA_DIR:** `{settings.data_dir}`")

    if st.button("Index DATA_DIR", type="primary"):
        try:
            with st.spinner("Indexing…"):
                show_report(ingest(settings, rag.embedder, rag.store, reset=True))
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
            show_report(ingest(settings, rag.embedder, rag.store, reset=False, data_dir=target))
            rag.refresh_corpus()
        except ConfigError as exc:
            st.error(str(exc))

    if st.button("Clear chat"):
        st.session_state.messages = []

# ----------------------------------------------------------------------------- chat
st.title("🧩 Modular RAG")
with st.expander("Pipeline for the selected modules", expanded=True):
    try:
        built = rag.pipeline_for(modules)
        st.graphviz_chart(dot(built.edges, built.inputs))
    except ConfigError as exc:
        st.warning(str(exc))

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_result(msg["content"], msg.get("modules"))

if question := st.chat_input("Ask, then swap a module and ask again…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Running the pipeline…"):
                result = rag.answer(question, modules)
        except ConfigError as exc:
            st.error(str(exc))
        else:
            render_result(result, modules)
            st.session_state.messages.append(
                {"role": "assistant", "content": result, "modules": modules}
            )
