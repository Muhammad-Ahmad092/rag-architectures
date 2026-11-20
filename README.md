<div align="center">

# RAG Types

### 14 Retrieval-Augmented Generation architectures, each one runnable, tested and evaluated

[![CI](https://github.com/Muhammad-Ahmad092/rag-architectures/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Ahmad092/rag-architectures/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/packaging-uv-purple.svg)](https://docs.astral.sh/uv/)
[![RAG types](https://img.shields.io/badge/RAG%20types-14-brightgreen.svg)](#the-14-architectures)

[Architectures](#the-14-architectures) • [Which one?](#which-one-should-i-use) • [Quick start](#quick-start) • [Evaluation](#evaluation) • [Notes](notes/) • [Branches](#branches)

</div>

---

**RAG** (Retrieval-Augmented Generation) lets an LLM answer from *your* data: it retrieves relevant passages first, then generates an answer grounded in them. The core idea is simple. The many variations are in **what happens around retrieval**: rewriting the query, grading results, routing, branching, building graphs, or handing control to an agent.

This repository implements **14 of those architectures side by side**. Each one is a complete, standalone project, from a minimal baseline to a tool-using agent, built on the same data and scored on the same questions, so you can compare them on equal terms.

## Highlights

- **14 standalone projects.** Every folder has its own `pyproject.toml`, lockfile, README, Streamlit UI, tests and eval script.
- **Swap providers without touching code.** 5 embedding providers × 7 vector stores, selected in `.env`.
- **Different stack per architecture:** plain Python, LangChain, LangGraph, LlamaIndex, Haystack and asyncio.
- **"Under the hood" panel** in every UI to inspect scores, rewritten queries, grader decisions, graph paths or agent tool calls.
- **Offline test suites.** Tests run without API keys or network. CI runs them on every push.
- **Fair-mode RAGAS evaluation.** Same generator, embeddings and store for every type.
- **Plain-language notes** for every architecture.

---

## The 14 architectures

### Basic: single-step retrieval

| Type | How it works | Stack | Code | Notes |
|------|--------------|-------|------|-------|
| **Naive RAG** | Embed question → top-k chunks → stuff prompt → answer | Plain Python · OpenAI | [`naive-rag`](naive-rag/) | [notes](notes/naive-rag.md) |
| **Simple RAG** | Sentence-aware chunks + fine-grained citations | LlamaIndex · Gemini | [`simple-rag`](simple-rag/) | [notes](notes/simple-rag.md) |
| **Simple RAG + Memory** | Chat history + query rewriting for follow-ups | LangChain · Groq | [`simple-rag-memory`](simple-rag-memory/) | [notes](notes/simple-rag-memory.md) |

### Advanced: extra steps for quality

| Type | How it works | Stack | Code | Notes |
|------|--------------|-------|------|-------|
| **Advanced RAG** | Query rewrite → hybrid search → RRF → rerank → answer check | LlamaIndex · Claude | [`advanced-rag`](advanced-rag/) | [notes](notes/advanced-rag.md) |
| **Modular RAG** | Swappable retriever / ranker / generator modules | Haystack · Mistral | [`modular-rag`](modular-rag/) | [notes](notes/modular-rag.md) |
| **Corrective RAG** | Grades passages and falls back to web search if needed | LangGraph · OpenAI | [`corrective-rag`](corrective-rag/) | [notes](notes/corrective-rag.md) |
| **Self-RAG** | Grades relevance, support and usefulness with loops | LangGraph · Claude | [`self-rag`](self-rag/) | [notes](notes/self-rag.md) |

### Structured: beyond plain similarity

| Type | How it works | Stack | Code | Notes |
|------|--------------|-------|------|-------|
| **Graph RAG** | LLM-extracted knowledge graph + multi-hop retrieval | LangChain · Neo4j | [`graph-rag`](graph-rag/) | [notes](notes/graph-rag.md) |
| **HyDE** | Searches using the embedding of a hypothetical answer | LangChain · OpenAI | [`hyde-rag`](hyde-rag/) | [notes](notes/hyde-rag.md) |

### Dynamic: adapts at runtime

| Type | How it works | Stack | Code | Notes |
|------|--------------|-------|------|-------|
| **Adaptive RAG** | Routes questions by complexity (no search / single / multi-step) | LangGraph · OpenAI | [`adaptive-rag`](adaptive-rag/) | [notes](notes/adaptive-rag.md) |
| **Speculative RAG** | Prefetches evidence for predicted follow-up questions | asyncio · Groq | [`speculative-rag`](speculative-rag/) | [notes](notes/speculative-rag.md) |
| **Branched RAG** | Explores multiple interpretations in parallel | LangGraph · Gemini | [`branched-rag`](branched-rag/) | [notes](notes/branched-rag.md) |

### Agentic & Multimodal

| Type | How it works | Stack | Code | Notes |
|------|--------------|-------|------|-------|
| **Agentic RAG** | Tool-using agent that decides when it has enough information | LangGraph · Claude | [`agentic-rag`](agentic-rag/) | [notes](notes/agentic-rag.md) |
| **Multimodal RAG** | Text + table + image indexes with vision model | LlamaIndex · Gemini | [`multimodal-rag`](multimodal-rag/) | [notes](notes/multimodal-rag.md) |

---

## Quick start

```bash
git clone https://github.com/Muhammad-Ahmad092/rag-architectures.git
cd rag-architectures/naive-rag

uv sync
cp .env.example .env          # add your API key
uv run python -m naive_rag.ingest
uv run streamlit run app.py
