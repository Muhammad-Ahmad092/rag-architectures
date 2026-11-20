<div align="center">

# RAG Types

### 14 Retrieval-Augmented Generation architectures, each one runnable, tested and evaluated

[![CI](https://github.com/Muhammad-Ahmad092/rag-architectures/actions/workflows/ci.yml/badge.svg)](https://github.com/Muhammad-Ahmad092/rag-architectures/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11%E2%80%933.13-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/packaging-uv-purple.svg)](https://docs.astral.sh/uv/)
[![RAG types](https://img.shields.io/badge/RAG%20types-14-brightgreen.svg)](#the-14-architectures)

[Architectures](#the-14-architectures) •
[Which one?](#which-one-should-i-use) •
[Quick start](#quick-start) •
[Evaluation](#evaluation) •
[Notes](notes/) •
[Branches](#branches)

</div>

---

**RAG** (Retrieval-Augmented Generation) lets an LLM answer from *your* data: it retrieves relevant passages first, then generates an answer grounded in them. The core idea is simple. The many variations are in **what happens around retrieval**: rewriting the query, grading results, routing, branching, building graphs, or handing control to an agent.

This repository implements **14 of those architectures side by side**. Each one is a complete, standalone project, from a minimal baseline to a tool-using agent, built on the same data and scored on the same questions, so you can compare them on equal terms.

## Highlights

- 🧩 **14 standalone projects.** Every folder has its own `pyproject.toml`, lockfile, README, Streamlit UI, tests and eval script. Clone one folder and run it.
- 🔌 **Swap providers without touching code.** 5 embedding providers × 7 vector stores, selected in `.env`.
- 🏗️ **Different stack per architecture:** plain Python, LangChain, LangGraph, LlamaIndex, Haystack and asyncio, with OpenAI, Anthropic Claude, Google Gemini, Groq and Mistral.
- 🔍 **"Under the hood" panel in every UI**: see the scores, rewritten queries, grader decisions, graph paths or agent tool calls behind each answer.
- ✅ **Offline test suites.** Every folder's tests run without API keys or network (fakes for LLMs, embeddings and stores), and CI runs them on every push.
- 📊 **Fair-mode RAGAS evaluation.** The same generator, embeddings and store for every type, and a judge from a different provider, so scores reflect the technique, not the model.
- 📚 **Plain-language [notes](notes/)** for every type: the idea, an ASCII diagram, real-life use cases and what our implementation does.

---

## The 14 architectures

Grouped by what they add to the basic retrieve-then-generate loop.

### 🟢 Basic: single-step retrieval

| Type | How it works | Stack | Code | Notes |
|---|---|---|---|---|
| **Naive RAG** | Embed question → top-k chunks → stuff prompt → answer. The baseline. | Plain Python · OpenAI | [`naive-rag`](naive-rag/) | [📖](notes/naive-rag.md) |
| **Simple RAG** | Sentence-aware chunks, one retrieval, fine-grained `[n]` citations | LlamaIndex · Gemini | [`simple-rag`](simple-rag/) | [📖](notes/simple-rag.md) |
| **Simple RAG + Memory** | Chat history + rewrites follow-ups into standalone questions | LangChain · Groq | [`simple-rag-memory`](simple-rag-memory/) | [📖](notes/simple-rag-memory.md) |

### 🔵 Advanced: extra steps for quality

| Type | How it works | Stack | Code | Notes |
|---|---|---|---|---|
| **Advanced RAG** | Query rewrite → hybrid vector + BM25 → RRF fusion → rerank → answer check | LlamaIndex · Claude · Cohere | [`advanced-rag`](advanced-rag/) | [📖](notes/advanced-rag.md) |
| **Modular RAG** | Swappable retriever / ranker / generator modules assembled into a pipeline | Haystack · Mistral | [`modular-rag`](modular-rag/) | [📖](notes/modular-rag.md) |
| **Corrective RAG** | Scores each passage; keeps, refines, or falls back to web search | LangGraph · OpenAI · Tavily | [`corrective-rag`](corrective-rag/) | [📖](notes/corrective-rag.md) |
| **Self-RAG** | Grades relevance, support and usefulness; loops to rewrite or regenerate | LangGraph · Claude | [`self-rag`](self-rag/) | [📖](notes/self-rag.md) |

### 🟣 Structured: beyond plain similarity

| Type | How it works | Stack | Code | Notes |
|---|---|---|---|---|
| **Graph RAG** | LLM-extracted knowledge graph; multi-hop facts + vector search | LangChain · Neo4j · OpenAI | [`graph-rag`](graph-rag/) | [📖](notes/graph-rag.md) |
| **HyDE** | Searches with the embedding of a hypothetical answer | LangChain · OpenAI | [`hyde-rag`](hyde-rag/) | [📖](notes/hyde-rag.md) |

### 🟠 Dynamic: adapts at runtime

| Type | How it works | Stack | Code | Notes |
|---|---|---|---|---|
| **Adaptive RAG** | Routes each question: no search, single step or multi-step; learns from feedback | LangGraph · OpenAI | [`adaptive-rag`](adaptive-rag/) | [📖](notes/adaptive-rag.md) |
| **Speculative RAG** | Predicts follow-up questions and prefetches their evidence in the background | asyncio · Groq | [`speculative-rag`](speculative-rag/) | [📖](notes/speculative-rag.md) |
| **Branched RAG** | Explores several interpretations in parallel, then selects or merges | LangGraph · Gemini | [`branched-rag`](branched-rag/) | [📖](notes/branched-rag.md) |

### 🔴 Agentic & Multimodal: most capable

| Type | How it works | Stack | Code | Notes |
|---|---|---|---|---|
| **Agentic RAG** | Tool-using agent (document search, web, calculator) that decides when it's done | LangGraph · Claude | [`agentic-rag`](agentic-rag/) | [📖](notes/agentic-rag.md) |
| **Multimodal RAG** | Text + table + image indexes; a vision model reads the retrieved images | LlamaIndex · Gemini · CLIP | [`multimodal-rag`](multimodal-rag/) | [📖](notes/multimodal-rag.md) |

---

## Which one should I use?

```
Is the answer usually in ONE passage?
├── yes ─► Do users chat with follow-ups?
│          ├── yes ─► Simple RAG + Memory   (add Speculative RAG for instant follow-ups)
│          └── no  ─► Simple RAG            (Naive RAG as a baseline)
└── no ──► Does the answer connect facts across documents?
           ├── via named entities (people, systems, incidents) ─► Graph RAG
           ├── in a sequence of lookups or calculations       ─► Agentic RAG / Adaptive RAG
           └── no, but retrieval quality is the problem:
               ├── questions worded unlike the docs  ─► HyDE
               ├── IDs + prose, need high precision  ─► Advanced RAG / Modular RAG
               ├── docs may lack the answer          ─► Corrective RAG (web fallback)
               ├── wrong answers are costly          ─► Self-RAG (verifies itself)
               └── questions are broad or ambiguous  ─► Branched RAG
Answers live in images or diagrams? ─► Multimodal RAG
```

| If you care most about… | Start with |
|---|---|
| Lowest cost and latency | Naive RAG, Simple RAG |
| Trustworthy, verified answers | Self-RAG, Advanced RAG |
| Questions your documents can't answer | Corrective RAG |
| Mixed easy and hard traffic | Adaptive RAG |
| Relationships and multi-hop reasoning | Graph RAG, Agentic RAG |
| Experimenting with components | Modular RAG |

Each [note](notes/) ends with two real-world use cases for its type.

---

## Quick start

**Prerequisites:** Python 3.11–3.13, [uv](https://docs.astral.sh/uv/getting-started/installation/), and an API key for the folder's LLM (see the table below).

```bash
git clone https://github.com/Muhammad-Ahmad092/rag-architectures.git
cd rag-architectures/naive-rag

uv sync                              # installs the default stack (local embeddings + Chroma)
cp .env.example .env                 # then add your API key
uv run python -m naive_rag.ingest    # index ../sample_data
uv run streamlit run app.py          # open http://localhost:8501
```

The same four commands work in every folder; only the package name changes (`<folder>` with `_` instead of `-`, e.g. `graph_rag`).

<details>
<summary><b>Keys required per folder</b></summary>

| Folder | Required | Optional |
|---|---|---|
| `naive-rag`, `hyde-rag`, `adaptive-rag` | `OPENAI_API_KEY` | – |
| `simple-rag`, `branched-rag`, `multimodal-rag` | `GEMINI_API_KEY` | `VOYAGE_API_KEY` / `COHERE_API_KEY` (multimodal image embedders) |
| `simple-rag-memory`, `speculative-rag` | `GROQ_API_KEY` | – |
| `self-rag` | `ANTHROPIC_API_KEY` | – |
| `advanced-rag` | `ANTHROPIC_API_KEY` | `COHERE_API_KEY` (rerank; falls back to a local cross-encoder) |
| `agentic-rag` | `ANTHROPIC_API_KEY` | `TAVILY_API_KEY` (web search tool) |
| `corrective-rag` | `OPENAI_API_KEY` | `TAVILY_API_KEY` (web fallback) |
| `modular-rag` | `MISTRAL_API_KEY` | `OPENAI_API_KEY` (alternative generator) |
| `graph-rag` | `OPENAI_API_KEY` + `NEO4J_PASSWORD` | `GRAPH_STORE=memory` runs without Neo4j |

Evaluation additionally needs `OPENAI_API_KEY` (common generator) and `ANTHROPIC_API_KEY` (judge).
</details>

<details>
<summary><b>Using pip instead of uv</b></summary>

```bash
cd naive-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
</details>

---

## Configuration

Every folder is configured through its own `.env` (copy from `.env.example`). The defaults need **only the LLM key**: embeddings run locally (`BAAI/bge-small-en-v1.5`) and vectors go to a local Chroma store.

| Setting | Options |
|---|---|
| `EMBEDDING_PROVIDER` | `huggingface` (default, local) · `openai` · `gemini` · `cohere` · `voyage` |
| `VECTOR_STORE` | Local: `chroma` (default) · `faiss` · `qdrant-local` · `lancedb`. Cloud: `pinecone` · `qdrant-cloud` · `weaviate-cloud` |
| `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | Retrieval and chunking |
| `DATA_DIR` | Folder to index (default `../sample_data`) |

Non-default backends are optional extras: `uv sync --extra faiss`, `--extra pinecone`, `--extra cohere`, … or `--extra all`. Each folder's README documents its architecture-specific settings (thresholds, loop limits, routes, hops, tool budgets).

---

## Repository structure

```
.
├── README.md                  ← you are here
├── LICENSE
├── .github/workflows/ci.yml   ← lint + tests for every RAG folder on the branch
├── notes/                     ← plain-language explainer for each RAG type
├── sample_data/               ← shared corpus + eval questions (fictional company)
├── scripts/
│   ├── compare.py             ← builds the evaluation table below
│   └── make_sample_data.py    ← regenerates sample_data/
│
├── naive-rag/                 ┐
├── simple-rag/                │
├── …                          │  one standalone project per architecture
└── agentic-rag/               ┘
```

Every architecture folder shares the same layout:

```
<type>-rag/
├── README.md          architecture, design notes, measured results, configuration
├── app.py             Streamlit UI
├── .env.example       every setting, documented
├── pyproject.toml     dependencies + optional extras (uv)
├── uv.lock            pinned versions
├── requirements.txt   pip alternative
├── src/<type>_rag/    config, loaders, embeddings, vector stores, ingest, pipeline
├── tests/             offline pytest suite
└── eval/run_eval.py   RAGAS evaluation (fair mode / as-shipped)
```

---

## Sample data

[`sample_data/`](sample_data/) is a self-written corpus about **Nimbus Robotics**, a fictional company. It covers every supported format: Markdown, text, HTML, DOCX, PDF, XLSX, CSV and PNG with captions. Facts are deliberately spread across files, so multi-hop questions can only be answered by combining sources. For example, the incident fixed in one release is described in a postmortem, whose author appears on the org chart.

[`sample_data/eval/questions.jsonl`](sample_data/eval/questions.jsonl) holds **20 evaluation questions** with reference answers, tagged by modality (text, table, image) and hop count, including one unanswerable question.

Point `DATA_DIR` at your own folder to use your own documents.

---

## Evaluation

Every type is scored on the same 20 questions with [RAGAS](https://docs.ragas.io/) (faithfulness, answer relevancy, context precision, context recall).

- **Fair mode** (default): the same generator LLM (`gpt-4o-mini`), the same embeddings (`bge-small`) and the same store (Chroma) for every type, so differences come from the *architecture*.
- **Judge:** a model from a different provider (Claude), to avoid self-grading bias.
- **As-shipped mode** (`--as-shipped`): each folder's own stack, for real-world numbers.

```bash
cd <type>-rag
uv sync --extra all --extra eval
uv run python eval/run_eval.py     # writes eval/results.json
cd .. && python scripts/compare.py # refreshes the table below
```

<!-- EVAL_TABLE -->

_No results yet._

<!-- /EVAL_TABLE -->

---

## Testing

```bash
cd <type>-rag
uv sync --extra all --extra dev
uv run pytest -q                                  # offline: no API keys, no network
uv run ruff check . && uv run ruff format --check .
```

Tests use scripted fakes for every LLM, bag-of-words embeddings and real local stores. They cover each architecture's control flow (loops, routes, branches, budgets), failure paths, provider wiring and the Streamlit app. [CI](.github/workflows/ci.yml) discovers every RAG folder on the branch and runs lint and tests for each in parallel.

---

## Branches

| Branch | Contents |
|---|---|
| `main` | All 14 architectures side by side |
| `basic-rag` | Naive RAG, Simple RAG, Simple RAG + Memory |
| `advanced-rag` | Advanced RAG, Modular RAG, Corrective RAG, Self-RAG |
| `structured-rag` | Graph RAG, HyDE |
| `dynamic-rag` | Adaptive RAG, Speculative RAG, Branched RAG |
| `agentic-multimodal-rag` | Agentic RAG, Multimodal RAG |

Every branch includes `sample_data/`, `scripts/` and CI. Use `main` for the full comparison.

---

## References

The architectures follow these papers, adapted as described in each folder's README:

- **HyDE**: Gao et al., 2022, *Precise Zero-Shot Dense Retrieval without Relevance Labels* ([arXiv:2212.10496](https://arxiv.org/abs/2212.10496))
- **Self-RAG**: Asai et al., 2023, *Learning to Retrieve, Generate, and Critique through Self-Reflection* ([arXiv:2310.11511](https://arxiv.org/abs/2310.11511))
- **Corrective RAG**: Yan et al., 2024, *Corrective Retrieval Augmented Generation* ([arXiv:2401.15884](https://arxiv.org/abs/2401.15884))
- **Adaptive-RAG**: Jeong et al., 2024, *Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity* ([arXiv:2403.14403](https://arxiv.org/abs/2403.14403))

## Contributing

Issues and pull requests are welcome. For a change to a RAG folder, please keep it standalone, add or update its offline tests, and make sure `uv run pytest -q` and `uv run ruff check .` pass in that folder.

## License

Released under the [Apache License 2.0](LICENSE).
#   r a g - a r c h i t e c t u r e s 
 
 