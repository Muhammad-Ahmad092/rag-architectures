# Advanced RAG

**Every standard fix for single-step RAG, in one pipeline.** It rewrites the query, retrieves with **hybrid** dense + keyword search, fuses the results with **reciprocal rank fusion**, **reranks** with Cohere, generates with citations, then **checks the answer** and revises it if a claim isn't supported. Built with **LlamaIndex** (retrievers, stores, rerankers) and **Anthropic Claude** (via the official SDK).

```mermaid
flowchart LR
    Q([question]) --> RW[rewrite:<br/>Claude → N search queries]
    RW --> V1[vector retriever] & B1[BM25 retriever]
    V1 & B1 --> F[RRF fuse<br/>Σ 1/(k+rank)]
    F --> RR[rerank<br/>Cohere rerank-v3.5<br/>or local cross-encoder]
    RR --> G[generate<br/>Claude, cites n]
    G --> C{check:<br/>claims supported?}
    C -->|yes| OK([✅ verified])
    C -->|no| RV[revise once] --> C2{re-check}
    C2 -->|yes| RE([✏️ revised])
    C2 -->|no| UN([⚠️ unsupported])
```

| Stage | Fixes | How |
|---|---|---|
| **Rewrite** | The user's wording ≠ the documents' wording | Claude (grader model, effort `low`) writes up to `NUM_QUERIES − 1` alternative queries; the original is always kept first |
| **Hybrid retrieve** | Vectors miss identifiers ("INC-2041"); BM25 misses paraphrases | For **every** query: LlamaIndex vector retriever on the configured store, plus `BM25Retriever` over the saved node corpus |
| **RRF fuse** | Scores from different retrievers aren't comparable | `score = Σ 1/(RRF_K + rank)` across all lists; passages that several queries or retrievers agree on rise |
| **Rerank** | Bi-encoder similarity is a rough relevance proxy | Cohere `rerank-v3.5` re-scores fused candidates against the **original** question and keeps `TOP_K`. Without `COHERE_API_KEY`, the local `ms-marco-MiniLM` cross-encoder |
| **Check** | Hallucinated or over-reaching claims | The grader lists unsupported claims. If there are any, one revision removes them and the answer is re-checked. Verdict: `verified` / `revised` / `unsupported` |

## On the sample data

With local `bge-small`, real BM25 and the local cross-encoder, top 3 per stage:

| Query | Vector only | BM25 only | RRF | Reranked |
|---|---|---|---|---|
| `INC-2041` | support_tickets · postmortem · pricing | **postmortem · release_notes** · faq | postmortem · support_tickets · release_notes | **postmortem · release_notes** · support_tickets |
| Which firmware fixed the navigation incident at the Rotterdam customer? | postmortem · support_tickets · release_notes | postmortem · release_notes · faq | postmortem · release_notes · support_tickets | postmortem · release_notes · support_tickets |

For the bare identifier, vector search alone drops `release_notes.txt`, the document that says it was *fixed in 3.4.2*, out of its top 3. BM25 finds it by exact match, and fusion and reranking keep it in the top two.

## Design notes

- **One search path for every store.** BM25 runs in-process over `INDEX_DIR/corpus.jsonl`, the same nodes ingest wrote to the vector store, so hybrid works on Chroma, FAISS, Qdrant, LanceDB, Pinecone and Weaviate alike. A missing corpus falls back to vector-only, with no crash.
- **RRF is a small pure function** with its own test (`rrf`); it's easier to read and trust than a framework fusion retriever that would also want to own query generation.
- **Claude through the official SDK**, sharing [self-rag](../self-rag/)'s tested `ClaudeLLM`: no `temperature` (current models reject sampling parameters), effort-based depth, JSON-schema structured output for the rewrite and the check, server-side refusal fallback on supported models.
- **Cost control:** `GRADER_MODEL=claude-haiku-5-5` moves the rewrite and the checks (2–3 calls per question) to the small model. `NUM_QUERIES=1` turns rewriting off, and `CHECK_ANSWER=false` skips the check.

## Stack

| Part | Choice |
|---|---|
| Framework | **LlamaIndex** 0.14: `VectorStoreIndex` retriever, `llama-index-retrievers-bm25`, `llama-index-postprocessor-cohere-rerank`, `SentenceTransformerRerank` |
| LLM | **Anthropic Claude** `claude-opus-5-5` (answers); grader same model at effort `low` (or `GRADER_MODEL`) |
| Reranker | **Cohere** `rerank-v3.5`; local fallback `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Embeddings | LlamaIndex integrations: `openai`, `gemini` (768-d), `cohere`, `voyage`, `huggingface` (default `BAAI/bge-small-en-v1.5`, local) |
| Vector store | LlamaIndex integrations: `chroma` (default), `faiss`, `qdrant-local`, `lancedb`, `pinecone`, `qdrant-cloud`, `weaviate-cloud` |
| UI | Streamlit: verdict badge, "searched" query list, per-stage outputs (each retriever list, fused, reranked, checks) under the hood |

## Run

```bash
uv sync                               # default stack: HF embeddings + Chroma
cp .env.example .env                  # set ANTHROPIC_API_KEY (+ COHERE_API_KEY for Cohere rerank)
uv run python -m advanced_rag.ingest  # index ../sample_data (also writes the BM25 corpus)
uv run streamlit run app.py
```

Optional extras: `uv sync --extra faiss`, `qdrant`, `lancedb`, `pinecone`, `weaviate`, `openai`, `gemini`, `cohere`, `voyage`, or `--extra all`. With pip: `pip install -r requirements.txt` and `pip install -e .`.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Required (LLM, grader, eval judge) |
| `LLM_MODEL` / `LLM_EFFORT` | `claude-opus-5-5` / `medium` | |
| `GRADER_MODEL` / `GRADER_EFFORT` | empty (= `LLM_MODEL`) / `low` | Rewriter + answer checker |
| `NUM_QUERIES` | `3` | Including the original; `1` = no rewriting |
| `CANDIDATE_K` | `8` | Per retriever per query, before fusion |
| `RRF_K` | `60` | Fusion constant |
| `RERANKER` | `cohere` | `cohere \| local \| none` (cohere without a key → local) |
| `RERANK_MODEL` / `LOCAL_RERANK_MODEL` | `rerank-v3.5` / `cross-encoder/ms-marco-MiniLM-L-6-v2` | |
| `CHECK_ANSWER` | `true` | |
| `COHERE_API_KEY` | – | Cohere reranker (and `cohere` embeddings) |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `huggingface` / provider default | `openai \| gemini \| cohere \| voyage \| huggingface` |
| `OPENAI_API_KEY` / `GEMINI_API_KEY` / `VOYAGE_API_KEY` | – | That embedding provider; OpenAI also for fair-mode eval |
| `VECTOR_STORE` | `chroma` | Local: `chroma \| faiss \| qdrant-local \| lancedb`; cloud: `pinecone \| qdrant-cloud \| weaviate-cloud` |
| `INDEX_DIR`, `DATA_DIR`, `TOP_K`, `CHUNK_SIZE`, `CHUNK_OVERLAP` | `.index`, `../sample_data`, `4`, `512`, `50` | Tokens (`SentenceSplitter`) |
| `EVAL_LLM`, `EVAL_JUDGE_MODEL` | `gpt-4o-mini`, `claude-sonnet-5-5` | Evaluation only |

## Tests

```bash
uv sync --extra all --extra dev
uv run pytest -q          # offline: bag-of-words embedding, scripted Claude fake, fake reranker
uv run ruff check . && uv run ruff format --check .
```

`tests/test_pipeline.py` covers:
- The RRF maths, including agreement across lists, with exact scores.
- The exact stage order, and rewriting that keeps the original question and caps the count.
- Vector **and** BM25 running for every query, and BM25 winning on exact keywords.
- The reranker seeing fused candidates and reranking against the original question.
- The `none` reranker, a revision when a claim is unsupported, the still-unsupported flag, and the check being off.
- Reranker selection (Cohere, local fallback, none), and the vector-only fallback without a corpus.

`tests/test_llm.py` pins the Claude request shape.

## Evaluation

```bash
uv sync --extra all --extra eval
uv run python eval/run_eval.py     # fair mode; needs OPENAI_API_KEY + ANTHROPIC_API_KEY (+ COHERE_API_KEY)
python ../scripts/compare.py       # refresh the root README table
```

Fair mode runs the rewrite, the answer and the check on the common `gpt-4o-mini`, with `bge-small` and `chroma`. Hybrid retrieval, RRF and the reranker are the technique and stay on: Cohere with a key, the local cross-encoder without one. `--as-shipped` uses Claude.

<!-- EVAL_RESULTS -->
_Not run yet. Run `eval/run_eval.py` with your keys._
<!-- /EVAL_RESULTS -->

## Limitations

- **Latency and cost add up.** A rewrite call, 2 × `NUM_QUERIES` retrievals, a rerank, the answer, then 1–3 check and revise calls. Use a cheap grader and turn stages off where they don't pay.
- **Still single-pass retrieval.** A question whose second hop depends on the first answer needs [adaptive-rag](../adaptive-rag/)'s multi-step or [agentic-rag](../agentic-rag/).
- **The check can't fix missing evidence.** It removes unsupported claims, but if the right passage was never retrieved, the honest result is a shorter answer or "I don't know" (see [corrective-rag](../corrective-rag/) for a web fallback).
- **The BM25 corpus is in memory**, which is fine for this scale but needs a search engine for large corpora.
