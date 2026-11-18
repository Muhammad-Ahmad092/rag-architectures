# Advanced RAG

> **In one line:** every standard upgrade in one pipeline: rewrite the query, search two ways, fuse, rerank, answer, then fact-check the answer.

## The idea

Each step of basic RAG has a known weakness, and each has a known fix:

| Step | Weakness | Fix |
|---|---|---|
| Query | The user's words ≠ the documents' words | **Rewrite** into several search queries |
| Search | Vectors miss IDs ("INC-2041"); keywords miss paraphrases | **Hybrid**: vector + BM25 keyword search |
| Combine | Scores from different searches don't compare | **RRF fusion**: rank-based merging |
| Order | Similarity scores are only a rough proxy for relevance | **Rerank** with a stronger model |
| Answer | The LLM may add unsupported claims | **Check** the answer, revise once if needed |

Advanced RAG applies **all** of these fixes, in order.

## How it works

```
question
   │
   ▼
rewrite ──► q1 (original), q2, q3
   │
   ▼  for EVERY query:
vector search   +   BM25 keyword search
   │
   ▼
RRF fuse:  score = Σ 1 / (60 + rank)      <- passages found by many lists rise
   │
   ▼
rerank against the ORIGINAL question (Cohere, or a local cross-encoder)
   │
   ▼
generate answer with [n] citations
   │
   ▼
check: is every claim supported?
   ├── yes ──────────────────────► ✅ verified
   └── no  ──► revise once ──► re-check ──► ✏️ revised / ⚠️ unsupported
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| High accuracy on single-step questions | Many calls: slower and more expensive |
| Finds exact IDs *and* paraphrases | Still one pass: can't chain facts across documents |
| Answers come with a verification verdict | The check can't fix evidence that was never retrieved |

## Best for (real life)

- **Technical support search:** queries mix error codes ("ERR-504") and plain language; hybrid search plus reranking finds both.
- **Enterprise document Q&A in production:** accuracy matters more than speed, and every answer should be fact-checked before display.

## What we built (`advanced-rag/`)

- **Stack:** LlamaIndex (retrievers, BM25, rerankers) and Anthropic Claude (`claude-opus-5-5`) for rewriting, answering and checking.
- **Settings:** `NUM_QUERIES` 3, `CANDIDATE_K` 8 per retriever, `RRF_K` 60.
- **Reranking:** Cohere `rerank-v3.5`. Without a Cohere key it falls back to the local cross-encoder `ms-marco-MiniLM`.
- **Answer check:** finds unsupported claims, revises once and gives a verdict: `verified`, `revised` or `unsupported`.
- **Cost control:** `GRADER_MODEL=claude-haiku-5-5` for the cheap steps. `NUM_QUERIES=1` or `CHECK_ANSWER=false` turns stages off.
- **Measured on our sample data:** for the bare ID "INC-2041", vector search alone missed `release_notes.txt`; BM25, fusion and rerank brought it into the top two.
- **UI:** each stage's output (each retriever's list, fused, reranked, checks) under the hood.
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores.
