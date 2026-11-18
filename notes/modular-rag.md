# Modular RAG

> **In one line:** build RAG from swappable parts (retriever, ranker, generator) so you can change one without touching the rest.

## The idea

Most RAG systems are one fixed chain. To try a different search method, you rewrite code.
Modular RAG treats each stage like a **LEGO brick** with a fixed shape:

```
[ Retriever ] ──documents──► [ Ranker ] ──top documents──► [ Generator ] ──► answer
  vector                       none                          Mistral
  bm25 (keywords)              cross-encoder                 OpenAI
  hybrid (both)                lost-in-the-middle
```

Pick one brick per slot and the pipeline is assembled for you. Every combination works because each brick takes and returns the same kind of data.

## How it works (the `hybrid` setup)

```
question ──► embed ──► vector search ──┐
    │                                  ├──► join (rank fusion) ──► ranker ──► prompt ──► LLM ──► answer
    └──────────────► BM25 keyword ─────┘
```

What the bricks do:

- **vector**: finds chunks with similar *meaning*.
- **bm25**: finds chunks with the same *words*, which is great for IDs like "INC-2041".
- **hybrid**: runs both and fuses the two lists, so you get the best of each.
- **cross-encoder ranker**: re-reads each (question, passage) pair and re-scores it more precisely.
- **lost-in-the-middle ranker**: puts the best passages at the start and end of the prompt, where LLMs pay most attention.

## Good at / weak at

| Good at | Weak at |
|---|---|
| Experimenting: swap a part, compare | More moving parts to understand |
| Fits different data (IDs, prose, both) | Every combination needs its own tuning |
| Clean, testable components | Overhead for a tiny project |

## Best for (real life)

- **Platform team serving many departments:** legal uses keyword search for clause numbers, marketing uses vector search, on one codebase.
- **Search quality experiments:** an ML team swaps the reranker or the LLM vendor and compares results without rewriting the pipeline.

## What we built (`modular-rag/`)

- **Stack:** Haystack 3 pipelines and Mistral (OpenAI available as a swap).
- **Modules:**
  - Retriever: `vector` / `bm25` / `hybrid`.
  - Ranker: `none` / `cross-encoder` / `lost-in-the-middle`.
  - Generator: `mistral` / `openai`.
- **Choose modules** in `.env` or per question in the UI sidebar. Each combination's pipeline is built once and cached.
- **Custom components:** our own embedders and stores are wrapped as Haystack components, so `vector` works with all 5 embedding providers and 7 vector stores.
- **BM25 corpus:** ingest saves the same chunks to `corpus.jsonl` for keyword search.
- **Measured on our sample data:** for "When is customer support available?", vector search alone ranks the support tickets first, while BM25, hybrid and cross-encoder all rank `faq.html` (the real hours) first.
- **UI:** draws the actual pipeline graph for the chosen modules and shows each stage's output.
