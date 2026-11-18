# Simple RAG

> **In one line:** naive RAG done carefully, with clean chunks and precise citations, still in one step.

## The idea

Naive RAG cuts documents into fixed-size pieces, which often slices a sentence in half, and cites whole big chunks.
Simple RAG keeps the same "search once, then answer" shape but does each part **properly**:

- **Sentence-aware chunking**: chunks end at sentence boundaries, so no idea is cut in the middle.
- **Fine-grained citations**: retrieved chunks are split into small numbered passages, so `[3]` points at a couple of sentences, not a whole page.

Think of it as the difference between "see chapter 4" and "see page 87, paragraph 2".

## How it works

```
INGEST
  documents ──► SentenceSplitter (512 tokens, 50 overlap) ──► embed ──► vector store

ASK
  question ──► retrieve top-k chunks
                     │
                     ▼
        split each chunk into small passages
        Source 1 | Source 2 | Source 3 | ...
                     │
                     ▼
        LLM: "answer, cite Source n" ──► answer with [n] citations
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| Clean, readable citations | Still one search: multi-document questions suffer |
| No broken sentences in the context | No check whether retrieved text is relevant |
| Same cost as naive RAG | Follow-up questions ("and its price?") lose context |

## Best for (real life)

- **Policy lookup for HR or compliance:** answers must cite the exact clause ("see Leave Policy, section 3.2"), not a whole document.
- **Product documentation search:** support agents look up a setting and need the precise sentence to paste to the customer.

## What we built (`simple-rag/`)

- **Stack:** LlamaIndex (`SentenceSplitter`, `VectorStoreIndex`, `CitationQueryEngine`) and Google Gemini (`gemini-flash-latest`, temperature 0).
- **Ingest:** the same loaders as naive-rag, so both folders index identical text and the comparison is fair.
- **Answer:** retrieves top-k nodes, splits them into citable passages and has Gemini answer with `[n]` citations.
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores, through LlamaIndex integrations.
- **UI:** Streamlit, with the retrieval scores, passage count and exact context under the hood.
