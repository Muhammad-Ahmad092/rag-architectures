# Naive RAG

> **In one line:** find the chunks that look most like the question, paste them into the prompt, let the LLM answer.

## The idea

An LLM only knows what it was trained on. It has never seen your company's documents.
RAG (Retrieval-Augmented Generation) fixes that by **looking things up first**: before the model answers, we search our documents and hand it the relevant pieces.

Naive RAG is the simplest possible version, a student taking an open-book exam:

1. Read the question.
2. Flip to the pages that *look* related.
3. Write the answer from those pages.

No double-checking, no second search, no judging whether the pages were actually useful.

## How it works

```
INGEST (once)
  documents ──► split into chunks ──► embed each chunk ──► vector store
                 (fixed size)          (text → numbers)

ASK (every question)
  question ──► embed ──► vector store: top-k most similar chunks
                                   │
                                   ▼
               prompt = "Answer using ONLY these passages:
                         [1] chunk ...  [2] chunk ...  + question"
                                   │
                                   ▼
                                  LLM ──► answer with [1] [2] citations
```

- **Embedding**: turns text into a list of numbers so that similar meanings sit close together.
- **Top-k**: the k closest chunks (we use 4).

## Good at / weak at

| Good at | Weak at |
|---|---|
| Simple fact lookups ("What does the X1 cost?") | Questions whose facts sit in several documents |
| Cheap: 1 search + 1 LLM call | Vague questions that don't share words with the docs |
| Easy to understand and debug | Irrelevant chunks still go into the prompt |

Every other RAG type in this repo fixes one of these weaknesses. Naive RAG is the **baseline** they are compared against.

## Best for (real life)

- **Company FAQ bot:** employees ask "How many leave days do I get?" and the answer sits in one handbook paragraph.
- **Quick prototype:** a team wants to test, in one afternoon, whether their docs can answer customer questions before building anything bigger.

## What we built (`naive-rag/`)

- **Stack:** plain Python, no framework, with small hand-written interfaces (`Embedder`, `VectorStore`, `LLM`) and OpenAI `gpt-4o-mini` at temperature 0.
- **Ingest:** loads `.txt .md .pdf .docx .csv .xlsx .html` (plus image captions), cuts fixed-size chunks, embeds them and stores them.
- **Answer:** embeds the question, takes the top-k chunks, builds one numbered prompt and asks the LLM to cite `[n]`.
- **Swappable through `.env`:** 5 embedding providers (OpenAI, Gemini, Cohere, Voyage, local HuggingFace) and 7 vector stores (Chroma, FAISS, Qdrant local/cloud, LanceDB, Pinecone, Weaviate).
- **UI:** Streamlit chat with an **Under the hood** panel showing the scores, the full prompt and the model used.
- **Eval:** `eval/run_eval.py` scores it with RAGAS, so every other folder can be compared to it.
