# HyDE RAG (Hypothetical Document Embeddings)

> **In one line:** let the LLM write a fake answer first, then search with *that*, because a fake answer looks more like the real documents than the question does.

## The idea

Questions and answers are worded very differently:

```
Question:  "Who should I talk to about the S2?"
Document:  "Mei Tanaka - S2 Product Lead, reports to ..."
```

They share almost no words, so their embeddings are not very close.
HyDE's trick: ask the LLM to **imagine** what the answer document might say:

```
Fake draft: "The S2 product line is led by the S2 product lead, who handles ..."
```

The facts in the draft may be **wrong**, and that's fine. It only has to *look like* the kind of document that holds the real answer. Searching with the draft's embedding lands much closer to the right document.

## How it works

```
question
   │
   ▼
LLM writes N hypothetical answer passages   (search-only, never shown to the user)
   │
   ▼
embed the drafts (+ the question) ──► average ──► one search vector
   │
   ▼
vector store: top-k REAL chunks closest to that vector
   │
   ▼
LLM answers from the REAL chunks only ──► answer [n]
```

**Important:** the final answer never sees the fake drafts, only the real retrieved chunks, so invented facts can't leak into the answer.

## Good at / weak at

| Good at | Weak at |
|---|---|
| Vague or conceptual questions | Extra LLM calls before searching (slower, costs more) |
| Questions worded differently from the docs | If the draft goes off-topic, retrieval follows it |
| "Who/what handles X?" questions | Not needed when the question already uses the docs' words |

## Best for (real life)

- **Medical or legal search by laypeople:** a patient types "my chest feels tight after running"; the papers say "exercise-induced bronchoconstriction".
- **Research paper discovery:** a student asks a broad conceptual question whose wording doesn't match any paper's title or abstract.

## What we built (`hyde-rag/`)

- **Stack:** LangChain 1.x and OpenAI (`gpt-4o-mini`).
- **HyDE step:** generates `HYDE_NUM_DOCS` drafts, embeds them, optionally adds the question (`HYDE_INCLUDE_QUERY`), then averages and normalises them into one search vector.
- **Baseline comparison:** a toggle also runs normal question-only retrieval and shows which chunks only HyDE found.
- **Safety:** a test checks that the drafts never reach the answer prompt.
- **Measured on our sample data:** for "Who should I talk to about the S2?", the org chart moved from 3rd place (0.516) to 1st (0.652).
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores.
