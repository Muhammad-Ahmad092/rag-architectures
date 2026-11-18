# Corrective RAG (CRAG)

> **In one line:** score how good the search results are; keep the good ones, clean up the so-so ones, and search the web when the documents don't have the answer.

## The idea

Sometimes your documents simply don't contain the answer. Basic RAG answers anyway, from bad passages.
Corrective RAG puts an **evaluator** between search and answer. It scores every retrieved passage from 0 to 1, then picks one of three actions:

```
          0.0 ─────────── 0.3 ─────────────── 0.7 ─────────── 1.0
              INCORRECT   │     AMBIGUOUS     │    CORRECT
          (go to the web) │ (docs + the web)  │ (trust the docs)
```

It also **refines** the passages it keeps, cutting them down to just the sentences that matter, so the LLM isn't distracted by noise.

## How it works

```
question ──► retrieve top-k ──► score each passage 0..1
                                      │
        ┌─────────────────────────────┼──────────────────────────────┐
        ▼                             ▼                              ▼
   CORRECT (any ≥ 0.7)          AMBIGUOUS                 INCORRECT (all < 0.3)
   refine: keep relevant        refine the docs            discard the docs
   sentences                    + web search               rewrite as web query
        │                             │                    web search (Tavily)
        └──────────────┬──────────────┴──────────────────────────────┘
                       ▼
            generate answer  ──► 📄 internal + 🌐 web citations
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| Questions your docs can't answer | Needs a web search API key |
| Cleaner context (refined sentences) | Web results may be less trustworthy than internal docs |
| Clear, explainable decision per question | The thresholds need tuning for your data |

## Best for (real life)

- **Sales assistant:** internal docs cover our products; for "how does a competitor price theirs?" it falls back to the web.
- **News or market research desk:** the internal archive is months old, so recent events get fetched from the web automatically.

## What we built (`corrective-rag/`)

- **Stack:** LangGraph, OpenAI `gpt-4o-mini` (evaluator, refiner, web-query rewriter and answer generator) and Tavily for web search.
- **Thresholds:** `UPPER_THRESHOLD` 0.7 and `LOWER_THRESHOLD` 0.3, set in `.env`.
- **Refinement:** keeps only the relevant sentences. If nothing survives, it goes to the web rather than answering from nothing.
- **Graceful fallback:** with no Tavily key, or if the web search fails, it answers from what it has and says so.
- **UI:** the action taken, each passage's score and reason, and 📄/🌐 tags on sources.
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores.
