# Speculative RAG

> **In one line:** while the user reads the answer, guess their next questions and fetch the evidence (or even the answers) in advance.

## The idea

Conversations are predictable. After "Which firmware fixed INC-2041?", people usually ask:

```
-> Who wrote the postmortem?
-> Which customer was affected?
-> When did the fix ship?
```

So why wait? Right after answering, the system **predicts likely follow-ups** and, **in the background**, searches for all of them at once. If the user then asks one of them, the work is already done: a **⚡ cache hit**.

Like a waiter who brings the dessert menu before you ask.

## How it works

```
USER                 MAIN PATH                       BACKGROUND
 │ question 1 ──►  cache? miss
 │                 retrieve + answer
 │ ◄── answer 1    (returned immediately) ──────► predict 3 follow-ups
 │                                                 search all 3 at once
 │  (user reads)                                   (optionally pre-answer)
 │                                                 store in cache
 │ question 2 ──►  cache? ⚡ HIT
 │                 (question 2 ≈ a predicted one)
 │ ◄── answer 2    no search needed
                   (and no LLM call in "answer" mode)
```

The cache matches **exactly**, or by **meaning** (cosine similarity ≥ 0.85), so a reworded question still hits.

## Good at / weak at

| Good at | Weak at |
|---|---|
| Fast follow-ups in a chat | Background work costs money even when the guess is wrong |
| Suggested next questions as buttons | Only helps conversations, not one-off questions |
| The user never waits for the speculation | Cached answers must be re-used carefully (strict threshold) |

> **Naming note:** a research paper with the same name (Wang et al., 2024) does something different: a small model drafts several answers and a big model picks one. This project uses the prefetching meaning.

## Best for (real life)

- **Banking or telecom chat:** after "What's my bill?", the next question is almost always "Why is it higher?", so it's ready instantly.
- **Product tour assistant:** while a visitor reads about pricing, the answers about plans and discounts are already prepared.

## What we built (`speculative-rag/`)

- **Stack:** plain Python + `asyncio` and Groq (Llama models, async client).
- **Background work:** runs on its own event loop thread, so the answer returns at once while speculation continues.
- **Settings:**
  - `NUM_FOLLOWUPS`: 3 predictions per answer.
  - `PREFETCH_MODE`: `retrieve` (prefetch passages) or `answer` (also pre-write the answers).
  - `CACHE_THRESHOLD` 0.85: how similar a new question must be to reuse prefetched passages.
  - `ANSWER_REUSE_THRESHOLD` 0.97: much stricter, for reusing a pre-written answer.
  - `CACHE_SIZE` 32: the cache keeps the 32 most recently used entries.
- **Thresholds measured** on real embeddings: rewordings scored about 0.94–0.98, unrelated questions about 0.36.
- **UI:** 🔮 predicted follow-ups as clickable buttons (guaranteed hits) and a ⚡ badge on cache hits.
- **Eval:** runs with speculation off, so its scores measure answer quality, not speed.
