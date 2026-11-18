# Branched RAG

> **In one line:** when a question can mean several things, explore each meaning in parallel, then keep the best answer or merge the good ones.

## The idea

Some questions are broad or ambiguous:

```
"Tell me about the X1"
   ├── its price?
   ├── its technical specs?
   └── its known incidents?
```

A single search picks just one of those meanings, whichever embeds closest.
Branched RAG makes the meanings **explicit**: the LLM lists the interpretations, and each one gets its **own branch** (its own search, its own draft answer and a quality score). All branches run **at the same time**.

At the end:
- one branch is clearly best → **select** it;
- several are close → **merge** them into one answer;
- all are weak → say **"I don't know"**.

## How it works

```
                      question
                         │
                         ▼
          interpret: up to 3 readings + search queries
             ┌───────────┼───────────┐        (run in parallel)
             ▼           ▼           ▼
         branch A    branch B    branch C
         retrieve    retrieve    retrieve
         draft       draft       draft
         score 0.8   score 0.5   score 0.2
             └───────────┼───────────┘
                         ▼
                      decide
       best leads by ≥ 0.2 ──► SELECT the winner's draft
       scores are close    ──► MERGE the good branches
       all below 0.3       ──► "I don't know"
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| Broad and ambiguous questions | More LLM calls (one draft per branch) |
| Covers several angles in one answer | Overkill for simple fact lookups |
| Parallel, so not much slower | Merged answers can get long |

## Best for (real life)

- **E-commerce search:** "Apple" could mean the phone, the laptop or the watch; each reading is explored and the best answer shown.
- **Executive briefing:** "Tell me about Project Atlas" covers budget, timeline and risks, with each angle searched and merged into one summary.

## What we built (`branched-rag/`)

- **Stack:** LangGraph (`Send` fans out the branches in parallel) and Google Gemini at temperature 0, so the same branch wins every run.
- **Settings:** `BRANCHES` 3 interpretations, `SELECT_MARGIN` 0.2, `MIN_BRANCH_SCORE` 0.3.
- **Deciding:** select or merge is a small, tested Python function. The LLM gives the scores, and plain code makes the call.
- **Merging:** combines the chosen drafts and cites the deduplicated union of their passages.
- **Parallelism:** a test proves the branches really run at the same time.
- **UI:** one card per branch (its reading, query and score, ✅ if used), with each draft and the grader's reason.
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores.
