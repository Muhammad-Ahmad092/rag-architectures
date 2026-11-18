# Adaptive RAG

> **In one line:** first decide how hard the question is, then use just enough effort: no search, one search, or a step-by-step plan.

## The idea

Using one pipeline for every question is wasteful or weak:

```
"thanks!"                                        -> needs NO search
"What does the X1 cost?"                         -> needs ONE search
"Who wrote the postmortem for the incident
 fixed in firmware 3.4.2?"                       -> needs SEVERAL searches, in order
```

Adaptive RAG puts a **router** in front. It classifies each question and sends it down the right path. Our version also **learns**: when a user marks a route as wrong, similar questions get routed better next time, with no retraining.

## How it works

```
question ──► router (LLM + similar past examples from user feedback)
                │
   ┌────────────┼──────────────────────┐
   ▼            ▼                      ▼
no_retrieval  single_step            multi_step
answer        retrieve once          plan sub-questions
directly      answer                 for each step:
                                       fill in what's known so far
                                       retrieve, answer the step
                                     synthesize the final answer
   └────────────┴──────────┬───────────┘
                           ▼
                         answer ──► 👍 / 👎 / "wrong route"
                                          │
                       stored with the question's embedding
                       -> steers future similar questions
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| Saves cost on easy questions | A wrong route gives a weak answer |
| Handles multi-hop questions step by step | The router adds one LLM call |
| Improves from user feedback over time | Feedback needs real users to accumulate |

## Best for (real life)

- **High-traffic support bot:** most messages are "hi" or simple lookups, so it saves cost there and spends effort only on the hard multi-step cases.
- **Enterprise knowledge assistant:** a mix of quick policy lookups and "compare Q2 vs Q3 results for region X" questions in one tool.

## What we built (`adaptive-rag/`)

- **Stack:** LangGraph and OpenAI `gpt-4o-mini` (router, planner and answers).
- **Routes:** `no_retrieval`, `single_step`, `multi_step`. The multi-step path is capped at `MAX_STEPS` (3).
- **Learning router:**
  - Feedback is saved to a JSON file with the question's embedding.
  - The `ROUTER_EXAMPLES` (4) most similar past questions are added to the router prompt as examples, if their similarity is at least `EXAMPLE_MIN_SIMILARITY` (0.7).
  - **👍** labels the route taken as right; **wrong route** labels the correct one; **👎** is stored for review only.
- **Threshold measured:** similar questions scored about 0.79–0.88, unrelated ones about 0.45–0.49, hence the 0.7 cut-off.
- **UI:** a route badge, the past examples that steered the decision, each step's findings, and the feedback buttons.
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores.
