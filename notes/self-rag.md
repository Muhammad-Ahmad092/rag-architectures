# Self-RAG

> **In one line:** the system grades its own search results and its own answer, and loops back to fix them when they're not good enough.

## The idea

Basic RAG trusts everything blindly: whatever the search returns goes into the prompt, and whatever the LLM writes goes to the user.
Self-RAG adds a **critic** that asks three questions:

| Check | Question the critic asks | If the answer is "no" |
|---|---|---|
| **Relevant?** | Is this passage about the question? | Drop it. If none are left, rewrite the query and search again |
| **Supported?** | Is every claim in the answer backed by the passages? | Write the answer again, naming the unsupported claims |
| **Useful?** | Does the answer actually address the question? | Rewrite the query and try again |

Like a careful student: check the sources, write, re-read, fix.

## How it works

```
question ──► retrieve ──► grade each passage: relevant?
                 ▲                 │
                 │      none relevant & rewrites left
                 └──── rewrite query ◄──┤
                                   │ none relevant & no rewrites left ──► "I don't know"
                                   │ some relevant
                                   ▼
                    generate from relevant passages only
                         ▲         │
     unsupported &       │         ▼
     attempts left ──────┴── supported? ── no & no attempts ──► flag: unsupported
                                   │ yes
                                   ▼
                               useful? ── no ──► rewrite query (loop)
                                   │ yes
                                   ▼
                              ✅ verified answer
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| Fewer made-up claims (hallucinations) | Many LLM calls: one grade per passage plus checks |
| Says "I don't know" instead of guessing | Slower: loops add latency |
| Every answer carries a verdict you can trust | The critic itself can be wrong |

## Best for (real life)

- **Legal or compliance assistant:** a wrong claim is costly, so every statement must be backed by a document, or the bot says "I don't know".
- **Healthcare knowledge assistant:** dosage or guideline answers must be verified against the source before a clinician sees them.

## What we built (`self-rag/`)

- **Stack:** LangGraph (a state graph with loops) and Anthropic Claude through the official SDK. Default model `claude-opus-5-5`; the grader can be a cheaper model (`GRADER_MODEL`).
- **Graders:** use Claude's structured JSON output, so each grade is a clean yes/no plus a reason.
- **Loop limits:** `MAX_REWRITES` (2) query rewrites and `MAX_GENERATIONS` (2) answer attempts.
- **Verdict badge** on every answer: `verified`, `unsupported`, `not useful` or `no relevant documents`.
- **UI:** a loop summary ("2 retrievals · 1 rewrite · 1 answer attempt") and every grader decision under the hood.
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores.
