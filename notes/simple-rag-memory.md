# Simple RAG with Memory

> **In one line:** a RAG chatbot that remembers the conversation and turns follow-ups into full questions before searching.

## The idea

People don't ask questions one at a time in isolation. They chat:

```
You: Which firmware fixed incident INC-2041?
Bot: Firmware 3.4.2.
You: Who wrote its postmortem?        <- "its" = INC-2041, but search doesn't know that
```

If we search for "Who wrote its postmortem?", the word *its* matches nothing useful.
The fix has two parts:

1. **Remember** the last few turns of the conversation.
2. **Rewrite** the follow-up into a standalone question ("Who wrote the postmortem for INC-2041?"), then search with that.

## How it works

```
follow-up question
      │
      ▼
 chat history (last N turns) ──► is there history?
                                   │ no  -> use the question as is
                                   │ yes -> LLM rewrites it:
                                   │        "Who wrote the postmortem for INC-2041?"
                                   ▼
                         retrieve top-k with the standalone question
                                   │
                                   ▼
            LLM: passages + history + original question ──► answer [n]
                                   │
                                   ▼
                         save this turn to history
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| Natural back-and-forth chat | 2 LLM calls per follow-up (rewrite + answer) |
| Pronouns like "it", "they", "that one" | A bad rewrite sends the search in the wrong direction |
| Answers that read like a conversation | Long chats need trimming (we keep a window of turns) |

## Best for (real life)

- **Customer support chat:** "My order hasn't arrived." ... "Can I cancel it?" The bot knows *it* means that order.
- **Internal IT helpdesk:** an employee asks about VPN setup, then "what about on my phone?" without repeating the context.

## What we built (`simple-rag-memory/`)

- **Stack:** LangChain 1.x core pieces and Groq (Llama models).
- **Memory:** a per-session message list, trimmed to the last `MEMORY_TURNS` turns.
- **Rewrite:** a condense prompt turns the follow-up into a standalone question. It is skipped on the first turn, which has no history and so costs nothing. It can use a cheaper `REWRITE_MODEL`.
- **UI:** shows "🔎 Searched for: ..." so you can see the rewritten query, and a toggle that turns rewriting off for comparison.
- **Measured on our sample data:** rewriting raised the top hit's score, for example from 0.452 to 0.597 for the postmortem question.
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores.
