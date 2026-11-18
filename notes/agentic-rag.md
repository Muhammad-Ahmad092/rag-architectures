# Agentic RAG

> **In one line:** give the LLM tools (search, web, calculator) and a goal, and let it decide what to look up and when it knows enough.

## The idea

Every other RAG type follows a **fixed recipe** written by the developer: search, then maybe check, then answer.
Agentic RAG hands control to the model. It works like a researcher:

```
"Who wrote the postmortem for the incident fixed in 3.4.2,
 and who is their manager?"

 think  -> I need the incident first.            search: "firmware 3.4.2 fix"
 read   -> it's INC-2041.
 think  -> now the postmortem author.            search: "INC-2041 postmortem"
 read   -> Priya Raman.
 think  -> now her manager.                      search: "Priya Raman reports to"
 read   -> Lukas Brandt.
 think  -> I have everything.                    -> final answer with citations
```

A greeting takes 0 searches, a simple lookup 1, a chain like this several. **The model decides.**

## How it works

```
             ┌─────────────────────────────────────────┐
             ▼                                         │
question ──► AGENT (Claude) ── wants tools? ── yes ──► TOOLS
                  │                                     ├─ search_documents (vector store)
                  │ no, it has enough                   ├─ web_search (Tavily, optional)
                  ▼                                     └─ calculator (safe math)
            final answer [n]                   results numbered [1] [2] ... go back

 Safety net: after MAX_TOOL_ROUNDS tool turns, the tools are switched off
 and the agent must answer from what it has found.
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| Multi-step research questions | Cost and speed vary a lot per question |
| Mixing tools: docs, web, math | Can stop too early or search too much |
| Recovers when a search misses | Harder to predict and test than a fixed pipeline |

## Best for (real life)

- **Financial analyst assistant:** "How much did revenue grow vs last year, in %?" means finding both figures, then calculating the change.
- **Due-diligence research:** it searches internal contracts, checks public news on the web, and keeps going until the question is fully answered.

## What we built (`agentic-rag/`)

- **Stack:** LangGraph (a two-node loop: `agent` ⇄ `tools`) and Anthropic Claude (`claude-opus-5-5`) through the official SDK with tool use.
- **Tools:**
  - `search_documents`: vector search, `TOP_K` passages per call.
  - `web_search`: Tavily. Only offered when `WEB_SEARCH=true` and `TAVILY_API_KEY` is set.
  - `calculator`: exact arithmetic. It parses expressions safely and never runs code.
- **Citations that work:** every passage any tool returns is numbered once, so `[3]` always points at a real source.
- **Errors don't crash it:** a failed tool (bad math, web quota) goes back to the agent as an error result, so it can try another way.
- **Budget:** `MAX_TOOL_ROUNDS` 6. After that, tools are turned off (`tool_choice: none`) and it must answer. Verdict: `answered`, `tool_budget` or `max_tokens`.
- **Current Claude API practice:**
  - No forced tool choice; `auto` plus the system prompt steer the agent.
  - Strict tool schemas.
  - Adaptive thinking with an effort level, and no temperature.
  - The conversation is append-only, so thinking blocks are passed back unchanged.
  - Server-side refusal fallback.
- **Fair-mode eval:** the same loop runs on OpenAI `gpt-4o-mini`, with web search off.
- **UI:** the **agent trajectory** (each round's tool, input and result, ❌ for errors) and the numbered evidence.
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores.
