# Graph RAG

> **In one line:** turn your documents into a map of *things* and *how they connect*, then answer by walking that map.

## The idea

Vector search finds text that *looks like* the question. It struggles when the answer is spread across documents and linked only through a shared name:

```
release_notes.txt :  "Firmware 3.4.2 fixes INC-2041."
postmortem.docx   :  "Priya Raman wrote the postmortem for INC-2041."
org_chart         :  "Priya Raman reports to Lukas Brandt."

Question: "Who wrote the postmortem for the incident fixed in 3.4.2?"
```

No single passage answers it. But as a **knowledge graph** it is a short walk:

```
 [Firmware 3.4.2] ──FIXES──► [INC-2041] ◄──WROTE_POSTMORTEM_FOR── [Priya Raman] ──REPORTS_TO──► [Lukas Brandt]
                    hop 1                         hop 2                              hop 3
```

Graph RAG builds that map once, with an LLM reading every passage, and at question time follows the links from the entities the question mentions.

## How it works

```
INGEST
  each chunk ──► embed ──────────────────────► vector store
            └──► LLM extracts:
                 entities  (Priya Raman: Person, INC-2041: Incident, ...)
                 relations (Priya Raman -WROTE_POSTMORTEM_FOR-> INC-2041)
                        ──► knowledge graph (Neo4j)

ASK
  question ──► LLM: which entities are mentioned? ──► match them in the graph
                    │
                    ├─► walk up to 2 hops ──► facts   [G1] [G2] ...
                    ├─► passages that mention those entities
  question ──► vector search ──────────────────────► passages [1] [2] ...
                    │
                    ▼
       LLM: facts + passages ──► answer citing [n] and [Gn]
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| "Connect the dots" multi-hop questions | Ingest costs one LLM call per chunk |
| Questions about people, systems, relationships | Graph quality depends on extraction quality |
| Explainable: you can see the path | Big "hub" entities can flood the facts |

## Best for (real life)

- **Fraud investigation:** "Which accounts share a phone number with this flagged customer?" means following links between people, accounts and devices.
- **IT incident analysis:** "Which services depend on the database that failed, and who owns them?" walks a dependency graph.

## What we built (`graph-rag/`)

- **Stack:** LangChain, Neo4j (graph database) and OpenAI `gpt-4o-mini` (extraction, question entities, answers).
- **Two graph backends:** Neo4j (default, `docker compose up -d` with the included `docker-compose.yml`), or an in-memory JSON graph (`GRAPH_STORE=memory`) that needs no Docker.
- **Extraction:** structured output, with relation names normalised to `UPPER_SNAKE_CASE`. Runs in parallel (`EXTRACT_WORKERS` 4), and all of it finishes before anything is overwritten, so a failure keeps the old indexes.
- **Settings:** `GRAPH_HOPS` 2 (1–3), `MAX_FACTS` 25, `GRAPH_CHUNKS` 2 linked passages.
- **Three modes to compare:** `hybrid` (graph + vector, default), `graph` only, `vector` only.
- **Safety:** all Cypher values are parameters, and the hop count is validated before use.
- **UI:** draws the subgraph behind each answer (matched entities highlighted), lists the numbered facts, and tags passages `vector` / `graph` / `vector+graph`.
- **Swappable through `.env`:** all 5 embedding providers and 7 vector stores.
