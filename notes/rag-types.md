# 14 Types of RAG (Retrieval-Augmented Generation)

---

## Overview

RAG lets an AI system **retrieve relevant information from external knowledge sources** (documents, databases) and use it to generate more accurate answers, instead of relying only on its training data.

RAG architectures range from simple (**retrieve once, then generate**) to multi-step (**retrieve, refine, regenerate**) for higher quality.

## Quick Comparison

| #  | Type                | Core idea                                         | Best for                        | Main trade-off                       |
| -- | ------------------- | ------------------------------------------------- | ------------------------------- | ------------------------------------ |
| 1  | Simple RAG          | One retrieval step, then generate                 | FAQs, basic Q&A, chatbots       | Weak on multi-source questions       |
| 2  | Simple RAG + Memory | Uses conversation history as context              | Assistants, support, tutoring   | Cost, stale context, privacy         |
| 3  | Agentic RAG         | Plans, uses tools, iterates                       | Legal/financial research        | Expensive, slow, complex             |
| 4  | Graph RAG           | Knowledge graph of entity relationships           | Investigations, business intel  | Graph building effort                |
| 5  | Self-RAG            | Rewrites query, self-checks answer                | Vague or incomplete queries     | Slower, can be over-cautious         |
| 6  | Branched RAG        | Explores several interpretations in parallel      | Open-ended research             | Hard to merge, info overload         |
| 7  | Multimodal RAG      | Text + images + audio + video                     | Mixed-media files               | Storage and compute heavy            |
| 8  | Adaptive RAG        | Classifies query, adapts the strategy             | Mixed query workloads           | Learning curve, can learn bad habits |
| 9  | Speculative RAG     | Pre-fetches for predicted follow-ups              | Real-time chat, autocomplete    | Wasted compute on wrong guesses      |
| 10 | Corrective RAG      | Evaluates the answer, re-searches if weak         | Legal, academic, policy         | Latency, possible loops              |
| 11 | Modular RAG         | Swappable retriever/reranker/generator            | Production systems              | Coordination complexity              |
| 12 | Naive RAG           | Top-k match passed straight to the LLM            | Narrow chatbots, FAQs           | No filtering or verification         |
| 13 | Advanced RAG        | Rewriting + reranking + feedback loops            | Enterprise, high-stakes         | Expertise, tuning, cost              |
| 14 | HyDE                | Generates a hypothetical answer, searches with it | Academic, legal, medical search | Hard to explain, can mislead         |

---

## 1. Simple RAG (Original)

The most basic implementation: retrieve relevant documents in **a single step**, then generate.

**How it works:** Query → embeddings → vector DB similarity search → top matches → LLM (question + retrieved text) → answer. Retrieval is predictable.

**Use cases:** Basic Q&A, chatbots, FAQ automation.

| Pros             | Cons                                  |
| ---------------- | ------------------------------------- |
| Fast responses   | Struggles with multi-source questions |
| Easy to set up   | No feedback after generation          |
| Low compute cost | Doesn't recover from poor retrieval   |

## 2. Simple RAG with Memory

Simple RAG that **remembers previous turns**: past questions, answers, and retrieved documents shape new searches.

**How it works:** Stores key parts of past conversation and combines them with new queries.
*Example:* "What's the capital of France?" → "What's **its** population?" The system infers Paris.

**Use cases:** Personal AI agents, conversational chatbots, customer support, tutoring.

| Pros                        | Cons                               |
| --------------------------- | ---------------------------------- |
| Less repetition for users   | Higher processing cost             |
| More human-like interaction | Risk of outdated/incorrect context |
| Personalized responses      | Data-privacy concerns              |

## 3. Agentic RAG

Acts like an **experienced researcher**: plans an approach, decides what to investigate, and takes actions with tools.

**How it works:** Breaks the task into steps, searches multiple sources, checks whether the findings answer the question, and **keeps searching if not**.

**Use cases:** Legal case analysis, financial analysis that combines market and regulatory data, any task needing methodical planning.

| Pros                                 | Cons                     |
| ------------------------------------ | ------------------------ |
| Strong multi-step reasoning          | Costly (many searches)   |
| Smart decisions about what to gather | Hard to build and manage |
| Better on complex queries            | Slower responses         |

## 4. Graph RAG

Uses a **knowledge graph** to model how information is connected, finding relationships rather than just matching words.

**How it works:** Maps interconnections between entities. A document without the exact search terms can still be retrieved if it is conceptually related.

**Use cases:** Investigative journalism (hidden connections), business intelligence (market relationships).

| Pros                              | Cons                                    |
| --------------------------------- | --------------------------------------- |
| Handles multi-concept questions   | Significant effort to build the graph   |
| Avoids scattered answers          | Slower than basic RAG                   |
| Surfaces unexpected relevant info | Only as good as its modeled connections |

## 5. Self-RAG

A researcher who **constantly checks their own work**.

**How it works:** An LLM **rewrites the user's query** (adding missing context and inferred intent), generates an answer, then uses evaluation modules to check that the answer is accurate and supported by the sources.

**Use cases:** Incomplete or under-specified questions.

| Pros                                | Cons                              |
| ----------------------------------- | --------------------------------- |
| Catches its own mistakes            | Higher cost from extra checks     |
| Better results on vague questions   | Slower (work done twice)          |
| More reliable when accuracy matters | Can be overly cautious and refuse |

## 6. Branched RAG

Explores **multiple lines of thought at once** before picking the best answer.

**How it works:** Generates different interpretations of the question, retrieves for each branch, compares the results, and selects the most relevant.

**Use cases:** Market research that needs technical specs, competitor insights, and customer feedback together.

| Pros                          | Cons                                  |
| ----------------------------- | ------------------------------------- |
| Good for open-ended questions | Hard to coordinate branch findings    |
| Less likely to miss aspects   | Can overwhelm users without filtering |
| More thoughtful final answer  |                                       |

## 7. Multimodal RAG

Works across **text, images, video, audio, charts, and documents**.

**How it works:** Converts each content type into a searchable representation, searches across all media, and merges the results into one answer.

**Use cases:** Analyzing files that mix text with other media.

| Pros                        | Cons                                              |
| --------------------------- | ------------------------------------------------- |
| Works with any content type | Complex to build and train                        |
| More complete answers       | More storage and processing                       |
| Great for visual topics     | Quality depends on each modality's interpretation |

## 8. Adaptive RAG

**Learns from experience** and improves on different kinds of questions over time.

**How it works:** Classifies the incoming query (simple/complex, broad/narrow) and adjusts the retrieval process and generation style to match.

**Use cases:** Support bots, research tools, digital assistants that handle varied queries.

| Pros                           | Cons                                    |
| ------------------------------ | --------------------------------------- |
| Improves with user preferences | Inconsistent early results              |
| Relevance tuned per query type | More complex than static pipelines      |
| Balances speed vs. depth       | Can learn bad habits from poor examples |

## 9. Speculative RAG

**Predicts the user's next question** and pre-fetches information in the background.

**How it works:** Analyzes the current question and history, predicts likely follow-ups, and retrieves documents for them while answering the current question.

**Use cases:** Real-time chatbots, autocomplete, customer service, anywhere speed matters.

| Pros                           | Cons                                    |
| ------------------------------ | --------------------------------------- |
| Faster responses               | Wrong predictions mean wrong retrievals |
| More natural conversation flow | Wastes compute on unused predictions    |
| Less waiting on complex topics |                                         |

## 10. Corrective RAG (CRAG)

**Double-checks answers and corrects them** when something is off.

**How it works:** Normal search → generate → evaluate "does this fully answer the question?" If not, it drops weak sources, re-searches, and updates the response.

**Use cases:** Legal research, academic writing, policy analysis.

| Pros                                         | Cons                              |
| -------------------------------------------- | --------------------------------- |
| Fixes poor retrieval before the user sees it | Slower (multiple search attempts) |
| More reliable and accurate                   | Can loop if never satisfied       |
| Extra quality-control layer                  | More compute                      |

## 11. Modular RAG

A **toolkit** of independent components (retriever, reranker, generator, and so on) that can be swapped or combined.

**How it works:** Splits the pipeline into separate modules, each customizable without rebuilding the whole system.

**Use cases:** Domain-specific research environments, production AI applications.

| Pros                                  | Cons                                  |
| ------------------------------------- | ------------------------------------- |
| Optimize each component independently | Harder to coordinate components       |
| Upgrade or replace parts easily       | Needs up-front architectural planning |
| Flexible custom workflows             |                                       |

## 12. Naive RAG

The simplest form: retrieve and pass the documents straight to the model with **no adjustments**.

**How it works:** Question → basic search terms → top matching documents → LLM. **No filtering or reranking**; just simple matching.

**Use cases:** Narrow-scope chatbots, FAQ systems with predictable questions.

| Pros                 | Cons                                 |
| -------------------- | ------------------------------------ |
| Very simple to build | Struggles with complex questions     |
| Fast                 | No verification of results           |
| Low compute cost     | Often retrieves irrelevant documents |

## 13. Advanced RAG

A refined pipeline that **layers multiple techniques**: query rewriting, reranking, memory, feedback loops, branching, and improved retrieval.

**How it works:** Can rewrite the query for clarity, rank results, check whether the answer makes sense, and revise if needed.

**Use cases:** Research tools and enterprise apps where mistakes are unacceptable.

| Pros                                   | Cons                  |
| -------------------------------------- | --------------------- |
| Better on complex questions            | Requires expertise    |
| Picks the right approach per situation | Careful tuning needed |
| Fine control over results              | Expensive to run      |

## 14. HyDE (Hypothetical Document Embeddings)

Works **in reverse**: first imagine the answer, then find real documents like it.

**How it works:** The LLM generates a hypothetical answer → that answer is embedded and used as the query → real documents similar to it are retrieved.

**Use cases:** Domains where keyword search struggles: academic research, legal databases, medical information.

| Pros                                      | Cons                                    |
| ----------------------------------------- | --------------------------------------- |
| Semantic matching, not term matching      | Hard to explain its results             |
| Good for hard-to-search technical domains | Slower (extra generation step)          |
| Helps when the query lacks key details    | A bad hypothesis can mislead the search |

---

## Why Are There Different RAG Architectures?

No single setup fits every situation. Some tasks need **speed and simplicity**; others need **deep analysis, multiple sources, or non-text inputs**.

- Quick, straightforward queries → **Simple / Naive RAG**
- Complex research or messy data → **Agentic RAG** or **Self-RAG**

## Why RAG Matters

RAG keeps LLMs **grounded in real information** and reduces hallucinations. Examples:

- **Tech support:** agents find answers to technical issues without digging through docs.
- **Law firms:** scan thousands of legal documents in seconds.
- **Hospitals:** match patient symptoms against medical literature to support diagnosis.

## Common RAG Tools

| Tool                  | Role                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| **LangChain**   | Open-source orchestration framework connecting retrievers, embedders, and generators            |
| **Weaviate**    | Production vector database with hybrid search, filtering, and scalability                       |
| **Faiss**       | Meta's library for efficient embedding indexing and similarity search                           |
| **Haystack**    | End-to-end RAG framework covering retrieval, QA, and generation                                 |

Tools group by the pipeline stage they serve: **retrieval, generation, orchestration, or storage**. Pick based on your stage and goal.

## Conclusion

Choosing a RAG approach means **matching the architecture to the problem**. Strong retrieval combined with generative models reduces errors and improves reliability, whether for quick lookups, complex research, or anything in between.
