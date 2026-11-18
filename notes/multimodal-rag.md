# Multimodal RAG

> **In one line:** search text, tables **and images**, then let a vision model *look* at the pictures while answering.

## The idea

Not all knowledge is text. An org chart, a labelled product diagram or a chart in a report holds answers that plain-text RAG can't see.

Multimodal RAG keeps **two indexes**:

- a **text index** for documents and table rows;
- an **image index**, where a special embedder (like CLIP) puts pictures **and** questions into the same vector space, so the question "who reports to the CTO?" lands near the org chart image.

At answer time, a **vision LLM** receives the text passages *and the actual image files*, so it reads the chart itself instead of trusting a caption.

## How it works

```
INGEST
  text, pdf, docx, html, csv/xlsx rows ──► text embedder ──► TEXT INDEX
  png / jpg images ─────────────────────► image embedder (CLIP) ──► IMAGE INDEX
  (image captions, if present, also go to the text index)

ASK
  question ──► text index  ──► top-k passages   [1] [2] [3]
           └─► image index ──► best image(s)    [I1]
                          │
                          ▼
     vision LLM sees: passages + the real image files
                          │
                          ▼
             answer citing [n] and [In]
```

## Good at / weak at

| Good at | Weak at |
|---|---|
| Diagrams, org charts, labelled pictures | Image search scores cluster, so it's hard to filter |
| Tables (rows indexed as text) | Image tokens cost more |
| One answer from mixed evidence | No audio, and no OCR for scanned pages |

## Best for (real life)

- **Field technician assistant:** "Where is the fuse on this model?" is answered by reading the wiring diagram, not just the manual text.
- **Insurance claims:** an adjuster asks about a claim, and the system checks the damage photos alongside the written report.

## What we built (`multimodal-rag/`)

- **Stack:** LlamaIndex and Google Gemini (a vision model, `gemini-flash-latest`).
- **Image embedders:** local CLIP `clip-ViT-B-32` (default, free), Voyage `voyage-multimodal-3` or Cohere `embed-v4.0`.
- **Two separate stores:** the image index has its own name per embedder, so vector spaces never mix. Both are embedded before either is reset, so a failed run keeps the old indexes.
- **Real images sent:** the PNG bytes go to Gemini labelled `[I1]`, not just captions.
- **Settings:** `TOP_K_IMAGES` 1, `MIN_IMAGE_SCORE` 0.2, `USE_CAPTIONS` true.
- **Measured with CLIP:** the right image ranked first for every test question, but all scores sat between 0.45 and 0.51 (an unrelated question still scored 0.465). That's why we send 1 image and tell the model to ignore it if irrelevant.
- **UI:** shows the retrieved images inline next to the answer.
- **Eval:** the only folder that answers the image questions in the shared eval set.
- **Swappable through `.env`:** all 5 text-embedding providers and 7 vector stores.
