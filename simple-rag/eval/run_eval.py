"""Evaluate Simple RAG on the shared question set with RAGAS.

Fair mode (default) forces the same generator, embeddings and store for every RAG type,
so scores reflect the technique rather than the model. The judge is a different provider
(Anthropic) than the generator (OpenAI) to avoid self-grading bias.

    uv run --extra eval python eval/run_eval.py [--as-shipped]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FOLDER / "src"))

from simple_rag.config import Settings, get_settings, require  # noqa: E402

SUPPORTS_IMAGES = False  # only multimodal-rag answers image questions
# RAGAS column name -> name used in results.json / README tables
METRIC_COLUMNS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "llm_context_precision_with_reference": "context_precision",
    "context_recall": "context_recall",
}


def build_eval_settings(base: Settings, as_shipped: bool) -> Settings:
    update = {"index_dir": Path(".index-eval")}
    if not as_shipped:
        update |= {
            "embedding_provider": "huggingface",
            "embedding_model": None,
            "vector_store": "chroma",
        }
    return base.model_copy(update=update)


def generator_for(base: Settings, as_shipped: bool) -> tuple[str, str]:
    """Fair mode swaps this folder's Gemini generator for the common OpenAI one."""
    return ("gemini", base.llm_model) if as_shipped else ("openai", base.eval_llm)


def load_questions(
    path: Path, folder_supports_images: bool = SUPPORTS_IMAGES
) -> tuple[list[dict], list[str]]:
    kept, skipped = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        if q.get("modality") == "image" and not folder_supports_images:
            skipped.append(q["id"])
        else:
            kept.append(q)
    return kept, skipped


def _ragas_components(settings: Settings, embedder):
    """Judge LLM + embeddings wrapped for RAGAS (imported lazily: needs --extra eval)."""
    from langchain_anthropic import ChatAnthropic
    from langchain_core.embeddings import Embeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    class _Emb(Embeddings):
        def embed_documents(self, texts):
            return embedder.get_text_embedding_batch(texts)

        def embed_query(self, text):
            return embedder.get_query_embedding(text)

    judge = ChatAnthropic(
        model=settings.eval_judge_model,
        api_key=require(settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
        # No temperature: current Claude models reject sampling params. Thinking is on
        # and counts toward max_tokens, so leave room.
        max_tokens=16000,
    )
    return LangchainLLMWrapper(judge), LangchainEmbeddingsWrapper(_Emb())


def run(as_shipped: bool) -> dict:
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    from simple_rag.ingest import ingest
    from simple_rag.llm import get_llm
    from simple_rag.pipeline import SimpleRAG

    base = get_settings()
    settings = build_eval_settings(base, as_shipped)
    provider, model = generator_for(base, as_shipped)
    rag = SimpleRAG(settings, llm=get_llm(settings, provider=provider, model=model))
    report = ingest(settings, rag.embed_spec, rag.store, reset=True)
    print(f"Indexed {len(report.loaded)} files ({rag.store.count()} nodes)")

    questions, skipped = load_questions(settings.eval_questions)
    rows, per_question = [], []
    for q in questions:
        result = rag.answer(q["question"])
        contexts = [s.chunk.text for s in result.sources]
        rows.append(
            {
                "user_input": q["question"],
                "response": result.answer,
                "retrieved_contexts": contexts,
                "reference": q["reference"],
            }
        )
        per_question.append(
            {
                "id": q["id"],
                "answer": result.answer,
                "sources": [s.chunk.source for s in result.sources],
            }
        )
        print(f"  {q['id']}: answered")

    llm, emb = _ragas_components(settings, rag.embed_spec.model)
    scores = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=llm,
        embeddings=emb,
    )
    df = scores.to_pandas()
    columns = [c for c in METRIC_COLUMNS if c in df.columns]
    for pq, (_, row) in zip(per_question, df.iterrows(), strict=True):
        pq["scores"] = {
            METRIC_COLUMNS[c]: (None if row[c] != row[c] else round(float(row[c]), 4))
            for c in columns
        }
    metrics = {METRIC_COLUMNS[c]: round(float(df[c].mean()), 4) for c in columns}

    out = {
        "folder": FOLDER.name,
        "mode": "as-shipped" if as_shipped else "fair",
        "generator": f"{provider}:{model}",
        "judge": f"anthropic:{settings.eval_judge_model}",
        "embedder": rag.embed_spec.name,
        "vector_store": settings.vector_store,
        "metrics": metrics,
        "per_question": per_question,
        "skipped": skipped,
    }
    (FOLDER / "eval" / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    rag.store.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--as-shipped", action="store_true", help="use this folder's own .env stack"
    )
    args = parser.parse_args()
    out = run(args.as_shipped)
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
