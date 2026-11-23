"""Modular RAG: one Haystack ``Pipeline`` per combination of swappable modules.

The point of modular RAG is that retrieval, ranking and generation are independent parts
behind fixed interfaces, so a deployment (or a user in the UI) can mix and match them per
question without touching the rest of the system. ``ModularRAG`` builds the Haystack
pipeline for each ``Modules`` choice on first use, caches it, and records what every stage
produced so the effect of each swap is visible.
"""

from __future__ import annotations

from dataclasses import asdict

from .config import ConfigError, Settings
from .corpus import corpus_path, load_corpus
from .embeddings import Embedder, get_embedder
from .models import RAGResult, Step
from .modules import (
    Built,
    Modules,
    bm25_store,
    build_pipeline,
    from_document,
    make_generator,
    make_ranker,
    mermaid,
)
from .vectorstore import VectorStore, get_vector_store

EMPTY_INDEX_MSG = (
    "Index is empty — click “Index DATA_DIR” or run: uv run python -m modular_rag.ingest"
)


class ModularRAG:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        generator_factory=make_generator,
        ranker_factory=make_ranker,
    ):
        self.settings = settings
        self.embedder = embedder or get_embedder(settings)
        self.store = store or get_vector_store(settings, self.embedder.name, self.embedder.dim)
        self._generator_factory = generator_factory
        self._ranker_factory = ranker_factory
        self._built: dict[Modules, Built] = {}
        self.refresh_corpus()

    def default_modules(self) -> Modules:
        s = self.settings
        return Modules(retriever=s.retriever, ranker=s.ranker, generator=s.generator)

    def refresh_corpus(self) -> None:
        """Reload the BM25 corpus (after ingest) and drop pipelines built on the old one."""
        self.corpus = load_corpus(corpus_path(self.settings))
        self.corpus_store = bm25_store(self.corpus)
        self._built.clear()

    def pipeline_for(self, modules: Modules) -> Built:
        if modules not in self._built:
            self._built[modules] = build_pipeline(
                modules,
                self.settings,
                self.embedder,
                self.store,
                self.corpus_store,
                generator_factory=self._generator_factory,
                ranker_factory=self._ranker_factory,
            )
        return self._built[modules]

    def answer(self, question: str, modules: Modules | None = None) -> RAGResult:
        modules = modules or self.default_modules()
        if self.store.count() == 0:
            return RAGResult(answer=EMPTY_INDEX_MSG, sources=[], steps=[])
        if modules.retriever in ("bm25", "hybrid") and not self.corpus:
            raise ConfigError(
                "The keyword (BM25) corpus is missing: click 'Index DATA_DIR' to rebuild it, "
                "or pick the 'vector' retriever."
            )
        built = self.pipeline_for(modules)
        data = {comp: {name: question for name in inputs} for comp, inputs in built.inputs.items()}
        out = built.pipeline.run(data, include_outputs_from={*built.stages, "prompt_builder"})

        steps = [
            Step("modules", asdict(modules)),
            Step("pipeline", mermaid(built.edges, built.inputs)),
        ]
        for stage in built.stages:
            docs = out[stage]["documents"]
            steps.append(Step(stage, [(d.id, round(float(d.score or 0.0), 4)) for d in docs]))
        prompt = out["prompt_builder"]["prompt"]
        steps.append(Step("prompt", "\n\n".join(f"[{m.role.value}] {m.text}" for m in prompt)))
        reply = out["generator"]["replies"][0]
        meta = getattr(reply, "meta", {}) or {}
        steps.append(
            Step("generate", {"generator": modules.generator, "model": meta.get("model", "?")})
        )
        sources = [from_document(d) for d in out["ranker"]["documents"]]
        return RAGResult(answer=(reply.text or "").strip(), sources=sources, steps=steps)
