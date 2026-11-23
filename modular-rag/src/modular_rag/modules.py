"""Swappable Haystack modules and the function that assembles them into a ``Pipeline``.

Modular RAG treats each stage as an interchangeable part with a fixed interface:

* **retriever** (``query`` in, ``documents`` out): ``vector`` (the configured store, via
  custom components), ``bm25`` (Haystack ``InMemoryBM25Retriever`` over the saved chunk
  corpus), or ``hybrid`` (both, fused by a ``DocumentJoiner`` with reciprocal rank fusion);
* **ranker** (``documents`` in, top ``TOP_K`` out): ``none``, ``cross-encoder`` (a custom
  component on ``sentence_transformers.CrossEncoder``: Haystack 3 moved its own ranker to an
  integration package whose dependencies clash with the eval extra), or
  ``lost-in-the-middle`` (``LostInTheMiddleRanker``: best passages at the start and end);
* **generator** (``messages`` in, ``replies`` out): ``mistral`` or ``openai``.

``build_pipeline`` wires the chosen modules with ``Pipeline.connect`` and records every
edge so the UI can draw the exact graph that ran.
"""

from __future__ import annotations

from dataclasses import dataclass

from haystack import Document, Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack.components.joiners import DocumentJoiner
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore

from .config import Settings, require
from .models import Chunk, ScoredChunk

RETRIEVERS = ("vector", "bm25", "hybrid")
RANKERS = ("none", "cross-encoder", "lost-in-the-middle")
GENERATORS = ("mistral", "openai")

SYSTEM = (
    "You answer questions using ONLY the numbered context passages. Cite the passages "
    "you used as [n]. If the answer is not in the context, say \"I don't know based on the "
    'provided documents."'
)
USER_TEMPLATE = (
    "Context:\n{% for doc in documents %}[{{ loop.index }}] ({{ doc.meta.source }}) "
    "{{ doc.content }}\n\n{% endfor %}Question: {{ query }}"
)


@dataclass(frozen=True)
class Modules:
    retriever: str = "hybrid"
    ranker: str = "none"
    generator: str = "mistral"

    def __post_init__(self):
        for value, allowed in (
            (self.retriever, RETRIEVERS),
            (self.ranker, RANKERS),
            (self.generator, GENERATORS),
        ):
            if value not in allowed:
                raise ValueError(f"'{value}' is not one of {allowed}")


def to_document(hit: ScoredChunk) -> Document:
    c = hit.chunk
    return Document(
        id=c.id, content=c.text, meta={"source": c.source, **c.metadata}, score=hit.score
    )


def from_document(doc: Document) -> ScoredChunk:
    meta = dict(doc.meta or {})
    source = meta.pop("source", "?")
    chunk = Chunk(id=doc.id, text=doc.content or "", source=source, metadata=meta)
    return ScoredChunk(chunk=chunk, score=float(doc.score or 0.0))


# ----------------------------------------------------------------------------- components


@component
class QueryEmbedder:
    """Embeds the query with this folder's embedder (any of the 5 providers)."""

    def __init__(self, embedder):
        self.embedder = embedder

    @component.output_types(embedding=list[float])
    def run(self, query: str):
        return {"embedding": self.embedder.embed_query(query)}


@component
class StoreRetriever:
    """Searches this folder's vector store (any of the 7) and returns Haystack Documents."""

    def __init__(self, store, top_k: int):
        self.store, self.top_k = store, top_k

    @component.output_types(documents=list[Document])
    def run(self, embedding: list[float]):
        return {"documents": [to_document(h) for h in self.store.search(embedding, self.top_k)]}


@component
class TopK:
    """The 'no ranker' module: keep the first k documents unchanged."""

    def __init__(self, top_k: int):
        self.top_k = top_k

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        return {"documents": documents[: self.top_k]}


@component
class CrossEncoderRanker:
    """Re-scores (query, passage) pairs with a cross-encoder and keeps the best ``top_k``.

    Cross-encoders read the query and passage together, so they rank far better than
    the bi-encoder similarity used for retrieval, at the cost of one forward pass per pair.
    The model downloads on first use (``warm_up``).
    """

    def __init__(self, model: str, top_k: int, model_obj=None):
        self.model_name, self.top_k, self._model = model, top_k, model_obj

    def warm_up(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    @component.output_types(documents=list[Document])
    def run(self, query: str, documents: list[Document]):
        if not documents:
            return {"documents": []}
        self.warm_up()
        scores = self._model.predict([(query, d.content or "") for d in documents])
        ranked = sorted(zip(documents, scores, strict=True), key=lambda p: -float(p[1]))
        return {
            "documents": [
                Document(id=d.id, content=d.content, meta=d.meta, score=float(s))
                for d, s in ranked[: self.top_k]
            ]
        }


def bm25_store(chunks: list[Chunk]) -> InMemoryDocumentStore:
    ds = InMemoryDocumentStore()
    if chunks:
        ds.write_documents([to_document(ScoredChunk(c, 0.0)) for c in chunks])
    return ds


def make_generator(name: str, settings: Settings):
    from haystack.utils import Secret

    if name == "mistral":
        from haystack_integrations.components.generators.mistral import MistralChatGenerator

        key = require(settings.mistral_api_key, "MISTRAL_API_KEY")
        return MistralChatGenerator(
            api_key=Secret.from_token(key),
            model=settings.llm_model,
            generation_kwargs={"temperature": 0},
        )
    if name == "openai":
        from haystack.components.generators.chat import OpenAIChatGenerator

        key = require(settings.openai_api_key, "OPENAI_API_KEY")
        return OpenAIChatGenerator(
            api_key=Secret.from_token(key),
            model=settings.openai_model,
            generation_kwargs={"temperature": 0},
        )
    raise ValueError(f"unknown generator '{name}'")


def make_ranker(name: str, settings: Settings):
    if name == "none":
        return TopK(settings.top_k)
    if name == "cross-encoder":
        return CrossEncoderRanker(settings.rerank_model, settings.top_k)
    if name == "lost-in-the-middle":
        from haystack.components.rankers import LostInTheMiddleRanker

        return LostInTheMiddleRanker(top_k=settings.top_k)
    raise ValueError(f"unknown ranker '{name}'")


# ----------------------------------------------------------------------------- assembly


@dataclass
class Built:
    pipeline: Pipeline
    inputs: dict  # component -> {input: "query"} : where the question goes
    edges: list[tuple[str, str]]  # (sender.socket, receiver.socket), for the diagram
    stages: list[str]  # components whose document outputs are traced


def build_pipeline(
    modules: Modules,
    settings: Settings,
    embedder,
    store,
    corpus_store: InMemoryDocumentStore,
    generator_factory=make_generator,
    ranker_factory=make_ranker,
) -> Built:
    p = Pipeline()
    edges: list[tuple[str, str]] = []
    inputs: dict = {}
    stages: list[str] = []

    def connect(a: str, b: str):
        p.connect(a, b)
        edges.append((a, b))

    k = settings.candidate_k
    retriever_out = []
    if modules.retriever in ("vector", "hybrid"):
        p.add_component("query_embedder", QueryEmbedder(embedder))
        p.add_component("vector_retriever", StoreRetriever(store, k))
        connect("query_embedder.embedding", "vector_retriever.embedding")
        inputs["query_embedder"] = {"query": "query"}
        retriever_out.append("vector_retriever")
    if modules.retriever in ("bm25", "hybrid"):
        p.add_component(
            "bm25_retriever", InMemoryBM25Retriever(document_store=corpus_store, top_k=k)
        )
        inputs["bm25_retriever"] = {"query": "query"}
        retriever_out.append("bm25_retriever")
    stages += retriever_out
    if modules.retriever == "hybrid":
        p.add_component("joiner", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=k))
        for r in retriever_out:
            connect(f"{r}.documents", "joiner.documents")
        stages.append("joiner")
        docs_from = "joiner"
    else:
        docs_from = retriever_out[0]

    p.add_component("ranker", ranker_factory(modules.ranker, settings))
    connect(f"{docs_from}.documents", "ranker.documents")
    if modules.ranker == "cross-encoder":
        inputs["ranker"] = {"query": "query"}
    stages.append("ranker")

    template = [ChatMessage.from_system(SYSTEM), ChatMessage.from_user(USER_TEMPLATE)]
    p.add_component(
        "prompt_builder",
        ChatPromptBuilder(template=template, required_variables=["query", "documents"]),
    )
    connect("ranker.documents", "prompt_builder.documents")
    inputs["prompt_builder"] = {"query": "query"}
    p.add_component("generator", generator_factory(modules.generator, settings))
    connect("prompt_builder.prompt", "generator.messages")
    return Built(pipeline=p, inputs=inputs, edges=edges, stages=stages)


def dot(edges: list[tuple[str, str]], entry: dict) -> str:
    """Graphviz DOT of the assembled pipeline (Streamlit renders it with graphviz_chart)."""
    lines = [
        'digraph G { rankdir=LR; node [shape=box, style=rounded, fontname="Helvetica"];',
        "  question [shape=oval]; answer [shape=oval];",
    ]
    lines += [f'  question -> {comp} [label="query"];' for comp in entry]
    for a, b in edges:
        (sa, sock), (rb, _) = a.split(".", 1), b.split(".", 1)
        lines.append(f'  {sa} -> {rb} [label="{sock}"];')
    lines.append('  generator -> answer [label="replies"];')
    return "\n".join([*lines, "}"])


def mermaid(edges: list[tuple[str, str]], entry: dict) -> str:
    """Mermaid flowchart of the assembled pipeline (question -> entry components -> ...)."""
    lines = ["flowchart LR", "    Q([question])"]
    for comp in entry:
        lines.append(f"    Q --> {comp}")
    for a, b in edges:
        (sa, sock), (rb, _) = a.split(".", 1), b.split(".", 1)
        lines.append(f"    {sa} -->|{sock}| {rb}")
    lines.append("    generator --> A([answer])")
    return "\n".join(lines)
