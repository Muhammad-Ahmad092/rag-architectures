import pytest
from conftest import BowEmbeddings, FakeChat

from simple_rag_memory.config import ConfigError, get_settings
from simple_rag_memory.ingest import build_documents, ingest
from simple_rag_memory.llm import get_llm
from simple_rag_memory.pipeline import EMPTY_INDEX_MSG, MemoryRAG, clean_rewrite
from simple_rag_memory.vectorstore import get_vector_store


@pytest.fixture
def store(settings, embed_spec):
    return get_vector_store(settings, embed_spec)


@pytest.fixture
def rag(settings, embed_spec, store, fake_chat):
    ingest(settings, embed_spec=embed_spec, store=store)
    return MemoryRAG(settings, embed_spec=embed_spec, store=store, llm=fake_chat)


def _retrieved_sources(result):
    return [src for src, _ in next(s.detail for s in result.steps if s.name == "retrieve")]


def test_first_turn_skips_rewrite(rag, fake_chat):
    result = rag.answer("Orion rover top speed", session_id="s")
    assert fake_chat.rewrite_calls() == []  # no history -> no condense call
    assert result.search_query == "Orion rover top speed"
    rewrite = next(s.detail for s in result.steps if s.name == "rewrite")
    assert rewrite == {"skipped": "no history yet"}
    assert result.answer == "answer [1]"


def test_follow_up_is_rewritten_and_rewrite_is_what_gets_retrieved(rag, fake_chat):
    rag.answer("Orion rover top speed", session_id="s")
    fake_chat.rewrite_reply = "When is Orion support open?"
    result = rag.answer("and when is it open?", session_id="s")
    assert len(fake_chat.rewrite_calls()) == 1
    condense = fake_chat.rewrite_calls()[0]
    assert [m.content for m in condense[1:3]] == ["Orion rover top speed", "answer [1]"]
    assert condense[-1].content == "and when is it open?"
    assert result.search_query == "When is Orion support open?"
    assert _retrieved_sources(result)[0] == "g.html"  # "Orion support is open 24/7"


def test_history_is_in_the_answer_prompt(rag, fake_chat):
    rag.answer("first question", session_id="s")
    rag.answer("second question", session_id="s")
    last = fake_chat.answer_calls()[-1]
    contents = [m.content for m in last]
    assert "Context:" in contents[0]
    assert contents[1:] == ["first question", "answer [1]", "second question"]


def test_answer_prompt_is_bounded_by_memory_turns(rag, fake_chat, settings):
    for i in range(5):
        rag.answer(f"question {i}", session_id="s")
    last = fake_chat.answer_calls()[-1]
    assert len(last) == 1 + 2 * settings.memory_turns + 1  # system + window + new question
    memory = next(s.detail for s in rag.answer("again", session_id="s").steps if s.name == "memory")
    # recorded before this turn is stored: 5 earlier turns, 2 of them in the window
    assert memory == {"session": "s", "turns_in_window": 2, "turns_total": 5}


def test_sessions_do_not_share_history(rag, fake_chat):
    rag.answer("Orion rover top speed", session_id="a")
    rag.answer("and its parts?", session_id="b")
    assert fake_chat.rewrite_calls() == []


def test_rewrite_disabled_uses_raw_follow_up(rag, fake_chat):
    rag.answer("Orion rover top speed", session_id="s")
    result = rag.answer("and its parts?", session_id="s", rewrite=False)
    assert fake_chat.rewrite_calls() == []
    assert result.search_query == "and its parts?"
    assert next(s.detail for s in result.steps if s.name == "rewrite") == {"skipped": "disabled"}


def test_separate_rewrite_model_is_used(settings, embed_spec, store):
    ingest(settings, embed_spec=embed_spec, store=store)
    main, small = FakeChat(calls=[]), FakeChat(calls=[], rewrite_reply="standalone")
    rag = MemoryRAG(settings, embed_spec=embed_spec, store=store, llm=main, rewrite_llm=small)
    rag.answer("one", session_id="s")
    rag.answer("two", session_id="s")
    assert len(small.rewrite_calls()) == 1 and main.rewrite_calls() == []


def test_steps_order(rag):
    names = [s.name for s in rag.answer("battery").steps]
    assert names == ["memory", "rewrite", "retrieve", "prompt", "generate"]


def test_top_k_respected(rag, settings):
    assert len(_retrieved_sources(rag.answer("Orion"))) == settings.top_k


def test_empty_index_message_no_llm_call_no_history(settings, embed_spec, store, fake_chat):
    rag = MemoryRAG(settings, embed_spec=embed_spec, store=store, llm=fake_chat)
    result = rag.answer("hi", session_id="s")
    assert result.answer == EMPTY_INDEX_MSG
    assert result.sources == [] and result.steps == []
    assert fake_chat.calls == []
    assert rag.memory.turns("s") == 0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"What fixed INC-2041?"', "What fixed INC-2041?"),
        ("Standalone question: What fixed INC-2041?", "What fixed INC-2041?"),
        ("  What fixed INC-2041?\n", "What fixed INC-2041?"),
        ("", "orig"),
        ('""', "orig"),
    ],
)
def test_clean_rewrite(raw, expected):
    assert clean_rewrite(raw, fallback="orig") == expected


def test_chunks_have_source_and_ids():
    docs = build_documents([("long.txt", "word " * 600)], chunk_size=200, chunk_overlap=20)
    assert len(docs) > 1
    assert all(d.metadata["source"] == "long.txt" for d in docs)
    assert [d.metadata["chunk_id"] for d in docs[:2]] == ["long.txt::0", "long.txt::1"]
    assert all(len(d.page_content) <= 200 for d in docs)


def test_ingest_report_and_count(settings, embed_spec, store):
    report = ingest(settings, embed_spec=embed_spec, store=store)
    n = store.count()
    assert n > 0
    assert "broken.pdf" in dict(report.skipped)
    ingest(settings, embed_spec=embed_spec, store=store, reset=True)
    assert store.count() == n


def test_ingest_failure_keeps_existing_index(settings, embed_spec, store):
    ingest(settings, embed_spec=embed_spec, store=store)
    n = store.count()

    class Failing(BowEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("API down")

    embed_spec.model = Failing()
    with pytest.raises(RuntimeError):
        ingest(settings, embed_spec=embed_spec, store=store, reset=True)
    assert store.count() == n


def test_ingest_missing_data_dir_raises_config_error(settings, embed_spec, store, tmp_path):
    bad = settings.model_copy(update={"data_dir": tmp_path / "nope"})
    with pytest.raises(ConfigError, match="DATA_DIR"):
        ingest(bad, embed_spec=embed_spec, store=store)


@pytest.mark.parametrize("provider,var", [("groq", "GROQ_API_KEY"), ("openai", "OPENAI_API_KEY")])
def test_llm_missing_key_raises_config_error(provider, var):
    s = get_settings(_env_file=None, groq_api_key=None, openai_api_key=None)
    with pytest.raises(ConfigError, match=var):
        get_llm(s, provider=provider)


def test_llm_builds_groq_with_model():
    s = get_settings(_env_file=None, groq_api_key="k")
    llm = get_llm(s, model="llama-3.1-8b-instant")
    assert type(llm).__name__ == "ChatGroq" and llm.model_name == "llama-3.1-8b-instant"
    assert llm.temperature < 1e-6  # ChatGroq stores temperature=0 as 1e-8
