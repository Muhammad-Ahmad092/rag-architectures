from langchain_core.messages import AIMessage, HumanMessage

from simple_rag_memory.memory import SessionMemory


def _fill(mem, sid, n):
    for i in range(n):
        mem.add_turn(sid, f"q{i}", f"a{i}")


def test_window_keeps_last_turns_starting_on_human():
    mem = SessionMemory(max_turns=2)
    _fill(mem, "s", 5)
    window = mem.window("s")
    assert [m.content for m in window] == ["q3", "a3", "q4", "a4"]
    assert isinstance(window[0], HumanMessage) and isinstance(window[-1], AIMessage)
    assert mem.turns("s") == 5  # full history is kept; only the prompt window is trimmed


def test_sessions_are_isolated():
    mem = SessionMemory(max_turns=3)
    _fill(mem, "a", 2)
    assert mem.window("b") == []
    assert mem.turns("a") == 2 and mem.turns("b") == 0


def test_clear_one_and_all():
    mem = SessionMemory(max_turns=3)
    _fill(mem, "a", 1)
    _fill(mem, "b", 1)
    mem.clear("a")
    assert mem.turns("a") == 0 and mem.turns("b") == 1
    mem.clear()
    assert mem.turns("b") == 0
