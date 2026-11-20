"""Per-session chat history with a sliding window.

Each session keeps its full list of LangChain messages; only the last ``max_turns``
question/answer pairs go into prompts (``trim_messages``), so prompt size stays bounded
however long the conversation gets. (``InMemoryChatMessageHistory`` is deprecated since
langchain-core 1.6.4; a plain message list per session is all this needs.)
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages


class SessionMemory:
    def __init__(self, max_turns: int):
        self.max_turns = max_turns
        self._sessions: dict[str, list[BaseMessage]] = {}

    def messages(self, session_id: str) -> list[BaseMessage]:
        return list(self._sessions.get(session_id, []))

    def window(self, session_id: str) -> list[BaseMessage]:
        return trim_messages(
            self._sessions.get(session_id, []),
            max_tokens=2 * self.max_turns,
            token_counter=len,  # count messages, not tokens: one turn = 2 messages
            strategy="last",
            start_on="human",
        )

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        self._sessions.setdefault(session_id, []).extend(
            [HumanMessage(question), AIMessage(answer)]
        )

    def turns(self, session_id: str) -> int:
        return len(self._sessions.get(session_id, [])) // 2

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)
