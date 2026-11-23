"""ClaudeLLM request shape, checked against a fake Anthropic client (no network)."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from self_rag.config import ConfigError, get_settings
from self_rag.llm import ClaudeLLM, LLMError, get_llms, strict_schema


class Verdict(BaseModel):
    ok: bool
    reason: str


class FakeMessages:
    def __init__(self, text='{"ok": true, "reason": "fine"}', stop_reason="end_turn"):
        self.calls = []
        self.text, self.stop_reason = text, stop_reason

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="cyber", explanation="x"),
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.text),
            ],
        )


def _client(**kw):
    messages = FakeMessages(**kw)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_complete_sends_no_sampling_params_and_effort():
    client, msgs = _client(text="hello")
    llm = ClaudeLLM(model="claude-opus-5-5", effort="medium", client=client)
    assert llm.complete("sys", "user") == "hello"  # thinking block skipped
    call = msgs.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "user"}]
    assert call["output_config"] == {"effort": "medium"}
    assert "temperature" not in call and "top_p" not in call  # rejected by current models
    assert call["max_tokens"] >= 4000  # thinking is always on and counts toward max_tokens


def test_refusal_fallback_on_supported_models_only():
    client, msgs = _client()
    ClaudeLLM(model="claude-opus-5-5", client=client).complete("s", "u")
    assert msgs.calls[-1]["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in msgs.calls[-1]["betas"]
    ClaudeLLM(model="claude-haiku-5-5", client=client).complete("s", "u")
    assert "fallbacks" not in msgs.calls[-1]  # Haiku 5.5 has no server-side fallback


def test_judge_uses_json_schema_output_and_validates():
    client, msgs = _client()
    out = ClaudeLLM(model="claude-opus-5-5", effort="low", client=client).judge(
        "sys", "user", Verdict
    )
    assert out == Verdict(ok=True, reason="fine")
    fmt = msgs.calls[0]["output_config"]
    assert fmt["effort"] == "low"
    assert fmt["format"]["type"] == "json_schema"
    assert fmt["format"]["schema"]["additionalProperties"] is False
    assert set(fmt["format"]["schema"]["required"]) == {"ok", "reason"}
    assert "tool_choice" not in msgs.calls[0]  # forced tool use 400s on Opus 5.5


def test_refusal_raises_llm_error():
    client, _ = _client(stop_reason="refusal")
    with pytest.raises(LLMError, match="declined"):
        ClaudeLLM(model="claude-opus-5-5", client=client).complete("s", "u")


def test_strict_schema_has_no_titles():
    schema = strict_schema(Verdict)
    assert schema["additionalProperties"] is False
    assert "title" not in schema and all("title" not in p for p in schema["properties"].values())


def test_get_llms_requires_key_and_uses_grader_model():
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        get_llms(get_settings(_env_file=None, anthropic_api_key=None))
    s = get_settings(_env_file=None, anthropic_api_key="k", grader_model="claude-haiku-5-5")
    gen, grader = get_llms(s)
    assert (gen.model, gen.effort) == ("claude-opus-5-5", "medium")
    assert (grader.model, grader.effort) == ("claude-haiku-5-5", "low")
    same = get_llms(get_settings(_env_file=None, anthropic_api_key="k"))[1]
    assert same.model == "claude-opus-5-5"  # empty GRADER_MODEL -> LLM_MODEL
