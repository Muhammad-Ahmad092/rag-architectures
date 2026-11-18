import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from naive_rag.config import get_settings

ROOT = Path(__file__).parent.parent
COMPARE = ROOT.parent / "scripts" / "compare.py"


def _load_run_eval():
    spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fair_mode_overrides():
    run_eval = _load_run_eval()
    base = get_settings(
        _env_file=None, vector_store="pinecone", embedding_provider="voyage", llm_model="x"
    )
    s = run_eval.build_eval_settings(base, as_shipped=False)
    assert s.vector_store == "chroma"
    assert s.embedding_provider == "huggingface"
    assert s.embedding_model is None
    assert s.llm_model == "gpt-4o-mini"
    assert str(s.index_dir) == ".index-eval"


def test_as_shipped_keeps_settings():
    run_eval = _load_run_eval()
    base = get_settings(_env_file=None, vector_store="faiss", llm_model="gpt-x")
    s = run_eval.build_eval_settings(base, as_shipped=True)
    assert s.vector_store == "faiss" and s.llm_model == "gpt-x"
    assert str(s.index_dir) == ".index-eval"


def test_image_questions_skipped(tmp_path):
    run_eval = _load_run_eval()
    p = tmp_path / "q.jsonl"
    rows = [
        {
            "id": "q1",
            "question": "a",
            "reference": "r",
            "modality": "image",
            "hops": 1,
            "sources": [],
        },
        {
            "id": "q2",
            "question": "b",
            "reference": "r",
            "modality": "text",
            "hops": 1,
            "sources": [],
        },
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    kept, skipped = run_eval.load_questions(p)
    assert [q["id"] for q in kept] == ["q2"]
    assert skipped == ["q1"]


@pytest.mark.skipif(
    not COMPARE.exists(), reason="repo-root scripts/ not present (folder used standalone)"
)
def test_compare_rewrites_marker(tmp_path):
    (tmp_path / "naive-rag" / "eval").mkdir(parents=True)
    (tmp_path / "naive-rag" / "eval" / "results.json").write_text(
        json.dumps(
            {
                "folder": "naive-rag",
                "mode": "fair",
                "metrics": {"faithfulness": 0.81234},
                "skipped": ["q15"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "other-rag" / "eval").mkdir(parents=True)
    (tmp_path / "other-rag" / "eval" / "results.json").write_text(
        json.dumps({"folder": "other-rag", "mode": "as-shipped", "metrics": {"faithfulness": 0.1}}),
        encoding="utf-8",
    )
    readme = tmp_path / "README.md"
    readme.write_text(
        "top\n<!-- EVAL_TABLE -->\nold\n<!-- /EVAL_TABLE -->\nbottom\n", encoding="utf-8"
    )
    subprocess.run([sys.executable, str(COMPARE), "--root", str(tmp_path)], check=True)
    text = readme.read_text(encoding="utf-8")
    assert "naive-rag" in text and "0.812" in text
    assert "other-rag" not in text  # only fair-mode results are compared
    assert "old" not in text and text.startswith("top") and text.rstrip().endswith("bottom")


def test_judge_sends_no_sampling_params(monkeypatch):
    """Current Claude models reject temperature/top_p/top_k with a 400."""
    pytest.importorskip("ragas")
    import langchain_anthropic

    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(langchain_anthropic, "ChatAnthropic", Recorder)
    run_eval = _load_run_eval()
    s = get_settings(_env_file=None, anthropic_api_key="k")
    run_eval._ragas_components(s, None)
    assert seen["model"] == "claude-sonnet-5-5"
    assert not {"temperature", "top_p", "top_k"} & set(seen)
    assert seen["max_tokens"] >= 16000
