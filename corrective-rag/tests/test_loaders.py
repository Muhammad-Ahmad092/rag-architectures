from pathlib import Path

from corrective_rag.loaders import load_documents

FIXTURES = Path(__file__).parent / "fixtures"


def _texts():
    docs, report = load_documents(FIXTURES)
    return dict(docs), report


def test_loads_every_supported_type():
    _, report = _texts()
    expected = {"a.txt", "b.md", "latin1.txt", "c.pdf", "d.docx", "e.csv", "f.xlsx", "g.html"}
    assert expected <= set(report.loaded)


def test_broken_pdf_skipped_not_raised():
    texts, report = _texts()
    skipped = dict(report.skipped)
    assert "broken.pdf" in skipped
    assert "unknown.xyz" in skipped
    assert "Orion rover top speed" in texts["c.pdf"]


def test_image_without_caption_skipped_with_reason():
    _, report = _texts()
    assert "multimodal" in dict(report.skipped)["h.png"]


def test_image_with_caption_uses_sidecar(tmp_path):
    (tmp_path / "pic.png").write_bytes(b"\x89PNG fake")
    (tmp_path / "pic.txt").write_text("A caption about a red robot.", encoding="utf-8")
    docs, report = load_documents(tmp_path)
    texts = dict(docs)
    assert texts["pic.png"] == "A caption about a red robot."
    assert "pic.txt" not in texts  # sidecar is not indexed twice


def test_latin1_text_decoded():
    texts, _ = _texts()
    assert "café" in texts["latin1.txt"]


def test_csv_rows_as_key_value():
    texts, _ = _texts()
    assert "part=wheel; price=40" in texts["e.csv"]
    assert "part=camera; price=300" in texts["f.xlsx"]


def test_html_visible_text_only():
    texts, _ = _texts()
    assert "Orion support is open 24/7" in texts["g.html"]
    assert "var x" not in texts["g.html"]


def test_root_readme_and_eval_dir_not_indexed(tmp_path):
    (tmp_path / "README.md").write_text("index of files", encoding="utf-8")
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "questions.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.md").write_text("real knowledge", encoding="utf-8")
    docs, report = load_documents(tmp_path)
    assert [s for s, _ in docs] == ["notes.md"]
