"""Load every supported file under a directory into plain text (chunking is done by LlamaIndex).

Loading never raises for a bad file: failures go into ``LoadReport.skipped`` with a reason,
so one corrupt PDF cannot block indexing of everything else.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _rows_to_text(rows: list[list]) -> str:
    if not rows:
        return ""
    header = [str(h) for h in rows[0]]
    lines = []
    for row in rows[1:]:
        pairs = [f"{h}={'' if v is None else v}" for h, v in zip(header, row, strict=False)]
        lines.append("; ".join(pairs))
    return "\n".join(lines)


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_csv(path: Path) -> str:
    return _rows_to_text(list(csv.reader(io.StringIO(_read_text(path)))))


def _load_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), read_only=True, data_only=True)
    parts = []
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        parts.append(_rows_to_text(rows))
    return "\n".join(parts)


def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text(path), "html.parser")
    for tag in soup(["script", "style", "nav", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


LOADERS: dict[str, Callable[[Path], str]] = {
    ".txt": _read_text,
    ".md": _read_text,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".csv": _load_csv,
    ".xlsx": _load_xlsx,
    ".html": _load_html,
    ".htm": _load_html,
}


def load_documents(data_dir: Path) -> tuple[list[tuple[str, str]], LoadReport]:
    """Return ``(source_relpath, text)`` pairs for every readable file plus a report."""
    data_dir = Path(data_dir)
    report = LoadReport()
    docs: list[tuple[str, str]] = []
    files = sorted(p for p in data_dir.rglob("*") if p.is_file())
    caption_sidecars = {p.with_suffix(".txt") for p in files if p.suffix.lower() in IMAGE_EXTS}

    for path in files:
        rel = path.relative_to(data_dir).as_posix()
        ext = path.suffix.lower()
        if path in caption_sidecars or path.name.startswith("."):
            continue
        if rel.startswith("eval/") or rel.lower() == "readme.md":
            continue  # eval questions and the corpus index are not knowledge
        if ext in IMAGE_EXTS:
            caption = path.with_suffix(".txt")
            if caption.exists():
                docs.append((rel, _read_text(caption).strip()))
                report.loaded.append(rel)
            else:
                report.skipped.append((rel, "image (no caption; multimodal only)"))
            continue
        loader = LOADERS.get(ext)
        if loader is None:
            report.skipped.append((rel, f"unsupported extension '{ext}'"))
            continue
        try:
            text = loader(path).strip()
        except Exception as exc:  # noqa: BLE001 - any parser failure is a skip, never a crash
            report.skipped.append((rel, f"parse error: {type(exc).__name__}: {exc}"[:200]))
            continue
        if not text:
            report.skipped.append((rel, "no extractable text"))
            continue
        docs.append((rel, text))
        report.loaded.append(rel)
    return docs, report
