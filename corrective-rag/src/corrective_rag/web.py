"""Tavily web search, returned as LangChain Documents tagged ``web=True``."""

from __future__ import annotations

from langchain_core.documents import Document

from .config import Settings


class WebSearchError(Exception):
    """Tavily failed (bad key, quota, timeout); the graph answers from what it has."""


class TavilyWeb:
    def __init__(self, api_key: str, max_results: int = 3, client=None):
        if client is None:
            from tavily import TavilyClient

            client = TavilyClient(api_key=api_key)
        self.client, self.max_results = client, max_results

    def search(self, query: str) -> list[Document]:
        try:
            response = self.client.search(query, max_results=self.max_results)
        except Exception as exc:  # noqa: BLE001 - every Tavily error degrades the same way
            raise WebSearchError(f"{type(exc).__name__}: {exc}"[:200]) from exc
        docs = []
        for i, r in enumerate(response.get("results", [])):
            content = (r.get("content") or "").strip()
            if not content:
                continue
            url = r.get("url") or f"web-{i}"
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "title": r.get("title") or url,
                        "chunk_id": f"web::{url}",
                        "web": True,
                        "web_score": float(r.get("score") or 0.0),
                    },
                )
            )
        return docs


def get_web(settings: Settings):
    """None when web search is off or TAVILY_API_KEY is missing (the graph notes why)."""
    if not settings.web_search or not settings.tavily_api_key:
        return None
    return TavilyWeb(settings.tavily_api_key, settings.web_results)
