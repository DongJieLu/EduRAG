"""检索器：query 编码 → 向量召回（Chroma，带 category 过滤）。

M3 仅实现向量召回；M5 在此之上扩展混合检索（向量 + 关键词）与策略引擎。
"""
from __future__ import annotations

from app.embeddings import get_encoder
from app.rag.vector_store import VectorStore

DEFAULT_TOP_K = 50


class Retriever:
    def __init__(self, vector_store=None, encoder=None) -> None:
        self._store = vector_store or VectorStore()
        self._encoder = encoder or get_encoder()

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """返回 [{id, text, metadata, score}]，按相似度降序。"""
        if not query or not query.strip():
            return []
        query_embedding = self._encoder.encode([query])[0]
        return self._store.search(query_embedding, top_k=top_k, category=category)
