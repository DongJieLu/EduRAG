"""检索器：向量召回 + 关键词召回 + RRF 混合融合。

M3 仅实现向量召回；M5 增加关键词召回（MySQL chunk 词元/中文 bigram 匹配）与 RRF 融合。
"""
from __future__ import annotations

import logging
import re

from app.embeddings import get_encoder
from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 50
RRF_K = 60  # RRF 平滑常数

_ASCII_TERM_RE = re.compile(r"[a-zA-Z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _keyword_score(query: str, text: str) -> float:
    """关键词命中分：ASCII 词元交集数 + 中文 bigram 交集数。"""
    q_terms = {t.lower() for t in _ASCII_TERM_RE.findall(query) if len(t) >= 2}
    t_terms = {t.lower() for t in _ASCII_TERM_RE.findall(text) if len(t) >= 2}
    ascii_hits = len(q_terms & t_terms)

    q_cn = "".join(_CJK_RE.findall(query))
    t_cn = "".join(_CJK_RE.findall(text))
    q_bg = {q_cn[i : i + 2] for i in range(len(q_cn) - 1)}
    t_bg = {t_cn[i : i + 2] for i in range(len(t_cn) - 1)}
    cn_hits = len(q_bg & t_bg)
    return float(ascii_hits + cn_hits)


class Retriever:
    def __init__(self, vector_store=None, encoder=None, repository=None) -> None:
        self._store = vector_store or VectorStore()
        self._encoder = encoder or get_encoder()
        self._repo = repository

    def _get_repo(self):
        if self._repo is None:
            from app.ingest.repository import KnowledgeRepository

            self._repo = KnowledgeRepository()
        return self._repo

    def retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """纯向量召回，返回 [{id, text, metadata, score}]，按相似度降序。"""
        if not query or not query.strip():
            return []
        query_embedding = self._encoder.encode([query])[0]
        return self._store.search(query_embedding, top_k=top_k, category=category)

    def keyword_retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """关键词召回：MySQL chunk 词元匹配，返回 [{id, text, metadata, score}]。"""
        if not query or not query.strip():
            return []
        try:
            chunks = self._get_repo().list_chunks(category)
        except Exception as exc:  # noqa: BLE001
            logger.warning("关键词召回读取 MySQL 失败，降级为空: %s", exc)
            return []
        scored = []
        for c in chunks:
            s = _keyword_score(query, c.get("chunk_text") or "")
            if s <= 0:
                continue
            scored.append(
                {
                    "id": str(c["chunk_id"]),
                    "text": c.get("chunk_text") or "",
                    "metadata": {
                        "chunk_id": c.get("chunk_id"),
                        "doc_id": c.get("doc_id"),
                        "doc_name": c.get("doc_name") or "",
                        "category": c.get("category") or "",
                        "title": c.get("title") or "",
                        "page_no": c.get("page_no") or 0,
                    },
                    "score": s,
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def hybrid_retrieve(
        self,
        query: str,
        category: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[dict]:
        """混合检索：向量召回 + 关键词召回做 RRF 融合，返回融合后 top_k。"""
        vec_hits = self.retrieve(query, category=category, top_k=top_k)
        kw_hits = self.keyword_retrieve(query, category=category, top_k=top_k)
        if not kw_hits:
            return vec_hits[:top_k]

        fused: dict[str, dict] = {}
        for rank, h in enumerate(vec_hits):
            key = self._hit_key(h)
            if key is None:
                continue
            entry = fused.setdefault(key, {"hit": h, "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank)
        for rank, h in enumerate(kw_hits):
            key = self._hit_key(h)
            if key is None:
                continue
            entry = fused.setdefault(key, {"hit": h, "score": 0.0})
            entry["score"] += 1.0 / (RRF_K + rank)

        merged = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return [m["hit"] for m in merged[:top_k]]

    @staticmethod
    def _hit_key(hit: dict) -> str | None:
        meta = hit.get("metadata") or {}
        chunk_id = meta.get("chunk_id")
        if chunk_id is None:
            return hit.get("id")
        return str(chunk_id)
