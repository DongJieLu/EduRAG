"""FAQ 通道：关键词命中 + BGE-M3 语义相似度 → 融合打分检索。

- 关键词层：query 与 FAQ question/keywords 的 ASCII 词元命中数
- 语义层：query embedding 与 FAQ 问题向量余弦相似度
- 融合：0.6*命中数(封顶1) + 0.4*语义分；命中判定 = 语义≥0.82 或 融合≥0.5
"""
from __future__ import annotations

import logging
import re

import numpy as np

from app.db.faq_repository import FAQRepository
from app.embeddings import get_encoder

logger = logging.getLogger(__name__)

FAQ_SEMANTIC_THRESHOLD = 0.82
FAQ_FUSION_THRESHOLD = 0.5
KW_WEIGHT = 0.6
SEM_WEIGHT = 0.4

_ASCII_TERM_RE = re.compile(r"[a-zA-Z0-9]+")


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def keyword_score(query: str, faq: dict) -> int:
    """关键词命中数：query 与 FAQ 的英文/数字词元交集大小。"""
    q_terms = {t.lower() for t in _ASCII_TERM_RE.findall(query) if len(t) >= 2}
    if not q_terms:
        return 0
    haystack = f"{faq.get('question', '')} {faq.get('keywords') or ''}".lower()
    hay_terms = {t for t in _ASCII_TERM_RE.findall(haystack) if len(t) >= 2}
    return len(q_terms & hay_terms)


def fusion(keyword_hits: int, semantic: float) -> float:
    return KW_WEIGHT * min(keyword_hits, 1) + SEM_WEIGHT * semantic


class FAQService:
    def __init__(self, repository=None, encoder=None) -> None:
        self._repo = repository or FAQRepository()
        self._encoder = encoder or get_encoder()
        self._emb_cache: dict[str, list[tuple[dict, np.ndarray]]] = {}

    def _embeddings(self, category: str | None) -> list[tuple[dict, np.ndarray]]:
        key = category or ""
        if key in self._emb_cache:
            return self._emb_cache[key]
        try:
            faqs = self._repo.get_faqs(category)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAQ 数据读取失败（MySQL 不可用？），语义层降级: %s", exc)
            return []
        if faqs:
            embs = self._encoder.encode([f["question"] for f in faqs])
            self._emb_cache[key] = list(zip(faqs, embs))
        else:
            self._emb_cache[key] = []
        return self._emb_cache[key]

    def semantic_scores(self, query: str, category: str | None = None) -> list[tuple[dict, float]]:
        """返回 [(faq, 语义相似度)]，按相似度降序。"""
        pairs = self._embeddings(category)
        if not pairs:
            return []
        q_emb = self._encoder.encode([query])[0]
        scored = [(faq, _cosine(q_emb, emb)) for faq, emb in pairs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def best_similarity(self, query: str, category: str | None = None) -> float:
        scored = self.semantic_scores(query, category)
        return scored[0][1] if scored else 0.0

    def increment_hit_count(self, faq_id: int) -> None:
        self._repo.increment_hit_count(faq_id)

    def search(self, query: str, category: str | None = None, top_k: int = 3) -> list[dict]:
        """返回命中的 FAQ（附分数），未命中返回空列表。"""
        scored = self.semantic_scores(query, category)
        matched: list[dict] = []
        for faq, sem in scored:
            kw = keyword_score(query, faq)
            fused = fusion(kw, sem)
            if sem >= FAQ_SEMANTIC_THRESHOLD or fused >= FAQ_FUSION_THRESHOLD:
                matched.append(
                    {
                        "faq_id": faq["faq_id"],
                        "question": faq["question"],
                        "answer": faq["answer"],
                        "category": faq["category"],
                        "score": round(fused, 4),
                        "semantic": round(sem, 4),
                        "keyword_hits": kw,
                    }
                )
            if len(matched) >= top_k:
                break
        return matched
