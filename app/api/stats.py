"""统计聚合：意图分布 / 平均延迟 / FAQ 热点（qa_log + faq + Redis 热点榜）。"""
from __future__ import annotations

from app.db.faq_repository import FAQRepository
from app.db.qa_log_repository import QALogRepository


class StatsService:
    def __init__(self, qa_log=None, faq_repo=None, session_store=None) -> None:
        self._qa_log = qa_log or QALogRepository()
        self._faq = faq_repo or FAQRepository()
        self._session = session_store

    def get_stats(self, days: int = 7) -> dict:
        base = self._qa_log.stats(days)
        # FAQ 热点：优先 Redis zset（实时），Redis 不可用回退 MySQL hit_count（持久）
        faq_top = self._session.top_hot(10) if self._session else []
        if not faq_top:
            faq_top = [
                {"question": f["question"], "hits": f["hit_count"], "category": f["category"]}
                for f in self._faq.top_faqs(10)
            ]
        return {
            "days": days,
            "total": base["total"],
            "intent_distribution": base["intent_distribution"],
            "strategy_distribution": base["strategy_distribution"],
            "avg_latency_ms": base["avg_latency_ms"],
            "daily": base["daily"],
            "faq_top": faq_top,
        }
