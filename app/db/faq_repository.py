"""FAQ 数据访问：MySQL faq 表查询（全部参数化 SQL）。"""
from __future__ import annotations

from sqlalchemy import text

from app.db.mysql import get_engine


class FAQRepository:
    def __init__(self, engine=None) -> None:
        self._engine = engine or get_engine()

    def get_faqs(self, category: str | None = None) -> list[dict]:
        """返回启用中的 FAQ 列表（可按方向过滤），供关键词/语义检索。"""
        sql = (
            "SELECT faq_id, question, keywords, answer, category, hit_count "
            "FROM faq WHERE status = 1"
        )
        params: dict = {}
        if category:
            sql += " AND category = :category"
            params["category"] = category
        sql += " ORDER BY faq_id ASC"
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params)
            return [dict(r._mapping) for r in rows]

    def increment_hit_count(self, faq_id: int) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("UPDATE faq SET hit_count = hit_count + 1 WHERE faq_id = :faq_id"),
                {"faq_id": faq_id},
            )

    def top_faqs(self, limit: int = 10) -> list[dict]:
        """按命中次数返回热点 FAQ Top N（供统计看板）。"""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT question, category, hit_count FROM faq "
                    "WHERE status = 1 ORDER BY hit_count DESC, faq_id ASC LIMIT :limit"
                ),
                {"limit": limit},
            ).fetchall()
            return [dict(r._mapping) for r in rows]
