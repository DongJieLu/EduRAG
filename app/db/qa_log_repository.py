"""问答日志仓储：qa_log 落库与统计聚合（全部参数化 SQL）。"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from sqlalchemy import text

from app.db.mysql import get_engine


class QALogRepository:
    def __init__(self, engine=None) -> None:
        self._engine = engine or get_engine()

    def insert_log(
        self,
        session_id: str,
        question: str,
        intent: str,
        strategy: str,
        route_detail: dict | None,
        evidence_ids: list | None,
        answer: str,
        latency_ms: int,
        cache_hit: bool,
    ) -> int:
        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    "INSERT INTO qa_log "
                    "(session_id, question, intent, strategy, route_detail, evidence_ids, "
                    " answer, latency_ms, cache_hit) "
                    "VALUES (:session_id, :question, :intent, :strategy, :route_detail, "
                    " :evidence_ids, :answer, :latency_ms, :cache_hit)"
                ),
                {
                    "session_id": session_id or "",
                    "question": question,
                    "intent": intent,
                    "strategy": strategy,
                    "route_detail": json.dumps(route_detail, ensure_ascii=False) if route_detail else None,
                    "evidence_ids": json.dumps(evidence_ids, ensure_ascii=False) if evidence_ids else None,
                    "answer": answer,
                    "latency_ms": latency_ms,
                    "cache_hit": 1 if cache_hit else 0,
                },
            )
            return result.lastrowid

    def stats(self, days: int = 7) -> dict:
        """聚合近 N 天：总问答数 / 意图分布 / 策略分布 / 平均延迟 / 日均序列。"""
        since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with self._engine.connect() as conn:
            intent_rows = conn.execute(
                text(
                    "SELECT intent, COUNT(*) AS cnt FROM qa_log "
                    "WHERE created_at >= :since GROUP BY intent"
                ),
                {"since": since},
            ).fetchall()
            strategy_rows = conn.execute(
                text(
                    "SELECT strategy, COUNT(*) AS cnt FROM qa_log "
                    "WHERE created_at >= :since GROUP BY strategy"
                ),
                {"since": since},
            ).fetchall()
            daily_rows = conn.execute(
                text(
                    "SELECT DATE(created_at) AS d, COUNT(*) AS cnt, AVG(latency_ms) AS avg_ms "
                    "FROM qa_log WHERE created_at >= :since GROUP BY DATE(created_at) ORDER BY d"
                ),
                {"since": since},
            ).fetchall()
            avg = conn.execute(
                text("SELECT AVG(latency_ms) FROM qa_log WHERE created_at >= :since"),
                {"since": since},
            ).scalar()
            total = conn.execute(
                text("SELECT COUNT(*) FROM qa_log WHERE created_at >= :since"),
                {"since": since},
            ).scalar()
        intent_dist = {r[0] or "unknown": int(r[1]) for r in intent_rows}
        strategy_dist = {r[0] or "其他": int(r[1]) for r in strategy_rows}
        daily = [
            {"date": str(r[0]), "count": int(r[1]), "avg_latency_ms": round(float(r[2]), 1) if r[2] is not None else 0.0}
            for r in daily_rows
        ]
        return {
            "total": int(total or 0),
            "intent_distribution": intent_dist,
            "strategy_distribution": strategy_dist,
            "avg_latency_ms": round(float(avg), 1) if avg is not None else 0.0,
            "daily": daily,
        }
