"""会话与热点存储：多轮历史(qa:list) + FAQ 热点榜(qa:faq:hot) + 并发锁(qa:lock)。

Redis 不可用时降级为进程内存（历史/锁），热点榜在内存降级下不维护（统计改走 MySQL hit_count 兜底）。
"""
from __future__ import annotations

import json
import logging

from app.llm.base import ChatMessage

logger = logging.getLogger(__name__)

HISTORY_MAX_TURNS = 20
HISTORY_TTL_SEC = 1800
LOCK_TTL_SEC = 30


class SessionStore:
    def __init__(self, redis_client=None) -> None:
        self._redis = redis_client
        self._memory: dict[str, list[dict]] = {}
        self._redis_available: bool | None = None

    def _redis_ok(self) -> bool:
        if self._redis_available is not None:
            return self._redis_available
        try:
            if self._redis is None:
                from app.db.redis import get_redis_client

                self._redis = get_redis_client()
            self._redis.ping()
            self._redis_available = True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis 不可用，会话/热点降级为进程内存: %s", exc)
            self._redis_available = False
        return self._redis_available

    @staticmethod
    def _list_key(session_id: str) -> str:
        return f"qa:list:{session_id}"

    def append(self, session_id: str, question: str, answer: str) -> None:
        """记录一轮 Q/A，仅保留最近 20 轮（滚动裁剪）。"""
        entry = json.dumps({"q": question, "a": answer}, ensure_ascii=False)
        if self._redis_ok():
            key = self._list_key(session_id)
            self._redis.rpush(key, entry)
            self._redis.ltrim(key, -HISTORY_MAX_TURNS, -1)
            self._redis.expire(key, HISTORY_TTL_SEC)
        else:
            buf = self._memory.setdefault(session_id, [])
            buf.append({"q": question, "a": answer})
            del buf[: max(0, len(buf) - HISTORY_MAX_TURNS)]

    def get_history(self, session_id: str) -> list[ChatMessage]:
        """返回该会话最近若干轮对话（user/assistant 交替的 ChatMessage 列表）。"""
        items: list[dict] = []
        if self._redis_ok():
            raw = self._redis.lrange(self._list_key(session_id), 0, -1)
            for r in raw or []:
                try:
                    items.append(json.loads(r))
                except (json.JSONDecodeError, TypeError):
                    continue
        else:
            items = self._memory.get(session_id, [])
        messages: list[ChatMessage] = []
        for it in items:
            messages.append(ChatMessage(role="user", content=it.get("q", "")))
            messages.append(ChatMessage(role="assistant", content=it.get("a", "")))
        return messages

    def increment_hot(self, faq_question: str) -> None:
        """FAQ 命中一次，热点榜 +1。"""
        if self._redis_ok():
            self._redis.zincrby("qa:faq:hot", 1, faq_question)

    def top_hot(self, limit: int = 10) -> list[dict]:
        """热点榜 Top N（仅 Redis 可用时有值，否则返回空）。"""
        if not self._redis_ok():
            return []
        items = self._redis.zrevrange("qa:faq:hot", 0, limit - 1, withscores=True)
        return [{"question": q, "hits": int(s)} for q, s in items]

    def acquire_lock(self, key: str) -> bool:
        """抢占并发锁（SET NX EX 30s）。Redis 不可用时返回 True（不阻塞，降级为不防抖）。"""
        if not self._redis_ok():
            return True
        return bool(self._redis.set(f"qa:lock:{key}", "1", nx=True, ex=LOCK_TTL_SEC))

    def release_lock(self, key: str) -> None:
        if self._redis_ok():
            self._redis.delete(f"qa:lock:{key}")
