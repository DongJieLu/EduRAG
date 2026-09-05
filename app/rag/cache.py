"""答案缓存：qa:ans:{category}:{md5(question)}（Redis，不可用时降级进程内存）。

category 作为明文段写入 key，便于文档更新后按方向清理缓存（规格 5.8）。
"""
from __future__ import annotations

import hashlib
import json
import logging

logger = logging.getLogger(__name__)

ANSWER_TTL = 3600
REJECT_TTL = 600


class AnswerCache:
    def __init__(self, redis_client=None, ttl: int = ANSWER_TTL, reject_ttl: int = REJECT_TTL) -> None:
        self._redis = redis_client
        self._ttl = ttl
        self._reject_ttl = reject_ttl
        self._memory: dict[str, dict] = {}
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
            logger.warning("Redis 不可用，答案缓存降级为进程内存: %s", exc)
            self._redis_available = False
        return self._redis_available

    @staticmethod
    def _key(question: str, category: str | None) -> str:
        digest = hashlib.md5(question.encode("utf-8")).hexdigest()
        return f"qa:ans:{category or 'all'}:{digest}"

    def get(self, question: str, category: str | None) -> dict | None:
        key = self._key(question, category)
        if self._redis_ok():
            raw = self._redis.get(key)
            try:
                return json.loads(raw) if raw else None
            except (json.JSONDecodeError, TypeError):
                return None
        return self._memory.get(key)

    def set(self, question: str, category: str | None, data: dict, rejected: bool = False) -> None:
        key = self._key(question, category)
        ttl = self._reject_ttl if rejected else self._ttl
        if self._redis_ok():
            self._redis.set(key, json.dumps(data, ensure_ascii=False), ex=ttl)
        else:
            self._memory[key] = data

    def invalidate_category(self, category: str | None) -> int:
        """清理某方向（或全部）的答案缓存，返回清理的 key 数。"""
        prefix = f"qa:ans:{category or 'all'}:"
        if not self._redis_ok():
            removed = sum(1 for k in list(self._memory) if k.startswith(prefix))
            self._memory = {k: v for k, v in self._memory.items() if not k.startswith(prefix)}
            return removed
        keys = list(self._redis.scan_iter(match=f"{prefix}*", count=100))
        if keys:
            self._redis.delete(*keys)
        return len(keys)
