"""答案缓存：qa:ans:{md5(question+category)}（Redis，不可用时降级进程内存）。"""
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
        raw = f"{question}|{category or ''}"
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()
        return f"qa:ans:{digest}"

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
