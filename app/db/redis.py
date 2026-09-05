"""Redis 客户端工厂。连接参数全部来自配置，禁止硬编码。"""
from __future__ import annotations

import redis

from app.config import get_settings


def get_redis_client() -> redis.Redis:
    s = get_settings()
    return redis.Redis(
        host=s.redis_host,
        port=s.redis_port,
        decode_responses=True,
        socket_connect_timeout=2,
    )
