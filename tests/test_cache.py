"""答案缓存单元测试：key 生成、读写往返、Redis 不可用降级内存。"""
from app.rag.cache import AnswerCache


class FakeRedis:
    def __init__(self, fail=False):
        self.data = {}
        self.fail = fail

    def ping(self):
        if self.fail:
            raise ConnectionError("down")
        return True

    def get(self, key):
        return self.data.get(key)

    def set(self, key, value, ex=None):
        self.data[key] = value


def test_key_deterministic():
    a = AnswerCache._key("问题", "ai")
    b = AnswerCache._key("问题", "ai")
    c = AnswerCache._key("问题", None)
    assert a == b
    assert a != c


def test_roundtrip_with_redis():
    cache = AnswerCache(redis_client=FakeRedis())
    cache.set("什么是 RAG？", "ai", {"answer": "A"})
    assert cache.get("什么是 RAG？", "ai") == {"answer": "A"}


def test_miss_returns_none():
    cache = AnswerCache(redis_client=FakeRedis())
    assert cache.get("不存在", None) is None


def test_redis_down_falls_back_to_memory():
    cache = AnswerCache(redis_client=FakeRedis(fail=True))
    cache.set("什么是 RAG？", None, {"answer": "内存答案"})
    assert cache.get("什么是 RAG？", None) == {"answer": "内存答案"}
