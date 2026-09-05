"""会话/热点存储单元测试：多轮历史、滚动裁剪、热点榜、并发锁。"""
from app.rag.session import SessionStore


class FakeRedis:
    def __init__(self, fail=False):
        self.fail = fail
        self.lists: dict[str, list] = {}
        self.zset: dict[str, dict] = {}
        self.locks: dict[str, str] = {}

    def ping(self):
        if self.fail:
            raise ConnectionError("down")
        return True

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def ltrim(self, key, start, end):
        if key in self.lists:
            self.lists[key] = self.lists[key][start:end]

    def expire(self, key, ttl):
        pass

    def lrange(self, key, start, end):
        return list(self.lists.get(key, []))

    def zincrby(self, key, amount, member):
        d = self.zset.setdefault(key, {})
        d[member] = d.get(member, 0) + amount

    def zrevrange(self, key, start, end, withscores=False):
        items = sorted(self.zset.get(key, {}).items(), key=lambda x: x[1], reverse=True)
        items = items[start : end + 1]
        return [(m, s) for m, s in items] if withscores else [m for m, _ in items]

    def set(self, key, value, nx=False, ex=None):
        if nx and key in self.locks:
            return None
        self.locks[key] = value
        return True

    def delete(self, *keys):
        for k in keys:
            self.locks.pop(k, None)


def test_append_and_get_history_memory_fallback():
    store = SessionStore(redis_client=FakeRedis(fail=True))
    store.append("s1", "什么是 RAG？", "RAG 是检索增强生成")
    hist = store.get_history("s1")
    assert len(hist) == 2
    assert hist[0].role == "user"
    assert hist[0].content == "什么是 RAG？"
    assert hist[1].role == "assistant"


def test_history_trimmed_to_recent_20_turns():
    store = SessionStore(redis_client=FakeRedis(fail=True))
    for i in range(25):
        store.append("s1", f"问题{i}", f"答案{i}")
    hist = store.get_history("s1")
    assert len(hist) == 20 * 2  # 最近 20 轮 × 2 条消息
    assert hist[0].content == "问题5"


def test_acquire_lock_true_when_redis_down():
    store = SessionStore(redis_client=FakeRedis(fail=True))
    assert store.acquire_lock("k") is True


def test_acquire_lock_sets_nx_and_rejects_duplicate():
    redis = FakeRedis()
    store = SessionStore(redis_client=redis)
    assert store.acquire_lock("q") is True
    assert store.acquire_lock("q") is False  # 已被持有
    store.release_lock("q")
    assert store.acquire_lock("q") is True


def test_increment_hot_and_top_hot():
    redis = FakeRedis()
    store = SessionStore(redis_client=redis)
    store.increment_hot("什么是 RAG？")
    store.increment_hot("什么是 RAG？")
    store.increment_hot("什么是 Docker？")
    top = store.top_hot()
    assert top[0] == {"question": "什么是 RAG？", "hits": 2}
    assert top[1] == {"question": "什么是 Docker？", "hits": 1}


def test_top_hot_empty_when_redis_down():
    store = SessionStore(redis_client=FakeRedis(fail=True))
    assert store.top_hot() == []
