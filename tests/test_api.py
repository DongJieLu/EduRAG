"""FastAPI 接口测试：统一响应、chat/stream/ingest/sources/stats/health（mock 依赖）。"""
import json

from fastapi.testclient import TestClient

from app.api import deps
from app.api.main import app


class FakeChatService:
    def __init__(self, result=None, events=None):
        self._result = result or {"intent": "rag", "answer": "答案", "citations": [], "strategy": "direct", "latency_ms": 5}
        self._events = events

    def chat(self, question, category=None, session_id=None):
        return self._result

    def stream_chat(self, question, category=None, session_id=None):
        yield from (self._events or [
            {"type": "route", "intent": "rag", "strategy": "direct"},
            {"type": "token", "content": "你"},
            {"type": "done", "latency_ms": 5, "rejected": False, "cache_hit": False},
        ])


class FakeSessionStore:
    def acquire_lock(self, key):
        return True

    def release_lock(self, key):
        pass


class FakeIngestService:
    def ingest_file(self, file_path, category, file_name=None):
        return {"doc_id": 1, "chunk_count": 3}


class FakeRepo:
    def list_documents(self, category=None):
        return [{"doc_id": 1, "file_name": "a.md", "category": "ai", "chunk_count": 7}]


class FakeStatsService:
    def get_stats(self, days=7):
        return {"days": days, "total": 10, "intent_distribution": {"rag": 10}, "avg_latency_ms": 120.5, "faq_top": []}


def _override(chat=None, session=None, ingest=None, repo=None, stats=None):
    if chat is not None:
        app.dependency_overrides[deps.get_chat_service] = lambda: chat
    if session is not None:
        app.dependency_overrides[deps.get_session_store] = lambda: session
    if ingest is not None:
        app.dependency_overrides[deps.get_ingest_service] = lambda: ingest
    if repo is not None:
        app.dependency_overrides[deps.get_knowledge_repository] = lambda: repo
    if stats is not None:
        app.dependency_overrides[deps.get_stats_service] = lambda: stats


def _client(**kwargs):
    _override(**kwargs)
    return TestClient(app)


def test_chat_ok():
    client = _client(chat=FakeChatService(), session=FakeSessionStore())
    resp = client.post("/api/v1/chat", json={"question": "什么是 RAG？", "category": "ai"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["intent"] == "rag"
    assert body["msg"] == "ok"


def test_chat_empty_question_param_error():
    client = _client()
    resp = client.post("/api/v1/chat", json={"question": "  "})
    assert resp.status_code == 400
    assert resp.json()["code"] == 1001


def test_chat_invalid_category_param_error():
    client = _client()
    resp = client.post("/api/v1/chat", json={"question": "hi", "category": "bad"})
    assert resp.status_code == 400
    assert resp.json()["code"] == 1001


def test_chat_stream_sse_format():
    client = _client(chat=FakeChatService())
    resp = client.post("/api/v1/chat/stream", json={"question": "hi"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data: " in resp.text
    # 首条为 route 事件
    first = json.loads(resp.text.split("\n\n")[0].removeprefix("data: "))
    assert first["type"] == "route"


def test_ingest_ok():
    client = _client(ingest=FakeIngestService())
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("a.txt", b"hello world", "text/plain")},
        data={"category": "ai"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["doc_id"] == 1


def test_ingest_bad_type():
    client = _client()
    resp = client.post(
        "/api/v1/ingest",
        files={"file": ("a.exe", b"x", "application/octet-stream")},
        data={"category": "ai"},
    )
    assert resp.json()["code"] == 4001


def test_sources_ok():
    client = _client(repo=FakeRepo())
    resp = client.get("/api/v1/sources", params={"category": "ai"})
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["count"] == 1


def test_stats_ok():
    client = _client(stats=FakeStatsService())
    resp = client.get("/api/v1/stats", params={"days": 7})
    body = resp.json()
    assert body["code"] == 0
    assert body["data"]["total"] == 10


def test_health_ok(monkeypatch):
    import app.api.main as main

    monkeypatch.setattr(main, "_check_mysql", lambda: True)
    monkeypatch.setattr(main, "_check_redis", lambda: True)
    monkeypatch.setattr(main, "_check_vector", lambda: True)
    monkeypatch.setattr(main, "_check_llm", lambda: True)
    client = TestClient(app)
    body = client.get("/api/v1/health").json()
    assert body["code"] == 0
    assert body["data"]["mysql"] is True
