"""检索器单元测试：query 编码 + 向量召回（mock 依赖）。"""
import numpy as np

from app.rag.retriever import Retriever


class FakeEncoder:
    def encode(self, texts):
        return np.zeros((len(texts), 3), dtype="float32")


class FakeStore:
    def __init__(self):
        self.calls = []

    def search(self, query_embedding, top_k=50, category=None):
        self.calls.append({"top_k": top_k, "category": category})
        return [{"id": "1", "text": "片段", "metadata": {"doc_name": "a"}, "score": 0.9}]


def test_retrieve_empty_query_returns_empty():
    r = Retriever(vector_store=FakeStore(), encoder=FakeEncoder())
    assert r.retrieve("   ") == []


def test_retrieve_calls_store_with_category():
    store = FakeStore()
    r = Retriever(vector_store=store, encoder=FakeEncoder())
    results = r.retrieve("什么是 RAG", category="ai")
    assert len(results) == 1
    assert store.calls[0]["top_k"] == 50
    assert store.calls[0]["category"] == "ai"


def test_retrieve_passes_none_category():
    store = FakeStore()
    r = Retriever(vector_store=store, encoder=FakeEncoder())
    r.retrieve("什么是 RAG", category=None)
    assert store.calls[0]["category"] is None


# --- 混合检索（M5）---

from app.rag.retriever import _keyword_score  # noqa: E402


class ConfigStore:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query_embedding, top_k=50, category=None):
        return self._hits


class FakeRepo:
    def __init__(self, chunks):
        self._chunks = chunks

    def list_chunks(self, category=None):
        if category:
            return [c for c in self._chunks if c["category"] == category]
        return self._chunks


def _chunk(cid, text, category="ai"):
    return {
        "chunk_id": cid,
        "doc_id": cid,
        "doc_name": f"d{cid}",
        "category": category,
        "title": "",
        "page_no": 0,
        "chunk_text": text,
    }


def test_keyword_score_cjk_and_ascii():
    assert _keyword_score("RAG 工作流程", "RAG 工作流程包含解析与分块") > 0
    assert _keyword_score("完全无关词", "RAG 工作流程") == 0


def test_keyword_retrieve_filters_hits():
    repo = FakeRepo([_chunk(1, "RAG 检索增强生成"), _chunk(2, "向量数据库存储高维向量")])
    r = Retriever(vector_store=ConfigStore([]), encoder=FakeEncoder(), repository=repo)
    hits = r.keyword_retrieve("RAG")
    assert len(hits) == 1
    assert hits[0]["metadata"]["chunk_id"] == 1


def test_hybrid_retrieve_merges_keyword_only_hit():
    repo = FakeRepo([_chunk(1, "RAG 检索增强生成"), _chunk(2, "向量数据库存储高维向量 embedding")])
    store = ConfigStore([{"id": "u1", "text": "RAG 检索增强生成", "metadata": {"chunk_id": 1}, "score": 0.9}])
    r = Retriever(vector_store=store, encoder=FakeEncoder(), repository=repo)
    hits = r.hybrid_retrieve("向量数据库")
    ids = {h["metadata"]["chunk_id"] for h in hits}
    assert ids == {1, 2}


def test_hybrid_retrieve_falls_back_to_vector_when_no_keyword():
    store = ConfigStore([{"id": "u1", "text": "x", "metadata": {"chunk_id": 1}, "score": 0.9}])
    r = Retriever(vector_store=store, encoder=FakeEncoder(), repository=FakeRepo([]))
    hits = r.hybrid_retrieve("任意")
    assert len(hits) == 1
    assert hits[0]["metadata"]["chunk_id"] == 1


def test_hybrid_retrieve_dedupes_by_chunk_id():
    repo = FakeRepo([_chunk(1, "RAG 检索增强生成")])
    store = ConfigStore([{"id": "u1", "text": "RAG 检索增强生成", "metadata": {"chunk_id": 1}, "score": 0.9}])
    r = Retriever(vector_store=store, encoder=FakeEncoder(), repository=repo)
    hits = r.hybrid_retrieve("RAG")
    assert len(hits) == 1
    assert hits[0]["metadata"]["chunk_id"] == 1
