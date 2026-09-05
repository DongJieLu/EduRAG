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
