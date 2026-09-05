"""FAQ 通道单元测试：关键词命中、融合打分、语义相似度、检索命中/未命中。"""
import numpy as np

from app.rag.faq import FAQService, _cosine, fusion, keyword_score


class FakeRepo:
    def __init__(self, faqs):
        self._faqs = faqs

    def get_faqs(self, category=None):
        if category:
            return [f for f in self._faqs if f["category"] == category]
        return self._faqs

    def increment_hit_count(self, faq_id):
        pass


class FakeEncoder:
    def __init__(self, emb_map):
        self._map = emb_map

    def encode(self, texts):
        return np.array([self._map[t] for t in texts], dtype="float32")


def test_keyword_score_ascii_overlap():
    faq = {"question": "什么是 RAG？", "keywords": "rag 检索增强生成"}
    assert keyword_score("什么是 RAG？", faq) == 1
    assert keyword_score("rag 是什么意思", faq) == 1


def test_keyword_score_multi_hit():
    faq = {"question": "HashMap 与 Hashtable 的区别？", "keywords": "hashmap hashtable 区别"}
    assert keyword_score("HashMap 和 Hashtable 区别", faq) == 2


def test_keyword_score_no_ascii_zero():
    faq = {"question": "什么是多态？", "keywords": "多态 面向对象"}
    assert keyword_score("什么是多态", faq) == 0


def test_fusion_formula():
    assert fusion(1, 0.9) == 0.6 * 1 + 0.4 * 0.9
    assert fusion(3, 0.8) == 0.6 * 1 + 0.4 * 0.8  # 命中数封顶 1


def test_cosine_orthogonal():
    assert _cosine(np.array([1, 0, 0]), np.array([0, 1, 0])) == 0.0


def test_cosine_identical():
    assert abs(_cosine(np.array([1, 0, 0]), np.array([1, 0, 0])) - 1.0) < 1e-6


def _make_service():
    faqs = [
        {"faq_id": 1, "question": "什么是 RAG？", "keywords": "rag 检索增强生成", "answer": "A1", "category": "ai"},
        {"faq_id": 2, "question": "什么是 Docker？", "keywords": "docker 容器", "answer": "A2", "category": "ops"},
    ]
    emb_map = {
        "什么是 RAG？": np.array([1, 0, 0], dtype="float32"),
        "什么是 Docker？": np.array([0, 1, 0], dtype="float32"),
        "rag 是什么": np.array([0.99, 0.0, 0.0], dtype="float32"),
        "今天天气": np.array([0.0, 0.0, 1.0], dtype="float32"),
    }
    return FAQService(repository=FakeRepo(faqs), encoder=FakeEncoder(emb_map))


def test_search_returns_match():
    svc = _make_service()
    result = svc.search("rag 是什么")
    assert len(result) == 1
    assert result[0]["faq_id"] == 1


def test_search_miss_when_unrelated():
    svc = _make_service()
    assert svc.search("今天天气") == []


def test_search_category_filter():
    svc = _make_service()
    # "rag 是什么" 语义接近 RAG(ai)，但限定 ops 后不应命中
    assert svc.search("rag 是什么", category="ops") == []


def test_best_similarity():
    svc = _make_service()
    assert svc.best_similarity("rag 是什么") > 0.9
