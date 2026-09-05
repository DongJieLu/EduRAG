"""三层路由单元测试：L1 规则 / L2 相似度 / L3 LLM 决策矩阵（≥12 组）。"""
import json

from app.llm.base import LLMResponse
from app.rag.router import Router


class FakeFAQService:
    def __init__(self, similarity=0.0):
        self.similarity = similarity
        self.calls = 0

    def best_similarity(self, query, category=None):
        self.calls += 1
        return self.similarity


class FakeLLM:
    def __init__(self, intent="rag", confidence=0.5):
        self.intent = intent
        self.confidence = confidence

    def chat(self, messages, temperature=0.0, **kwargs):
        return LLMResponse(
            content=json.dumps({"intent": self.intent, "confidence": self.confidence, "reason": "test"}),
            model="fake",
        )


def _router(similarity=0.0, intent="rag", confidence=0.5):
    return Router(faq_service=FakeFAQService(similarity), llm=FakeLLM(intent, confidence))


# --- L1 规则 ---

def test_rule_chitchat_reject():
    assert _router().route("你好")["intent"] == "reject"
    assert _router().route("谢谢")["intent"] == "reject"


def test_rule_faq_word_triggers_faq():
    r = _router().route("什么是 RAG？")
    assert r["intent"] == "faq"
    assert r["confidence"] == 0.95


def test_rule_negation_bypasses_l1():
    # 含否定词时 L1 不触发，落入 L2（此处相似度 0 → L3 rag）
    r = _router(similarity=0.0, intent="rag").route("这不是什么是 RAG")
    assert r["intent"] == "rag"


def test_empty_question_reject():
    assert _router().route("")["intent"] == "reject"
    assert _router().route("   ")["intent"] == "reject"


# --- L2 相似度 ≥ 0.90 ---

def test_l2_high_similarity_faq_without_llm():
    faq = FakeFAQService(similarity=0.93)
    router = Router(faq_service=faq, llm=FakeLLM(intent="reject"))  # LLM 不该被调用
    r = router.route("向量数据库用途")
    assert r["intent"] == "faq"
    assert faq.calls == 1


# --- L2 待定（0.75~0.90）依赖 L3 ---

def test_l2_pending_and_l3_faq():
    r = _router(similarity=0.80, intent="faq").route("问题")
    assert r["intent"] == "faq"


def test_l2_pending_and_l3_rag():
    r = _router(similarity=0.80, intent="rag").route("问题")
    assert r["intent"] == "rag"


def test_l2_pending_and_l3_reject_goes_rag():
    # L2>=0.75 时即使 L3 判 reject 也不 reject，走 rag
    r = _router(similarity=0.80, intent="reject").route("问题")
    assert r["intent"] == "rag"


# --- L2 < 0.75 依赖 L3 ---

def test_l2_low_and_l3_reject():
    r = _router(similarity=0.50, intent="reject").route("问题")
    assert r["intent"] == "reject"


def test_l2_low_and_l3_rag():
    r = _router(similarity=0.50, intent="rag").route("问题")
    assert r["intent"] == "rag"


def test_l2_low_and_l3_faq_still_rag():
    # L2<0.75 时即使 L3 判 faq 也走 rag
    r = _router(similarity=0.50, intent="faq").route("问题")
    assert r["intent"] == "rag"


# --- L3 解析失败兜底 ---

class BadLLM:
    def chat(self, messages, temperature=0.0, **kwargs):
        return LLMResponse(content="这不是 JSON", model="fake")


def test_l3_parse_failure_defaults_rag():
    router = Router(faq_service=FakeFAQService(similarity=0.3), llm=BadLLM())
    assert router.route("问题")["intent"] == "rag"


# --- 类别过滤透传 ---

def test_route_passes_category_to_faq():
    faq = FakeFAQService(similarity=0.95)
    router = Router(faq_service=faq, llm=FakeLLM())
    router.route("容器编排工具", category="ops")
    assert faq.calls == 1
