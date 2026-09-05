"""RAG 聊天编排单元测试：缓存→路由→FAQ/RAG/reject 分发（mock 全部依赖）。"""
from app.rag.chat_service import ChatService
from app.rag.generator import GenerationResult, REJECT_ANSWER


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, query, category=None, top_k=50):
        return self._hits

    def hybrid_retrieve(self, query, category=None, top_k=50):
        return self._hits


class FakeReranker:
    def __init__(self, scores):
        self._scores = scores

    def rerank(self, query, documents):
        return self._scores[: len(documents)]


class FakeGenerator:
    def __init__(self, result=None, stream_tokens=None):
        self._result = result or GenerationResult(answer="答案", citations=[])
        self._stream = stream_tokens or []

    def generate(self, question, contexts):
        return self._result

    def stream(self, question, contexts):
        yield from self._stream

    def parse_plain_result(self, content, contexts, confidence=0.0):
        return self._result


class FakeRouter:
    def __init__(self, intent="rag", confidence=0.8, reason="test"):
        self.intent = intent
        self.confidence = confidence
        self.reason = reason

    def route(self, question, category=None):
        return {"intent": self.intent, "confidence": self.confidence, "reason": self.reason}


class FakeStrategyEngine:
    def __init__(self, strategy="direct", queries=None):
        self.strategy = strategy
        self.queries = queries

    def plan(self, question, history=None):
        from app.rag.strategy import StrategyPlan

        return StrategyPlan(self.strategy, self.queries or [question], "test")


class FakeFAQService:
    def __init__(self, matched=None):
        self._matched = matched or []
        self.hit_increments = []

    def search(self, query, category=None, top_k=1):
        return self._matched

    def increment_hit_count(self, faq_id):
        self.hit_increments.append(faq_id)


class FakeCache:
    def __init__(self):
        self.store = {}

    def get(self, question, category):
        return self.store.get((question, category))

    def set(self, question, category, data, rejected=False):
        self.store[(question, category)] = data


def _hit(doc_name="a", text="片段", score=0.8):
    return {"id": "1", "text": text, "metadata": {"doc_name": doc_name, "title": "t", "category": "ai"}, "score": score}


def _svc(intent="rag", hits=None, scores=None, generator=None, faq=None, cache=None, strategy=None):
    return ChatService(
        retriever=FakeRetriever(hits if hits is not None else []),
        reranker=FakeReranker(scores if scores is not None else []),
        generator=generator or FakeGenerator(),
        router=FakeRouter(intent=intent),
        faq_service=faq or FakeFAQService(),
        cache=cache or FakeCache(),
        strategy_engine=strategy or FakeStrategyEngine(),
    )


# --- RAG 无证据拒答 ---

def test_chat_rejects_when_low_evidence():
    svc = _svc(hits=[_hit(score=0.1), _hit(score=0.2)], scores=[0.1, 0.2])
    resp = svc.chat("问题")
    assert resp["rejected"] is True
    assert resp["intent"] == "rag"
    assert resp["cache_hit"] is False


# --- RAG 正常生成 ---

def test_chat_generates_when_evidence_sufficient():
    gen = FakeGenerator(result=GenerationResult(answer="正确答案", citations=[{"doc_name": "a"}]))
    svc = _svc(hits=[_hit(score=0.9)], scores=[0.9], generator=gen)
    resp = svc.chat("问题")
    assert resp["rejected"] is False
    assert resp["answer"] == "正确答案"
    assert resp["intent"] == "rag"
    assert resp["strategy"] == "direct"


# --- FAQ 通道 ---

def test_chat_faq_route_returns_faq_answer():
    faq = FakeFAQService(matched=[{"faq_id": 1, "answer": "FAQ 答案", "score": 0.9}])
    svc = _svc(intent="faq", faq=faq)
    resp = svc.chat("什么是 RAG？")
    assert resp["intent"] == "faq"
    assert resp["answer"] == "FAQ 答案"
    assert resp["strategy"] == "faq"
    assert faq.hit_increments == [1]


def test_chat_faq_miss_falls_back_to_rag():
    gen = FakeGenerator(result=GenerationResult(answer="RAG 深度回答", citations=[]))
    svc = _svc(intent="faq", faq=FakeFAQService(matched=[]), hits=[_hit(score=0.9)], scores=[0.9], generator=gen)
    resp = svc.chat("问题")
    assert resp["intent"] == "rag"
    assert resp["answer"] == "RAG 深度回答"


# --- 拒答路由 ---

def test_chat_reject_route():
    svc = _svc(intent="reject")
    resp = svc.chat("你好")
    assert resp["intent"] == "reject"
    assert resp["rejected"] is True
    assert resp["answer"] == REJECT_ANSWER


# --- 缓存 ---

def test_chat_cache_hit_returns_cached():
    cache = FakeCache()
    cache.store[("问题", None)] = {"intent": "faq", "answer": "缓存答案", "rejected": False}
    svc = _svc(cache=cache)
    resp = svc.chat("问题")
    assert resp["cache_hit"] is True
    assert resp["answer"] == "缓存答案"


def test_chat_writes_cache_on_answer():
    cache = FakeCache()
    faq = FakeFAQService(matched=[{"faq_id": 1, "answer": "FAQ 答案", "score": 0.9}])
    svc = _svc(intent="faq", faq=faq, cache=cache)
    svc.chat("什么是 RAG？")
    assert ("什么是 RAG？", None) in cache.store


# --- 重排降级 ---

def test_rerank_fallback_to_vector_score_when_no_reranker():
    hits = [_hit(score=0.8), _hit(score=0.6)]
    svc = ChatService(retriever=FakeRetriever(hits), generator=FakeGenerator(), router=FakeRouter(), cache=FakeCache())
    svc._reranker = None
    svc._reranker_tried = True
    top = svc._rerank("q", hits)
    assert [s for _, s in top] == [0.8, 0.6]


# --- 流式事件 ---

def test_stream_chat_emits_events():
    gen = FakeGenerator(stream_tokens=["你", "好"], result=GenerationResult(answer="你好", citations=[]))
    svc = _svc(hits=[_hit(score=0.9)], scores=[0.9], generator=gen)
    events = list(svc.stream_chat("问题"))
    types = [e["type"] for e in events]
    assert types[0] == "route"
    assert types[-1] == "done"
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == "你好"


def test_stream_chat_reject_emits_no_llm():
    svc = _svc(hits=[_hit(score=0.1)], scores=[0.1])
    events = list(svc.stream_chat("问题"))
    types = [e["type"] for e in events]
    assert types == ["route", "token", "citation", "done"]
    assert events[-1]["rejected"] is True


def test_stream_chat_cache_hit():
    cache = FakeCache()
    cache.store[("问题", None)] = {"intent": "faq", "answer": "缓存答案", "citations": [], "rejected": False}
    svc = _svc(cache=cache)
    events = list(svc.stream_chat("问题"))
    assert events[-1]["cache_hit"] is True
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == "缓存答案"
