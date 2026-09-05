"""RAG 聊天编排单元测试：检索→重排→拒答/生成（mock 全部依赖）。"""
from app.llm.base import LLMResponse
from app.rag.chat_service import ChatService
from app.rag.generator import GenerationResult


class FakeRetriever:
    def __init__(self, hits):
        self._hits = hits

    def retrieve(self, query, category=None, top_k=50):
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


def _hit(doc_name="a", text="片段", score=0.8):
    return {"id": "1", "text": text, "metadata": {"doc_name": doc_name, "title": "t", "category": "ai"}, "score": score}


# --- 无证据拒答 ---

def test_chat_rejects_when_low_evidence():
    hits = [_hit(score=0.1), _hit(score=0.2)]
    svc = ChatService(
        retriever=FakeRetriever(hits),
        reranker=FakeReranker([0.1, 0.2]),
        generator=FakeGenerator(),
    )
    resp = svc.chat("问题")
    assert resp["rejected"] is True
    assert resp["citations"] == []


# --- 正常生成 ---

def test_chat_generates_when_evidence_sufficient():
    hits = [_hit(score=0.9)]
    svc = ChatService(
        retriever=FakeRetriever(hits),
        reranker=FakeReranker([0.9]),
        generator=FakeGenerator(result=GenerationResult(answer="正确答案", citations=[{"doc_name": "a"}])),
    )
    resp = svc.chat("问题")
    assert resp["rejected"] is False
    assert resp["answer"] == "正确答案"
    assert resp["intent"] == "rag"
    assert resp["strategy"] == "direct"


# --- 重排降级（无 reranker） ---

def test_rerank_fallback_to_vector_score_when_no_reranker():
    hits = [_hit(score=0.8), _hit(score=0.6)]
    svc = ChatService(retriever=FakeRetriever(hits), generator=FakeGenerator())
    svc._reranker = None
    svc._reranker_tried = True
    top = svc._rerank("q", hits)
    assert [s for _, s in top] == [0.8, 0.6]


# --- 流式事件 ---

def test_stream_chat_emits_events():
    hits = [_hit(score=0.9)]
    gen = FakeGenerator(stream_tokens=["你", "好"], result=GenerationResult(answer="你好", citations=[]))
    svc = ChatService(
        retriever=FakeRetriever(hits),
        reranker=FakeReranker([0.9]),
        generator=gen,
    )
    events = list(svc.stream_chat("问题"))
    types = [e["type"] for e in events]
    assert types[0] == "route"
    assert types[-1] == "done"
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert tokens == "你好"


def test_stream_chat_reject_emits_no_llm():
    hits = [_hit(score=0.1)]
    svc = ChatService(
        retriever=FakeRetriever(hits),
        reranker=FakeReranker([0.1]),
        generator=FakeGenerator(stream_tokens=["不", "该", "出", "现"]),
    )
    events = list(svc.stream_chat("问题"))
    types = [e["type"] for e in events]
    assert types == ["route", "token", "citation", "done"]
    assert events[-1]["rejected"] is True
