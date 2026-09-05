"""RAG 聊天服务编排：检索 → 重排 → 无证据拒答 → 生成。

M3 固定 intent=rag、strategy=direct；M4 在此前置 FAQ 通道与三层路由，M5 扩展策略引擎。
"""
from __future__ import annotations

import logging
import time

from app.rag.generator import MIN_EVIDENCE_SCORE, REJECT_ANSWER, Generator
from app.rag.retriever import Retriever
from app.rerank import get_reranker

logger = logging.getLogger(__name__)

TOP_K_RECALL = 50
TOP_K_RERANK = 5


class ChatService:
    def __init__(self, retriever=None, reranker=None, generator=None) -> None:
        self._retriever = retriever or Retriever()
        self._generator = generator or Generator()
        self._reranker = reranker
        self._reranker_tried = False

    def _get_reranker(self):
        if self._reranker is None and not self._reranker_tried:
            self._reranker_tried = True
            try:
                self._reranker = get_reranker()
            except Exception as exc:  # 重排模型不可用则降级为向量分
                logger.warning("重排模型加载失败，降级为向量分排序: %s", exc)
                self._reranker = None
        return self._reranker

    def _rerank(self, question: str, hits: list[dict], top_k: int = TOP_K_RERANK) -> list[tuple[dict, float]]:
        """返回 [(hit, score)]，score 为精排分（降序）。"""
        if not hits:
            return []
        reranker = self._get_reranker()
        if reranker is None:
            return [(h, h.get("score", 0.0)) for h in hits[:top_k]]
        docs = [h.get("text", "") for h in hits]
        scores = reranker.rerank(question, docs)
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    @staticmethod
    def _to_context(hit: dict, score: float) -> dict:
        meta = hit.get("metadata") or {}
        return {
            "doc_name": meta.get("doc_name", ""),
            "title": meta.get("title", ""),
            "category": meta.get("category", ""),
            "text": hit.get("text", ""),
            "score": round(float(score), 4),
        }

    def _build_reject(self, latency_ms: int, answer: str = REJECT_ANSWER) -> dict:
        return {
            "intent": "rag",
            "answer": answer,
            "citations": [],
            "strategy": "direct",
            "latency_ms": latency_ms,
            "confidence": 0.0,
            "rejected": True,
        }

    def chat(self, question: str, category: str | None = None, session_id: str | None = None) -> dict:
        start = time.perf_counter()
        hits = self._retriever.retrieve(question, category=category, top_k=TOP_K_RECALL)
        top = self._rerank(question, hits)
        if not top:
            return self._build_reject(self._elapsed(start))

        scores = [s for _, s in top]
        avg = sum(scores) / len(scores)
        if avg < MIN_EVIDENCE_SCORE:
            logger.info("检索证据不足（avg=%.3f），触发拒答", avg)
            return self._build_reject(self._elapsed(start))

        contexts = [self._to_context(h, s) for h, s in top]
        result = self._generator.generate(question, contexts)
        return {
            "intent": "rag",
            "answer": result.answer,
            "citations": result.citations,
            "strategy": "direct",
            "latency_ms": self._elapsed(start),
            "confidence": result.confidence,
            "rejected": result.rejected,
        }

    def stream_chat(self, question: str, category: str | None = None, session_id: str | None = None):
        """SSE 风格事件流：route / token / citation / done。"""
        start = time.perf_counter()
        hits = self._retriever.retrieve(question, category=category, top_k=TOP_K_RECALL)
        top = self._rerank(question, hits)
        yield {"type": "route", "intent": "rag", "strategy": "direct"}

        scores = [s for _, s in top]
        avg = sum(scores) / len(scores)
        if not top or avg < MIN_EVIDENCE_SCORE:
            yield {"type": "token", "content": REJECT_ANSWER}
            yield {"type": "citation", "citations": []}
            yield {"type": "done", "latency_ms": self._elapsed(start), "rejected": True}
            return

        contexts = [self._to_context(h, s) for h, s in top]
        buf: list[str] = []
        for token in self._generator.stream(question, contexts):
            buf.append(token)
            yield {"type": "token", "content": token}

        result = self._generator.parse_plain_result("".join(buf), contexts, confidence=avg)
        yield {"type": "citation", "citations": result.citations}
        yield {
            "type": "done",
            "latency_ms": self._elapsed(start),
            "rejected": result.rejected,
            "confidence": result.confidence,
        }

    @staticmethod
    def _elapsed(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
