"""RAG 聊天服务编排：缓存 → 路由 → FAQ/RAG/reject 分发 → 写缓存 + 落库 qa_log。

M4 增加 FAQ 通道 + 三层路由 + 答案缓存；M5 扩展策略引擎；
M6 增加会话历史（多轮改写）、FAQ 热点榜、qa_log 落库。
"""
from __future__ import annotations

import logging
import time

from app.rag.cache import AnswerCache
from app.rag.faq import FAQService
from app.rag.generator import MIN_EVIDENCE_SCORE, REJECT_ANSWER, Generator
from app.rag.retriever import Retriever
from app.rag.router import Router
from app.rag.strategy import StrategyEngine, StrategyPlan
from app.rerank import get_reranker

logger = logging.getLogger(__name__)

TOP_K_RECALL = 50
TOP_K_RERANK = 5


class ChatService:
    def __init__(
        self,
        retriever=None,
        reranker=None,
        generator=None,
        router=None,
        faq_service=None,
        cache=None,
        strategy_engine=None,
        hybrid: bool = True,
        session_store=None,
        qa_log=None,
    ) -> None:
        self._retriever = retriever or Retriever()
        self._generator = generator or Generator()
        self._reranker = reranker
        self._reranker_tried = False
        self._faq = faq_service or FAQService()
        self._router = router or Router(faq_service=self._faq)
        self._cache = cache or AnswerCache()
        self._strategy = strategy_engine or StrategyEngine()
        self._hybrid = hybrid
        self._session = session_store
        self._qa_log = qa_log

    # --- 重排 ---

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
            "chunk_id": meta.get("chunk_id"),
            "text": hit.get("text", ""),
            "score": round(float(score), 4),
        }

    @staticmethod
    def _reject_response(intent: str, latency_ms: int, answer: str = REJECT_ANSWER) -> dict:
        return {
            "intent": intent,
            "answer": answer,
            "citations": [],
            "strategy": "",
            "latency_ms": latency_ms,
            "confidence": 0.0,
            "rejected": True,
            "evidence_ids": [],
        }

    # --- 会话 / 日志辅助 ---

    def _load_history(self, session_id: str | None) -> list:
        if not session_id or self._session is None:
            return []
        try:
            return self._session.get_history(session_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话历史读取失败: %s", exc)
            return []

    def _remember(self, session_id: str | None, question: str, answer: str) -> None:
        if not session_id or self._session is None:
            return
        try:
            self._session.append(session_id, question, answer)
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话历史写入失败: %s", exc)

    def _bump_hot(self, faq_question: str) -> None:
        if self._session is None:
            return
        try:
            self._session.increment_hot(faq_question)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAQ 热点计数失败: %s", exc)

    def _record_log(
        self,
        session_id: str | None,
        question: str,
        intent: str,
        strategy: str,
        route_detail: dict,
        evidence_ids: list,
        answer: str,
        latency_ms: int,
        cache_hit: bool,
    ) -> None:
        if self._qa_log is None:
            return
        try:
            self._qa_log.insert_log(
                session_id=session_id or "",
                question=question,
                intent=intent,
                strategy=strategy,
                route_detail=route_detail,
                evidence_ids=evidence_ids,
                answer=answer,
                latency_ms=latency_ms,
                cache_hit=cache_hit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("qa_log 写入失败: %s", exc)

    # --- 非流式 ---

    def chat(self, question: str, category: str | None = None, session_id: str | None = None) -> dict:
        start = time.perf_counter()
        cached = self._cache.get(question, category)
        if cached is not None:
            cached["cache_hit"] = True
            cached["latency_ms"] = self._elapsed(start)
            self._record_log(
                session_id, question,
                cached.get("intent", "rag"), cached.get("strategy", ""), {},
                cached.get("evidence_ids", []), cached.get("answer", ""),
                cached["latency_ms"], True,
            )
            return cached

        history = self._load_history(session_id)
        route = self._router.route(question, category)
        result = self._dispatch(question, category, start, route, history)
        result["cache_hit"] = False
        self._cache.set(question, category, result, rejected=result.get("rejected", False))
        self._remember(session_id, question, result.get("answer", ""))
        self._record_log(
            session_id, question,
            result.get("intent", "rag"), result.get("strategy", ""),
            route.get("route_detail", {}), result.get("evidence_ids", []),
            result.get("answer", ""), result.get("latency_ms", 0), False,
        )
        return result

    def _dispatch(self, question: str, category: str | None, start: float, route: dict, history: list) -> dict:
        if route["intent"] == "reject":
            return self._reject_response("reject", self._elapsed(start))
        if route["intent"] == "faq":
            faq_result = self._answer_faq(question, category, start)
            if faq_result is not None:
                return faq_result
            logger.info("FAQ 未命中，降级 RAG")
        return self._answer_rag(question, category, start, history)

    def _answer_faq(self, question: str, category: str | None, start: float) -> dict | None:
        matched = self._faq.search(question, category, top_k=1)
        if not matched:
            return None
        best = matched[0]
        try:
            self._faq.increment_hit_count(best["faq_id"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAQ hit_count 更新失败: %s", exc)
        self._bump_hot(best.get("question") or question)
        return {
            "intent": "faq",
            "answer": best["answer"],
            "citations": [],
            "strategy": "faq",
            "latency_ms": self._elapsed(start),
            "confidence": best["score"],
            "rejected": False,
            "evidence_ids": [best["faq_id"]],
        }

    def _answer_rag(self, question: str, category: str | None, start: float, history: list) -> dict:
        plan = self._strategy.plan(question, history)
        hits = self._retrieve_for_plan(plan, category)
        top = self._rerank(question, hits)
        if not top:
            return self._reject_response("rag", self._elapsed(start))

        scores = [s for _, s in top]
        avg = sum(scores) / len(scores)
        if avg < MIN_EVIDENCE_SCORE:
            logger.info("检索证据不足（avg=%.3f），触发拒答", avg)
            return self._reject_response("rag", self._elapsed(start))

        contexts = [self._to_context(h, s) for h, s in top]
        evidence_ids = [str(c["chunk_id"]) for c in contexts if c.get("chunk_id") is not None]
        result = self._generator.generate(question, contexts, history=history)
        return {
            "intent": "rag",
            "answer": result.answer,
            "citations": result.citations,
            "strategy": plan.strategy,
            "latency_ms": self._elapsed(start),
            "confidence": result.confidence,
            "rejected": result.rejected,
            "evidence_ids": evidence_ids,
        }

    def _retrieve_for_plan(self, plan: StrategyPlan, category: str | None) -> list[dict]:
        """按策略产出的 query 列表检索，多个 query 的结果按 chunk_id 去重合并。"""
        seen: dict[str, dict] = {}
        for q in plan.queries:
            if self._hybrid:
                hits = self._retriever.hybrid_retrieve(q, category=category, top_k=TOP_K_RECALL)
            else:
                hits = self._retriever.retrieve(q, category=category, top_k=TOP_K_RECALL)
            for h in hits:
                key = str((h.get("metadata") or {}).get("chunk_id") or h.get("id"))
                if key not in seen:
                    seen[key] = h
        return list(seen.values())

    # --- 流式 ---

    def stream_chat(self, question: str, category: str | None = None, session_id: str | None = None):
        """SSE 风格事件流：route / token / citation / done。"""
        start = time.perf_counter()
        cached = self._cache.get(question, category)
        if cached is not None:
            yield {"type": "route", "intent": cached.get("intent", "rag"), "strategy": cached.get("strategy", "")}
            yield {"type": "token", "content": cached.get("answer", "")}
            yield {"type": "citation", "citations": cached.get("citations", [])}
            yield {
                "type": "done",
                "latency_ms": self._elapsed(start),
                "rejected": cached.get("rejected", False),
                "cache_hit": True,
            }
            self._record_log(
                session_id, question,
                cached.get("intent", "rag"), cached.get("strategy", ""), {},
                cached.get("evidence_ids", []), cached.get("answer", ""),
                self._elapsed(start), True,
            )
            return

        history = self._load_history(session_id)
        route = self._router.route(question, category)
        route_detail = route.get("route_detail", {})

        if route["intent"] == "reject":
            yield {"type": "route", "intent": "reject", "strategy": ""}
            yield {"type": "token", "content": REJECT_ANSWER}
            yield {"type": "citation", "citations": []}
            yield {"type": "done", "latency_ms": self._elapsed(start), "rejected": True, "cache_hit": False}
            self._record_log(
                session_id, question, "reject", "", route_detail, [],
                REJECT_ANSWER, self._elapsed(start), False,
            )
            self._remember(session_id, question, REJECT_ANSWER)
            return

        if route["intent"] == "faq":
            matched = self._faq.search(question, category, top_k=1)
            if matched:
                best = matched[0]
                try:
                    self._faq.increment_hit_count(best["faq_id"])
                except Exception:  # noqa: BLE001
                    pass
                self._bump_hot(best.get("question") or question)
                self._cache.set(question, category, {
                    "intent": "faq", "answer": best["answer"], "citations": [],
                    "strategy": "faq", "latency_ms": 0, "confidence": best["score"],
                    "rejected": False, "evidence_ids": [best["faq_id"]],
                })
                yield {"type": "route", "intent": "faq", "strategy": "faq"}
                yield {"type": "token", "content": best["answer"]}
                yield {"type": "citation", "citations": []}
                yield {"type": "done", "latency_ms": self._elapsed(start), "rejected": False, "cache_hit": False}
                self._record_log(
                    session_id, question, "faq", "faq", route_detail, [best["faq_id"]],
                    best["answer"], self._elapsed(start), False,
                )
                self._remember(session_id, question, best["answer"])
                return

        # RAG 流式：先做策略决策（决定检索 query），再检索
        plan = self._strategy.plan(question, history)
        yield {"type": "route", "intent": "rag", "strategy": plan.strategy}

        hits = self._retrieve_for_plan(plan, category)
        top = self._rerank(question, hits)
        scores = [s for _, s in top]
        avg = sum(scores) / len(scores) if scores else 0.0
        if not top or avg < MIN_EVIDENCE_SCORE:
            yield {"type": "token", "content": REJECT_ANSWER}
            yield {"type": "citation", "citations": []}
            yield {"type": "done", "latency_ms": self._elapsed(start), "rejected": True, "cache_hit": False}
            self._record_log(
                session_id, question, "rag", plan.strategy, route_detail, [],
                REJECT_ANSWER, self._elapsed(start), False,
            )
            self._remember(session_id, question, REJECT_ANSWER)
            return

        contexts = [self._to_context(h, s) for h, s in top]
        evidence_ids = [str(c["chunk_id"]) for c in contexts if c.get("chunk_id") is not None]
        buf: list[str] = []
        for token in self._generator.stream(question, contexts, history=history):
            buf.append(token)
            yield {"type": "token", "content": token}

        result = self._generator.parse_plain_result("".join(buf), contexts, confidence=avg)
        final = {
            "intent": "rag",
            "answer": result.answer,
            "citations": result.citations,
            "strategy": plan.strategy,
            "latency_ms": self._elapsed(start),
            "confidence": result.confidence,
            "rejected": result.rejected,
            "evidence_ids": evidence_ids,
        }
        self._cache.set(question, category, final, rejected=result.rejected)
        yield {"type": "citation", "citations": result.citations}
        yield {
            "type": "done",
            "latency_ms": self._elapsed(start),
            "rejected": result.rejected,
            "confidence": result.confidence,
            "cache_hit": False,
        }
        self._record_log(
            session_id, question, "rag", plan.strategy, route_detail, evidence_ids,
            result.answer, final["latency_ms"], False,
        )
        self._remember(session_id, question, result.answer)

    @staticmethod
    def _elapsed(start: float) -> int:
        return int((time.perf_counter() - start) * 1000)
