"""三层查询路由：L1 规则词 / L2 FAQ 语义相似度 / L3 LLM 分类。

输出 {intent: faq|rag|reject, confidence, reason}。
决策：L1 命中→faq；否则 L2≥0.90→faq；0.75~0.90 且 L3==faq→faq；
L3==reject 且 L2<0.75→reject；其余→rag。
"""
from __future__ import annotations

import logging

from app.llm import get_llm
from app.llm.base import BaseLLM, ChatMessage
from app.rag.faq import FAQService
from app.rag.generator import _extract_json, _to_float

logger = logging.getLogger(__name__)

# L1 规则词（技术问答领域的 faq 倾向词，避开第三方品牌/人名等字样）
# 注意：刻意排除「是什么」「如何」「原理」等宽泛词——它们也高频出现在深度/开放问题中，
# 会误判 complex 类问题为 faq，交由 L2/L3 兜底更准确。
FAQ_RULE_WORDS = ("什么是", "定义", "区别", "怎么用", "介绍", "概念", "有什么用")
CHITCHAT_WORDS = ("你好", "您好", "谢谢", "再见", "在吗", "嗨", "hello", "hi")
NEGATION_WORDS = ("不是", "不要", "没有", "并非", "难道", "别")

L2_FAQ_THRESHOLD = 0.90
L2_PENDING_THRESHOLD = 0.75

CLASSIFY_SYSTEM_PROMPT = (
    "你是查询意图分类器。将用户问题分为三类并只输出一个 JSON 对象：\n"
    "- faq：可被常见问答直接回答的定义/概念/区别类问题\n"
    "- rag：需要检索文档深度回答的复杂或开放问题\n"
    "- reject：寒暄、闲聊或与知识库完全无关的问题\n"
    '输出格式：{"intent":"faq|rag|reject","confidence":0.0,"reason":"简短理由"}'
)


class Router:
    def __init__(self, faq_service=None, llm=None) -> None:
        self._faq = faq_service or FAQService()
        self._llm = llm

    def _get_llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    @staticmethod
    def _rule_route(question: str) -> dict | None:
        q = question.strip()
        if len(q) <= 20 and any(w in q.lower() for w in CHITCHAT_WORDS):
            return {"intent": "reject", "confidence": 0.95, "reason": "寒暄/闲聊"}
        has_rule = any(w in q for w in FAQ_RULE_WORDS)
        has_neg = any(w in q for w in NEGATION_WORDS)
        if has_rule and not has_neg:
            return {"intent": "faq", "confidence": 0.95, "reason": "规则词命中"}
        return None

    def _llm_classify(self, question: str) -> dict:
        try:
            messages = [
                ChatMessage(role="system", content=CLASSIFY_SYSTEM_PROMPT),
                ChatMessage(role="user", content=question),
            ]
            resp = self._get_llm().chat(messages, temperature=0.0)
            data = _extract_json(resp.content)
            if data and data.get("intent") in ("faq", "rag", "reject"):
                return {
                    "intent": data["intent"],
                    "confidence": _to_float(data.get("confidence"), 0.5),
                    "reason": data.get("reason") or "",
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("L3 LLM 分类失败，默认 rag: %s", exc)
        return {"intent": "rag", "confidence": 0.5, "reason": "LLM 分类失败，默认 rag"}

    def route(self, question: str, category: str | None = None) -> dict:
        detail: dict = {"l1": None, "l2": None, "l3": None}
        if not question or not question.strip():
            detail["l1"] = {"hit": True, "intent": "reject", "reason": "空问题"}
            return {"intent": "reject", "confidence": 1.0, "reason": "空问题", "route_detail": detail}
        l1 = self._rule_route(question)
        if l1:
            detail["l1"] = {"hit": True, "intent": l1["intent"], "reason": l1["reason"]}
            l1["route_detail"] = detail
            return l1

        sim = self._faq.best_similarity(question, category)
        detail["l2"] = {"similarity": round(sim, 4)}

        if sim >= L2_FAQ_THRESHOLD:
            detail["l2"]["decision"] = "faq"
            return {"intent": "faq", "confidence": min(round(sim, 4), 0.95), "reason": "语义相似度≥0.90", "route_detail": detail}

        if sim >= L2_PENDING_THRESHOLD:
            detail["l2"]["decision"] = "pending"
            l3 = self._llm_classify(question)
            detail["l3"] = {"intent": l3["intent"], "confidence": l3["confidence"], "reason": l3["reason"]}
            if l3["intent"] == "faq":
                return {"intent": "faq", "confidence": l3["confidence"], "reason": "语义待定，LLM 确认 faq", "route_detail": detail}
            return {"intent": "rag", "confidence": l3["confidence"], "reason": "语义待定，非 faq 走 RAG", "route_detail": detail}

        l3 = self._llm_classify(question)
        detail["l3"] = {"intent": l3["intent"], "confidence": l3["confidence"], "reason": l3["reason"]}
        if l3["intent"] == "reject":
            return {"intent": "reject", "confidence": l3["confidence"], "reason": "语义低，LLM 判拒答", "route_detail": detail}
        return {"intent": "rag", "confidence": l3["confidence"], "reason": "默认 RAG", "route_detail": detail}
