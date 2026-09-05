"""检索策略引擎：LLM 依据问题特征选择 direct / hyde / subquery / rewrite。

- direct：问题明确、术语完整，原问题直接向量检索
- hyde：术语少、口语化，LLM 生成假设性答案文档 → 用其向量检索
- subquery：复合问题（含"和/或者/对比"或多个主题），LLM 拆 2~3 个子问题各自检索
- rewrite：多轮追问（代词"它/这个/上面"），携带历史改写为独立问题

plan() 返回 StrategyPlan(strategy, queries, reason)，queries 为实际用于检索的 query 列表。
任何 LLM 调用或解析失败均回退 direct，保证链路不因策略层阻塞。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.llm import get_llm
from app.llm.base import BaseLLM, ChatMessage
from app.rag.generator import _extract_json

logger = logging.getLogger(__name__)

STRATEGY_SYSTEM_PROMPT = (
    "你是检索策略决策器。根据用户问题的特征，从以下四种策略中选择一种，并只输出一个 JSON 对象：\n"
    "- direct：问题明确、术语完整，直接用原问题检索\n"
    "- hyde：问题口语化、术语少（如：这个报错咋整），先生成一段假设性答案再检索\n"
    "- subquery：复合问题（含多个主题、并列或对比），拆成 2~3 个子问题分别检索\n"
    "- rewrite：多轮追问（含它/这个/上面等指代词），结合历史改写为独立问题\n"
    '输出格式：{"strategy":"direct|hyde|subquery|rewrite","queries":["子问题或改写后的问题"],"reason":"简短理由"}'
)

HYDE_SYSTEM_PROMPT = (
    "你是文档片段生成助手。根据用户问题，写一段像是从知识库文档中摘录出来的假设性回答文本，"
    "用于检索相似文档。只输出正文本身，不要解释、不要编号。"
)

VALID_STRATEGIES = ("direct", "hyde", "subquery", "rewrite")


@dataclass
class StrategyPlan:
    strategy: str
    queries: list[str] = field(default_factory=list)
    reason: str = ""


def _build_classify_user(question: str, history: list[ChatMessage] | None) -> str:
    if not history:
        return question
    lines = []
    for m in history[-6:]:
        tag = "用户" if m.role == "user" else "助手"
        lines.append(f"{tag}: {m.content}")
    lines.append(f"当前问题: {question}")
    return "\n".join(lines)


class StrategyEngine:
    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._llm = llm or get_llm()

    def plan(self, question: str, history: list[ChatMessage] | None = None) -> StrategyPlan:
        decision = self._select(question, history)
        strategy = decision["strategy"]
        reason = decision.get("reason", "")

        if strategy == "hyde":
            hypo = self._hypothesis(question)
            return StrategyPlan("hyde", [hypo] if hypo else [question], reason)
        if strategy == "subquery":
            queries = self._clean_queries(decision.get("queries")) or [question]
            return StrategyPlan("subquery", queries, reason)
        if strategy == "rewrite":
            queries = self._clean_queries(decision.get("queries")) or [question]
            return StrategyPlan("rewrite", queries, reason)
        # direct（含解析失败默认）
        return StrategyPlan("direct", [question], reason)

    def _select(self, question: str, history: list[ChatMessage] | None) -> dict:
        try:
            user = _build_classify_user(question, history)
            messages = [
                ChatMessage(role="system", content=STRATEGY_SYSTEM_PROMPT),
                ChatMessage(role="user", content=user),
            ]
            resp = self._llm.chat(messages, temperature=0.0)
            data = _extract_json(resp.content)
            if data and data.get("strategy") in VALID_STRATEGIES:
                return {
                    "strategy": data["strategy"],
                    "queries": data.get("queries") or [],
                    "reason": data.get("reason") or "",
                }
        except Exception as exc:  # noqa: BLE001
            logger.warning("策略决策 LLM 调用失败，默认 direct: %s", exc)
        return {"strategy": "direct", "queries": [question], "reason": "策略决策失败，默认 direct"}

    def _hypothesis(self, question: str) -> str:
        try:
            messages = [
                ChatMessage(role="system", content=HYDE_SYSTEM_PROMPT),
                ChatMessage(role="user", content=question),
            ]
            resp = self._llm.chat(messages, temperature=0.0)
            return (resp.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("HyDE 假设文档生成失败: %s", exc)
            return ""

    @staticmethod
    def _clean_queries(queries) -> list[str]:
        if not isinstance(queries, list):
            return []
        return [str(q).strip() for q in queries if str(q).strip()]
