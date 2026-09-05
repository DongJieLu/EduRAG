"""生成器：拼接 Prompt（参考片段 + 问题）→ 调用 LLM → 解析 JSON 输出（含引用/拒答）。"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from app.llm import get_llm
from app.llm.base import BaseLLM, ChatMessage

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是 EduQA 课程问答助手。请严格依据用户提供的「参考片段」回答问题，"
    "不得使用片段之外的知识编造内容。回答正文中引用片段处需标注来源编号（如 [1]）。"
    "若参考片段不足以回答问题，请在 answer 中说明“未在资料中找到”，并将 needs_human 设为 true。"
    "最终只输出一个 JSON 对象，不要输出其他内容。"
)

OUTPUT_SCHEMA_HINT = (
    '{"answer":"正文，引用处附[来源N]","citations":[{"doc_name":"...","title":"...","text":"片段摘要"}],'
    '"confidence":0.0,"needs_human":false}'
)

STREAM_SYSTEM_PROMPT = (
    "你是 EduQA 课程问答助手。请严格依据用户提供的「参考片段」回答问题，"
    "不得使用片段之外的知识编造内容。回答正文中引用片段处需标注来源编号（如 [1]）。"
    "若参考片段不足以回答问题，请直接说明“未在资料中找到”。"
    "只输出回答正文本身，不要输出 JSON 或任何额外格式。"
)

REJECT_ANSWER = "抱歉，我暂时无法根据当前知识库回答这个问题。您可以尝试换个问法，或补充更多背景信息。"

NOT_FOUND_MARKERS = ("未在资料中找到", "未找到", "找不到", "无法回答")

MIN_EVIDENCE_SCORE = 0.35  # Top5 平均分低于此值视为无证据（拒答）


@dataclass
class GenerationResult:
    answer: str
    citations: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    needs_human: bool = False
    rejected: bool = False

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "citations": self.citations,
            "confidence": self.confidence,
            "needs_human": self.needs_human,
            "rejected": self.rejected,
        }


def _extract_json(text: str) -> dict | None:
    """从 LLM 输出中提取 JSON 对象，容忍代码块与前后杂质文本。"""
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:  # noqa: PERF203
        logger.warning("LLM 输出 JSON 解析失败: %s", exc)
        return None


def _to_float(value, default: float) -> float:
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return default


class Generator:
    def __init__(self, llm: BaseLLM | None = None) -> None:
        self._llm = llm or get_llm()

    @staticmethod
    def _format_refs(contexts: list[dict]) -> str:
        refs = []
        for i, ctx in enumerate(contexts, 1):
            doc_name = ctx.get("doc_name") or ""
            title = ctx.get("title") or ""
            category = ctx.get("category") or ""
            text = (ctx.get("text") or "").strip()
            refs.append(f"[{i}] ({doc_name}, {title}, {category})\n{text}")
        return "\n\n".join(refs) if refs else "（无参考片段）"

    @staticmethod
    def _format_history(history: list[ChatMessage] | None) -> str:
        if not history:
            return ""
        lines = []
        for m in history[-6:]:
            tag = "用户" if m.role == "user" else "助手"
            lines.append(f"{tag}: {m.content}")
        return "\n".join(lines)

    def build_prompt(self, question: str, contexts: list[dict], history: list[ChatMessage] | None = None) -> list[ChatMessage]:
        """contexts: [{doc_name, title, category, text}]；history: 最近若干轮 ChatMessage。"""
        ref_block = self._format_refs(contexts)
        history_block = self._format_history(history)
        parts = [f"参考片段：\n{ref_block}"]
        if history_block:
            parts.append(f"对话历史：\n{history_block}")
        parts.append(f"用户问题：{question}\n\n请按如下 JSON 格式输出：{OUTPUT_SCHEMA_HINT}")
        return [ChatMessage(role="system", content=SYSTEM_PROMPT), ChatMessage(role="user", content="\n\n".join(parts))]

    def _build_stream_prompt(self, question: str, contexts: list[dict], history: list[ChatMessage] | None = None) -> list[ChatMessage]:
        """流式用：只要求输出回答正文（带 [N] 引用），不要求 JSON。"""
        ref_block = self._format_refs(contexts)
        history_block = self._format_history(history)
        parts = [f"参考片段：\n{ref_block}"]
        if history_block:
            parts.append(f"对话历史：\n{history_block}")
        parts.append(f"用户问题：{question}")
        return [ChatMessage(role="system", content=STREAM_SYSTEM_PROMPT), ChatMessage(role="user", content="\n\n".join(parts))]

    def generate(self, question: str, contexts: list[dict], history: list[ChatMessage] | None = None) -> GenerationResult:
        messages = self.build_prompt(question, contexts, history)
        resp = self._llm.chat(messages, temperature=0.0)
        return self.parse_result(resp.content, contexts)

    def stream(self, question: str, contexts: list[dict], history: list[ChatMessage] | None = None):
        """流式产出回答正文 token；调用方自行累积后交给 parse_plain_result。"""
        messages = self._build_stream_prompt(question, contexts, history)
        yield from self._llm.stream(messages, temperature=0.0)

    def parse_plain_result(self, answer: str, contexts: list[dict], confidence: float = 0.0) -> GenerationResult:
        """解析流式输出（纯文本回答）：引用由 contexts 兜底推导，拒答靠关键词/空答。"""
        answer = (answer or "").strip()
        not_found = any(marker in answer for marker in NOT_FOUND_MARKERS)
        if not answer or not_found:
            return GenerationResult(
                answer=answer or REJECT_ANSWER,
                citations=[],
                confidence=confidence,
                needs_human=True,
                rejected=True,
            )
        citations = self._normalize_citations(None, contexts)
        return GenerationResult(
            answer=answer, citations=citations, confidence=confidence, needs_human=False, rejected=False
        )

    def parse_result(self, content: str, contexts: list[dict]) -> GenerationResult:
        data = _extract_json(content)
        if data is None:
            logger.warning("LLM 输出无法解析为 JSON，按拒答处理")
            return GenerationResult(
                answer=REJECT_ANSWER, citations=[], confidence=0.0, needs_human=True, rejected=True
            )
        answer = (data.get("answer") or "").strip()
        citations = self._normalize_citations(data.get("citations"), contexts)
        confidence = _to_float(data.get("confidence"), 0.5)
        needs_human = bool(data.get("needs_human", False))

        not_found = any(marker in answer for marker in NOT_FOUND_MARKERS)
        if not answer or needs_human or not_found:
            return GenerationResult(
                answer=answer or REJECT_ANSWER,
                citations=citations,
                confidence=confidence,
                needs_human=True,
                rejected=True,
            )
        return GenerationResult(
            answer=answer, citations=citations, confidence=confidence, needs_human=False, rejected=False
        )

    @staticmethod
    def _normalize_citations(raw, contexts: list[dict]) -> list[dict]:
        if isinstance(raw, list) and raw:
            out = []
            for item in raw:
                if isinstance(item, dict):
                    out.append(
                        {
                            "doc_name": item.get("doc_name") or "",
                            "title": item.get("title") or "",
                            "text": item.get("text") or "",
                        }
                    )
            if out:
                return out
        # 兜底：LLM 未返回引用时，直接用传入的片段构造引用
        return [
            {
                "doc_name": ctx.get("doc_name") or "",
                "title": ctx.get("title") or "",
                "text": (ctx.get("text") or "")[:200],
            }
            for ctx in contexts
        ]
