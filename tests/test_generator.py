"""生成器单元测试：JSON 解析、拒答条件、引用兜底、Prompt 拼接。"""
import json

from app.llm.base import LLMResponse
from app.rag.generator import (
    REJECT_ANSWER,
    Generator,
    _extract_json,
    _to_float,
)


class FakeLLM:
    def __init__(self, content: str = "", stream_tokens=None):
        self._content = content
        self._stream = stream_tokens or []

    def chat(self, messages, temperature=0.0, **kwargs):
        return LLMResponse(content=self._content, model="fake")

    def stream(self, messages, temperature=0.0, **kwargs):
        yield from self._stream


CONTEXTS = [
    {"doc_name": "a.md", "title": "标题A", "category": "ai", "text": "片段内容甲"},
    {"doc_name": "b.pdf", "title": "标题B", "category": "java", "text": "片段内容乙"},
]


def _answer_payload(answer="答案", needs_human=False, citations=None):
    return json.dumps(
        {
            "answer": answer,
            "citations": citations if citations is not None else [],
            "confidence": 0.8,
            "needs_human": needs_human,
        },
        ensure_ascii=False,
    )


# --- _extract_json ---

def test_extract_json_plain():
    assert _extract_json('{"a":1}') == {"a": 1}


def test_extract_json_markdown_fence():
    assert _extract_json('```json\n{"a":1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_text():
    assert _extract_json('前缀说明 {"a":1} 后缀') == {"a": 1}


def test_extract_json_empty_returns_none():
    assert _extract_json("") is None
    assert _extract_json("没有json") is None


def test_extract_json_invalid_returns_none():
    assert _extract_json("{broken") is None


# --- _to_float ---

def test_to_float_clamps():
    assert _to_float("0.7", 0.5) == 0.7
    assert _to_float("1.5", 0.5) == 1.0
    assert _to_float("-0.2", 0.5) == 0.0
    assert _to_float("bad", 0.5) == 0.5


# --- parse_result ---

def test_parse_result_valid_not_rejected():
    gen = Generator(llm=FakeLLM())
    result = gen.parse_result(_answer_payload(answer="这是答案"), CONTEXTS)
    assert result.answer == "这是答案"
    assert result.rejected is False
    assert result.needs_human is False


def test_parse_result_needs_human_rejected():
    gen = Generator(llm=FakeLLM())
    result = gen.parse_result(_answer_payload(answer="未在资料中找到", needs_human=True), CONTEXTS)
    assert result.rejected is True


def test_parse_result_not_found_marker_rejected():
    gen = Generator(llm=FakeLLM())
    result = gen.parse_result(_answer_payload(answer="很抱歉，未在资料中找到相关内容"), CONTEXTS)
    assert result.rejected is True


def test_parse_result_invalid_json_rejected():
    gen = Generator(llm=FakeLLM())
    result = gen.parse_result("这不是 JSON", CONTEXTS)
    assert result.rejected is True
    assert result.answer == REJECT_ANSWER


def test_parse_result_empty_answer_rejected():
    gen = Generator(llm=FakeLLM())
    result = gen.parse_result(_answer_payload(answer=""), CONTEXTS)
    assert result.rejected is True
    assert result.answer == REJECT_ANSWER


def test_normalize_citations_fallback_to_contexts():
    gen = Generator(llm=FakeLLM())
    citations = gen._normalize_citations(None, CONTEXTS)
    assert len(citations) == 2
    assert citations[0]["doc_name"] == "a.md"


def test_normalize_citations_uses_raw():
    gen = Generator(llm=FakeLLM())
    raw = [{"doc_name": "x", "title": "t", "text": "s"}]
    citations = gen._normalize_citations(raw, CONTEXTS)
    assert citations == raw


# --- build_prompt ---

def test_build_prompt_contains_question_and_refs():
    gen = Generator(llm=FakeLLM())
    messages = gen.build_prompt("什么是 RAG？", CONTEXTS)
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert "什么是 RAG？" in messages[1].content
    assert "[1] (a.md, 标题A, ai)" in messages[1].content
    assert "[2] (b.pdf, 标题B, java)" in messages[1].content


def test_stream_prompt_is_plain_text_without_json_hint():
    gen = Generator(llm=FakeLLM())
    messages = gen._build_stream_prompt("什么是 RAG？", CONTEXTS)
    assert "JSON" not in messages[1].content
    assert "[1] (a.md, 标题A, ai)" in messages[1].content


# --- parse_plain_result ---

def test_parse_plain_result_ok_derives_citations_from_contexts():
    gen = Generator(llm=FakeLLM())
    result = gen.parse_plain_result("这是基于[1]的回答", CONTEXTS, confidence=0.7)
    assert result.rejected is False
    assert result.answer == "这是基于[1]的回答"
    assert result.confidence == 0.7
    assert [c["doc_name"] for c in result.citations] == ["a.md", "b.pdf"]


def test_parse_plain_result_not_found_rejected():
    gen = Generator(llm=FakeLLM())
    result = gen.parse_plain_result("未在资料中找到相关内容", CONTEXTS)
    assert result.rejected is True


def test_parse_plain_result_empty_rejected():
    gen = Generator(llm=FakeLLM())
    result = gen.parse_plain_result("  ", CONTEXTS)
    assert result.rejected is True
    assert result.answer == REJECT_ANSWER
