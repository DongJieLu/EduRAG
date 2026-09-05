"""检索策略引擎单元测试：四种策略决策 + 解析失败兜底 direct。"""
import json

from app.llm.base import LLMResponse
from app.rag.strategy import StrategyEngine, StrategyPlan


class FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat(self, messages, temperature=0.0, **kwargs):
        self.calls += 1
        content = self._responses.pop(0) if self._responses else "{}"
        return LLMResponse(content=content, model="fake")


class RaisingLLM:
    def chat(self, messages, temperature=0.0, **kwargs):
        raise RuntimeError("LLM 挂了")


def _engine(llm):
    return StrategyEngine(llm=llm)


def test_direct_uses_original_question():
    llm = FakeLLM([json.dumps({"strategy": "direct", "queries": [], "reason": "明确"})])
    plan = _engine(llm).plan("什么是 RAG 的工作流程？")
    assert plan.strategy == "direct"
    assert plan.queries == ["什么是 RAG 的工作流程？"]


def test_hyde_generates_hypothesis_document():
    llm = FakeLLM([
        json.dumps({"strategy": "hyde", "reason": "口语化"}),
        "RAG 工作流程包括解析、分块、向量化、检索、重排、生成等步骤。",
    ])
    plan = _engine(llm).plan("这个报错咋整")
    assert plan.strategy == "hyde"
    assert plan.queries == ["RAG 工作流程包括解析、分块、向量化、检索、重排、生成等步骤。"]


def test_subquery_uses_split_queries():
    llm = FakeLLM([
        json.dumps({"strategy": "subquery", "queries": ["什么是 RAG", "什么是向量数据库"], "reason": "复合"}),
    ])
    plan = _engine(llm).plan("RAG 和向量数据库的区别")
    assert plan.strategy == "subquery"
    assert plan.queries == ["什么是 RAG", "什么是向量数据库"]


def test_rewrite_uses_rewritten_query():
    llm = FakeLLM([
        json.dumps({"strategy": "rewrite", "queries": ["Docker 容器的网络模式有哪些"], "reason": "指代"}),
    ])
    plan = _engine(llm).plan("它的网络模式呢")
    assert plan.strategy == "rewrite"
    assert plan.queries == ["Docker 容器的网络模式有哪些"]


def test_parse_failure_defaults_direct():
    plan = _engine(FakeLLM(["这不是 JSON"])).plan("问题")
    assert plan.strategy == "direct"
    assert plan.queries == ["问题"]


def test_invalid_strategy_defaults_direct():
    plan = _engine(FakeLLM([json.dumps({"strategy": "unknown"})])).plan("问题")
    assert plan.strategy == "direct"
    assert plan.queries == ["问题"]


def test_llm_exception_defaults_direct():
    plan = _engine(RaisingLLM()).plan("问题")
    assert plan.strategy == "direct"
    assert plan.queries == ["问题"]


def test_empty_subquery_falls_back_to_question():
    llm = FakeLLM([json.dumps({"strategy": "subquery", "queries": []})])
    plan = _engine(llm).plan("问题")
    assert plan.strategy == "subquery"
    assert plan.queries == ["问题"]


def test_plan_result_is_dataclass():
    plan = StrategyPlan("direct", ["q"], "r")
    assert plan.strategy == "direct"
    assert plan.queries == ["q"]
    assert plan.reason == "r"
