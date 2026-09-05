"""LLM 客户端单元测试。"""
from app.config import get_settings
from app.llm import get_llm
from app.llm.base import ChatMessage
from app.llm.mock import MockLLM


def test_mock_llm_echoes_question():
    llm = MockLLM()
    resp = llm.chat([ChatMessage(role="user", content="什么是 RAG？")])
    assert "什么是 RAG" in resp.content
    assert resp.model == "mock"


def test_mock_llm_extracts_last_user_message():
    llm = MockLLM()
    messages = [
        ChatMessage(role="system", content="你是助手"),
        ChatMessage(role="user", content="第一个问题"),
        ChatMessage(role="assistant", content="回答"),
        ChatMessage(role="user", content="第二个问题"),
    ]
    resp = llm.chat(messages)
    assert "第二个问题" in resp.content


def test_get_llm_mock_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    assert isinstance(get_llm(), MockLLM)


def test_get_llm_deepseek_without_key_falls_back(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")  # 显式置空，覆盖 .env 中的真实 key
    get_settings.cache_clear()
    # 无 key 时应自动降级 mock，避免运行时报错
    assert isinstance(get_llm(), MockLLM)
