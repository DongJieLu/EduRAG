"""LLM 客户端工厂：按 provider 返回实现，缺 key 时自动降级 mock。"""
import logging

from app.config import get_settings
from app.llm.base import BaseLLM
from app.llm.deepseek import DeepSeekLLM
from app.llm.mock import MockLLM

logger = logging.getLogger(__name__)


def get_llm() -> BaseLLM:
    provider = get_settings().llm_provider.lower()
    if provider == "deepseek":
        if not get_settings().deepseek_api_key:
            logger.warning("DEEPSEEK_API_KEY 未配置，降级为 MockLLM")
            return MockLLM()
        return DeepSeekLLM()
    if provider == "mock":
        return MockLLM()
    # dashscope / ollama 在后续里程碑接入
    raise ValueError(f"当前不支持的 LLM provider: {provider}")
