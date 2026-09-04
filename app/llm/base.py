"""LLM 客户端抽象基类与通用数据结构。"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str  # system | user | assistant
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str = ""
    usage: dict = field(default_factory=dict)


class BaseLLM(ABC):
    """所有 LLM 实现的统一接口，便于 provider 切换与 mock。"""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        """发送多轮消息，返回模型回复。"""
        raise NotImplementedError
