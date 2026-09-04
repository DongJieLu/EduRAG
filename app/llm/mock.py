"""Mock LLM：无真实 key 时用于打通链路与跑测试。"""
from app.llm.base import BaseLLM, ChatMessage, LLMResponse


class MockLLM(BaseLLM):
    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        user_msg = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "",
        )
        return LLMResponse(
            content=f"[mock] 收到问题：{user_msg[:50]}",
            model="mock",
            usage={},
        )
