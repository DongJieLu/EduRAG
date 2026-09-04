"""DeepSeek 实现：走 OpenAI 兼容接口。"""
from openai import OpenAI

from app.config import get_settings
from app.llm.base import BaseLLM, ChatMessage, LLMResponse


class DeepSeekLLM(BaseLLM):
    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=settings.llm_timeout_sec,
        )
        self._model = settings.deepseek_model

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.0,
        **kwargs,
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            **kwargs,
        )
        choice = resp.choices[0].message
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0),
        }
        return LLMResponse(content=choice.content or "", model=resp.model, usage=usage)
