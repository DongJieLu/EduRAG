"""API 请求模型（Pydantic）。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: str | None = Field(None, description="会话 id（客户端 UUID）")
    category: str | None = Field(None, description="知识方向 ai|java|test|ops|bigdata，空为全部")
