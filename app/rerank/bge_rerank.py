"""bge-reranker-large 重排器：本地 CrossEncoder，无 GPU 自动 CPU。"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import get_settings


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_settings().rerank_model_name
        self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """返回每个文档与 query 的相关性分数（越高越相关）。"""
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
