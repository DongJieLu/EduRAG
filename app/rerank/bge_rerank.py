"""bge-reranker-large 重排器：本地 CrossEncoder，无 GPU 自动 CPU。"""
from functools import lru_cache

import numpy as np

from app.config import get_settings  # 先加载 .env（HF_ENDPOINT 等）再 import 模型库
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_settings().rerank_model_name
        self._model = CrossEncoder(self._model_name)

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """返回每个文档与 query 的相关性分数（sigmoid 归一化到 0~1，越高越相关）。"""
        pairs = [[query, doc] for doc in documents]
        scores = self._model.predict(pairs)
        scores = np.asarray(scores, dtype="float32")
        probs = 1.0 / (1.0 + np.exp(-scores))  # sigmoid 归一化，便于与阈值比较
        return [float(p) for p in probs]


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
