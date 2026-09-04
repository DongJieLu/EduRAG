"""BGE-M3 编码器：本地 sentence-transformers，向量 1024 维，normalize 后入库。"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class BGEEncoder:
    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or get_settings().embed_model_name
        self._model = SentenceTransformer(self._model_name)

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        bs = batch_size or get_settings().embed_batch_size
        vectors = self._model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()


@lru_cache
def get_encoder() -> BGEEncoder:
    return BGEEncoder()
