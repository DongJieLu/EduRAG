"""Chroma 向量存储封装（开发期）。

collection: eduqa_chunks
schema 与生产 Milvus 对齐：vector(1024) + chunk_id + category + 溯源字段(doc_id/doc_name/title/page_no)。
"""
from __future__ import annotations

from typing import Any

from app.config import get_settings

COLLECTION_NAME = "eduqa_chunks"


class VectorStore:
    def __init__(self) -> None:
        import chromadb

        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: Any,
        metadatas: list[dict],
    ) -> None:
        """批量写入。embeddings 接受 np.ndarray 或 list[list[float]]。"""
        emb_list = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
        self._collection.add(
            ids=ids,
            documents=texts,
            embeddings=emb_list,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: Any,
        top_k: int = 50,
        category: str | None = None,
    ) -> list[dict]:
        """向量检索，返回 [{id, text, metadata, score}]，score 为余弦相似度近似。"""
        where = {"category": category} if category else None
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        return [
            {
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i] or {},
                "score": 1.0 - float(dists[i]),  # cosine distance -> similarity
            }
            for i in range(len(docs))
        ]

    def delete_by_doc(self, doc_id: int) -> None:
        """按文档 id 删除该文档全部向量（配合增量入库的先删后插）。"""
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._collection.count()
