"""入库编排：解析 → 分块 → 向量化 → Chroma + MySQL 落库，支持增量（先删后插）。"""
from __future__ import annotations

import uuid
from pathlib import Path

from app.embeddings import get_encoder
from app.ingest.chunker import chunk_document
from app.ingest.parser import parse_document
from app.ingest.repository import KnowledgeRepository
from app.rag.vector_store import VectorStore

MAX_FILE_SIZE = 20 * 1024 * 1024  # 单文件 ≤ 20MB
ALLOWED_TYPES = {"pdf", "docx", "txt", "md", "markdown"}


class IngestService:
    def __init__(self, repository=None, vector_store=None, encoder=None) -> None:
        self._repo = repository or KnowledgeRepository()
        self._store = vector_store or VectorStore()
        self._encoder = encoder or get_encoder()

    def ingest_file(
        self,
        file_path: Path,
        category: str,
        file_name: str | None = None,
    ) -> dict:
        """入库单个文档，返回 {"doc_id", "chunk_count"}。"""
        file_path = Path(file_path)
        file_type = file_path.suffix.lstrip(".").lower()
        if file_type not in ALLOWED_TYPES:
            raise ValueError(f"不支持的文件类型: {file_type}")
        if file_path.stat().st_size > MAX_FILE_SIZE:
            raise ValueError(f"文件超过大小限制 {MAX_FILE_SIZE // 1024 // 1024}MB")

        name = file_name or file_path.name
        parsed = parse_document(file_path, file_type)
        strategy = "markdown" if file_type in ("md", "markdown") else "recursive"
        chunks = chunk_document(parsed, strategy=strategy)
        if not chunks:
            raise ValueError("文档未提取到有效内容")

        doc_id = self._repo.insert_document(name, category, file_type)

        texts = [c.text for c in chunks]
        embeddings = self._encoder.encode(texts)

        ids: list[str] = []
        metadatas: list[dict] = []
        for c in chunks:
            vid = uuid.uuid4().hex
            chunk_id = self._repo.insert_chunk(
                doc_id=doc_id,
                doc_name=name,
                category=category,
                title=c.metadata.get("title") or "",
                page_no=c.metadata.get("page_no"),
                chunk_text=c.text,
                milvus_id=vid,
            )
            ids.append(vid)
            metadatas.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "doc_name": name,
                    "category": category,
                    "title": c.metadata.get("title") or "",
                    "page_no": c.metadata.get("page_no") or 0,  # Chroma metadata 不接受 None，无页码用 0
                }
            )

        self._store.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
        self._repo.update_doc_chunk_count(doc_id, len(chunks))
        self._invalidate_cache(category)
        return {"doc_id": doc_id, "chunk_count": len(chunks)}

    def delete_document(self, doc_id: int, category: str | None = None) -> None:
        """删除文档（MySQL chunks + doc 标记删除 + Chroma 向量）。"""
        milvus_ids = self._repo.delete_document(doc_id)
        if milvus_ids:
            self._store.delete_by_doc(doc_id)
        if category:
            self._invalidate_cache(category)

    def _invalidate_cache(self, category: str | None) -> None:
        """文档更新后清理该方向的答案缓存（qa:ans:{category}:*）。"""
        try:
            from app.rag.cache import AnswerCache

            AnswerCache().invalidate_category(category)
        except Exception as exc:  # noqa: BLE001
            # 缓存清理失败不应阻断入库主流程
            import logging

            logging.getLogger(__name__).warning("答案缓存清理失败: %s", exc)
