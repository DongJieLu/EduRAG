"""MySQL 元数据仓储：knowledge_doc / knowledge_chunk 的写入与查询（全部参数化 SQL）。"""
from __future__ import annotations

from sqlalchemy import text

from app.db.mysql import get_engine


class KnowledgeRepository:
    def __init__(self, engine=None) -> None:
        self._engine = engine or get_engine()

    def insert_document(self, file_name: str, category: str, file_type: str) -> int:
        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    "INSERT INTO knowledge_doc (file_name, category, file_type) "
                    "VALUES (:file_name, :category, :file_type)"
                ),
                {"file_name": file_name, "category": category, "file_type": file_type},
            )
            return result.lastrowid

    def insert_chunk(
        self,
        doc_id: int,
        doc_name: str,
        category: str,
        title: str,
        page_no: int | None,
        chunk_text: str,
        milvus_id: str,
    ) -> int:
        with self._engine.begin() as conn:
            result = conn.execute(
                text(
                    "INSERT INTO knowledge_chunk "
                    "(doc_id, doc_name, category, title, page_no, chunk_text, milvus_id) "
                    "VALUES (:doc_id, :doc_name, :category, :title, :page_no, :chunk_text, :milvus_id)"
                ),
                {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "category": category,
                    "title": title,
                    "page_no": page_no,
                    "chunk_text": chunk_text,
                    "milvus_id": milvus_id,
                },
            )
            return result.lastrowid

    def update_doc_chunk_count(self, doc_id: int, count: int) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("UPDATE knowledge_doc SET chunk_count = :count WHERE doc_id = :doc_id"),
                {"count": count, "doc_id": doc_id},
            )

    def list_documents(self, category: str | None = None) -> list[dict]:
        sql = (
            "SELECT doc_id, file_name, category, file_type, chunk_count, status, created_at "
            "FROM knowledge_doc WHERE status = 1"
        )
        params: dict = {}
        if category:
            sql += " AND category = :category"
            params["category"] = category
        sql += " ORDER BY doc_id DESC"
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params)
            return [dict(r._mapping) for r in rows]

    def delete_document(self, doc_id: int) -> list[str]:
        """删除文档及其 chunks，返回被删 chunk 的 milvus_id 列表（用于同步删向量）。"""
        with self._engine.begin() as conn:
            rows = conn.execute(
                text("SELECT milvus_id FROM knowledge_chunk WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            ).fetchall()
            milvus_ids = [r[0] for r in rows if r[0]]
            conn.execute(
                text("DELETE FROM knowledge_chunk WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            )
            conn.execute(
                text("UPDATE knowledge_doc SET status = 0 WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            )
            return milvus_ids
