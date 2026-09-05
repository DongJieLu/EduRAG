"""EduQA FastAPI 服务：规格 5.10 API 契约（REST + SSE）。

统一响应包 {"code":0,"data":{...},"msg":"ok"}；
错误码：1001 参数错 / 2001 无证据 / 3001 服务内部错 / 4001 文档入库失败。

启动：uvicorn app.api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.deps import (
    get_chat_service,
    get_ingest_service,
    get_knowledge_repository,
    get_session_store,
    get_stats_service,
)
from app.api.schemas import ChatRequest
from app.ingest.service import ALLOWED_TYPES, MAX_FILE_SIZE

logger = logging.getLogger(__name__)

CATEGORIES = ("ai", "java", "test", "ops", "bigdata")

app = FastAPI(title="EduQA 课程问答助手 API", version="0.1.0")


# --- 统一响应 ---

def _resp(code: int, data, msg: str, status: int = 200) -> JSONResponse:
    return JSONResponse(status_code=status, content={"code": code, "data": data, "msg": msg})


def ok(data) -> JSONResponse:
    return _resp(0, data, "ok")


def fail(code: int, msg: str, status: int = 200) -> JSONResponse:
    return _resp(code, None, msg, status)


# --- 会话问答 ---

@app.post("/api/v1/chat")
def chat(req: ChatRequest, svc=Depends(get_chat_service), session=Depends(get_session_store)):
    question = req.question.strip()
    if not question:
        return fail(1001, "问题不能为空", 400)
    if len(question) > 2000:
        return fail(1001, "问题过长（>2000 字符）", 400)
    if req.category and req.category not in CATEGORIES:
        return fail(1001, f"不支持的 category: {req.category}", 400)

    lock_key = f"{req.category or 'all'}:{question}"
    if not session.acquire_lock(lock_key):
        return ok({
            "intent": "reject", "answer": "同一问题正在处理中，请稍后重试",
            "citations": [], "strategy": "", "latency_ms": 0, "busy": True,
        })
    try:
        result = svc.chat(question, category=req.category, session_id=req.session_id)
        return ok(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat 接口异常")
        return fail(3001, f"服务内部错误: {exc}", 500)
    finally:
        session.release_lock(lock_key)


@app.post("/api/v1/chat/stream")
def chat_stream(req: ChatRequest, svc=Depends(get_chat_service)):
    question = req.question.strip()
    if not question:
        return fail(1001, "问题不能为空", 400)
    if len(question) > 2000:
        return fail(1001, "问题过长（>2000 字符）", 400)
    if req.category and req.category not in CATEGORIES:
        return fail(1001, f"不支持的 category: {req.category}", 400)

    def gen():
        try:
            for event in svc.stream_chat(question, category=req.category, session_id=req.session_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream 接口异常")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- 文档入库 / 检索 ---

@app.post("/api/v1/ingest")
async def ingest(file: UploadFile = File(...), category: str = Form(...), svc=Depends(get_ingest_service)):
    if not category or category not in CATEGORIES:
        return fail(4001, f"不支持的 category: {category}", 400)
    suffix = Path(file.filename or "").suffix.lstrip(".").lower()
    if suffix not in ALLOWED_TYPES:
        return fail(4001, f"不支持的文件类型: {suffix or '未知'}", 400)

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return fail(4001, f"文件超过大小限制 {MAX_FILE_SIZE // 1024 // 1024}MB", 400)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        result = svc.ingest_file(tmp_path, category, file_name=file.filename)
        return ok(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("ingest 接口异常")
        return fail(4001, f"入库失败: {exc}", 400)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@app.get("/api/v1/sources")
def sources(category: str | None = None, repo=Depends(get_knowledge_repository)):
    try:
        docs = repo.list_documents(category or None)
        return ok({"documents": docs, "count": len(docs)})
    except Exception as exc:  # noqa: BLE001
        logger.exception("sources 接口异常")
        return fail(3001, f"服务内部错误: {exc}", 500)


# --- 统计 / 健康 ---

@app.get("/api/v1/stats")
def stats(days: int = 7, svc=Depends(get_stats_service)):
    try:
        return ok(svc.get_stats(days=days))
    except Exception as exc:  # noqa: BLE001
        logger.exception("stats 接口异常")
        return fail(3001, f"服务内部错误: {exc}", 500)


@app.get("/api/v1/health")
def health():
    checks = {
        "mysql": _check_mysql(),
        "redis": _check_redis(),
        "vector_store": _check_vector(),
        "llm": _check_llm(),
    }
    return ok(checks)


def _check_mysql() -> bool:
    try:
        from sqlalchemy import text

        from app.db.mysql import get_engine

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


def _check_redis() -> bool:
    try:
        from app.db.redis import get_redis_client

        return bool(get_redis_client().ping())
    except Exception:  # noqa: BLE001
        return False


def _check_vector() -> bool:
    try:
        from app.rag.vector_store import VectorStore

        VectorStore().count()
        return True
    except Exception:  # noqa: BLE001
        return False


def _check_llm() -> bool:
    try:
        from app.llm import get_llm

        get_llm()
        return True
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=True)
