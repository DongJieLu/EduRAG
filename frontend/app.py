"""Gradio 前端：对话（流式 + 引用） + 知识库管理。

用法: python frontend/app.py
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import gradio as gr
import pandas as pd

from app.ingest.repository import KnowledgeRepository
from app.ingest.service import IngestService
from app.rag.chat_service import ChatService

CATEGORIES = ["ai", "java", "test", "ops", "bigdata"]
CATEGORY_CHOICES = ["全部"] + CATEGORIES


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService()


@lru_cache
def get_ingest_service() -> IngestService:
    return IngestService()


def render_citations(citations: list[dict]) -> str:
    if not citations:
        return "（无引用来源）"
    lines = []
    for i, c in enumerate(citations, 1):
        doc_name = c.get("doc_name") or "未知文档"
        title = c.get("title") or ""
        text = (c.get("text") or "").strip()
        head = f"**[{i}] {doc_name}**" + (f" — {title}" if title else "")
        lines.append(f"{head}\n\n> {text}")
    return "\n\n---\n\n".join(lines)


def chat_respond(message: str, history: list, category: str):
    """流式对话：逐 token 更新助手气泡，结束后填充引用。"""
    history = history or []
    if not message or not message.strip():
        yield history, ""
        return
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]
    cat = None if category == "全部" else category
    citations_md = ""
    for event in get_chat_service().stream_chat(message, category=cat):
        etype = event.get("type")
        if etype == "token":
            history[-1]["content"] += event.get("content", "")
            yield history, citations_md
        elif etype == "citation":
            citations_md = render_citations(event.get("citations", []))
            yield history, citations_md
        elif etype == "done":
            if not history[-1]["content"]:
                history[-1]["content"] = "（无输出）"
            yield history, citations_md


def upload_file(file, category: str) -> str:
    if file is None:
        return "请先选择要上传的文件"
    if not category or category == "全部":
        return "请选择知识方向"
    try:
        result = get_ingest_service().ingest_file(Path(file), category)
        return f"入库成功：doc_id={result['doc_id']}，分块 {result['chunk_count']} 个"
    except Exception as exc:  # noqa: BLE001
        return f"入库失败：{exc}"


def list_documents(category: str):
    cat = None if category == "全部" else category
    docs = KnowledgeRepository().list_documents(cat)
    if not docs:
        return pd.DataFrame(columns=["doc_id", "file_name", "category", "file_type", "chunk_count", "created_at"])
    return pd.DataFrame(docs)


def delete_document(doc_id) -> str:
    if not doc_id:
        return "请输入要删除的 doc_id"
    try:
        get_ingest_service().delete_document(int(doc_id))
        return f"已删除 doc_id={doc_id}"
    except Exception as exc:  # noqa: BLE001
        return f"删除失败：{exc}"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="EduQA 课程问答助手") as demo:
        gr.Markdown("# EduQA 课程问答助手")
        gr.Markdown("FAQ 秒答 + 复杂问题 RAG 深度回答，回答可溯源。")

        with gr.Tab("对话"):
            chatbot = gr.Chatbot(height=480, label="对话")
            with gr.Row():
                category_dd = gr.Dropdown(CATEGORY_CHOICES, value="全部", label="知识方向")
                msg = gr.Textbox(placeholder="输入你的问题，回车发送", label="问题", scale=4)
                send_btn = gr.Button("发送")
            with gr.Accordion("引用来源", open=False):
                citations = gr.Markdown("（暂无引用）")
            clear_btn = gr.Button("清空对话")

            def _respond(m, h, c):
                yield from chat_respond(m, h, c)

            send_btn.click(_respond, [msg, chatbot, category_dd], [chatbot, citations])
            msg.submit(_respond, [msg, chatbot, category_dd], [chatbot, citations])
            clear_btn.click(lambda: ([], ""), None, [chatbot, citations])

        with gr.Tab("知识库"):
            gr.Markdown("上传文档（PDF / DOCX / TXT / MD）到指定知识方向，入库后可被检索。")
            with gr.Row():
                file_in = gr.File(label="选择文件", file_types=[".pdf", ".docx", ".txt", ".md"])
                kb_category = gr.Dropdown(CATEGORIES, value="ai", label="知识方向")
            upload_btn = gr.Button("上传入库")
            upload_out = gr.Textbox(label="结果", interactive=False)

            with gr.Row():
                refresh_btn = gr.Button("刷新文档列表")
                list_category = gr.Dropdown(CATEGORY_CHOICES, value="全部", label="筛选方向")
            docs_table = gr.Dataframe(label="文档列表", interactive=False)

            with gr.Row():
                del_id = gr.Number(label="doc_id", precision=0)
                del_btn = gr.Button("删除文档")
            del_out = gr.Textbox(label="删除结果", interactive=False)

            upload_btn.click(upload_file, [file_in, kb_category], upload_out)
            refresh_btn.click(list_documents, [list_category], docs_table)
            list_category.change(list_documents, [list_category], docs_table)
            del_btn.click(delete_document, [del_id], del_out)

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue()
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
