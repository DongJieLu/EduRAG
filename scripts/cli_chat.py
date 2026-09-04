"""CLI 一问一答：验证 LLM 连通。用法: python scripts/cli_chat.py "问题"。"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows 终端默认 GBK，强制 UTF-8 输出避免中文乱码
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.config import get_settings
from app.llm import get_llm
from app.llm.base import ChatMessage


def main() -> None:
    parser = argparse.ArgumentParser(description="EduQA CLI 一问一答")
    parser.add_argument("question", nargs="?", help="要提问的内容")
    args = parser.parse_args()

    logging.basicConfig(level=get_settings().log_level.upper())

    question = args.question
    if not question:
        question = input("请输入问题: ").strip()
    if not question:
        print("问题不能为空")
        sys.exit(1)

    llm = get_llm()
    resp = llm.chat([ChatMessage(role="user", content=question)])
    print(resp.content)


if __name__ == "__main__":
    main()
