"""RAG 链路端到端演示：检索→重排→生成（含拒答）。

用法:
  python scripts/chat_demo.py "什么是 RAG？" --category ai
  python scripts/chat_demo.py "今天天气怎么样"           # 预期触发拒答
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.rag.chat_service import ChatService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--category", default=None)
    args = parser.parse_args()

    service = ChatService()
    result = service.chat(args.question, category=args.category)
    print(f"intent    : {result['intent']}")
    print(f"strategy  : {result['strategy']}")
    print(f"rejected  : {result['rejected']}")
    print(f"confidence: {result.get('confidence')}")
    print(f"latency_ms: {result['latency_ms']}")
    print("-" * 60)
    print(f"answer:\n{result['answer']}")
    if result.get("citations"):
        print("-" * 60)
        print("citations:")
        for i, c in enumerate(result["citations"], 1):
            print(f"  [{i}] {c.get('doc_name')} | {c.get('title')}")
            print(f"      {(c.get('text') or '')[:120]}")


if __name__ == "__main__":
    main()
