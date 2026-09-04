"""验证 Embedding 连通：编码一句话并打印向量维度。首次运行会下载 BGE-M3 权重。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.embeddings import get_encoder


def main() -> None:
    encoder = get_encoder()
    texts = ["职业教育中的 RAG 是什么"]
    vectors = encoder.encode(texts)
    print(f"模型: {encoder._model_name}")
    print(f"向量维度: {encoder.dim}")
    print(f"编码条数: {len(vectors)}")
    print(f"首向量前 5 维: {vectors[0][:5].tolist()}")


if __name__ == "__main__":
    main()
