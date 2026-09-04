"""M2 入库验证：入库示例 MD + PDF，并做向量检索验证。首次运行会下载 BGE-M3 权重。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.embeddings import get_encoder
from app.ingest.service import IngestService
from app.rag.vector_store import VectorStore


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    svc = IngestService()

    md = root / "data" / "docs" / "sample_rag.md"
    pdf = root / "data" / "docs" / "sample_java.pdf"

    r1 = svc.ingest_file(md, "ai")
    print(f"[MD 入库] doc_id={r1['doc_id']} chunk_count={r1['chunk_count']}")

    r2 = svc.ingest_file(pdf, "java")
    print(f"[PDF 入库] doc_id={r2['doc_id']} chunk_count={r2['chunk_count']}")

    encoder = get_encoder()
    store = VectorStore()

    for question, category in [("什么是 RAG？", "ai"), ("什么是多态？", "java")]:
        emb = encoder.encode([question])[0]
        results = store.search(emb, top_k=3, category=category)
        print(f"\n[检索] '{question}' (category={category}) Top3:")
        for r in results:
            title = r["metadata"].get("title") or ""
            text = r["text"][:40].replace("\n", " ")
            print(f"  score={r['score']:.3f} title={title!r} text={text}...")


if __name__ == "__main__":
    main()
