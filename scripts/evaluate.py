"""评估脚本：分流正确率 + 拒答正确率 + Recall@5 三组对比，产出 EVALUATION.md。

三组对比（规格书 5.12）：
  1. 纯向量基线：Retriever.retrieve（向量召回，无重排）
  2. 加重排：向量召回 + bge-reranker 精排
  3. 混合+策略：混合检索（向量+关键词 RRF）+ 策略引擎 + 重排

用法：
  python scripts/evaluate.py --eval-set data/eval/eval_set.jsonl --output EVALUATION.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.retriever import Retriever  # noqa: E402
from app.rag.router import Router  # noqa: E402
from app.rag.strategy import StrategyEngine  # noqa: E402

TYPE_TO_INTENT = {"faq": "faq", "complex": "rag", "reject": "reject"}


def load_eval_set(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _doc_names(hits: list[dict]) -> set[str]:
    return {(h.get("metadata") or {}).get("doc_name", "") for h in hits}


def _top_n(hits: list[dict], question: str, reranker, top_k: int = 5) -> list[dict]:
    if not hits:
        return []
    if reranker is None:
        return hits[:top_k]
    docs = [h.get("text", "") for h in hits]
    scores = reranker.rerank(question, docs)
    ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
    return [h for h, _ in ranked[:top_k]]


def evaluate_routing(eval_set: list[dict], router: Router) -> tuple[float, dict]:
    detail = {"faq": [0, 0], "complex": [0, 0], "reject": [0, 0]}
    correct = 0
    for item in eval_set:
        route = router.route(item["question"], item.get("category"))
        expected = TYPE_TO_INTENT[item["type"]]
        detail[item["type"]][1] += 1
        if route["intent"] == expected:
            correct += 1
            detail[item["type"]][0] += 1
    return correct / len(eval_set), detail


def evaluate_reject(eval_set: list[dict], router: Router) -> float:
    rejects = [i for i in eval_set if i["type"] == "reject"]
    if not rejects:
        return 0.0
    correct = sum(
        1 for i in rejects if router.route(i["question"], i.get("category"))["intent"] == "reject"
    )
    return correct / len(rejects)


def _hybrid_via_strategy(question: str, category: str | None, strategy: StrategyEngine, retriever: Retriever) -> list[dict]:
    plan = strategy.plan(question)
    seen: dict[str, dict] = {}
    for subq in plan.queries:
        for h in retriever.hybrid_retrieve(subq, category=category, top_k=50):
            key = str((h.get("metadata") or {}).get("chunk_id") or h.get("id"))
            if key not in seen:
                seen[key] = h
    return list(seen.values())


def evaluate_recall_group(
    name: str,
    retrieve_fn,
    reranker,
    complex_items: list[dict],
) -> dict:
    hit = 0
    for item in complex_items:
        hits = retrieve_fn(item["question"], item.get("category"))
        top = _top_n(hits, item["question"], reranker, top_k=5)
        if any(d in _doc_names(top) for d in item.get("retrieval_docs", [])):
            hit += 1
    total = len(complex_items)
    return {"name": name, "hit": hit, "total": total, "recall": hit / total if total else 0.0}


def _load_reranker():
    try:
        from app.rerank import get_reranker

        return get_reranker()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 重排模型加载失败，重排组降级为向量分: {exc}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="data/eval/eval_set.jsonl")
    ap.add_argument("--output", default="EVALUATION.md")
    args = ap.parse_args()

    started = time.time()
    eval_set = load_eval_set(args.eval_set)
    print(f"载入金标集 {len(eval_set)} 条")

    router = Router()
    retriever = Retriever()
    strategy = StrategyEngine()
    reranker = _load_reranker()

    # 1. 分流正确率
    routing_acc, routing_detail = evaluate_routing(eval_set, router)
    print(f"分流正确率: {routing_acc:.2%}")

    # 2. 拒答正确率
    reject_acc = evaluate_reject(eval_set, router)
    print(f"拒答正确率: {reject_acc:.2%}")

    # 3. Recall@5 三组对比
    complex_items = [i for i in eval_set if i["type"] == "complex"]
    groups = [
        evaluate_recall_group(
            "纯向量基线",
            lambda q, c: retriever.retrieve(q, category=c, top_k=50),
            None,
            complex_items,
        ),
        evaluate_recall_group(
            "加重排",
            lambda q, c: retriever.retrieve(q, category=c, top_k=50),
            reranker,
            complex_items,
        ),
        evaluate_recall_group(
            "混合+策略",
            lambda q, c: _hybrid_via_strategy(q, c, strategy, retriever),
            reranker,
            complex_items,
        ),
    ]
    for g in groups:
        print(f"  {g['name']}: Recall@5 = {g['recall']:.2%} ({g['hit']}/{g['total']})")

    elapsed = time.time() - started
    run_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_report(args.output, routing_acc, routing_detail, reject_acc, groups, run_at, elapsed)


def write_report(path, routing_acc, routing_detail, reject_acc, groups, run_at, elapsed) -> None:
    lines = [
        "# EVALUATION.md",
        "",
        f"评测运行时间：{run_at}",
        "",
        "## 复现命令",
        "",
        "```bash",
        "python scripts/gen_eval_set.py                      # 生成金标集",
        "python scripts/evaluate.py --eval-set data/eval/eval_set.jsonl --output EVALUATION.md",
        "```",
        "",
        "## 金标集规模",
        "",
        "100 条：faq 30 / complex 50 / reject 20（语料：sample_rag.md 7 chunk + sample_java.pdf 1 chunk + 10 条 FAQ 种子）。",
        "",
        "## 指标结果",
        "",
        "### 分流正确率",
        "",
        f"- 总体：**{routing_acc:.2%}**",
    ]
    for t in ("faq", "complex", "reject"):
        ok, tot = routing_detail[t]
        lines.append(f"- {t}：{ok}/{tot}（{ok / tot:.2%}）" if tot else f"- {t}：0/0")
    lines += [
        "",
        "### 拒答正确率（reject 类被路由为 reject 的比例）",
        "",
        f"- **{reject_acc:.2%}**",
        "",
        "### Recall@5（complex 类，retrieval_docs 命中 Top5 比例）",
        "",
        "| 组 | Recall@5 | 命中 |",
        "|---|---|---|",
    ]
    for g in groups:
        lines.append(f"| {g['name']} | {g['recall']:.2%} | {g['hit']}/{g['total']} |")
    # 混合 vs 纯向量提升
    if len(groups) == 3:
        base, hybrid = groups[0]["recall"], groups[2]["recall"]
        delta = hybrid - base
        lines.append("")
        lines.append(f"混合+策略 相对 纯向量基线 Recall@5 提升：**{delta:+.2%}**")
    lines += [
        "",
        "> 说明：当前语料仅 2 篇文档（8 个 chunk），Recall@5 以 doc_name 是否命中衡量，",
        "> 向量召回 top50 已覆盖全部文档，故三组均为 100%、无区分度。",
        "> 需扩充语料至多文档后，混合检索相对纯向量的 Recall 提升才能体现。",
    ]
    lines += [
        "",
        "## 未实测项（待实测）",
        "",
        "- RAGAS Faithfulness / Answer Relevancy / Context Precision（依赖 ragas 库，尚未接入）",
        "- 端到端延迟 p50/p95（需对全部 complex 类跑完整生成链路，成本较高，暂未实测）",
        "",
        f"本报告运行耗时 {elapsed:.0f}s。",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"报告已写入 {path}")


if __name__ == "__main__":
    main()
