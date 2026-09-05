# EVALUATION.md

评测运行时间：2026-09-05 17:43:39

## 复现命令

```bash
python scripts/gen_eval_set.py                      # 生成金标集
python scripts/evaluate.py --eval-set data/eval/eval_set.jsonl --output EVALUATION.md
```

## 金标集规模

100 条：faq 30 / complex 50 / reject 20（语料：sample_rag.md 7 chunk + sample_java.pdf 1 chunk + 10 条 FAQ 种子）。

## 指标结果

### 分流正确率

- 总体：**94.00%**
- faq：29/30（96.67%）
- complex：48/50（96.00%）
- reject：17/20（85.00%）

### 拒答正确率（reject 类被路由为 reject 的比例）

- **85.00%**

### Recall@5（complex 类，retrieval_docs 命中 Top5 比例）

| 组 | Recall@5 | 命中 |
|---|---|---|
| 纯向量基线 | 100.00% | 50/50 |
| 加重排 | 100.00% | 50/50 |
| 混合+策略 | 100.00% | 50/50 |

混合+策略 相对 纯向量基线 Recall@5 提升：**+0.00%**

> 说明：当前语料仅 2 篇文档（8 个 chunk），Recall@5 以 doc_name 是否命中衡量，
> 向量召回 top50 已覆盖全部文档，故三组均为 100%、无区分度。
> 需扩充语料至多文档后，混合检索相对纯向量的 Recall 提升才能体现。

## 未实测项（待实测）

- RAGAS Faithfulness / Answer Relevancy / Context Precision（依赖 ragas 库，尚未接入）
- 端到端延迟 p50/p95（需对全部 complex 类跑完整生成链路，成本较高，暂未实测）

本报告运行耗时 341s。
