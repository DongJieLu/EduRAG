"""生成评估金标集 data/eval/eval_set.jsonl。

规模与分布（规格书 5.12）：100 条，faq : complex : reject = 30 : 50 : 20。
用法：python scripts/gen_eval_set.py
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_set.jsonl"

# (question, category, expected_faq)
FAQ_ITEMS = [
    ("什么是 RAG？", "ai", "什么是 RAG？"),
    ("RAG 是什么的缩写？", "ai", "什么是 RAG？"),
    ("解释一下 RAG 技术", "ai", "什么是 RAG？"),
    ("什么是向量数据库？", "ai", "什么是向量数据库？"),
    ("向量数据库是做什么的？", "ai", "什么是向量数据库？"),
    ("介绍一下向量数据库", "ai", "什么是向量数据库？"),
    ("什么是 Java 的多态？", "java", "什么是 Java 的多态？"),
    ("Java 多态是什么意思？", "java", "什么是 Java 的多态？"),
    ("解释 Java 面向对象的多态", "java", "什么是 Java 的多态？"),
    ("HashMap 与 Hashtable 的区别？", "java", "HashMap 与 Hashtable 的区别？"),
    ("HashMap 和 Hashtable 有什么不同？", "java", "HashMap 与 Hashtable 的区别？"),
    ("对比一下 HashMap 和 Hashtable", "java", "HashMap 与 Hashtable 的区别？"),
    ("什么是单元测试？", "test", "什么是单元测试？"),
    ("单元测试是干什么的？", "test", "什么是单元测试？"),
    ("介绍一下单元测试", "test", "什么是单元测试？"),
    ("黑盒测试与白盒测试的区别？", "test", "黑盒测试与白盒测试的区别？"),
    ("黑盒测试和白盒测试有什么不同？", "test", "黑盒测试与白盒测试的区别？"),
    ("对比黑盒测试与白盒测试", "test", "黑盒测试与白盒测试的区别？"),
    ("什么是 Docker？", "ops", "什么是 Docker？"),
    ("Docker 是做什么的？", "ops", "什么是 Docker？"),
    ("介绍一下 Docker 容器", "ops", "什么是 Docker？"),
    ("什么是 CI/CD？", "ops", "什么是 CI/CD？"),
    ("CI/CD 是什么意思？", "ops", "什么是 CI/CD？"),
    ("解释一下持续集成和持续部署", "ops", "什么是 CI/CD？"),
    ("什么是 Hadoop？", "bigdata", "什么是 Hadoop？"),
    ("Hadoop 是做什么的？", "bigdata", "什么是 Hadoop？"),
    ("介绍一下 Hadoop 框架", "bigdata", "什么是 Hadoop？"),
    ("什么是 Spark？", "bigdata", "什么是 Spark？"),
    ("Spark 是做什么的？", "bigdata", "什么是 Spark？"),
    ("介绍一下 Spark 引擎", "bigdata", "什么是 Spark？"),
]

# (question, [expected_answer_keys], doc_name, category)
COMPLEX_RAG = [
    ("RAG 的核心思想是什么？请详细说明", ["检索", "生成"], "sample_rag.md", "ai"),
    ("RAG 如何降低大语言模型的幻觉问题？", ["检索", "幻觉"], "sample_rag.md", "ai"),
    ("为什么 RAG 能实现答案可溯源？", ["溯源", "片段"], "sample_rag.md", "ai"),
    ("RAG 相比直接使用大模型有哪些优势？", ["检索", "证据"], "sample_rag.md", "ai"),
    ("检索增强生成中的检索阶段起到什么作用？", ["检索", "片段"], "sample_rag.md", "ai"),
    ("RAG 系统的工作流程包含哪些步骤？", ["解析", "分块", "向量化", "检索", "重排", "生成"], "sample_rag.md", "ai"),
    ("RAG 的五个步骤分别是什么？", ["解析", "分块", "向量化", "检索", "生成"], "sample_rag.md", "ai"),
    ("文档解析与分块在 RAG 中做什么？", ["解析", "分块"], "sample_rag.md", "ai"),
    ("向量化编码在 RAG 中的作用是什么？", ["向量", "编码"], "sample_rag.md", "ai"),
    ("RAG 中向量检索是如何工作的？", ["向量", "相似"], "sample_rag.md", "ai"),
    ("重排序在 RAG 流程中处于什么位置？", ["重排", "检索", "生成"], "sample_rag.md", "ai"),
    ("RAG 的答案生成阶段做了什么？", ["生成", "片段"], "sample_rag.md", "ai"),
    ("RAG 系统中常见的分块策略有哪些？", ["递归", "语义", "结构"], "sample_rag.md", "ai"),
    ("分块过大或过小会带来什么问题？", ["过大", "过小", "无关"], "sample_rag.md", "ai"),
    ("递归字符分块是如何工作的？", ["递归", "分隔符"], "sample_rag.md", "ai"),
    ("语义分块的特点是什么？", ["语义", "完整性"], "sample_rag.md", "ai"),
    ("结构分块适合什么场景？", ["结构", "标题"], "sample_rag.md", "ai"),
    ("为什么分块是 RAG 的关键环节？", ["分块", "关键"], "sample_rag.md", "ai"),
    ("向量检索的基本原理是什么？", ["向量", "相似度"], "sample_rag.md", "ai"),
    ("为什么语义相近的文本向量距离更近？", ["语义", "向量", "空间"], "sample_rag.md", "ai"),
    ("向量检索用什么度量相似度？", ["余弦", "内积"], "sample_rag.md", "ai"),
    ("余弦相似度和内积有什么区别？", ["余弦", "内积"], "sample_rag.md", "ai"),
    ("向量检索在 RAG 系统中的作用？", ["向量", "检索"], "sample_rag.md", "ai"),
    ("向量检索如何找到最相似的片段？", ["向量", "相似", "片段"], "sample_rag.md", "ai"),
    ("重排序为什么能提升检索精度？", ["重排", "精度"], "sample_rag.md", "ai"),
    ("向量检索和重排序有什么区别？", ["向量", "重排", "精度"], "sample_rag.md", "ai"),
    ("重排模型如何给候选片段打分？", ["重排", "打分"], "sample_rag.md", "ai"),
    ("为什么需要重排序这一步骤？", ["重排", "精度"], "sample_rag.md", "ai"),
    ("近似最近邻算法有什么局限？", ["近似", "精度"], "sample_rag.md", "ai"),
    ("重排序在 RAG 中如何工作？", ["重排", "候选", "打分"], "sample_rag.md", "ai"),
    ("评估 RAG 系统通常关注哪些维度？", ["检索", "生成", "延迟"], "sample_rag.md", "ai"),
    ("RAG 的检索质量如何评估？", ["召回", "命中"], "sample_rag.md", "ai"),
    ("RAG 的生成质量关注哪些指标？", ["忠实", "相关"], "sample_rag.md", "ai"),
    ("RAG 评估中的忠实度指什么？", ["忠实"], "sample_rag.md", "ai"),
    ("如何构造金标集来评估 RAG？", ["金标", "评估"], "sample_rag.md", "ai"),
    ("RAG 与知识库更新有什么关系？", ["更新", "入库"], "sample_rag.md", "ai"),
    ("为什么知识库更新时不需要重新训练模型？", ["更新", "训练"], "sample_rag.md", "ai"),
    ("RAG 如何保证答案基于证据？", ["证据", "片段"], "sample_rag.md", "ai"),
    ("向量数据库在 RAG 系统中起什么作用？", ["向量", "数据库"], "sample_rag.md", "ai"),
    ("RAG 适合解决什么问题？", ["检索", "生成"], "sample_rag.md", "ai"),
]

COMPLEX_JAVA = [
    ("Java 面向对象的三大特性是什么？", ["封装", "继承", "多态"], "sample_java.pdf", "java"),
    ("Java 封装的含义是什么？", ["封装", "隐藏"], "sample_java.pdf", "java"),
    ("封装如何隐藏对象内部实现细节？", ["封装", "隐藏", "细节"], "sample_java.pdf", "java"),
    ("Java 中封装的好处有哪些？", ["封装", "接口"], "sample_java.pdf", "java"),
    ("面向对象的三大特性分别指什么？", ["封装", "继承", "多态"], "sample_java.pdf", "java"),
    ("Java 中如何实现封装？", ["封装", "方法"], "sample_java.pdf", "java"),
    ("为什么面向对象需要封装？", ["封装", "隐藏"], "sample_java.pdf", "java"),
    ("面向对象编程的核心特性有哪些？", ["封装", "继承", "多态"], "sample_java.pdf", "java"),
    ("Java 基础知识点包括哪些？", ["封装", "面向对象"], "sample_java.pdf", "java"),
    ("封装与接口之间是什么关系？", ["封装", "接口"], "sample_java.pdf", "java"),
]

# (question, category)
REJECT_ITEMS = [
    ("你好", None),
    ("谢谢", None),
    ("今天天气怎么样", None),
    ("帮我写一首诗", None),
    ("你是谁", None),
    ("现在几点了", None),
    ("推荐一部电影", None),
    ("怎么做红烧肉", None),
    ("明天的股市怎么样", None),
    ("讲个笑话", None),
    ("你会唱歌吗", None),
    ("最近有什么新闻", None),
    ("帮我订个外卖", None),
    ("周末去哪里玩", None),
    ("聊聊爱情观", None),
    ("怎么减肥", None),
    ("炒股有什么技巧", None),
    ("世界杯冠军是谁", None),
    ("给我讲个童话故事", None),
    ("再见", None),
]


def main() -> None:
    items: list[dict] = []
    idx = 0
    for question, category, expected_faq in FAQ_ITEMS:
        idx += 1
        items.append({
            "id": f"e{idx:03d}",
            "question": question,
            "category": category,
            "type": "faq",
            "expected_faq": expected_faq,
            "expected_answer_keys": [],
            "retrieval_docs": [],
        })
    for question, keys, doc, category in COMPLEX_RAG + COMPLEX_JAVA:
        idx += 1
        items.append({
            "id": f"e{idx:03d}",
            "question": question,
            "category": category,
            "type": "complex",
            "expected_answer_keys": keys,
            "retrieval_docs": [doc],
        })
    for question, category in REJECT_ITEMS:
        idx += 1
        items.append({
            "id": f"e{idx:03d}",
            "question": question,
            "category": category,
            "type": "reject",
            "expected_answer_keys": [],
            "retrieval_docs": [],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    n_faq = sum(1 for i in items if i["type"] == "faq")
    n_cpx = sum(1 for i in items if i["type"] == "complex")
    n_rej = sum(1 for i in items if i["type"] == "reject")
    print(f"生成 {len(items)} 条金标 -> {OUT}")
    print(f"  faq={n_faq} complex={n_cpx} reject={n_rej}")


if __name__ == "__main__":
    main()
