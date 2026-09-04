"""初始化数据库：建表 + FAQ 种子导入。用法: python scripts/init_db.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from sqlalchemy import text

from app.db.mysql import get_engine

DDL_STATEMENTS = [
    """CREATE TABLE IF NOT EXISTS faq (
      faq_id       BIGINT PRIMARY KEY AUTO_INCREMENT,
      question     VARCHAR(500) NOT NULL,
      keywords     VARCHAR(500),
      answer       TEXT NOT NULL,
      category     VARCHAR(50) NOT NULL,
      hit_count    INT DEFAULT 0,
      status       TINYINT DEFAULT 1,
      created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
      updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
      KEY idx_category (category),
      KEY idx_status (status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS knowledge_doc (
      doc_id       BIGINT PRIMARY KEY AUTO_INCREMENT,
      file_name    VARCHAR(255) NOT NULL,
      category     VARCHAR(50) NOT NULL,
      file_type    VARCHAR(10),
      chunk_count  INT DEFAULT 0,
      status       TINYINT DEFAULT 1,
      created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
      KEY idx_category (category)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS knowledge_chunk (
      chunk_id     BIGINT PRIMARY KEY AUTO_INCREMENT,
      doc_id       BIGINT NOT NULL,
      doc_name     VARCHAR(255) NOT NULL,
      category     VARCHAR(50) NOT NULL,
      title        VARCHAR(500),
      page_no      INT,
      chunk_text   TEXT NOT NULL,
      milvus_id    VARCHAR(64),
      created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
      KEY idx_doc (doc_id),
      KEY idx_category (category)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
    """CREATE TABLE IF NOT EXISTS qa_log (
      log_id       BIGINT PRIMARY KEY AUTO_INCREMENT,
      session_id   VARCHAR(64) NOT NULL,
      question     TEXT NOT NULL,
      intent       VARCHAR(20),
      strategy     VARCHAR(20),
      route_detail TEXT,
      evidence_ids JSON,
      answer       MEDIUMTEXT,
      latency_ms   INT,
      cache_hit    TINYINT DEFAULT 0,
      created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
      KEY idx_session (session_id),
      KEY idx_created (created_at)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4""",
]

# 通用技术问答种子，不含任何机构/课程/讲师字样
SEED_FAQS = [
    ("什么是 RAG？", "rag 检索增强生成", "RAG（检索增强生成）是一种将信息检索与大语言模型生成相结合的技术：先从知识库检索相关片段，再交给模型基于片段生成答案，从而降低幻觉、支持溯源。", "ai"),
    ("什么是向量数据库？", "向量 数据库 embedding", "向量数据库是用于存储和检索高维向量（embedding）的数据库，支持按相似度（如余弦、内积）快速查找，是 RAG 系统的核心存储组件。", "ai"),
    ("什么是 Java 的多态？", "java 多态 面向对象", "多态是面向对象的三大特性之一，指同一操作作用于不同对象时产生不同行为，Java 中通过方法重写和父类引用指向子类对象实现。", "java"),
    ("HashMap 与 Hashtable 的区别？", "hashmap hashtable 区别", "HashMap 允许 null 键值、非线程安全、性能更高；Hashtable 不允许 null、方法加锁线程安全。并发场景推荐 ConcurrentHashMap。", "java"),
    ("什么是单元测试？", "单元测试 junit", "单元测试是针对最小可测试单元（如方法、函数）的自动化测试，用于验证其行为是否符合预期，通常隔离外部依赖。", "test"),
    ("黑盒测试与白盒测试的区别？", "黑盒 白盒 测试 区别", "黑盒测试不关注内部实现，只看输入输出是否符合需求；白盒测试基于内部逻辑结构设计用例，覆盖分支路径。", "test"),
    ("什么是 Docker？", "docker 容器", "Docker 是一种容器化平台，将应用及其依赖打包为镜像，在任何支持容器的环境一致运行，解决环境一致性问题。", "ops"),
    ("什么是 CI/CD？", "ci cd 持续集成 部署", "CI（持续集成）自动构建测试代码，CD（持续交付/部署）自动发布。CI/CD 流水线提升交付效率与质量。", "ops"),
    ("什么是 Hadoop？", "hadoop 大数据", "Hadoop 是分布式大数据处理框架，核心包括 HDFS（分布式存储）和 MapReduce（分布式计算），适合海量数据批处理。", "bigdata"),
    ("什么是 Spark？", "spark 大数据 计算", "Spark 是内存计算的大数据处理引擎，相比 MapReduce 更快，支持批处理、流处理、SQL、机器学习等统一接口。", "bigdata"),
]


def main() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        for stmt in DDL_STATEMENTS:
            conn.execute(text(stmt))
        existing = conn.execute(text("SELECT COUNT(*) FROM faq")).scalar()
        if existing == 0:
            for question, keywords, answer, category in SEED_FAQS:
                conn.execute(
                    text(
                        "INSERT INTO faq (question, keywords, answer, category) "
                        "VALUES (:question, :keywords, :answer, :category)"
                    ),
                    {
                        "question": question,
                        "keywords": keywords,
                        "answer": answer,
                        "category": category,
                    },
                )
            print(f"已导入 {len(SEED_FAQS)} 条 FAQ 种子")
        else:
            print(f"faq 表已有 {existing} 条数据，跳过种子导入")
    print("init_db 完成")


if __name__ == "__main__":
    main()
