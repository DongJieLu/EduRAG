---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 2a076a6593cf5cbed52f4e0afa2e4c78_de29d603a82f11f187fd525400826444
    ReservedCode1: 4tksQO9q3nTK2WkpLXyaSbLACOl7yz0Qqe1Qviga1POV/kmHAR62Ml/2QGzcj6MOhTbuywuxsTReBKPlMdq2EGoPDzPYwPuf1imlPff3rdiLUl0AiZq3cU+boKpQkhDtxXZ2WkPfhckCR5o8qKnAxi4d3RX+bnkpbrdzc8WnqeNIHaMn1x12Wl3911s=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 2a076a6593cf5cbed52f4e0afa2e4c78_de29d603a82f11f187fd525400826444
    ReservedCode2: 4tksQO9q3nTK2WkpLXyaSbLACOl7yz0Qqe1Qviga1POV/kmHAR62Ml/2QGzcj6MOhTbuywuxsTReBKPlMdq2EGoPDzPYwPuf1imlPff3rdiLUl0AiZq3cU+boKpQkhDtxXZ2WkPfhckCR5o8qKnAxi4d3RX+bnkpbrdzc8WnqeNIHaMn1x12Wl3911s=
---

# EduQA 职业教育课程智能问答平台 · 独立开发规格书

> 本文档是项目 A（EduQA）的独立开发规格，供 Claude Code 在**单独仓库**中从零实现。
> 项目 B（PetCare Agent）见另一份独立文档，两者互不依赖。
> 硬约束：仓库/代码/注释/README/commit 中禁止出现任何培训机构、课程平台、讲师字样；本规格中"基线 R"仅指实现参考，不写入任何交付物。

---

## 0. 交付总览

| 项 | 值 |
|---|---|
| 仓库建议名 | `eduqa`（私有先行，公开前自查） |
| 语言/版本 | Python 3.10+（建议 3.11） |
| 交付形式 | 可一键 Docker Compose 启动的完整前后端 + 离线入库脚本 + 评估脚本 |
| 完成标志 | README / EVALUATION.md / INTERVIEW.md 齐备，验收清单全部可勾选 |

---

## 1. 项目目标与边界

### 1.1 目标

面向职业教育学员的技术知识问答平台：FAQ 秒答 + 复杂问题 RAG 深度回答，支持按知识方向过滤、回答可溯源、多轮对话、流式输出。

### 1.2 范围（In Scope）

1. FAQ 快速通道（MySQL 精确/模糊检索）
2. RAG 深度问答链路（文档向量检索 + 重排 + 生成）
3. 三层查询路由（规则 / 相似度 / LLM）
4. 四种检索策略引擎（直接 / HyDE / 子查询 / 回溯）
5. Redis 三级缓存
6. 多方向知识库（AI / Java / 测试 / 运维 / 大数据）与 metadata 过滤
7. 文档上传入库（PDF / DOCX / TXT / MD）
8. FastAPI 服务 + SSE 流式 + Gradio 界面
9. RAGAS 自动化评估 + 金标集
10. Docker Compose 一键部署

### 1.3 非目标（Out of Scope）

- 不做 Web 端复杂后台管理系统（Gradio 足够演示）
- 不做账号体系/权限系统（单用户演示版）
- 不做图片/音视频等非文本文档理解
- 不做在线文档实时协作

### 1.4 质量约束

- 所有指标可复现：EVALUATION.md 必须给出评测命令与运行日期
- 禁止虚构指标；未实测处写"待实测"，实测后回填
- 代码模块化分层，核心逻辑必须可单测（tests/ 覆盖路由、分块、缓存、评估）

---

## 2. 总体架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────┐
│ 表现层   Gradio UI / FastAPI REST+SSE                   │
├─────────────────────────────────────────────────────────┤
│ 应用层   chat 服务 / ingest 服务 / 会话管理 / 统计服务    │
├─────────────────────────────────────────────────────────┤
│ 路由层   三层查询路由 → FAQ 通道 | RAG 链路              │
├─────────────────────────────────────────────────────────┤
│ RAG 引擎  策略选择 → 检索(向量+关键词) → 重排 → 生成      │
├─────────────────────────────────────────────────────────┤
│ 数据访问  MySQL / Milvus / Redis 客户端                  │
└─────────────────────────────────────────────────────────┘
离线链路：文档 → 解析 → 分块 → 编码(BGE-M3) → Milvus + MySQL
```

### 2.2 在线主流程时序

1. 用户提问 → 查 Redis 答案缓存（key 设计见 5.8），命中直接返回
2. 未命中 → 三层路由判定 intent ∈ {faq, rag, reject}
3. reject → 返回拒答模板（并写入缓存防止重复骚扰）
4. faq → FAQ 通道检索 → 命中返回；未命中降级转 rag
5. rag → 策略引擎选策略（direct/hyde/subquery/rewrite）→ 召回 → 重排 → 生成
6. 结果结构化落库（qa_log），答案写回缓存

### 2.3 离线入库流程时序

1. 接收文件（本地路径或上传）
2. 校验格式与大小（限制单文件 ≤ 20MB）
3. 解析为纯文本 + 标题结构
4. 按所选策略分块，生成 chunk 记录（含来源/标题/页码 metadata）
5. BGE-M3 编码
6. 写入 Milvus（collection `eduqa_chunks`）+ MySQL `chunk` 元数据表 + `knowledge_doc` 登记
7. 支持增量入库（按文档 id 先删后插）

---

## 3. 技术栈与版本基线

| 组件 | 选型 | 版本/参数基线 |
|---|---|---|
| 语言 | Python | 3.10+ / 3.11 |
| Web | FastAPI + Uvicorn | fastapi≥0.110；uvicorn[standard] |
| RAG 编排 | LangChain | 0.3.x（与 langchain-openai/community 配套锁定） |
| LLM | DeepSeek / Qwen（DashScope）API 或 Ollama 本地 | 默认 deepseek-chat；Ollama 可切换 qwen2.5:14b |
| Embedding | BGE-M3 | 本地 sentence-transformers 或 API；向量维度 1024 |
| 重排 | bge-reranker-large | 本地加载；GPU 无则 CPU |
| 向量库 | Milvus | 2.4+（Docker）；开发可切换 Chroma 同 schema |
| 关系库 | MySQL | 8.0（utf8mb4） |
| 缓存 | Redis | 7.x |
| 前端 | Gradio | ≥4.0 |
| 评估 | RAGAS | ≥0.1（按官方接口适配） |
| 部署 | Docker Compose | MySQL+Redis+Milvus(app 可本地跑) |

---

## 4. 数据设计

### 4.1 MySQL 表结构（DDL 基线）

```sql
-- FAQ 表
CREATE TABLE faq (
  faq_id       BIGINT PRIMARY KEY AUTO_INCREMENT,
  question     VARCHAR(500) NOT NULL,
  keywords     VARCHAR(500),          -- 空格分隔，检索加速
  answer       TEXT NOT NULL,
  category     VARCHAR(50) NOT NULL,  -- ai/java/test/ops/bigdata
  hit_count    INT DEFAULT 0,
  status       TINYINT DEFAULT 1,     -- 1 启用 0 停用
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY idx_category (category),
  KEY idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 知识文档登记表
CREATE TABLE knowledge_doc (
  doc_id       BIGINT PRIMARY KEY AUTO_INCREMENT,
  file_name    VARCHAR(255) NOT NULL,
  category     VARCHAR(50) NOT NULL,
  file_type    VARCHAR(10),           -- pdf/docx/txt/md
  chunk_count  INT DEFAULT 0,
  status       TINYINT DEFAULT 1,     -- 1 已入库 0 已删除
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- chunk 元数据表（向量在 Milvus，文本与元数据在此）
CREATE TABLE knowledge_chunk (
  chunk_id     BIGINT PRIMARY KEY AUTO_INCREMENT,
  doc_id       BIGINT NOT NULL,
  doc_name     VARCHAR(255) NOT NULL,
  category     VARCHAR(50) NOT NULL,
  title        VARCHAR(500),          -- 所在标题层级
  page_no      INT,                   -- PDF 页码，可为空
  chunk_text   TEXT NOT NULL,
  milvus_id    VARCHAR(64),           -- Milvus 主键
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  KEY idx_doc (doc_id),
  KEY idx_category (category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 问答日志表
CREATE TABLE qa_log (
  log_id       BIGINT PRIMARY KEY AUTO_INCREMENT,
  session_id   VARCHAR(64) NOT NULL,
  question     TEXT NOT NULL,
  intent       VARCHAR(20),           -- faq/rag/reject
  strategy     VARCHAR(20),           -- direct/hyde/subquery/rewrite
  route_detail TEXT,                  -- JSON：每层路由的分数与理由
  evidence_ids JSON,                  -- 命中的 chunk/faq id 列表
  answer       MEDIUMTEXT,
  latency_ms   INT,
  cache_hit    TINYINT DEFAULT 0,
  created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
  KEY idx_session (session_id),
  KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 4.2 Milvus Collection Schema

```
collection: eduqa_chunks
字段:
  id         INT64        (主键,自增)
  vector     FLOAT_VECTOR (dim=1024, BGE-M3)
  chunk_id   INT64        (对应 MySQL knowledge_chunk.chunk_id)
  category   VARCHAR      (过滤字段)
参数: metric_type=IP（或 COSINE）；index_type=HNSW 或 IVF_FLAT
```

### 4.3 Redis Key 设计

| Key | 类型 | 说明 | TTL |
|---|---|---|---|
| `qa:ans:{md5(question+category)}` | string | FAQ/答案缓存 | 3600s |
| `qa:faq:hot` | zset | FAQ 热点榜（命中计数） | 永久 |
| `qa:list:{session_id}` | list | 用户问题列表（第 2 级） | 1800s |
| `qa:lock:{md5}` | string | 防并发重复请求 | 30s |

三级缓存语义：
1. 第 1 级：答案缓存 `qa:ans:*`（最优先）
2. 第 2 级：该 session 历史问题列表 `qa:list:*`，同问题复问走历史答案
3. 第 3 级：未命中上述时直接查 MySQL FAQ 表做精确兜底

---

## 5. 模块详细设计

### 5.1 文档解析与分块

接口：
```python
def parse_document(file_path: Path, file_type: str) -> list[dict]:
    """返回 [{text, page_no?, headers:[...]}]"""

def chunk_document(parsed, strategy: str = "recursive",
                   chunk_size: int = 400, chunk_overlap: int = 80) -> list[Chunk]:
    """返回 Chunk（含 text/metadata:doc_id,doc_name,category,title,page_no,chunk_index）"""
```

分块策略：
- `recursive`：RecursiveCharacterTextSplitter，chunk_size=400±50、overlap=80±20（以 token 计）
- `markdown`：先按标题层级切到二级标题块，再对超大块递归切分；title 记录所在标题链
- PDF 解析用 pypdf 按页读，page_no 写入 metadata；DOCX 用 python-docx 按段落；MD/TXT 直接读

入库规则：
- 单文档 chunk 数上限 2000，超限拒绝并提示拆分
- 入库前清洗：去多余空白、去页码/页眉脚（启发式），长度 <10 字符的 chunk 丢弃

### 5.2 向量化与索引

- 编码模型：BGE-M3（本地部署权重或 API），维度 1024，normalize 后入库
- 批处理：batch_size=64，失败批次重试 1 次，单条重试失败则跳过并记录
- 索引参数（HNSW）：M=16，efConstruction=200；查询 ef=64

### 5.3 FAQ 通道

检索步骤：
1. 关键词层：question 分词后与 `keywords` 用 MySQL FULLTEXT 或 LIKE 多词命中评分
2. 语义层：question Embedding 与 FAQ 问题向量比对（同 Milvus 另一 collection 或单独内存向量），相似度阈值 ≥0.82
3. 融合：关键词命中数 ×0.6 + 语义分 ×0.4，取 Top3，最高分 <0.5 判定未命中

命中后：answer 返回 + hit_count+1 + 写 `qa:ans:*` 缓存。

### 5.4 三层查询路由

统一输出：
```json
{ "intent": "faq|rag|reject", "confidence": 0.0-1.0, "reason": "..." }
```

| 层 | 实现 | 判定逻辑 | 置信度规则 |
|---|---|---|---|
| L1 规则 | 关键词正则库 | 含"多少钱/报名/时长/开班/几周"等 20+ 规则词 → faq 倾向；含"你好/谢谢/再见" → chitchat 归 reject | 规则词命中且无否定词：0.95 |
| L2 相似度 | FAQ 问题向量库 | 最高相似 ≥0.90 → faq；≥0.75 待定 | 映射到 0.75~0.95 |
| L3 LLM | 分类 prompt + 结构化输出 | 输出 intent+confidence+reason | 取 LLM 值 |

路由决策：L1 命中则直接 faq；否则 L2 相似 ≥0.90 faq，0.75~0.90 且 L3 也 faq 才 faq；L3 判 reject 且 L2 <0.75 → reject；其余 rag。
每轮路由详情写入 qa_log.route_detail，供统计看板。

### 5.5 检索策略引擎

策略表：

| 策略 | 触发条件（LLM 依据问题特征选） | 执行方式 | 输出 |
|---|---|---|---|
| direct | 问题明确、术语完整 | 原问题直接向量检索 | query 原样 |
| hyde | 术语少、口语化（"这个报错咋整"） | LLM 生成假设性答案文档 → 用其向量检索 | hyde_doc |
| subquery | 复合问题（含"和/或者/对比"或多个主题） | LLM 拆 2~3 个子问题，各检索后合并 | queries[] |
| rewrite | 多轮追问（代词"它/这个/上面"） | 携带历史改写为独立问题 | rewritten |

LLM 决策 prompt 输出 JSON：`{"strategy":"direct|hyde|subquery|rewrite","queries":["..."],"reason":"..."}`；解析失败默认 direct。

### 5.6 混合检索与重排

召回：
- 向量召回：query 编码 → Milvus 取 top_k=50（带 category 过滤）
- （基线 R 含 MySQL 精准匹配；本仓库向量为主，FAQ 类已被路由分流）
- 文档相关性分数：IP 相似度

精排：
- 使用 bge-reranker-large 对 top50 打分，取 Top5 进 Prompt
- reranker 不可用时降级：原向量分排序取 Top5（在 EVALUATION.md 注明降级条件）

融合与兜底：
- 检索 Top5 平均分 < 0.35 → 视为无证据，走拒答模板
- metadata category 过滤：用户指定方向时，过滤其他方向 chunk

### 5.7 答案生成

Prompt 结构（按序）：
1. System：你是 EduQA 课程问答助手；只依据提供的"参考片段"回答；不得编造；无法回答时输出拒答语
2. 参考片段：`[1] (doc_name, title, category) 内容...` 至 `[5]`
3. 对话历史：最近 6 轮（截断超长）
4. 用户问题
5. 输出约束：标注引用编号；JSON 包装

输出 JSON schema：
```json
{
  "answer": "正文，引用处附[来源N]",
  "citations": [{"doc_name": "...", "title": "...", "text": "片段摘要"}],
  "confidence": 0.0-1.0,
  "needs_human": false
}
```

拒答条件（任一）：
- 无证据（Top5 平均分 <0.35）
- LLM 输出中声明"未在资料中找到"
- 解析失败且内容为空 → 统一拒答模板

### 5.8 缓存层

- 写策略：FAQ/RAG 成功回答后写 `qa:ans:*`；拒答也写（TTL 短 600s）
- 失效：管理员文档更新后按 category 前缀清理 `qa:ans:*` 中该方向键（scan + del）
- 热点：`qa:faq:hot` zset 每命中 +1；每 10 分钟将 Top20 热点问题的答案主动续期
- 缓存一致性说明写入 README（容许多端短暂不一致）

### 5.9 会话与多轮

- session_id 由客户端生成（UUID）或服务端返回
- 会话状态仅存 Redis `qa:list:{session_id}`（最近 20 条 Q，超限滚动）
- 多轮改写仅 RAG 链路使用；FAQ 判定始终基于当轮问题原文（改写后的独立问题）做判定

### 5.10 API 定义（契约）

统一响应包：`{"code":0,"data":{...},"msg":"ok"}`；错误码：1001 参数错 / 2001 无证据 / 3001 服务内部错 / 4001 文档入库失败。

POST `/api/v1/chat`
```json
请求: {"question":"...","session_id":"uuid","category":"ai|java|test|ops|bigdata|null"}
响应data: {"intent":"faq|rag|reject","answer":"...","citations":[...],"strategy":"direct|...","latency_ms":123}
```

POST `/api/v1/chat/stream`（SSE，事件格式）
```
data: {"type":"route","intent":"rag","strategy":"direct"}
data: {"type":"token","content":"..."}
data: {"type":"citation","citations":[...]}
data: {"type":"done","latency_ms":123}
```

POST `/api/v1/ingest`（multipart：file + category）→ 入库结果 `{"doc_id":1,"chunk_count":120}`

GET `/api/v1/sources?category=` → 文档列表
GET `/api/v1/stats?days=7` → 意图分布 / 平均延迟 / FAQ Top
GET `/api/v1/health` → 各组件连通性

### 5.11 Gradio 前端

- 页面 1 对话：聊天窗口、方向下拉（全部/AI/Java/测试/运维/大数据）、流式显示、来源折叠展示
- 页面 2 知识库：文件上传 + 方向选择、文档列表、删除入口
- 页面 3 统计：意图饼图、策略分布、Top FAQ、日均延迟（用 gradio plots 或 echarts 均可）

### 5.12 评估系统

金标集文件格式（data/eval/eval_set.jsonl）：
```json
{"id":"e001","question":"...","category":"ai","type":"faq|complex|reject","expected_faq":"可选","expected_answer_keys":["..."],"retrieval_docs":["doc_name"]}
```
规模与分布：100 条；faq:complex:reject = 3:5:2。

指标：
| 指标 | 计算 | 目标 |
|---|---|---|
| 分流正确率 | 路由 intent 与金标 type 一致比例 | ≥85% |
| Recall@5 | 金标 retrieval_docs 出现在 Top5 比例 | 混合 ≥ 纯向量 +15% |
| RAGAS Faithfulness / Answer Relevancy / Context Precision | ragas 官方库 | ≥0.8 / ≥0.85 / ≥0.75 |
| 端到端延迟 | 日志均值 p50/p95 | FAQ<300ms / RAG 首字<2s |
| 拒答正确率 | reject 金标被正确拒答比例 | ≥80% |

运行方式：`python scripts/evaluate.py --eval-set data/eval/eval_set.jsonl --output EVALUATION.md`，脚本输出逐项数值 + 复现命令 + 运行时间。必须跑三组对比：纯向量基线 / 加重排 / 混合+策略（若基线实现为开关，写清如何切换）。

---

## 6. 配置管理

`.env.example` 字段（全部必须有默认/占位与说明）：
```
LLM_PROVIDER=deepseek|dashscope|ollama
DEEPSEEK_API_KEY=
DASHSCOPE_API_KEY=
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:14b
EMBED_MODEL_NAME=BAAI/bge-m3          # 或 API 配置
RERANK_MODEL_NAME=BAAI/bge-reranker-large
MYSQL_HOST=localhost  MYSQL_PORT=3306  MYSQL_USER=eduqa  MYSQL_PASSWORD=...  MYSQL_DB=eduqa
REDIS_HOST=localhost  REDIS_PORT=6379
MILVUS_HOST=localhost MILVUS_PORT=19530
LOG_LEVEL=INFO
```
配置读取用 pydantic-settings，禁止硬编码。提供 `scripts/init_db.py` 建表 + FAQ 种子导入 + 索引初始化。

---

## 7. 日志与可观测

- 统一 JSON 行日志（time/level/module/session_id/latency_ms）
- 每次路由输出 route_detail JSON；统计接口从 qa_log 聚合
- 检索无结果、路由解析失败、LLM 超时等异常必须 error 级日志
- 可视化不强制，但 qa_log 需可按日清理（保留 90 天脚本）

---

## 8. 部署

docker-compose.yml 服务：
- mysql:8.0（挂 volume，初始化 SQL）
- redis:7（无持久化要求，可 appendonly yes）
- milvus 官方单机 compose（etcd + minio + standalone，按官方模板裁剪）
- app：FastAPI 服务（可选，开发时本地跑）

健康检查：/api/v1/health 返回 mysql/redis/milvus/llm 四连通状态。
资源建议：CPU 4 核 / 内存 8G（含 Milvus）；重排与 BGE-M3 可仅 CPU。

---

## 9. 安全与合规

- 密钥仅 .env，.gitignore 忽略；公开前 grep 检查 `sk-`/`password`
- 上传文件类型白名单 + 大小限制 + 恶意内容不做执行/渲染
- Prompt 注入防护：用户问题含"忽略以上指令/系统提示"等字样时在路由层标记，不进 system 段（片段与问题拼接时做隔离）
- 语料仅用公开合法内容；data/SOURCES.md 记录来源与授权情况

---

## 10. 失败与边界处理清单

| 场景 | 处理 |
|---|---|
| LLM 超时（15s） | 返回 3001 + 日志；缓存不写 |
| 向量库不可用 | health 置红；接口返回 3001 并提示稍后 |
| 路由三层全部异常 | 默认 rag 链路（不阻断） |
| 文档解析失败 | ingest 返回 4001 + 具体文件与原因 |
| 空问题 / 超长问题（>2000 字符） | 参数校验 1001 |
| 并发相同问题 | qa:lock 防抖 30s，后到请求直接复用进行中的结果或返回排队提示 |

---

## 11. 测试计划

- 单元测试：分块参数、三级路由决策矩阵（至少 12 组用例）、缓存读写、策略解析 fallback、拒答条件
- 集成测试：mock LLM 的端到端 chat（FAQ/RAG/拒答 各 3 条）
- 评估测试：`pytest tests/` 全绿 + `scripts/evaluate.py` 输出报告
- CI 不强制；本地可运行即可，测试命令写入 README

---

## 12. 里程碑与验收清单

| 阶段 | 目标 | 验收（全部满足才算完成） |
|---|---|---|
| M1 | 骨架 + 模型连通 | 环境变量加载、LLM/Embedding 各返回一次成功；CLI 一问一答 |
| M2 | 入库链路 | 上传 1 篇 PDF + 1 篇 MD 后 Milvus 可检索到片段；MySQL 元数据齐全 |
| M3 | RAG 主链路 + UI | Gradio 对话流式输出且带引用；拒答可触发 |
| M4 | FAQ + 路由 + 缓存 | 12 组路由用例通过；FAQ 二次提问命中缓存（延迟显著下降） |
| M5 | 策略 + 重排 + 评估 | 三组对比跑完并产出 EVALUATION.md；指标达 5.12 目标 |
| M6 | 工程收尾 | API 契约全通、docker compose 一键起、README 截图齐全、Git 提交规范 |

---

## 13. 交付文件清单

```
README.md          架构图/快速开始/截图/环境变量表/目录说明
EVALUATION.md      指标 + 复现命令 + 三组对比
INTERVIEW.md       10 个高频追问与回答要点
data/SOURCES.md    语料来源与授权
docker-compose.yml
scripts/  app/  frontend/  tests/  data/eval/
```

README 必须含：Mermaid 架构图、两张效果截图（问答+知识库管理）、一键启动命令、FAQ 通道与 RAG 链路的分流示意图。
INTERVIEW.md 建议覆盖：为什么做双通道 / 路由阈值怎么定的 / HyDE 何时有效 / 缓存一致性取舍 / 混合检索融合方案 / 拒答如何设计 / 评测集怎么建 / 失败案例与改进。

---

## 14. 给 Claude Code 的执行约束

1. 先读完整份规格，按 M1→M6 顺序实现，禁止跳过测试
2. 每个里程碑完成即提交 Git，commit message 含里程碑号（`M1: ...`）
3. 遇规格歧义：优先按规格默认值实现并在 README/代码注释标注假设；不要询问阻塞
4. 不安装未经列表允许的重量级依赖；新增依赖需在 requirements.txt 注明用途
5. 生成文件后自查一遍"机构字样/密钥/虚构指标"三类问题
6. 最终交付前运行全部测试与一次端到端演示，产出截图
*（内容由AI生成，仅供参考）*
