# 知识库模块

为 IsoBase 提供厂商中立的知识库和 RAG（检索增强生成）能力。

## 概述

本模块提供文档索引和相关上下文检索工具，用于 LLM 应用。遵循 IsoBase 的设计原则：

- **厂商中立**：支持任何 OpenAI 兼容的嵌入 API
- **模块化**：分块、嵌入、存储和检索之间清晰分离
- **最小依赖**：从简单的内存存储开始，支持 SQL 持久化
- **IsoBase 规范**：与现有 `llm/` 和 `database/` 模块无缝集成

## 快速开始

```python
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.stores import MemoryKnowledgeStore

# 初始化组件
embedding_client = OpenAIEmbeddingClient(
    api_key="your_api_key",
    base_url="https://api.aimlapi.com/v1",
    model="alibaba/qwen-text-embedding-v4",
    dimensions=1024,
)

chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=50)
store = MemoryKnowledgeStore()

service = KnowledgeBaseService(
    embedding_client=embedding_client,
    store=store,
    chunker=chunker,
    embed_batch_size=20,  # 每次 API 调用的最大块数（默认 20，兼容 DashScope）
)

# 创建知识库
kb = service.create_knowledge_base(
    name="产品文档",
    description="内部产品文档和常见问题",
)

# 索引文档
service.index_text(
    knowledge_base_id=kb.id,
    text="RAG 代表检索增强生成。它结合了检索和生成...",
    title="RAG 概述",
    source_uri="docs/rag.md",
)

# 检索相关块
results = service.retrieve(
    query="什么是 RAG？",
    knowledge_base_id=kb.id,
    top_k=3,
)

for result in results:
    print(f"分数: {result.score:.3f}")
    print(f"来源: {result.document.title}")
    print(f"内容: {result.chunk.content[:100]}...")
```

## LLM 集成

### 方式 1：基于工具（推荐）

让 LLM 自动决定何时搜索知识。
使用 `isobase.llm.tools.knowledge` 中提供的便捷工厂函数：

```python
from isobase.llm import OpenAIChat
from isobase.llm.tools import ToolSet
from isobase.llm.tools.knowledge import create_knowledge_search_tool

# 创建知识搜索工具
kb_tool = create_knowledge_search_tool(service, kb.id, top_k=3)

# 添加到工具集
toolset = ToolSet()
toolset.add_tool(kb_tool)

# 初始化 LLM 客户端
llm = OpenAIChat(api_key="your_openai_key")

# 使用工具生成
response = llm.generate(
    messages=[
        {"role": "user", "content": "RAG 是什么以及如何工作？"}
    ],
    tools=toolset.to_openai_schema(),
)

# 处理工具调用
if response.tool_calls:
    tool_results, _ = toolset.execute_tool_calls(response.tool_calls)

    # 将工具结果返回给 LLM
    messages = [
        {"role": "user", "content": "RAG 是什么以及如何工作？"},
        {"role": "assistant", "tool_calls": response.tool_calls},
        {"role": "tool", "content": str(tool_results[0])}
    ]

    final_response = llm.generate(messages=messages)
    print(final_response.content)
```

### 方式 2：手动上下文注入

直接将检索到的上下文注入提示：

```python
query = "什么是 RAG？"
context = service.retrieve_as_context(
    query=query,
    knowledge_base_id=kb.id,
    top_k=3,
)

prompt = f"""使用提供的上下文回答以下问题。

上下文：
{context}

问题：{query}

回答："""

response = llm.generate(messages=[{"role": "user", "content": prompt}])
print(response.content)
```

## 支持的嵌入提供商

### Qwen Text Embedding v4 (AIMLAPI)

```python
from isobase.knowledge.embeddings import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient(
    api_key="your_aimlapi_key",
    base_url="https://api.aimlapi.com/v1",
    model="alibaba/qwen-text-embedding-v4",
    dimensions=1024,  # 选项: 512, 1024, 2048
)
```

**特性：**

- 上下文: 32K tokens
- 多语言: 100+ 种语言
- 性能: MTEB 68.36 (英文), 70.14 (中文) @ 1024 维

### 阿里云百炼 DashScope

```python
import os
from isobase.knowledge.embeddings import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v4",
    dimensions=1024,
)
```

**特性：**

- 中文原生优化
- 批处理: 最多 20 个文本/请求（`KnowledgeBaseService` 自动分批）
- 模型: v3, v4

### OpenAI

```python
import os
from isobase.knowledge.embeddings import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small",  # 或 text-embedding-3-large
)
```

**模型：**

- `text-embedding-3-small`: 1536 维，更快，更便宜
- `text-embedding-3-large`: 3072 维，更高质量

## 架构

```
isobase/llm/tools/knowledge/
│   ├── __init__.py      # create_knowledge_search_tool 工厂函数
│   └── tools.py         # 工具实现

isobase/knowledge/
├── entities.py          # 核心 DTO（KnowledgeBase, Document, Chunk, RetrievalResult, RetrievalOption）
├── embeddings/
│   ├── base.py         # BaseEmbeddingClient 抽象基类
│   └── openai.py # OpenAI 兼容客户端
├── chunking/
│   ├── base.py         # BaseChunker ABC + ChunkSection
│   ├── fixed.py        # 固定大小分块器（带重叠）
│   └── markdown.py     # Markdown 标题感知分块器
├── stores/
│   ├── base.py         # BaseKnowledgeStore 抽象基类
│   ├── memory.py       # 内存存储（MVP）
│   ├── sql.py          # SQL 后端存储
│   └── mongo.py        # MongoDB 后端存储
├── retrieval/
│   ├── __init__.py     # 检索模块导出
│   ├── base.py         # BaseRetriever 抽象基类 + DenseRetrievalItem
│   ├── dense.py        # 稠密向量检索器（余弦相似度）
│   ├── sparse.py       # 稀疏检索器（BM25，可插拔 tokenizer，停用词）
│   ├── pipeline.py     # RetrievalPipeline（dense/sparse/hybrid，metadata filter，top_k trim）
│   ├── reranker.py     # BaseReranker 抽象基类 + NoOpReranker
│   ├── rank_fusion.py  # RankFusion（RRF 融合）+ FusedResult
│   └── providers/      # 厂商特定实现
│       └── openai_reranker.py
├── service.py          # KnowledgeBaseService 编排
└── tools.py            # LLM 工具包装器
```

## 核心概念

### 知识库

带配置的索引文档逻辑集合。知识库实体是**分块参数的唯一真相来源**：
`index_text()` 会读取 KB 的 `chunk_size` / `chunk_overlap` 并传给 chunker，
因此同一个 KB 内的所有文档始终使用相同参数分块。更换 service 上的 chunker
实例**不会影响**已创建的 KB 的行为。

- 嵌入模型标识符
- 分块策略（大小、重叠）
- 向量维度

### 文档

包含以下内容的顶级索引单元：

- 原始内容
- 标题和来源 URI
- 元数据（作者、日期、标签）

### 块

检索单元：

- 从父文档拆分
- 作为向量嵌入
- 通过相似性搜索检索

### 检索结果

包含以下内容的搜索结果：

- 块内容
- 相似度分数
- 父文档引用
- 分数来源（`"dense"`、`"sparse"`、`"fused"`、`"rerank"`）

### 检索选项 (RetrievalOption)

控制检索行为：

- `top_k` — 返回结果数量（默认 5）
- `candidate_k` — 最终筛选前获取的候选数（默认等于 `top_k`）
- `metadata_filter` — 块元数据等值过滤
- `use_sparse` — 使用 BM25 稀疏检索代替稠密检索（默认 `False`）
- `use_hybrid` — 双路稠密 + 稀疏检索，通过 RRF 融合（默认 `False`）

### 检索管线 (RetrievalPipeline)

编排检索流程：从存储获取候选 → 应用 metadata filter → （可选）rerank → 返回 `top_k`。管线是检索逻辑的**唯一入口点**——存储后端只需提供 `search(kb_id, query_embedding, top_k)`，无需关心 metadata filter、candidate/final_k 或检索模式。

**检索模式：**

| 模式           | 选项                                 | 说明                                                 |
| -------------- | ------------------------------------ | ---------------------------------------------------- |
| 纯稠密（默认） | `use_sparse=False, use_hybrid=False` | 余弦相似度向量检索                                   |
| 纯稀疏         | `use_sparse=True`                    | BM25 关键词匹配（需要 `query_text`）                 |
| 混合检索       | `use_hybrid=True`                    | 稠密 + 稀疏 → RRF 融合 → rerank（需要 `query_text`） |

### 稀疏检索 (BM25)

`SparseRetriever` 提供零外部依赖的关键词检索：

```python
from isobase.knowledge.retrieval import SparseRetriever, curated_stopwords

retriever = SparseRetriever(
    k1=1.5,
    b=0.75,
    stopwords=curated_stopwords(),  # 400+ 精选中英文停用词
)
results = retriever.retrieve("python 装饰器", items, top_k=5)
```

**特性：**

- 纯 Python BM25 — 零外部依赖
- 可插拔 `tokenizer` — 可注入 `jieba.cut` 支持中文分词
- 内置精选中英文停用词表
- 可配置 BM25 参数（`k1`、`b`）

### 倒数排名融合 (RRF)

`RankFusion` 类按排名位置合并稠密和稀疏结果：

```python
from isobase.knowledge.retrieval import RankFusion

fusion = RankFusion(k=60)  # 平滑常数
fused = fusion.fuse(dense_results, sparse_results, top_k=20)
# 融合结果 score_source="fused"
```

### 重排序器钩子 (Reranker Hook)

通过 `BaseReranker` 接口实现检索后重排序：

```python
from isobase.knowledge.retrieval.providers import OpenAIReranker

reranker = OpenAIReranker(
    api_key="...",
    base_url="https://.../compatible-api/v1",
    model="qwen3-rerank",
)
pipeline = RetrievalPipeline(store, reranker=reranker)
# 结果 score_source="rerank"
```

### 分数来源语义

每个 `RetrievalResult` 携带明确的 `score_source`：

| 来源       | 含义                      |
| ---------- | ------------------------- |
| `"dense"`  | 余弦相似度（嵌入向量）    |
| `"sparse"` | BM25 关键词相关性分数     |
| `"fused"`  | 稠密 + 稀疏的倒数排名融合 |
| `"rerank"` | 模型重排序器分数          |

## 分块策略

分块参数由**知识库实体**而非 chunker 实例拥有。创建知识库时，chunker 的当前默认值会被写入 KB 实体；后续每次调用 `index_text()` 都会从 KB 实体读取存储的参数并传递给 chunker。这使得 **同一个 service 实例可以管理多份配置不同的知识库** ——每个 KB 使用自己的参数分块，chunker 实例只提供默认值。

### 固定大小分块器

带重叠的简单基于字符的分割：

```python
from isobase.knowledge.chunking import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=512,    # 默认每块最大字符数
    chunk_overlap=50,  # 默认块之间的重叠
)

# 调用时 KB 的参数会覆盖 chunker 默认值：
chunks = chunker.chunk("长文本内容...",
                        chunk_size=1024,
                        chunk_overlap=100)
```

**优点：**

- 简单可预测
- 快速处理
- 无依赖

**缺点：**

- 可能在句子中间分割
- 不尊重文档结构

### Markdown 分块器

结构感知切分，按 ATX 标题边界切分并保留标题路径元数据：

```python
from isobase.knowledge.chunking import MarkdownChunker

chunker = MarkdownChunker(
    chunk_size=500,    # 默认每块最大字符数
    chunk_overlap=60,  # 默认重叠（仅在同一 section 内生效）
)

# 每个 chunk 携带 heading_path 元数据：
chunks = chunker.chunk(markdown_text)
# chunks[0].content       → "# 引言\n\n你好世界"
# chunks[0].metadata      → {"chunk_strategy": "markdown",
#                             "heading_path": ["引言"]}
```

**关键行为：**

- **chunk_size 是上限** — 短标题 section 不会跨标题边界合并
- **chunk_overlap 仅在 section 内部生效** — section 间为硬边界，无重叠
- **超大 section** 由 `FixedSizeChunker` 进一步细分，子 chunk 继承相同 `heading_path`
- **文本规范化**（BOM 去除、行尾统一、ATX 间距等）由 `isobase.utils.MarkdownFixer` 处理

### 未来策略

`BaseChunker.chunk()` 返回 `List[ChunkSection]` — 每个 section 携带 `content` 和
`metadata`（strategy 名称、heading path 等）。`KnowledgeBaseService.index_text()`
自动将这些 metadata 存入 `KnowledgeChunk.metadata`。

## 存储后端

### 内存存储（MVP）

用于测试和演示的内存存储：

```python
from isobase.knowledge.stores import MemoryKnowledgeStore

store = MemoryKnowledgeStore()
```

**优点：**

- 零设置
- 快速
- 非常适合测试

**缺点：**

- 重启后数据丢失
- 非线程安全
- 仅限于可用 RAM

**使用场景：** 测试、演示、小数据集（<10K 块）

### SQL 存储

使用 SQLAlchemy 的持久存储（SQLite / PostgreSQL）：

```python
from isobase.knowledge.stores import SqlKnowledgeStore
from isobase.database.sql import SqlDbService

db_service = SqlDbService(connection_url="sqlite:///knowledge.db")
store = SqlKnowledgeStore(db_service)
```

**特性：**

- 持久存储
- SQLite 和 PostgreSQL 支持
- 与现有 `database/sql.py` 集成
- `document.content` 是全文的唯一持久来源；chunk 内容在读取时派生

### MongoDB 存储

使用 MongoDB 的持久化存储：

```python
from isobase.knowledge.stores import MongoKnowledgeStore
from isobase.database.mongo import MongoDbService

mongo_service = MongoDbService(
    uri="mongodb://localhost:27017",
    database_name="isobase",
)
store = MongoKnowledgeStore(mongo_service)

# 也可以直接使用全局默认 mongo_db 实例：
# store = MongoKnowledgeStore()
```

**特性：**

- 持久存储，使用原生 BSON 类型（embedding 无需 JSON 序列化）
- MongoDB 自动生成 BSON `ObjectId` 作为文档标识符
- BSON `array<double>` 以紧凑二进制形式存储 embedding 向量——比 JSON 文本更省空间
- 与现有 `database/mongo.py` 集成
- 适合生产级服务端部署

## 路线图

### Phase 1: MVP

- ✅ 核心接口（`entities.py`）
- ✅ 固定大小分块器（`FixedSizeChunker`）
- ✅ OpenAI 兼容嵌入客户端（支持 `dimensions` 自动检测）
- ✅ `ModelRegistry` — 嵌入模型注册表，支持 lazy 维度获取
- ✅ 内存存储（`MemoryKnowledgeStore`）
- ✅ `KnowledgeBaseService` — 高层索引 + 检索编排
- ✅ LLM 工具集成（`create_knowledge_search_tool`）

### Phase 2: 持久化

- ✅ SQL 后端存储 (`SqlKnowledgeStore`)
- ✅ MongoDB 后端存储 (`MongoKnowledgeStore`)
- ✅ 生命周期 API: 文档和知识库的 `get` / `list` / `delete`
- ✅ `KnowledgeBaseService` 构造函数 `embed_batch_size`
- ✅ 共享 store 契约测试（27 项，参数化覆盖 memory / SQL / Mongo）
- [ ] 模式迁移

### Phase 3: Markdown 结构感知分块

- ✅ `ChunkSection(content, metadata)` 返回类型，`BaseChunker.chunk() → List[ChunkSection]`
- ✅ `FixedSizeChunker` — 返回带 `chunk_strategy: "fixed"` 的 `ChunkSection`
- ✅ `MarkdownChunker` — ATX 标题感知分块，附带 `heading_path` 追踪
- ✅ `TextFixer.normalize()` — BOM 去除、行尾统一、ATX 间距规范化
- ✅ `TextFixer.extract_front_matter()` — YAML 前置元数据提取
- ✅ `KnowledgeBaseService.index_text()` — 将 chunker 元数据存入 `KnowledgeChunk`
- ✅ 42 个单元测试（6 chunking + 36 text fixer）

### Phase 4: 检索增强

- ✅ `RetrievalOption` — `candidate_k`、`top_k`、metadata filter、`use_sparse`、`use_hybrid`
- ✅ `RetrievalPipeline` — dense / sparse / hybrid 检索的唯一入口
- ✅ `RetrievalResult.score_source` — 明确分数语义（`"dense"`、`"sparse"`、`"fused"`、`"rerank"`）
- ✅ `SparseRetriever` — BM25 关键词检索，可插拔 tokenizer + 停用词
- ✅ `RankFusion` — 倒数排名融合（RRF），合并稠密 + 稀疏结果
- ✅ `BaseReranker` / `NoOpReranker` — 重排序器钩子接口（可选，默认无操作）
- ✅ `OpenAIReranker` — OpenAI 兼容重排序提供商（qwen3-rerank 等）
- ✅ `list_chunks(kb_id)` — store 契约扩展（sparse / hybrid 检索需要）
- ✅ 62 个检索单元测试（dense, sparse, pipeline, reranker, RRF）

### Phase 5: 文档解析器

- [ ] 文本解析器
- [ ] PDF 提取
- [ ] 网页提取

### Phase 6: 高级功能

- [ ] 异步/批量索引
- [ ] 增量更新
- [ ] 多知识库搜索
- ✅ 元数据过滤
- ✅ 混合检索（稠密 + 稀疏 + RRF）
- ✅ `candidate_k` / `top_k` 管线控制（`RetrievalOption.effective_candidate_k()`）
- [ ] pgvector 迁移
- [ ] MongoDB Atlas Vector Search

## 参考

- [架构设计](../../docs/design/knowledge_base_architecture.md)
- [Qwen v4 API 参考](https://docs.aimlapi.com/api-references/embedding-models/alibaba-cloud/qwen-text-embedding-v4)
- [DashScope API 参考](https://help.aliyun.com/zh/model-studio/embedding)
- [AstrBot Knowledge Base](https://github.com/AstrBotDevs/AstrBot/tree/master/astrbot/core/knowledge_base)
- [Cherry Studio Knowledge](https://github.com/cherryhq/cherry-studio)
