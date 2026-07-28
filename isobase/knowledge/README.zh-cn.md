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
from isobase.knowledge.embeddings import OpenAICompatEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.stores import MemoryKnowledgeStore

# 初始化组件
embedding_client = OpenAICompatEmbeddingClient(
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

让 LLM 自动决定何时搜索知识：

```python
from isobase.llm.providers.openai_chat import OpenAIChat
from isobase.llm.tools.base import ToolSet
from isobase.knowledge.tools import create_knowledge_search_tool

# 创建知识搜索工具
kb_tool = create_knowledge_search_tool(
    service=service,
    knowledge_base_id=kb.id,
    top_k=3,
)

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
from isobase.knowledge.embeddings import OpenAICompatEmbeddingClient

client = OpenAICompatEmbeddingClient(
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
from isobase.knowledge.embeddings import OpenAICompatEmbeddingClient

client = OpenAICompatEmbeddingClient(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v4",
    dimensions=1024,
)
```

**特性：**
- 中文原生优化
- 批处理: 最多 25 个文本/请求
- 模型: v3, v4

### OpenAI

```python
import os
from isobase.knowledge.embeddings import OpenAICompatEmbeddingClient

client = OpenAICompatEmbeddingClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small",  # 或 text-embedding-3-large
)
```

**模型：**
- `text-embedding-3-small`: 1536 维，更快，更便宜
- `text-embedding-3-large`: 3072 维，更高质量

## 架构

```
isobase/knowledge/
├── entities.py          # 核心 DTO（KnowledgeBase, Document, Chunk, RetrievalResult）
├── embeddings/
│   ├── base.py         # BaseEmbeddingClient 抽象基类
│   └── openai_compat.py # OpenAI 兼容客户端
├── chunking/
│   ├── base.py         # BaseChunker 抽象基类
│   └── fixed.py        # 固定大小分块器（带重叠）
├── stores/
│   ├── base.py         # BaseKnowledgeStore 抽象基类
│   ├── memory.py       # 内存存储（MVP）
│   └── sql.py          # SQL 后端存储（未来）
├── retrieval/          # 混合检索（未来）
├── service.py          # KnowledgeBaseService 编排
└── tools.py            # LLM 工具包装器
```

## 核心概念

### 知识库

带配置的索引文档逻辑集合：
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

## 分块策略

### 固定大小分块器

带重叠的简单基于字符的分割：

```python
from isobase.knowledge.chunking import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=512,    # 每块最大字符数
    chunk_overlap=50,  # 块之间的重叠
)

chunks = chunker.chunk("长文本内容...")
```

**优点：**
- 简单可预测
- 快速处理
- 无依赖

**缺点：**
- 可能在句子中间分割
- 不尊重文档结构

**未来：** 递归字符分割器、语义分块、结构感知分块。

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

### SQL 存储（未来）

使用 SQLAlchemy 的持久存储：

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

## 路线图

### Phase 1: MVP（当前）
- ✅ 核心接口
- ✅ 固定大小分块器
- ✅ OpenAI 兼容嵌入客户端
- ✅ 内存存储
- ✅ KnowledgeBaseService
- ✅ LLM 工具集成

### Phase 2: SQL 持久化
- [ ] SQL 后端存储
- [ ] 模式迁移
- [ ] 向量存储（初始为 JSON）

### Phase 3: 增强检索
- [ ] 混合检索（稠密 + 稀疏）
- [ ] BM25 稀疏检索器
- [ ] RRF（倒数排名融合）
- [ ] 重排序器接口

### Phase 4: 文档解析器
- [ ] 文本解析器
- [ ] Markdown 解析器
- [ ] PDF 提取
- [ ] 网页提取

### Phase 5: 高级功能
- [ ] 异步/批量索引
- [ ] 增量更新
- [ ] 多知识库搜索
- [ ] 元数据过滤
- [ ] pgvector 迁移

## 参考

- [架构设计](../../docs/design/knowledge_base_architecture.md)
- [Qwen v4 API 参考](https://docs.aimlapi.com/api-references/embedding-models/alibaba-cloud/qwen-text-embedding-v4)
- [DashScope API 参考](https://help.aliyun.com/zh/model-studio/embedding)
- [AstrBot Knowledge Base](https://github.com/AstrBotDevs/AstrBot/tree/master/astrbot/core/knowledge_base)
- [Cherry Studio Knowledge](https://github.com/cherryhq/cherry-studio)
