# IsoBase Knowledge Base — Quick Start

- 汉语 [QUICKSTART.zh-cn.md](QUICKSTART.zh-cn.md)
- English [QUICKSTART.md](QUICKSTART.md)

## 安装

IsoBase 知识库模块已包含在主包中，基本路径（内存存储、稠密检索）无需额外依赖。

如需使用真实嵌入 API 或重排序模型：

```bash
pip install openai
```

## 5 分钟快速上手

### 1. 虚拟嵌入（无需 API 调用）

```python
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.embeddings import BaseEmbeddingClient
from isobase.knowledge.stores import MemoryKnowledgeStore

class FakeEmbedding(BaseEmbeddingClient):
    def embed_texts(self, texts, **kwargs):
        return [[float(i)] * 128 for i in range(len(texts))]
    def embed_query(self, query, **kwargs):
        return [0.5] * 128
    @property
    def dimensions(self):
        return 128

service = KnowledgeBaseService(
    embedding_client=FakeEmbedding(),
    store=MemoryKnowledgeStore(),
    chunker=FixedSizeChunker(chunk_size=512, chunk_overlap=50),
)

kb = service.create_knowledge_base(name="我的知识库")
service.index_text(kb.id, "RAG 是面向 LLM 的检索增强技术。", title="文档 1")
results = service.retrieve("什么是 RAG？", kb.id, top_k=3)
print(results[0].chunk.content)
```

### 2. 生产环境（真实嵌入 API）

```python
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.stores import MemoryKnowledgeStore

# 任选一家：
client = OpenAIEmbeddingClient(
    api_key="...",
    base_url="https://api.aimlapi.com/v1",          # AIMLAPI
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # DashScope
    model="alibaba/qwen-text-embedding-v4",          # 或 "text-embedding-v4"
    dimensions=1024,
)

service = KnowledgeBaseService(
    embedding_client=client,
    store=MemoryKnowledgeStore(),
    chunker=FixedSizeChunker(chunk_size=512, chunk_overlap=50),
)
```

## 检索模式

### 稠密检索（默认）

```python
results = service.retrieve("什么是 RAG？", kb.id, top_k=5)
```

### 稀疏检索 (BM25)

```python
from isobase.knowledge.entities import RetrievalOption

results = service.retrieve(
    "什么是 RAG？", kb.id,
    options=RetrievalOption(top_k=5, use_sparse=True),
)
```

### 混合检索（稠密 + 稀疏 + RRF 融合）

```python
results = service.retrieve(
    "什么是 RAG？", kb.id,
    options=RetrievalOption(top_k=5, use_hybrid=True),
)
```

### 带元数据过滤

```python
results = service.retrieve(
    "函数", kb.id,
    options=RetrievalOption(
        top_k=5,
        metadata_filter={"heading_path": ["安装说明"]},
    ),
)
```

### 带重排序器

```python
from isobase.knowledge.retrieval import RetrievalPipeline
from isobase.knowledge.retrieval.providers import OpenAIReranker

reranker = OpenAIReranker(
    api_key="...",
    base_url="https://{workspace}.cn-beijing.maas.aliyuncs.com/compatible-api/v1",
    model="qwen3-rerank",
)
service._pipeline = RetrievalPipeline(service.store, reranker=reranker)

results = service.retrieve("什么是 RAG？", kb.id, top_k=5)
# 结果 score_source="rerank"
```

## 使用底层检索组件

```python
from isobase.knowledge.retrieval import (
    RankFusion,
    SparseRetriever,
    SparseRetrievalItem,
    curated_stopwords,
)

# BM25 稀疏检索
retriever = SparseRetriever(
    k1=1.5, b=0.75,
    stopwords=curated_stopwords(),
)
items = [SparseRetrievalItem(chunk=c, document=doc) for c, doc in get_my_chunks()]
sparse_results = retriever.retrieve("查询文本", items, top_k=5)

# RRF 融合（与 store 的稠密结果合并）
fusion = RankFusion(k=60)
fused = fusion.fuse(dense_results, sparse_results, top_k=10)
# 融合结果 score_source="fused"
```

## 存储后端

```python
# 内存（开发 / 测试）
from isobase.knowledge.stores import MemoryKnowledgeStore
store = MemoryKnowledgeStore()

# SQLite / PostgreSQL
from isobase.knowledge.stores import SqlKnowledgeStore
from isobase.database.sql import SqlDbService
store = SqlKnowledgeStore(SqlDbService(connection_url="sqlite:///knowledge.db"))

# MongoDB
from isobase.knowledge.stores import MongoKnowledgeStore
store = MongoKnowledgeStore()
```

## LLM 集成

```python
from isobase.llm.tools import ToolSet
from isobase.llm.tools.knowledge import create_knowledge_search_tool

toolset = ToolSet()
toolset.add_tool(create_knowledge_search_tool(service, kb.id, top_k=3))

client.ask("什么是 RAG？", tools=list(toolset))
```

## 运行示例

```bash
# 单元测试（无需 API 密钥）
python -m pytest test/knowledge/ -v

# 虚拟嵌入 — 无需 API 或数据库
python -m test.knowledge.live.run_knowledge_basic

# --- 以下联调测试需要 API 密钥和/或数据库 ---

# Markdown 分块器联调测试
python -m test.knowledge.live.run_markdown_chunking
# 运行条件：.env 中配置 DASHSCOPE_API_KEY，localhost:27017 可访问 MongoDB

# 重排序器联调测试
python -m test.knowledge.live.run_reranker
# 运行条件：.env 中配置 EMBEDDING_API_KEY、RERANK_API_KEY、RERANK_BASE_URL

# 生命周期联调测试
python -m test.knowledge.live.run_lifecycle_sql
# 运行条件：localhost:5432 可访问 PostgreSQL

python -m test.knowledge.live.run_lifecycle_mongo
# 运行条件：localhost:27017 可访问 MongoDB
```

## 常见问题

### ModuleNotFoundError: No module named 'openai'

```bash
pip install openai
```

### 混合 / 稀疏检索无结果

稀疏和混合检索需要知识库在索引时已嵌入文本。默认的 `KnowledgeBaseService.retrieve()` 会自动对查询文本进行嵌入，需要时同时传递 `query_embedding` 和 `query_text`。

### 检索结果为空

请检查：

1. 文档是否已索引：`service.index_text(kb.id, ...)`
2. 查询字符串是否非空
3. 知识库 ID 是否正确
4. 嵌入维度是否在客户端与存储间匹配
