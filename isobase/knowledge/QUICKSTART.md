# IsoBase Knowledge Base — Quick Start

- 汉语 [QUICKSTART.zh-cn.md](QUICKSTART.zh-cn.md)
- English [QUICKSTART.md](QUICKSTART.md)

## Installation

IsoBase knowledge module is included in the main package. No additional
dependencies are required for the basic path (memory store, dense retrieval).

For production use with real embedding APIs or rerank models:

```bash
pip install openai
```

## 5-Minute Quick Start

### 1. Fake Embeddings (no API calls)

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

kb = service.create_knowledge_base(name="My Knowledge Base")
service.index_text(kb.id, "RAG is a technique for LLMs.", title="Doc 1")
results = service.retrieve("What is RAG?", kb.id, top_k=3)
print(results[0].chunk.content)
```

### 2. Production Setup (Real Embedding APIs)

```python
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.stores import MemoryKnowledgeStore

# Pick one provider:
client = OpenAIEmbeddingClient(
    api_key="...",
    base_url="https://api.aimlapi.com/v1",          # AIMLAPI
    # base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # DashScope
    model="alibaba/qwen-text-embedding-v4",          # or "text-embedding-v4"
    dimensions=1024,
)

service = KnowledgeBaseService(
    embedding_client=client,
    store=MemoryKnowledgeStore(),
    chunker=FixedSizeChunker(chunk_size=512, chunk_overlap=50),
)
```

## Retrieval Modes

### Dense (default)

```python
results = service.retrieve("What is RAG?", kb.id, top_k=5)
```

### Sparse (BM25)

```python
from isobase.knowledge.entities import RetrievalOption

results = service.retrieve(
    "What is RAG?", kb.id,
    options=RetrievalOption(top_k=5, use_sparse=True),
)
```

### Hybrid (Dense + Sparse + RRF Fusion)

```python
results = service.retrieve(
    "What is RAG?", kb.id,
    options=RetrievalOption(top_k=5, use_hybrid=True),
)
```

### With Metadata Filter

```python
results = service.retrieve(
    "function", kb.id,
    options=RetrievalOption(
        top_k=5,
        metadata_filter={"heading_path": ["Installation"]},
    ),
)
```

### With Reranker

```python
from isobase.knowledge.retrieval import RetrievalPipeline
from isobase.knowledge.retrieval.providers import OpenAIReranker

reranker = OpenAIReranker(
    api_key="...",
    base_url="https://{workspace}.cn-beijing.maas.aliyuncs.com/compatible-api/v1",
    model="qwen3-rerank",
)
service._pipeline = RetrievalPipeline(service.store, reranker=reranker)

results = service.retrieve("What is RAG?", kb.id, top_k=5)
# results have score_source="rerank"
```

## Using Low-Level Retrieval Components

```python
from isobase.knowledge.retrieval import (
    RankFusion,
    SparseRetriever,
    SparseRetrievalItem,
    curated_stopwords,
)

# BM25 sparse retrieval
retriever = SparseRetriever(
    k1=1.5, b=0.75,
    stopwords=curated_stopwords(),
)
items = [SparseRetrievalItem(chunk=c, document=doc) for c, doc in get_my_chunks()]
sparse_results = retriever.retrieve("query text", items, top_k=5)

# RRF fusion (combine with dense results from store)
fusion = RankFusion(k=60)
fused = fusion.fuse(dense_results, sparse_results, top_k=10)
# fused results have score_source="fused"
```

## Storage Backends

```python
# Memory (dev / testing)
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

## LLM Integration

```python
from isobase.llm.tools import ToolSet
from isobase.llm.tools.knowledge import create_knowledge_search_tool

toolset = ToolSet()
toolset.add_tool(create_knowledge_search_tool(service, kb.id, top_k=3))

client.ask("What is RAG?", tools=list(toolset))
```

## Run Examples

```bash
# Unit tests (no API keys required)
python -m pytest test/knowledge/ -v

# Fake embeddings — no API or database required
python -m test.knowledge.live.run_knowledge_basic

# --- Live smoke tests (require API keys and/or databases) ---

# Markdown chunker live test
python -m test.knowledge.live.run_markdown_chunking
# Prerequisites: .env with DASHSCOPE_API_KEY, MongoDB accessible at localhost:27017

# Reranker live test
python -m test.knowledge.live.run_reranker
# Prerequisites: .env with EMBEDDING_API_KEY, RERANK_API_KEY, RERANK_BASE_URL

# Lifecycle live tests
python -m test.knowledge.live.run_lifecycle_sql
# Prerequisites: PostgreSQL accessible at localhost:5432

python -m test.knowledge.live.run_lifecycle_mongo
# Prerequisites: MongoDB accessible at localhost:27017
```

## Common Issues

### ModuleNotFoundError: No module named 'openai'

```bash
pip install openai
```

### No results from hybrid / sparse search

Sparse and hybrid modes require knowledge to have been indexed **with
`query_text` metadata**. The default `KnowledgeBaseService.retrieve()`
auto-embeds the query and passes both `query_embedding` and `query_text`
when needed.

### Empty search results

1. Documents were indexed: `service.index_text(kb.id, ...)`
2. Query string is not empty
3. Knowledge base ID is correct
4. Embedding dimensions match between client and store
