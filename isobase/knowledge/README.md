# Knowledge Base Module

Provider-neutral knowledge base and RAG (Retrieval-Augmented Generation) capabilities for IsoBase.

## Overview

This module provides tools for indexing documents and retrieving relevant context for LLM applications. It follows IsoBase's design principles:

- **Provider neutrality**: Works with any OpenAI-compatible embedding API
- **Modularity**: Clean separation between chunking, embedding, storage, and retrieval
- **Minimal dependencies**: Starts with simple in-memory storage, supports SQL persistence
- **IsoBase conventions**: Integrates seamlessly with existing `llm/` and `database/` modules

## Quick Start

```python
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.stores import MemoryKnowledgeStore

# Initialize components
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
    embed_batch_size=20,  # chunks per API call (default 20 — DashScope-safe)
)

# Create knowledge base
kb = service.create_knowledge_base(
    name="Product Documentation",
    description="Internal product docs and FAQs",
)

# Index documents
service.index_text(
    knowledge_base_id=kb.id,
    text="RAG stands for Retrieval-Augmented Generation. It combines retrieval and generation...",
    title="RAG Overview",
    source_uri="docs/rag.md",
)

service.index_text(
    knowledge_base_id=kb.id,
    text="Vector databases store high-dimensional embeddings for similarity search...",
    title="Vector Databases",
    source_uri="docs/vector-db.md",
)

# Retrieve relevant chunks
results = service.retrieve(
    query="What is RAG?",
    knowledge_base_id=kb.id,
    top_k=3,
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Source: {result.document.title}")
    print(f"Content: {result.chunk.content[:100]}...")
    print("---")
```

## LLM Integration

### Option 1: Tool-based (Recommended)

Let the LLM automatically decide when to search knowledge.
Use the convenience factory shipped in `isobase.llm.tools.knowledge`:

```python
from isobase.llm import OpenAIChat
from isobase.llm.tools import ToolSet
from isobase.llm.tools.knowledge import create_knowledge_search_tool

# Create knowledge search tool
kb_tool = create_knowledge_search_tool(service, kb.id, top_k=3)

# Add to toolset
toolset = ToolSet()
toolset.add_tool(kb_tool)

# Initialize LLM client
from isobase.llm import OpenAIChat
llm = OpenAIChat(api_key="your_openai_key")

# Generate with tool access
response = llm.generate(
    messages=[
        {"role": "user", "content": "What is RAG and how does it work?"}
    ],
    tools=toolset.to_openai_schema(),
)

# Handle tool calls
if response.tool_calls:
    tool_results, _ = toolset.execute_tool_calls(response.tool_calls)

    # Send tool results back to LLM
    messages = [
        {"role": "user", "content": "What is RAG and how does it work?"},
        {"role": "assistant", "tool_calls": response.tool_calls},
        {"role": "tool", "content": str(tool_results[0])}
    ]

    final_response = llm.generate(messages=messages)
    print(final_response.content)
```

### Option 2: Manual Context Injection

Directly inject retrieved context into prompts:

```python
query = "What is RAG?"
context = service.retrieve_as_context(
    query=query,
    knowledge_base_id=kb.id,
    top_k=3,
)

prompt = f"""Answer the following question using the provided context.

Context:
{context}

Question: {query}

Answer:"""

response = llm.generate(messages=[{"role": "user", "content": prompt}])
print(response.content)
```

## Supported Embedding Providers

### Qwen Text Embedding v4 (AIMLAPI)

```python
from isobase.knowledge.embeddings import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient(
    api_key="your_aimlapi_key",
    base_url="https://api.aimlapi.com/v1",
    model="alibaba/qwen-text-embedding-v4",
    dimensions=1024,  # Options: 512, 1024, 2048
)
```

**Features:**

- Context: 32K tokens
- Multilingual: 100+ languages
- Performance: MTEB 68.36 (EN), 70.14 (ZH) @ 1024-dim

### Alibaba DashScope (百炼)

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

**Features:**

- Native Chinese optimization
- Batch processing: up to 20 texts/request (auto-batched by `KnowledgeBaseService`)
- Models: v3, v4

### OpenAI

```python
import os
from isobase.knowledge.embeddings import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small",  # or text-embedding-3-large
)
```

**Models:**

- `text-embedding-3-small`: 1536-dim, faster, cheaper
- `text-embedding-3-large`: 3072-dim, higher quality

## Architecture

```
isobase/llm/tools/knowledge/
│   ├── __init__.py      # create_knowledge_search_tool factory
│   └── tools.py         # Tool implementation

isobase/knowledge/
├── entities.py          # Core DTOs (KnowledgeBase, Document, Chunk, RetrievalResult, RetrievalOption)
├── embeddings/
│   ├── base.py         # BaseEmbeddingClient ABC
│   └── openai.py # OpenAI-compatible client
├── chunking/
│   ├── base.py         # BaseChunker ABC + ChunkSection
│   ├── fixed.py        # Fixed-size chunker with overlap
│   └── markdown.py     # Markdown heading-aware chunker
├── stores/
│   ├── base.py         # BaseKnowledgeStore ABC
│   ├── memory.py       # In-memory store (MVP)
│   ├── sql.py          # SQL-backed store
│   └── mongo.py        # MongoDB-backed store
├── retrieval/
│   ├── __init__.py     # Retrieval exports
│   ├── base.py         # BaseRetriever ABC + DenseRetrievalItem
│   ├── dense.py        # Dense retriever (cosine similarity)
│   ├── sparse.py       # Sparse retriever (BM25, pluggable tokenizer, stopwords)
│   ├── pipeline.py     # RetrievalPipeline (dense/sparse/hybrid, metadata filter, top_k trim)
│   ├── reranker.py     # BaseReranker ABC + NoOpReranker
│   ├── rank_fusion.py  # RankFusion (RRF) + FusedResult
│   └── providers/      # Provider-specific implementations
│       └── openai_reranker.py
├── service.py          # KnowledgeBaseService orchestration
└── tools.py            # LLM tool wrappers
```

## Core Concepts

### Knowledge Base

A logical collection of indexed documents with configuration. The knowledge
base entity is the **single source of truth** for chunking parameters:
`index_text()` reads the KB's `chunk_size` / `chunk_overlap` and passes
them to the chunker, so every document in a given KB is split with the same
parameters. Changing the chunker instance on the service **does not** change
the behaviour of already-created KBs.

- Embedding model identifier
- Chunking strategy (size, overlap)
- Vector dimensions

### Document

Top-level indexing unit containing:

- Original content
- Title and source URI
- Metadata (author, date, tags)

### Chunk

Unit of retrieval:

- Split from parent document
- Embedded as vector
- Retrieved by similarity search

### Retrieval Result

Search result containing:

- Chunk content
- Similarity score
- Parent document reference
- Score source (`"dense"`, `"sparse"`, `"fused"`, `"rerank"`)

### RetrievalOption

Controls retrieval behavior:

- `top_k` — number of results to return (default 5)
- `candidate_k` — candidates to fetch before final trim (defaults to `top_k`)
- `metadata_filter` — chunk metadata equality filters
- `use_sparse` — use BM25 sparse retrieval instead of dense (default `False`)
- `use_hybrid` — dual dense + sparse retrieval via RRF fusion (default `False`)

### RetrievalPipeline

Orchestrates retrieval flow: gather candidates from store → apply metadata filter → (optional) rerank → return `top_k`. The pipeline is the **single entry point** for retrieval logic — store backends only provide `search(kb_id, query_embedding, top_k)` and don't need to know about metadata filters, candidate/final_k, or retrieval modes.

**Retrieval modes:**

| Mode                 | Option                               | Description                                                  |
| -------------------- | ------------------------------------ | ------------------------------------------------------------ |
| Dense-only (default) | `use_sparse=False, use_hybrid=False` | Cosine similarity vector search                              |
| Sparse-only          | `use_sparse=True`                    | BM25 keyword matching (requires `query_text`)                |
| Hybrid               | `use_hybrid=True`                    | Dense + sparse → RRF fusion → rerank (requires `query_text`) |

### Sparse Retrieval (BM25)

The `SparseRetriever` provides keyword-based retrieval without external dependencies:

```python
from isobase.knowledge.retrieval import SparseRetriever, curated_stopwords

retriever = SparseRetriever(
    k1=1.5,
    b=0.75,
    stopwords=curated_stopwords(),  # 400+ curated EN/ZH stopwords
)
results = retriever.retrieve("python 装饰器", items, top_k=5)
```

**Features:**

- Pure Python BM25 — zero external dependencies
- Pluggable `tokenizer` — inject `jieba.cut` for CJK support
- Built-in curated stopword list (English + Chinese)
- Configurable BM25 parameters (`k1`, `b`)

### Reciprocal Rank Fusion (RRF)

The `RankFusion` class merges dense and sparse result lists by rank position:

```python
from isobase.knowledge.retrieval import RankFusion

fusion = RankFusion(k=60)  # smoothing constant
fused = fusion.fuse(dense_results, sparse_results, top_k=20)
# fused results have score_source="fused"
```

### Reranker Hook

Post-retrieval reranking via the `BaseReranker` interface:

```python
from isobase.knowledge.retrieval.providers import OpenAIReranker

reranker = OpenAIReranker(
    api_key="...",
    base_url="https://.../compatible-api/v1",
    model="qwen3-rerank",
)
pipeline = RetrievalPipeline(store, reranker=reranker)
# results have score_source="rerank"
```

### Score Sources

Each `RetrievalResult` carries an explicit `score_source`:

| Source     | Meaning                                  |
| ---------- | ---------------------------------------- |
| `"dense"`  | Cosine similarity from embedding vectors |
| `"sparse"` | BM25 keyword relevance score             |
| `"fused"`  | Reciprocal Rank Fusion of dense + sparse |
| `"rerank"` | Model-based reranker score               |

## Chunking Strategies

Chunking parameters are owned by the **knowledge base**, not by the chunker
instance. When you create a knowledge base the current chunker's defaults are
snapped into the entity; every subsequent `index_text()` call reads the
KB's stored values and passes them to the chunker. This means you can manage
multiple knowledge bases with different chunk sizes through a single service
instance — the KB entity is the single source of truth.

### Fixed-Size Chunker

Simple character-based splitting with overlap:

```python
from isobase.knowledge.chunking import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=512,    # Default max characters per chunk
    chunk_overlap=50,  # Default overlap between chunks
)

# At call time the KB's own values override the chunker defaults:
chunks = chunker.chunk("Long text content...",
                        chunk_size=1024,
                        chunk_overlap=100)
```

**Pros:**

- Simple and predictable
- Fast processing
- No dependencies

**Cons:**

- May split mid-sentence
- Doesn't respect document structure

### Markdown Chunker

Structure-aware splitting that respects ATX heading boundaries
and preserves heading path metadata:

```python
from isobase.knowledge.chunking import MarkdownChunker

chunker = MarkdownChunker(
    chunk_size=500,    # Default max characters per chunk
    chunk_overlap=60,  # Default overlap between chunks (within a section)
)

# Each chunk carries heading_path metadata:
chunks = chunker.chunk(markdown_text)
# chunks[0].content       → "# Introduction\n\nHello world"
# chunks[0].metadata      → {"chunk_strategy": "markdown",
#                             "heading_path": ["Introduction"]}
```

**Key behaviors:**

- **chunk_size is an upper bound** — short heading sections are never merged across heading boundaries
- **chunk_overlap applies within a section only** — sections are hard semantic boundaries, no overlap between them
- **Oversized sections** are further split by `FixedSizeChunker`, with all sub-chunks inheriting the same `heading_path`
- **Text normalization** (BOM removal, line ending unification, ATX spacing, etc.) is handled by `isobase.utils.MarkdownFixer`

### Future Strategies

The `BaseChunker.chunk()` signature returns `List[ChunkSection]` — each section
carries `content` and `metadata` (strategy name, heading path, etc.).
`KnowledgeBaseService.index_text()` automatically stores this metadata on
`KnowledgeChunk.metadata`.

## Storage Backends

### Memory Store (MVP)

In-memory storage for testing and demos:

```python
from isobase.knowledge.stores import MemoryKnowledgeStore

store = MemoryKnowledgeStore()
```

**Pros:**

- Zero setup
- Fast
- Perfect for testing

**Cons:**

- Data lost on restart
- Not thread-safe
- Limited to available RAM

**Use when:** Testing, demos, small datasets (<10K chunks)

### SQL Store

Persistent storage using SQLAlchemy (SQLite / PostgreSQL):

```python
from isobase.knowledge.stores import SqlKnowledgeStore
from isobase.database.sql import SqlDbService

db_service = SqlDbService(connection_url="sqlite:///knowledge.db")
store = SqlKnowledgeStore(db_service)
```

**Features:**

- Persistent storage
- SQLite and PostgreSQL support
- Integration with existing `database/sql.py`
- `document.content` is the single source of truth for full text; chunk content is derived at read time

### Mongo Store

Persistent storage using MongoDB:

```python
from isobase.knowledge.stores import MongoKnowledgeStore
from isobase.database.mongo import MongoDbService

mongo_service = MongoDbService(
    uri="mongodb://localhost:27017",
    database_name="isobase",
)
store = MongoKnowledgeStore(mongo_service)

# Or simply use the global default mongo_db instance:
# store = MongoKnowledgeStore()
```

**Features:**

- Persistent storage with native BSON types (no JSON serialization overhead for embeddings)
- MongoDB automatically generates BSON `ObjectId` as document identifiers
- BSON `array<double>` stores embedding vectors in compact binary form — more space-efficient than JSON text
- Integration with existing `database/mongo.py`
- Suitable for production server-side deployments

## Design Principles

### Provider Neutrality

Knowledge base functionality is independent of specific LLM providers:

- Embedding clients are separate from chat clients
- Storage abstractions allow backend switching
- Tools integrate via `FunctionTool` standard interface

### Separation of Concerns

Clear boundaries between layers:

- `knowledge/` is a capability, not a provider
- Service layer orchestrates components
- Tools are thin wrappers, not business logic

### IsoBase Conventions

Follows existing patterns:

- Dataclass DTOs (like `llm/entities.py`)
- ABC abstractions (like `llm/providers/base.py`)
- Synchronous API (matches current `llm` and `database`)
- Minimal dependencies

## Performance Considerations

### Indexing Performance

**Bottlenecks:**

- Embedding API calls (network latency)
- Database writes (for SQL stores)

**Optimizations:**

- Batch embedding is automatic — `KnowledgeBaseService` splits chunks into `embed_batch_size`-sized batches (default 20, configurable at construction time)
- Small batches reduce network payload without increasing token cost
- Connection pooling for SQL stores
- Async indexing (future)

### Search Performance

**Memory Store:**

- O(n) linear scan over all chunks
- Suitable for <10K chunks
- ~100ms for 1K chunks on typical hardware

**SQL Store:**

- Full table scan without vector index
- pgvector enables HNSW/IVFFlat indexes
- Sub-100ms for 100K+ chunks with proper indexing

**Mongo Store:**

- Multi-collection lookup per-chunk (chunks + embeddings + documents)
- MongoDB Atlas Vector Search enables ANN indexes (cloud-hosted only)
- Local community edition uses Python-side cosine similarity

### Memory Usage

**Memory Store:**

- ~4KB per chunk (content + metadata)
- ~4KB per embedding (1024-dim float32)
- Total: ~8KB per chunk → 80MB for 10K chunks

## Roadmap

### Phase 1: MVP

- ✅ Core interfaces (`entities.py`)
- ✅ Fixed-size chunker (`FixedSizeChunker`)
- ✅ OpenAI-compatible embedding client (with `dimensions` auto-detection)
- ✅ `ModelRegistry` — embedding model registry with lazy dimensions fetch
- ✅ In-memory store (`MemoryKnowledgeStore`)
- ✅ `KnowledgeBaseService` — high-level indexing + retrieval orchestration
- ✅ LLM tool integration (`create_knowledge_search_tool`)

### Phase 2: Persistence

- ✅ SQL-backed store (`SqlKnowledgeStore`)
- ✅ MongoDB-backed store (`MongoKnowledgeStore`)
- ✅ Lifecycle APIs: `get` / `list` / `delete` for documents and knowledge bases
- ✅ `embed_batch_size` in `KnowledgeBaseService` constructor
- ✅ Shared store contract tests (27 items, parametrized across memory / SQL / Mongo)
- [ ] Schema migrations

### Phase 3: Markdown Structure-Aware Chunking

- ✅ `ChunkSection(content, metadata)` return type, `BaseChunker.chunk() → List[ChunkSection]`
- ✅ `FixedSizeChunker` — returns `ChunkSection` with `chunk_strategy: "fixed"`
- ✅ `MarkdownChunker` — ATX heading-aware splitter with `heading_path` tracking
- ✅ `TextFixer.normalize()` — BOM removal, line ending unification, ATX spacing
- ✅ `TextFixer.extract_front_matter()` — YAML front matter extraction
- ✅ `KnowledgeBaseService.index_text()` — stores chunker-provided metadata on `KnowledgeChunk`
- ✅ 42 unit tests (6 chunking + 36 text fixer)

### Phase 4: Retrieval Enhancement

- ✅ `RetrievalOption` — `candidate_k`, `top_k`, metadata filter, `use_sparse`, `use_hybrid`
- ✅ `RetrievalPipeline` — single entry point for dense / sparse / hybrid retrieval
- ✅ `RetrievalResult.score_source` — explicit score semantics (`"dense"`, `"sparse"`, `"fused"`, `"rerank"`)
- ✅ `SparseRetriever` — BM25 keyword retrieval with pluggable tokenizer + stopwords
- ✅ `RankFusion` — Reciprocal Rank Fusion (RRF) for dense + sparse merging
- ✅ `BaseReranker` / `NoOpReranker` — reranker hook interface (optional, default no-op)
- ✅ `OpenAIReranker` — OpenAI-compatible rerank provider (qwen3-rerank, etc.)
- ✅ `list_chunks(kb_id)` — store contract extension for sparse + hybrid retrieval
- ✅ 62 retrieval unit tests (dense, sparse, pipeline, reranker, RRF)

### Phase 5: Document Parsers

- [ ] Text parser
- [ ] PDF extraction
- [ ] Web page extraction

### Phase 6: Advanced Features

- [ ] Async/batch indexing
- [ ] Incremental updates
- [ ] Multi-knowledge-base search
- ✅ Metadata filtering
- ✅ Hybrid retrieval (dense + sparse + RRF)
- ✅ `candidate_k` / `top_k` pipeline control (`RetrievalOption.effective_candidate_k()`)
- [ ] pgvector migration
- [ ] MongoDB Atlas Vector Search

## References

- [Architecture Design](../../docs/design/knowledge_base_architecture.md)
- [Qwen v4 API Reference](https://docs.aimlapi.com/api-references/embedding-models/alibaba-cloud/qwen-text-embedding-v4)
- [DashScope API Reference](https://help.aliyun.com/zh/model-studio/embedding)
- [AstrBot Knowledge Base](https://github.com/AstrBotDevs/AstrBot/tree/master/astrbot/core/knowledge_base)
- [Cherry Studio Knowledge](https://github.com/cherryhq/cherry-studio)
