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

Let the LLM automatically decide when to search knowledge:

```python
from isobase.llm.providers.openai_chat import OpenAIChat
from isobase.llm.tools.base import ToolSet
from isobase.knowledge.tools import create_knowledge_search_tool

# Create knowledge search tool
kb_tool = create_knowledge_search_tool(
    service=service,
    knowledge_base_id=kb.id,
    top_k=3,
)

# Add to toolset
toolset = ToolSet()
toolset.add_tool(kb_tool)

# Initialize LLM client
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
- Batch processing: up to 25 texts/request
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
isobase/knowledge/
├── entities.py          # Core DTOs (KnowledgeBase, Document, Chunk, RetrievalResult)
├── embeddings/
│   ├── base.py         # BaseEmbeddingClient ABC
│   └── openai.py # OpenAI-compatible client
├── chunking/
│   ├── base.py         # BaseChunker ABC
│   └── fixed.py        # Fixed-size chunker with overlap
├── stores/
│   ├── base.py         # BaseKnowledgeStore ABC
│   ├── memory.py       # In-memory store (MVP)
│   └── sql.py          # SQL-backed store (future)
├── retrieval/          # Hybrid retrieval (future)
├── service.py          # KnowledgeBaseService orchestration
└── tools.py            # LLM tool wrappers
```

## Core Concepts

### Knowledge Base

A logical collection of indexed documents with configuration:

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

## Chunking Strategies

### Fixed-Size Chunker

Simple character-based splitting with overlap:

```python
from isobase.knowledge.chunking import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=512,    # Max characters per chunk
    chunk_overlap=50,  # Overlap between chunks
)

chunks = chunker.chunk("Long text content...")
```

**Pros:**

- Simple and predictable
- Fast processing
- No dependencies

**Cons:**

- May split mid-sentence
- Doesn't respect document structure

**Future:** Recursive character splitter, semantic chunking, structure-aware chunking.

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

### SQL Store (Future)

Persistent storage using SQLAlchemy:

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

- Batch embedding calls (up to 25 texts for DashScope)
- Connection pooling for SQL stores
- Async indexing (future)

### Search Performance

**Memory Store:**

- O(n) linear scan over all chunks
- Suitable for <10K chunks
- ~100ms for 1K chunks on typical hardware

**SQL Store (future):**

- Full table scan without vector index
- pgvector enables HNSW/IVFFlat indexes
- Sub-100ms for 100K+ chunks with proper indexing

### Memory Usage

**Memory Store:**

- ~4KB per chunk (content + metadata)
- ~4KB per embedding (1024-dim float32)
- Total: ~8KB per chunk → 80MB for 10K chunks

## Roadmap

### Phase 1: MVP (Current)

- ✅ Core interfaces
- ✅ Fixed-size chunker
- ✅ OpenAI-compatible embedding client
- ✅ In-memory store
- ✅ KnowledgeBaseService
- ✅ LLM tool integration

### Phase 2: SQL Persistence

- [ ] SQL-backed store
- [ ] Schema migrations
- [ ] Vector storage (JSON initially)

### Phase 3: Enhanced Retrieval

- [ ] Hybrid retrieval (dense + sparse)
- [ ] BM25 sparse retriever
- [ ] RRF (Reciprocal Rank Fusion)
- [ ] Reranker interface

### Phase 4: Document Parsers

- [ ] Text parser
- [ ] Markdown parser
- [ ] PDF extraction
- [ ] Web page extraction

### Phase 5: Advanced Features

- [ ] Async/batch indexing
- [ ] Incremental updates
- [ ] Multi-knowledge-base search
- [ ] Metadata filtering
- [ ] pgvector migration

## References

- [Architecture Design](../../docs/design/knowledge_base_architecture.md)
- [Qwen v4 API Reference](https://docs.aimlapi.com/api-references/embedding-models/alibaba-cloud/qwen-text-embedding-v4)
- [DashScope API Reference](https://help.aliyun.com/zh/model-studio/embedding)
- [AstrBot Knowledge Base](https://github.com/AstrBotDevs/AstrBot/tree/master/astrbot/core/knowledge_base)
- [Cherry Studio Knowledge](https://github.com/cherryhq/cherry-studio)
