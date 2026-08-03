# Knowledge Base Architecture for IsoBase

## 1. Overview

This document outlines the architectural design for adding RAG (Retrieval-Augmented Generation) capabilities to IsoBase. The design prioritizes:

- **Provider neutrality**: Knowledge base functionality remains independent of specific LLM providers
- **Modularity**: Clear separation between indexing, retrieval, and LLM integration
- **Minimal dependencies**: Start with simple implementations, defer complex backends
- **IsoBase conventions**: Follow existing patterns in `llm/` and `database/` modules

## 2. Reference Implementation Analysis

### 2.1 AstrBot Knowledge Base

**Strengths observed:**

- Clear layering: models → chunking → retrieval → tools
- Hybrid retrieval: dense (FAISS) + sparse (BM25) + fusion (RRF) + rerank
- SQLite for metadata, separate vector storage
- Clean abstraction: `RetrievalManager` coordinates all retrieval strategies

**Key files reviewed:**

- `astrbot/core/knowledge_base/models.py`: SQLModel schemas for KB/Document/Chunk
- `astrbot/core/knowledge_base/retrieval/manager.py`: Orchestrates dense/sparse/rerank
- `astrbot/core/knowledge_base/chunking/base.py`: Abstract chunker interface
- `astrbot/core/tools/knowledge_base_tools.py`: LLM tool integration

### 2.2 Cherry Studio Knowledge Base

**Strengths observed:**

- Multi-tool approach: `kb_list` → `kb_search` → `kb_read` → `kb_grep`
- Error distinction: infrastructure failure vs. "no results found"
- Scoped access: knowledge bases bound to assistants
- Cancellation support for long-running searches

**Key insight:**

- Don't just provide "search" — provide browsing (`list`), searching, and deep reading tools

### 2.3 Embedding API Details (from references/)

#### Qwen Text Embedding v4 (AIMLAPI)

```python
# Endpoint: https://api.aimlapi.com/v1/embeddings
# Model: alibaba/qwen-text-embedding-v4
# Dimensions: 512 / 1024 / 2048 (configurable)
# Context: 32K tokens

from openai import OpenAI
client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://api.aimlapi.com/v1"
)

response = client.embeddings.create(
    model="alibaba/qwen-text-embedding-v4",
    input=["text to embed"],
    dimensions=1024  # optional: 512, 1024, 2048
)
embedding_vector = response.data[0].embedding  # list[float]
```

**Key parameters:**

- `dimensions`: 512 (faster) / 1024 (balanced) / 2048 (best quality)
- `encoding_format`: "float" (default) or "base64"
- Pricing: $0.0001 per 1M tokens (via AIMLAPI credits)

#### Alibaba DashScope (百炼)

```python
# Endpoint: https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding
# Models: text-embedding-v3, text-embedding-v4
# OpenAI-compatible endpoint available

from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.embeddings.create(
    model="text-embedding-v4",
    input="衣服的质量杠杠的",
    dimensions=1024
)
```

**Key parameters:**

- `input`: string or list[string], max 20 texts per request
- `dimensions`: 512 / 1024 / 2048 for v4; 768 / 1024 / 1536 for v3
- Supports async batch processing for large document sets

**Performance comparison (from reference docs):**

| Model     | MTEB (en) | C-MTEB (zh) |
| --------- | --------- | ----------- |
| v3 (1024) | 63.39     | 68.92       |
| v4 (1024) | 68.36     | 70.14       |
| v4 (2048) | 71.58     | 71.99       |

## 3. Proposed Architecture

### 3.1 Module Structure

```
isobase/knowledge/
├── __init__.py
├── entities.py          # Core DTOs
├── chunking/
│   ├── __init__.py
│   ├── base.py         # BaseChunker ABC
│   ├── fixed.py        # Fixed-size chunker
│   └── recursive.py    # Recursive character splitter (Planned)
├── embeddings/
│   ├── __init__.py
│   ├── base.py         # BaseEmbeddingClient ABC
│   └── openai.py  # OpenAI-compatible embedding client
├── stores/
│   ├── __init__.py
│   ├── base.py         # BaseKnowledgeStore ABC
│   ├── memory.py       # In-memory vector store (MVP)
│   ├── sql.py          # SQLAlchemy-backed metadata + vectors
│   └── mongo.py        # MongoDB-backed metadata + vectors
├── retrieval/
│   ├── __init__.py
│   ├── base.py         # BaseRetriever ABC
│   └── dense.py        # Dense retriever (cosine similarity)
├── service.py          # KnowledgeBaseService
```

**LLM-side tool integration (`isobase/llm/tools/knowledge.py`):**

```
isobase/llm/tools/knowledge.py  # factory wrapping isobase.knowledge into FunctionTool
```

### 3.2 Design Principles

**Separation of concerns:**

- `knowledge/` is independent of `llm/providers/` — it's a capability, not a provider
- Embedding clients are separate from chat clients (different interface, different purpose)
- Storage abstractions allow switching between memory, SQL, pgvector without changing business logic

**Integration points:**

- `isobase/llm/tools/knowledge.py` provides `FunctionTool` factories that consume `knowledge/`
- Embedding clients can use dedicated APIs (via `OpenAIEmbeddingClient`)
- `SqlDbService` from `database/sql.py` manages connections for SQL-backed stores

## 4. Core Interfaces

### 4.1 Data Transfer Objects (`entities.py`)

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class KnowledgeBase:
    """Represents a knowledge base container."""

    id: str
    name: str
    description: str = ""
    embedding_model_id: str = ""
    dimensions: int = 1024
    chunk_size: int = 512
    chunk_overlap: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None


@dataclass
class KnowledgeDocument:
    """Represents a source document in a knowledge base."""

    id: str
    knowledge_base_id: str
    title: str
    source_uri: str = ""
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: Optional[datetime] = None


@dataclass
class KnowledgeChunk:
    """Represents an indexed text chunk."""

    id: str
    document_id: str
    knowledge_base_id: str
    content: str
    index: int  # Position in document
    token_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Represents a retrieved chunk with provenance."""

    chunk: KnowledgeChunk
    score: float
    document: Optional[KnowledgeDocument] = None

    def __str__(self) -> str:
        """Format for LLM context injection."""
        source = f" ({self.document.title})" if self.document else ""
        return f"[Score: {self.score:.3f}{source}]\n{self.chunk.content}"
```

### 4.2 Embedding Client (`embeddings/base.py`)

```python
from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingClient(ABC):
    """Abstract embedding client interface.

    Separated from BaseLLMClient because embedding is a different task
    with different input/output contracts and pricing models.
    """

    @abstractmethod
    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embeds a batch of texts.

        Args:
            texts: List of text strings to embed.
            **kwargs: Model-specific parameters (e.g., dimensions).

        Returns:
            List of embedding vectors (list of floats).

        Raises:
            ValueError: If texts is empty or exceeds batch size.
            RuntimeError: If API call fails.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str, **kwargs) -> List[float]:
        """Embeds a single query string.

        Some models distinguish between document and query embeddings.
        Default implementation can delegate to embed_texts()[0].

        Args:
            query: Query text to embed.
            **kwargs: Model-specific parameters.

        Returns:
            Embedding vector (list of floats).
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Returns the embedding vector dimensions."""
        pass
```

### 4.3 OpenAI-Compatible Embedding Client (`embeddings/openai.py`)

```python
from typing import List, Optional
from openai import OpenAI
from .base import BaseEmbeddingClient


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI-compatible embedding client.

    Supports:
    - OpenAI (text-embedding-3-small, text-embedding-3-large)
    - Alibaba DashScope (text-embedding-v3, text-embedding-v4)
    - AIMLAPI (alibaba/qwen-text-embedding-v4)
    - Any OpenAI-compatible embedding endpoint

    Example usage:
        # Qwen v4 via AIMLAPI
        client = OpenAIEmbeddingClient(
            api_key="aimlapi_key",
            base_url="https://api.aimlapi.com/v1",
            model="alibaba/qwen-text-embedding-v4",
            dimensions=1024
        )

        # Qwen v4 via DashScope
        client = OpenAIEmbeddingClient(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="text-embedding-v4",
            dimensions=1024
        )
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dimensions: Optional[int] = None,
        **kwargs
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._dimensions = dimensions
        self.default_kwargs = kwargs

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Embeds a batch of texts."""
        if not texts:
            raise ValueError("texts cannot be empty")

        params = {**self.default_kwargs, **kwargs}
        if self._dimensions:
            params["dimensions"] = self._dimensions

        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            **params
        )

        return [item.embedding for item in response.data]

    def embed_query(self, query: str, **kwargs) -> List[float]:
        """Embeds a single query."""
        return self.embed_texts([query], **kwargs)[0]

    @property
    def dimensions(self) -> int:
        """Returns embedding dimensions."""
        if self._dimensions:
            return self._dimensions
        # Infer from model name if not explicitly set
        if "text-embedding-3-large" in self.model:
            return 3072
        elif "text-embedding-3-small" in self.model:
            return 1536
        elif "text-embedding-v4" in self.model:
            return 1024  # Default for Qwen v4
        elif "text-embedding-v3" in self.model:
            return 1024  # Default for Qwen v3
        return 1536  # Conservative default
```

### 4.4 Chunking (`chunking/base.py`)

```python
from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    """Abstract chunker interface."""

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[str]:
        """Splits text into chunks.

        Args:
            text: Input text to split.
            **kwargs: Chunker-specific parameters.

        Returns:
            List of text chunks.
        """
        pass
```

**Fixed-size implementation (`chunking/fixed.py`):**

```python
from typing import List
from .base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunker with overlap.

    Simple character-based splitting. Does not respect sentence
    or word boundaries.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, **kwargs) -> List[str]:
        """Splits text into fixed-size chunks with overlap."""
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            if end >= text_len:
                break

            start = end - self.chunk_overlap

        return chunks
```

### 4.5 Knowledge Store (`stores/base.py`)

```python
from abc import ABC, abstractmethod
from typing import List
from ..entities import KnowledgeBase, KnowledgeDocument, KnowledgeChunk, RetrievalResult


class BaseKnowledgeStore(ABC):
    """Abstract storage interface for knowledge base data."""

    @abstractmethod
    def create_knowledge_base(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Creates a new knowledge base."""
        pass

    @abstractmethod
    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        """Retrieves a knowledge base by ID."""
        pass

    @abstractmethod
    def add_document(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        """Stores a document."""
        pass

    @abstractmethod
    def add_chunks(
        self,
        chunks: List[KnowledgeChunk],
        embeddings: List[List[float]]
    ) -> None:
        """Stores chunks and their embeddings.

        Args:
            chunks: List of chunks to store.
            embeddings: Corresponding embedding vectors.

        Raises:
            ValueError: If len(chunks) != len(embeddings).
        """
        pass

    @abstractmethod
    def search(
        self,
        kb_id: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """Searches for similar chunks by vector similarity.

        Args:
            kb_id: Knowledge base ID to search within.
            query_embedding: Query vector.
            top_k: Maximum number of results.

        Returns:
            List of retrieval results, sorted by score descending.
        """
        pass
```

**In-memory implementation (`stores/memory.py`) - MVP:**

```python
import uuid
from typing import Dict, List
from datetime import datetime, timezone
from ..entities import (
    KnowledgeBase, KnowledgeDocument, KnowledgeChunk, RetrievalResult
)
from .base import BaseKnowledgeStore


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Computes cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class MemoryKnowledgeStore(BaseKnowledgeStore):
    """In-memory knowledge store for testing and demos.

    Not suitable for production use. All data is lost on restart.
    """

    def __init__(self):
        self.knowledge_bases: Dict[str, KnowledgeBase] = {}
        self.documents: Dict[str, KnowledgeDocument] = {}
        self.chunks: Dict[str, KnowledgeChunk] = {}
        self.embeddings: Dict[str, List[float]] = {}  # chunk_id -> vector

    def create_knowledge_base(self, kb: KnowledgeBase) -> KnowledgeBase:
        if not kb.id:
            kb.id = str(uuid.uuid4())
        kb.created_time = datetime.now(timezone.utc)
        kb.updated_time = kb.created_time
        self.knowledge_bases[kb.id] = kb
        return kb

    def get_knowledge_base(self, kb_id: str) -> KnowledgeBase:
        if kb_id not in self.knowledge_bases:
            raise KeyError(f"Knowledge base {kb_id} not found")
        return self.knowledge_bases[kb_id]

    def add_document(self, doc: KnowledgeDocument) -> KnowledgeDocument:
        if not doc.id:
            doc.id = str(uuid.uuid4())
        doc.created_time = datetime.now(timezone.utc)
        self.documents[doc.id] = doc
        return doc

    def add_chunks(
        self,
        chunks: List[KnowledgeChunk],
        embeddings: List[List[float]]
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count ({len(chunks)}) must match "
                f"embedding count ({len(embeddings)})"
            )

        for chunk, embedding in zip(chunks, embeddings):
            if not chunk.id:
                chunk.id = str(uuid.uuid4())
            self.chunks[chunk.id] = chunk
            self.embeddings[chunk.id] = embedding

    def search(
        self,
        kb_id: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        # Filter chunks by knowledge base
        kb_chunks = [
            (chunk_id, chunk)
            for chunk_id, chunk in self.chunks.items()
            if chunk.knowledge_base_id == kb_id
        ]

        if not kb_chunks:
            return []

        # Compute similarities
        results = []
        for chunk_id, chunk in kb_chunks:
            embedding = self.embeddings[chunk_id]
            score = cosine_similarity(query_embedding, embedding)

            # Optionally fetch document
            doc = self.documents.get(chunk.document_id)

            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                document=doc
            ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]
```

### 4.6 Knowledge Base Service (`service.py`)

```python
import uuid
from typing import Dict, List, Optional, Any
from .entities import (
    KnowledgeBase, KnowledgeDocument, KnowledgeChunk, RetrievalResult
)
from .embeddings.base import BaseEmbeddingClient
from .chunking.base import BaseChunker
from .stores.base import BaseKnowledgeStore


class KnowledgeBaseService:
    """High-level service for indexing and retrieving knowledge.

    Orchestrates chunking, embedding, and storage.
    """

    def __init__(
        self,
        embedding_client: BaseEmbeddingClient,
        store: BaseKnowledgeStore,
        chunker: BaseChunker,
        embed_batch_size: int = 20,
    ):
        """Initializes the knowledge base service.

        Args:
            embedding_client: Client for generating text embeddings.
            store: Storage backend for knowledge bases and vectors.
            chunker: Chunking strategy for splitting documents.
            embed_batch_size: Max chunks per embedding API call.
                Defaults to a conservative value that works across
                common providers (DashScope: ≤20, OpenAI: ≤2048).
        """
        self.embedding_client = embedding_client
        self.store = store
        self.chunker = chunker
        self.embed_batch_size = embed_batch_size

    def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBase:
        """Creates a new knowledge base."""
        kb = KnowledgeBase(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            embedding_model_id=self.embedding_client.model,
            dimensions=self.embedding_client.dimensions,
            metadata=metadata or {},
        )
        return self.store.create_knowledge_base(kb)

    def index_text(
        self,
        knowledge_base_id: str,
        text: str,
        title: str = "",
        source_uri: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeDocument:
        """Indexes a text document into a knowledge base.

        The document is chunked, embedded in batches, and stored.
        embed_batch_size (set at construction time) controls how
        many chunks are sent per API call.

        Args:
            knowledge_base_id: Target knowledge base ID.
            text: Text content to index.
            title: Document title.
            source_uri: Source URL or file path.
            metadata: Additional metadata.

        Returns:
            The created document with its indexed chunks.
        """
        # Create document record
        doc = KnowledgeDocument(
            id=str(uuid.uuid4()),
            knowledge_base_id=knowledge_base_id,
            title=title or "Untitled",
            source_uri=source_uri,
            content=text,
            metadata=metadata or {},
        )
        doc = self.store.add_document(doc)

        # Chunk text
        chunk_texts = self.chunker.chunk(text)

        # Create chunk records
        chunks = [
            KnowledgeChunk(
                id=str(uuid.uuid4()),
                document_id=doc.id,
                knowledge_base_id=knowledge_base_id,
                content=chunk_text,
                index=idx,
                token_count=len(chunk_text.split()),
                metadata={},
            )
            for idx, chunk_text in enumerate(chunk_texts)
        ]

        # Embed chunks in batches to respect provider limits
        all_embeddings: List[List[float]] = []
        for batch_start in range(0, len(chunks), self.embed_batch_size):
            batch = chunks[batch_start:batch_start + self.embed_batch_size]
            batch_embeddings = self.embedding_client.embed_texts(
                [c.content for c in batch]
            )
            all_embeddings.extend(batch_embeddings)

        # Store chunks and embeddings
        self.store.add_chunks(chunks, all_embeddings)

        return doc

    def retrieve(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Retrieves relevant chunks for a query.

        Args:
            query: Query text.
            knowledge_base_id: Knowledge base to search.
            top_k: Maximum number of results.

        Returns:
            List of retrieval results, sorted by relevance.
        """
        # Embed query
        query_embedding = self.embedding_client.embed_query(query)

        # Search
        results = self.store.search(
            kb_id=knowledge_base_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results

    def retrieve_as_context(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 5,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """Retrieves and formats results as LLM context.

        Args:
            query: Query text.
            knowledge_base_id: Knowledge base to search.
            top_k: Maximum number of results.
            separator: Separator between chunks.

        Returns:
            Formatted context string ready for LLM injection.
        """
        results = self.retrieve(query, knowledge_base_id, top_k)

        if not results:
            return ""

        return separator.join(str(result) for result in results)
```

### 4.7 LLM Tool Integration (`isobase/llm/tools/knowledge.py`)

The tool factory lives in `isobase/llm/tools/knowledge.py` so that
`knowledge/` never imports from `llm/`. The dependency flows
only one way: `llm/ → knowledge/`.

```python
from isobase.llm.tools.base import FunctionTool
from isobase.knowledge import KnowledgeBaseService


def create_knowledge_search_tool(
    service: KnowledgeBaseService,
    knowledge_base_id: str,
    top_k: int = 5,
) -> FunctionTool:
    """Creates a knowledge base search tool for LLM integration.

    Args:
        service: Knowledge base service instance.
        knowledge_base_id: Target knowledge base ID.
        top_k: Default number of results.

    Returns:
        A FunctionTool that can be added to a ToolSet.

    Example usage:
        from isobase.llm.tools.knowledge import create_knowledge_search_tool

        kb_tool = create_knowledge_search_tool(service, kb.id, top_k=3)
        toolset = ToolSet()
        toolset.add_tool(kb_tool)
    """

    def search_knowledge_base(query: str, limit: Optional[int] = None) -> str:
        k = limit if limit is not None else top_k
        context = service.retrieve_as_context(query, knowledge_base_id, top_k=k)
        return context or "No relevant knowledge found for this query."

    return FunctionTool(
        mapped_callable=search_knowledge_base,
        name="search_knowledge_base",
        description=(
            "Search indexed private knowledge. Use when the user asks about "
            "uploaded documents, project notes, or domain-specific information."
        ),
    )
```

## 5. Implementation Phases

### Phase 1: MVP (Minimal Viable Product)

**Goal:** Establish core interfaces and prove end-to-end flow.

**Scope:**

1. ✅ `entities.py` - Core DTOs
2. ✅ `chunking/base.py` + `chunking/fixed.py` - Fixed-size chunker
3. ✅ `embeddings/base.py` + `embeddings/openai.py` - OpenAI-compatible client
4. ✅ `stores/base.py` + `stores/memory.py` - In-memory store
5. ✅ `service.py` - KnowledgeBaseService
6. ✅ `isobase/llm/tools/knowledge.py` - LLM tool integration

**Dependencies:**

- Existing: `isobase.llm.tools.base.FunctionTool`, `isobase.llm.entities`
- New: `openai` (already used by `llm/providers/openai_chat.py`)

**Validation criteria:**

- Unit tests for chunking (overlap, boundary conditions)
- Unit tests for embedding client (mock API)
- Unit tests for memory store (CRUD + search)
- Integration test: index 3 documents → retrieve relevant chunk
- Integration test: KnowledgeSearchTool executed via ToolSet

**Documentation:**

- `isobase/knowledge/README.md` - Module overview
- `isobase/knowledge/README.zh-cn.md` - Chinese version
- Code examples in README

### Phase 2: SQL Persistence ✅ (core done)

**Goal:** Replace memory store with production-ready SQL backend.

**Completed:**

1. ✅ `stores/sql.py` - SQLAlchemy-backed implementation
2. ✅ Lifecycle APIs: get/list/delete for documents and knowledge bases
3. ✅ `embed_batch_size` in `KnowledgeBaseService` constructor (default 20)

**Remaining:** 4. Schema migrations (if using Alembic) 5. Vector storage strategy:

- SQLite: Store as JSON text (simple, no extensions)
- PostgreSQL: Store as JSON initially, migrate to pgvector later

**Integration:**

- Use `isobase.database.sql.SqlDbService` for connection management
- Define models using `SqlDbModelMixin`

**New dependencies:**

- None (SQLAlchemy already in `database/sql.py`)

**Validation criteria:**

- All Phase 1 tests pass with SQL store
- Concurrent writes don't corrupt data
- Search performance acceptable for 1K-10K chunks

### Phase 3: Enhanced Retrieval

**Goal:** Add hybrid retrieval and reranking.

**Scope:**

1. `retrieval/base.py` - BaseRetriever ABC
2. `retrieval/dense.py` - Dense retriever (refactor from store)
3. `retrieval/sparse.py` - BM25 sparse retriever
4. `retrieval/fusion.py` - RRF (Reciprocal Rank Fusion)
5. `retrieval/rerank.py` - Reranker interface (no implementation yet)

**Validation criteria:**

- Hybrid retrieval outperforms dense-only on test queries
- Fusion correctly merges ranked lists

### Phase 4: Document Parsers

**Goal:** Support structured document ingestion.

**Scope:**

1. `parsers/base.py` - BaseDocumentParser
2. `parsers/text.py` - Plain text
3. `parsers/markdown.py` - Markdown with heading preservation
4. `parsers/pdf.py` - PDF extraction
5. `parsers/url.py` - Web page extraction

**New dependencies:**

- `pypdf` or `pdfplumber` (PDF)
- `beautifulsoup4` (HTML)
- `markitdown` (optional, unified parser)

### Phase 5: Advanced Features (Future)

- Async/batch indexing
- Incremental updates (update/delete chunks)
- Multi-knowledge-base search
- Query expansion
- Metadata filtering
- pgvector migration for PostgreSQL

## 6. Embedding API Integration Details

### 6.1 Qwen Text Embedding v4 (AIMLAPI)

**Endpoint:** `https://api.aimlapi.com/v1/embeddings`

**Configuration:**

```python
from isobase.knowledge.embeddings.openai import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient(
    api_key="your_aimlapi_key",
    base_url="https://api.aimlapi.com/v1",
    model="alibaba/qwen-text-embedding-v4",
    dimensions=1024,  # 512 / 1024 / 2048
)
```

**Key features:**

- Context length: 32K tokens
- Dimensions: 512 (fast), 1024 (balanced), 2048 (best quality)
- Multilingual: 100+ languages
- Pricing: ~$0.0001 per 1M tokens (via AIMLAPI credits)

**Performance (MTEB benchmarks):**

- 1024-dim: 68.36 (English), 70.14 (Chinese)
- 2048-dim: 71.58 (English), 71.99 (Chinese)

**Recommendations:**

- Use 1024-dim for most use cases (balanced quality/speed)
- Use 2048-dim for high-stakes retrieval (legal, medical)
- Use 512-dim for real-time applications with latency constraints

### 6.2 Alibaba DashScope (百炼)

**Endpoint:** `https://dashscope.aliyuncs.com/compatible-mode/v1`

**Configuration:**

```python
import os
from isobase.knowledge.embeddings.openai import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v4",
    dimensions=1024,
)
```

**Key features:**

- Native Chinese language optimization
- OpenAI-compatible interface
- Batch processing: up to 20 texts per request
- Models: `text-embedding-v3`, `text-embedding-v4`

**Batch processing example:**

```python
# Embedding is automatically batched by KnowledgeBaseService
texts = ["doc1 content", "doc2 content", "doc3 content"]
embeddings = client.embed_texts(texts)  # Single API call if ≤20 texts
```

**Rate limits and best practices:**

- Max 20 texts per request
- `KnowledgeBaseService` batches automatically via `embed_batch_size` (set at construction time, defaults to conservative 20)
- For large document sets (>1000 docs), use batch processing with delays
- Store dimensions in KnowledgeBase metadata to prevent mismatch

### 6.3 OpenAI Native

**Endpoint:** `https://api.openai.com/v1`

**Configuration:**

```python
from isobase.knowledge.embeddings.openai import OpenAIEmbeddingClient

client = OpenAIEmbeddingClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small",  # or text-embedding-3-large
    dimensions=1536,  # 3-small: 1536, 3-large: 3072
)
```

**Model comparison:**

- `text-embedding-3-small`: 1536-dim, faster, cheaper
- `text-embedding-3-large`: 3072-dim, higher quality
- Both support dimension truncation (e.g., 512-dim from 3-large)

## 7. Usage Examples

### 7.1 Basic Indexing and Retrieval

```python
from isobase.knowledge.service import KnowledgeBaseService
from isobase.knowledge.embeddings.openai import OpenAIEmbeddingClient
from isobase.knowledge.chunking.fixed import FixedSizeChunker
from isobase.knowledge.stores.memory import MemoryKnowledgeStore

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
    text="RAG stands for Retrieval-Augmented Generation...",
    title="RAG Overview",
    source_uri="docs/rag.md",
)

service.index_text(
    knowledge_base_id=kb.id,
    text="Vector databases store embeddings...",
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

### 7.2 LLM Integration

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
llm = OpenAIChat(api_key="your_openai_key")

# Generate with tool access
response = llm.generate(
    messages=[
        {"role": "user", "content": "What is RAG and how does it work?"}
    ],
    tools=toolset.to_openai_schema(),
)

# If LLM calls the tool
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

### 7.3 Manual Context Injection (No Tools)

```python
# For simple use cases, directly inject context
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

## 8. Testing Strategy

### 8.1 Store Contract Tests (Shared)

All stores that implement :class:`BaseKnowledgeStore` must pass the same set of
shared contract tests defined in `test/knowledge/stores/test_store_contract.py`.

**Purpose:** Store contract tests verify external behaviour, not internal
implementation details. A new backend that passes the full contract suite is
guaranteed to behave identically to every other backend from the perspective of
`KnowledgeBaseService` — no matter whether it stores embeddings as JSON text,
BSON arrays, or something else entirely.

**Data boundary (implementor constraints):**

- The store persists an _indexing snapshot_, not a real-time mirror of the
  original file system. `source_uri` records provenance only; stores must
  not assume the external resource still exists.
- `KnowledgeChunk.content` is the direct source of retrieval context and
  must round-trip through `search()` results.
- `KnowledgeDocument.content` is an optional document-text snapshot useful
  for debugging, display, or future re-indexing. It is **not** the single
  source of truth for the original file.
- Deleting a document / knowledge base removes index data from the store
  only; it does not delete any external file.

**Contract checklist (27 items):**

1. Create knowledge base
2. Get knowledge base
3. List knowledge bases
4. Raise `ValueError` on duplicate KB id
5. Raise `KeyError` on `get_knowledge_base("missing")`
6. List knowledge bases returns `[]` when store is empty
7. Add document
8. Get document
9. List documents
10. Raise `KeyError` on `add_document` to missing KB
11. Raise `KeyError` on `get_document("missing")`
12. Raise `KeyError` on `list_documents("missing")`
13. `list_documents` returns `[]` when KB has no documents
14. Add chunks and embeddings
15. Raise `ValueError` when chunk/embedding counts mismatch
16. Raise `KeyError` when adding chunks to a missing KB
17. Search returns results sorted by descending similarity
18. Search respects `top_k`
19. Search returns `[]` for empty or non-existent KB
20. Search does not return chunks from a different KB
21. Raise `ValueError` on vector dimension mismatch during search
22. `delete_document` cascade-deletes its chunks and embeddings
23. `delete_document` does not affect other documents in the same KB
24. `delete_knowledge_base` cascade-deletes documents, chunks, and embeddings
25. Raise `KeyError` on `delete_document("missing")`
26. Raise `KeyError` on `delete_knowledge_base("missing")`
27. Persistent stores can reload data across store instances

**Adding a new backend:** See `test/knowledge/stores/__init__.py` for the
step-by-step onboarding checklist.

**Backends covered:** `MemoryKnowledgeStore`, `SqlKnowledgeStore`,
`MongoKnowledgeStore` (parameters: `["memory", "sql", "mongo"]`).
Mongo tests are auto-skipped when no MongoDB server is reachable on
`localhost:27017`.

### 8.2 Unit Tests

**Test chunking (`tests/knowledge/test_chunking.py`):**

```python
def test_fixed_chunker_no_overlap():
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
    text = "0123456789abcdefghij"
    chunks = chunker.chunk(text)
    assert chunks == ["0123456789", "abcdefghij"]

def test_fixed_chunker_with_overlap():
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
    text = "0123456789abcdefghij"
    chunks = chunker.chunk(text)
    assert chunks[1].startswith("789")  # Overlap from previous chunk
```

**Test embedding client (mock API):**

```python
@patch("openai.OpenAI")
def test_embed_texts(mock_openai):
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_openai.return_value.embeddings.create.return_value = mock_response

    client = OpenAIEmbeddingClient(api_key="test", model="test-model")
    embeddings = client.embed_texts(["test text"])

    assert len(embeddings) == 1
    assert embeddings[0] == [0.1, 0.2, 0.3]
```

**Test memory store:**

```python
def test_memory_store_search():
    store = MemoryKnowledgeStore()
    kb = store.create_knowledge_base(KnowledgeBase(id="kb1", name="Test KB"))

    chunks = [
        KnowledgeChunk(id="c1", knowledge_base_id="kb1", document_id="d1", content="test", index=0)
    ]
    embeddings = [[1.0, 0.0, 0.0]]
    store.add_chunks(chunks, embeddings)

    # Search with similar vector
    results = store.search("kb1", query_embedding=[0.9, 0.1, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "c1"
```

### 8.3 Integration Tests

**End-to-end indexing and retrieval:**

```python
def test_e2e_index_and_retrieve():
    # Use fake embedding client for deterministic tests
    class FakeEmbeddingClient(BaseEmbeddingClient):
        def embed_texts(self, texts):
            return [[float(i), 0.0] for i in range(len(texts))]
        def embed_query(self, query):
            return [0.0, 1.0]
        @property
        def dimensions(self):
            return 2

    service = KnowledgeBaseService(
        embedding_client=FakeEmbeddingClient(),
        store=MemoryKnowledgeStore(),
        chunker=FixedSizeChunker(chunk_size=50, chunk_overlap=10),
    )

    kb = service.create_knowledge_base(name="Test KB")
    service.index_text(kb.id, text="The quick brown fox jumps over the lazy dog.")

    results = service.retrieve(query="fox", knowledge_base_id=kb.id, top_k=1)
    assert len(results) > 0
```

## 9. Migration Path from Existing Systems

### 9.1 From AstrBot

**Similarities:**

- Both use SQLite/SQL for metadata
- Both support FAISS for vectors (future phase)
- Similar chunking strategies

**Migration steps:**

1. Export documents from `KBDocument` table
2. Re-index using `KnowledgeBaseService.index_text()`
3. Map `kb_id` → new knowledge base IDs
4. Optionally migrate embeddings if dimensions match

### 9.2 From Cherry Studio

**Similarities:**

- Tool-based LLM integration
- Multi-knowledge-base support

**Migration steps:**

1. Export knowledge items via DataApi
2. Create corresponding knowledge bases
3. Index items as documents
4. Update assistant bindings to new tool definitions

## 10. Performance Considerations

### 10.1 Indexing Performance

**Bottlenecks:**

- Embedding API calls (network latency)
- Database writes (for SQL stores)

**Optimizations:**

- Batch embedding calls (up to 20 texts per call; `KnowledgeBaseService` batches automatically)
- Async indexing (Phase 5)
- Connection pooling for SQL stores

### 10.2 Search Performance

**Bottlenecks:**

- Vector similarity computation (scales with chunk count)
- Memory store: O(n) linear scan
- SQL store: Full table scan without vector index
- Mongo store: Multi-collection lookup, same O(n) linear scan without vector index

**Optimizations:**

- Limit chunk count per knowledge base (< 10K for memory store)
- Use pgvector for PostgreSQL (enables HNSW/IVFFlat indexes)
- Use MongoDB Atlas Vector Search for Mongo store (enables ANN indexes)
- Pre-filter by metadata before similarity computation

### 10.3 Memory Usage

**Memory store:**

- ~4KB per chunk (content + metadata)
- ~4KB per embedding (1024-dim float32)
- Total: ~8KB per chunk → 80MB for 10K chunks

**SQL store:**

- Bounded by connection pool
- Embeddings stored as JSON (less efficient than binary)

## 11. Security and Privacy

### 11.1 API Key Management

- Never hardcode API keys in examples
- Use environment variables: `os.getenv("DASHSCOPE_API_KEY")`
- Document key setup in README

### 11.2 Data Privacy

- Embeddings are sent to external APIs (OpenAI, Alibaba)
- Sensitive documents should use self-hosted embedding models
- Consider local embedding options (Phase 5):
  - Sentence Transformers
  - FastEmbed
  - Local LLM providers via Ollama

### 11.3 Input Validation

- Validate chunk size > overlap
- Validate embedding dimensions match across operations
- Validate knowledge base exists before indexing

## 12. Open Questions and Future Decisions

### 12.1 Async vs Sync

**Current decision:** Start with sync API (matches existing `isobase.llm` and `isobase.database` patterns).

**Future consideration:** Add `AsyncKnowledgeBaseService` when async LLM providers are added.

### 12.2 Vector Database Selection

**Current decision:** Start with in-memory (MVP), then SQL with JSON storage.

**Future options:**

- pgvector (PostgreSQL extension)
- FAISS (local file-based)
- Qdrant / Milvus (dedicated vector DBs)
- MongoDB Atlas Vector Search (cloud-hosted ANN for existing Mongo stores)

**Decision criteria:**

- Deployment complexity
- Query performance
- Filtering capabilities

### 12.3 Embedding Model Selection

**Current decision:** Support OpenAI-compatible APIs via `OpenAIEmbeddingClient`.

**Future options:**

- Add `SentenceTransformerEmbedding` for local models
- Add `OllamaEmbedding` for self-hosted LLMs
- Support multi-modal embeddings (text + image)

### 12.4 Chunking Strategies

**Current decision:** Fixed-size with overlap (simple, predictable).

**Future options:**

- Recursive character splitter (respects sentence boundaries)
- Semantic chunking (LLM-based)
- Document-structure-aware (Markdown headers, PDF sections)

**Decision point:** Phase 4 (after parsers are implemented).

## 13. Success Metrics

### Phase 1 (MVP) Success Criteria

- [x] All interfaces defined and documented
- [x] Fixed-size chunker passes unit tests
- [x] OpenAI-compatible embedding client works with Qwen v4
- [x] Memory store passes CRUD and search tests
- [x] End-to-end test: index 3 docs → retrieve correct chunk
- [x] KnowledgeSearchTool integrates with existing ToolSet
- [x] Lifecycle APIs: get/list/delete for documents and knowledge bases
- [x] `embed_batch_size` in constructor for provider-aware batch control
- [x] README with runnable examples

### Phase 2 (SQL) Success Criteria

- ✅ SQL store passes all Phase 1 tests
- ✅ 10K chunks indexed without errors
- ✅ Lifecycle APIs: get/list/delete for documents and knowledge bases
- ✅ `embed_batch_size` controls batch size (default 20, adjustable per provider)
- [ ] Search completes in <1s for 10K chunks
- ✅ Store contract tests covering both memory and SQL backends

### Phase 3 (Hybrid Retrieval) Success Criteria

- [ ] Hybrid retrieval outperforms dense-only by >10% on test queries
- [ ] RRF fusion correctly merges ranked lists
- [ ] Sparse retrieval finds exact keyword matches missed by dense

## 14. References

- AstrBot knowledge base: `/home/yinjie/repositories/AstrBot/astrbot/core/knowledge_base/`
- Cherry Studio knowledge: `/home/yinjie/repositories/cherry-studio/src/main/ai/tools/knowledgeLookup.ts`
- Qwen v4 docs: `docs/references/Qwen Text Embedding v4 AIMLAPI.md`
- DashScope docs: `docs/references/向量化 - 阿里云百炼.md`
- IsoBase LLM module: `isobase/llm/`
- IsoBase database module: `isobase/database/`

---

## Appendix: Alternative Architectures Considered

### A1. Embedding as LLM Provider Method

**Rejected approach:**

```python
class BaseLLMClient(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        pass
```

**Why rejected:**

- Embedding is a different task with different pricing/rate limits
- Not all chat models support embeddings (Anthropic Claude)
- Not all embedding models support chat (Qwen embedding-only models)
- Violates single responsibility principle

### A2. Knowledge Base as LLM Provider

**Rejected approach:** Add RAG capabilities directly to `OpenAIChat` / `AnthropicMessages`.

**Why rejected:**

- Tight coupling between LLM and knowledge store
- Cannot use knowledge base without an LLM
- Cannot switch providers without reconfiguring knowledge base
- Violates modularity principle

### A3. Tool-First Design

**Rejected approach:** Start with tools, implement service layer later.

**Why rejected:**

- Tools are integration layer, not core functionality
- Need testable service layer independent of LLM
- Risk of coupling business logic to tool call format

**Chosen approach:** Service layer first, tools as thin wrappers.
