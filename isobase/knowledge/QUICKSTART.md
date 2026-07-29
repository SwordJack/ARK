# IsoBase Knowledge Base - Quick Start

## Installation

IsoBase knowledge module is included in the main package. No additional dependencies required for basic usage (memory store).

For production use with real embedding APIs, you'll need:

```bash
# OpenAI-compatible embedding APIs (OpenAI, DashScope, AIMLAPI)
pip install openai
```

## 5-Minute Quick Start

### 1. Basic Setup (Fake Embeddings)

```python
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import BaseEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.stores import MemoryKnowledgeStore
from typing import List

# Fake client for testing (no API calls)
class FakeEmbedding(BaseEmbeddingClient):
    def embed_texts(self, texts: List[str], **kwargs):
        return [[float(i)] * 128 for i in range(len(texts))]
    def embed_query(self, query: str, **kwargs):
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
service.index_text(kb.id, "RAG is a technique for LLMs.", "Doc 1")
results = service.retrieve("What is RAG?", kb.id, top_k=3)

print(results[0].chunk.content)
```

### 2. Production Setup (Real APIs)

#### Option A: Qwen v4 via AIMLAPI

```python
from isobase.knowledge.embeddings import OpenAIEmbeddingClient

embedding_client = OpenAIEmbeddingClient(
    api_key="your_aimlapi_key",
    base_url="https://api.aimlapi.com/v1",
    model="alibaba/qwen-text-embedding-v4",
    dimensions=1024,
)
```

#### Option B: Alibaba DashScope

```python
import os

embedding_client = OpenAIEmbeddingClient(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="text-embedding-v4",
    dimensions=1024,
)
```

#### Option C: OpenAI

```python
import os

embedding_client = OpenAIEmbeddingClient(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-small",
)
```

### 3. LLM Integration

```python
from isobase.llm.providers.openai_chat import OpenAIChat
from isobase.llm.tools.base import ToolSet
from isobase.knowledge.tools import create_knowledge_search_tool

# Create tool
kb_tool = create_knowledge_search_tool(service, kb.id, top_k=3)

# Add to toolset
toolset = ToolSet()
toolset.add_tool(kb_tool)

# Use with LLM
llm = OpenAIChat(api_key="your_openai_key")
response = llm.generate(
    messages=[{"role": "user", "content": "What is RAG?"}],
    tools=toolset.to_openai_schema(),
)

if response.tool_calls:
    results, _ = toolset.execute_tool_calls(response.tool_calls)
    print(results[0])
```

## Run Examples

```bash
# Manual test with fake embeddings (no API needed)
python -m test.knowledge.live.run_knowledge_basic

# All unit tests
python -m pytest test/knowledge/ -v
```

## Next Steps

- Read [README.md](README.md) for full documentation
- Read [README.zh-cn.md](README.zh-cn.md) for Chinese documentation
- Read [Architecture Design](../../docs/design/knowledge_base_architecture.md) for implementation details
- Check [Phase 1 Summary](../../docs/design/knowledge_base_phase1_summary.md) for what's implemented

## Common Issues

### ModuleNotFoundError: No module named 'openai'

Install the OpenAI SDK:

```bash
pip install openai
```

### API Key Not Set

Set environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export DASHSCOPE_API_KEY="sk-..."
```

Or pass directly:

```python
client = OpenAIEmbeddingClient(api_key="your_key", ...)
```

### Empty Search Results

Check:

1. Documents are indexed: `service.index_text(...)`
2. Query is not empty
3. Knowledge base ID is correct
4. Embeddings are working (test with fake client first)

## Support

- GitHub Issues: [Create an issue](https://github.com/SwordJack/IsoBase/issues)
- Documentation: [docs/design/](../../docs/design/)
- Examples: [test/knowledge/live/](../../test/knowledge/live/)
