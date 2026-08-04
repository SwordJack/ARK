# IsoBase LLM — Quick Start

- 汉语 [QUICKSTART.zh-cn.md](QUICKSTART.zh-cn.md)
- English [QUICKSTART.md](QUICKSTART.md)

## Installation

```bash
pip install openai anthropic
```

## 5-Minute Quick Start

### 1. Basic Chat

```python
from isobase.llm import OpenAIChat, AnthropicMessages

# OpenAI
client = OpenAIChat(api_key="sk-...", default_model="gpt-4o-mini")
resp = client.ask("Say hello in French.")
print(resp.content)          # final answer
print(resp.usage.total_tokens)

# Anthropic (drop-in interchangeable)
client = AnthropicMessages(api_key="sk-ant-...", default_model="claude-haiku-4-5")
resp = client.ask("Say hello in French.")
print(resp.content)
```

### 2. Streaming

```python
for chunk in client.ask("Count from 1 to 5.", stream=True):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 3. Multi-Turn Conversation

Use `generate()` when you need full control over messages:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]
# method 1: generate() — returns LLMResponse
resp = client.generate(messages=messages, model="gpt-4o-mini")
messages.append({"role": "assistant", "content": resp.content})
messages.append({"role": "user", "content": "Tell me a joke."})
resp = client.generate(messages=messages)

# method 2: ask() — handles history internally for single-turn convenience
resp = client.ask("Tell me a joke.")
```

### 4. Tool Calling (Function Calling)

Define your tool once — works with any provider.

```python
from isobase.llm.tools import FunctionTool, ToolSet

def get_weather(city: str) -> tuple[bool, str]:
    """Get the current weather for a city.

    Args:
        city: The name of the city.
    """
    return True, f"It is 22°C and sunny in {city}."

# Auto-generates name, description, and JSON schema from the callable
tool = FunctionTool(get_weather)
toolset = ToolSet()
toolset.add_tool(tool)

# ask() handles the tool-calling loop automatically
client = OpenAIChat(api_key="sk-...", tools=list(toolset))
resp = client.ask("What's the weather in Paris?")
print(resp.content)  # "It is 22°C and sunny in Paris."
```

For explicit schema control:

```python
tool = FunctionTool(
    mapped_callable=get_weather,
    name="get_weather_override",
    description="Fetch current weather for a city.",
    parameters_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name (e.g. 'Paris')"}
        },
        "required": ["city"],
    },
)
```

### 5. Web Search

```python
from isobase.llm.tools.search import SearchTool

search = SearchTool()
toolset = ToolSet()
toolset.add_tool(search.to_function_tool())

client = OpenAIChat(api_key="sk-...", tools=list(toolset))
resp = client.ask("Latest news about AI")
```

For custom search backends (non-native providers):

```python
from isobase.llm.tools.search.providers import TavilySearchProvider

search = SearchTool(search_provider=TavilySearchProvider(api_key="tvly-..."))
```

### 6. Execution Callbacks

```python
from isobase.llm.callbacks import BaseLLMCallback
from typing import Any, Dict

class MyCallback(BaseLLMCallback):
    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        print(f"[start] {tool_name}({arguments})")

    def on_tool_end(self, tool_name: str, result: Any) -> None:
        print(f"[done]  {tool_name}")

resp = client.ask("Search for Kyoto weather.", callbacks=[MyCallback()])
```

### 7. Multimodal (Images)

```python
from PIL import Image
from isobase.llm.providers.base import build_user_message_content

img = Image.open("photo.jpg")
messages = [
    {"role": "user", "content": build_user_message_content("Describe this:", [img])},
]
resp = client.generate(messages=messages)
```

### 8. Extended Thinking (Anthropic)

```python
client = AnthropicMessages(
    api_key="sk-ant-...",
    thinking={"type": "adaptive"},
)
resp = client.ask("Solve 27 * 453 step by step.")
print(resp.reasoning_content)  # model's chain-of-thought
print(resp.content)            # final answer
```

### 9. Knowledge Base (RAG)

Search your indexed private documents during conversations.

```python
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.stores import MemoryKnowledgeStore
from isobase.llm.tools import ToolSet
from isobase.llm.tools.knowledge import create_knowledge_search_tool

# 1. Set up the knowledge service
service = KnowledgeBaseService(
    embedding_client=OpenAIEmbeddingClient(api_key="...", base_url="...", model="text-embedding-v4"),
    store=MemoryKnowledgeStore(),
    chunker=FixedSizeChunker(),
)
kb = service.create_knowledge_base(name="Tech Docs")
service.index_text(kb.id, "IsoBase is a provider-neutral automation framework.", title="Intro")

# 2. Create a search tool bound to this knowledge base
toolset = ToolSet()
toolset.add_tool(create_knowledge_search_tool(service, kb.id, top_k=3))

# 3. Let the LLM use it
client = OpenAIChat(api_key="sk-...", tools=list(toolset))
resp = client.ask("What is IsoBase?")
print(resp.content)
```

**Multiple knowledge bases?** Use distinct `tool_name` per base:

```python
toolset.add_tool(create_knowledge_search_tool(service, kb_tech.id, tool_name="search_tech_docs"))
toolset.add_tool(create_knowledge_search_tool(service, kb_hr.id,   tool_name="search_hr_policies"))
```

**Custom description?** Tell the model what each base contains:

```python
toolset.add_tool(create_knowledge_search_tool(
    service, kb.id,
    description="Search the internal product wiki for API references and changelogs.",
))
```

For advanced retrieval modes (sparse / hybrid / rerank), see [Knowledge Base Quick Start](../knowledge/QUICKSTART.md).

## Key Types

| Type              | Description                                                                                                       |
| ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| `LLMResponse`     | `.content: str`, `.tool_calls: list[ToolCall]`, `.reasoning_content: str`, `.usage: TokenUsage`, `.success: bool` |
| `ToolCall`        | Neutral tool-call representation (vendor-independent)                                                             |
| `FunctionTool`    | Wraps a Python callable into an LLM tool                                                                          |
| `ToolSet`         | Collection of tools; `.execute_tool_calls()` runs them                                                            |
| `BaseLLMCallback` | Hook interface for tool execution progress                                                                        |

## Providers

| Class               | API                                                                         |
| ------------------- | --------------------------------------------------------------------------- |
| `OpenAIChat`        | OpenAI Chat Completion (and any `/v1/chat/completions` compatible endpoint) |
| `AnthropicMessages` | Anthropic Messages API                                                      |

## Run Examples

```bash
# Unit tests (no API keys required)
python -m pytest test/llm/ -v

# --- Live smoke tests (require API keys) ---

# Copy .env.example → .env, fill in credentials, then:
python -m test.llm.live.run_llm_basic
# Prerequisites: OpenAI-compatible API key in .env

# Knowledge-base agent live test
python -m test.llm.live.run_agent_knowledge
# Prerequisites: .env with EMBEDDING_API_KEY, EMBEDDING_BASE_URL,
#                LLM_API_KEY (OpenAI-compatible), MODEL_ID, 
#                and the knowledge base content has already been stored in the database.
```
