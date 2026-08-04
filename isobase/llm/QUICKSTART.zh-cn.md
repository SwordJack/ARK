# IsoBase LLM — Quick Start

- 汉语 [QUICKSTART.zh-cn.md](QUICKSTART.zh-cn.md)
- English [QUICKSTART.md](QUICKSTART.md)

## 安装

```bash
pip install openai anthropic
```

## 5 分钟快速上手

### 1. 基本对话

```python
from isobase.llm import OpenAIChat, AnthropicMessages

# OpenAI
client = OpenAIChat(api_key="sk-...", default_model="gpt-4o-mini")
resp = client.ask("用法语说你好。")
print(resp.content)          # 最终回答
print(resp.usage.total_tokens)

# Anthropic（可直接互换）
client = AnthropicMessages(api_key="sk-ant-...", default_model="claude-haiku-4-5")
resp = client.ask("用法语说你好。")
print(resp.content)
```

### 2. 流式输出

```python
for chunk in client.ask("从 1 数到 5。", stream=True):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 3. 多轮对话

当你需要完整控制 messages 时，使用 `generate()`：

```python
messages = [
    {"role": "system", "content": "你是一个有用的助手。"},
    {"role": "user", "content": "你好！"},
]
# 方式 1：generate() — 返回 LLMResponse
resp = client.generate(messages=messages, model="gpt-4o-mini")
messages.append({"role": "assistant", "content": resp.content})
messages.append({"role": "user", "content": "给我讲个笑话。"})
resp = client.generate(messages=messages)

# 方式 2：ask() — 内部管理历史，适合单轮场景
resp = client.ask("给我讲个笑话。")
```

### 4. 工具调用（函数调用）

定义一次工具——适用于所有提供商。

```python
from isobase.llm.tools import FunctionTool, ToolSet

def get_weather(city: str) -> tuple[bool, str]:
    """获取指定城市的当前天气。

    Args:
        city: 目标城市名称。
    """
    return True, f"{city} 当前 22°C，晴。"

# 自动从 Python 函数签名和 Docstring 生成 name、description 和参数 Schema
tool = FunctionTool(get_weather)
toolset = ToolSet()
toolset.add_tool(tool)

# ask() 自动处理工具调用循环
client = OpenAIChat(api_key="sk-...", tools=list(toolset))
resp = client.ask("巴黎天气怎么样？")
print(resp.content)  # "巴黎当前 22°C，晴。"
```

手动控制 Schema：

```python
tool = FunctionTool(
    mapped_callable=get_weather,
    name="get_weather_override",
    description="获取指定城市的当前天气状况和温度。",
    parameters_schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "目标城市名称（例如：'Paris', '东京'）",
            }
        },
        "required": ["city"],
    },
)
```

### 5. 联网搜索

```python
from isobase.llm.tools.search import SearchTool

search = SearchTool()
toolset = ToolSet()
toolset.add_tool(search.to_function_tool())

client = OpenAIChat(api_key="sk-...", tools=list(toolset))
resp = client.ask("关于 AI 的最新新闻")
```

对于不支持原生搜索的模型，可注入自定义搜索引擎：

```python
from isobase.llm.tools.search.providers import TavilySearchProvider

search = SearchTool(search_provider=TavilySearchProvider(api_key="tvly-..."))
```

### 6. 执行回调

```python
from isobase.llm.callbacks import BaseLLMCallback
from typing import Any, Dict

class MyCallback(BaseLLMCallback):
    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        print(f"[开始] {tool_name}({arguments})")

    def on_tool_end(self, tool_name: str, result: Any) -> None:
        print(f"[完成] {tool_name}")

resp = client.ask("搜索京都的天气。", callbacks=[MyCallback()])
```

### 7. 厂商中立的消息历史

保存对话历史为厂商中立格式，并可在不同 provider 之间转移。

```python
from isobase.llm import (
    AnthropicMessages, OpenAIChat,
    LLMMessage, LLMMessageHistory, MessageContentBlock, ToolCall,
)

# 构建中立历史
history = LLMMessageHistory()
history.append(LLMMessage(role="user", content="巴黎天气怎么样？"))
history.append(LLMMessage(role="assistant", content=[
    MessageContentBlock(type="tool_use",
        tool_call=ToolCall(id="c1", name="get_weather", arguments='{"city":"Paris"}')),
]))

# 保存为 JSON
saved = history.to_list()

# 重新加载并在 OpenAI 上使用
reloaded = LLMMessageHistory.from_list(saved)
openai = OpenAIChat(api_key="sk-...")
oai_msgs = [OpenAIChat.from_neutral_message(m) for m in reloaded.messages]
resp = openai.ask("结合刚才的工具调用结果回答我。", stream=False)

# 切换到 Anthropic（system 自动提取至顶层参数）
anthropic = AnthropicMessages(api_key="sk-ant-...")
ant_msgs, system_prompt = AnthropicMessages.from_neutral_history(reloaded)
anthropic.instructions = system_prompt
resp = anthropic.ask("结合刚才的工具调用结果回答我。", stream=False)
```

### 8. 多模态（图像输入）

```python
from PIL import Image
from isobase.llm.providers.base import build_user_message_content

img = Image.open("photo.jpg")
messages = [
    {"role": "user", "content": build_user_message_content("描述这张图片：", [img])},
]
resp = client.generate(messages=messages)
```

### 9. 扩展思考（Anthropic）

```python
client = AnthropicMessages(
    api_key="sk-ant-...",
    thinking={"type": "adaptive"},
)
resp = client.ask("请逐步计算 27 × 453。")
print(resp.reasoning_content)  # 模型的思维链
print(resp.content)            # 最终回答
```

### 10. 知识库 (RAG)

在对话中搜索你已索引的私有文档。

```python
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.stores import MemoryKnowledgeStore
from isobase.llm.tools import ToolSet
from isobase.llm.tools.knowledge import create_knowledge_search_tool

# 1. 搭建知识库服务
service = KnowledgeBaseService(
    embedding_client=OpenAIEmbeddingClient(api_key="...", base_url="...", model="text-embedding-v4"),
    store=MemoryKnowledgeStore(),
    chunker=FixedSizeChunker(),
)
kb = service.create_knowledge_base(name="技术文档")
service.index_text(kb.id, "IsoBase 是一个厂商中立的自动化框架。", title="简介")

# 2. 创建绑定到该知识库的搜索工具
toolset = ToolSet()
toolset.add_tool(create_knowledge_search_tool(service, kb.id, top_k=3))

# 3. 让 LLM 使用该工具
client = OpenAIChat(api_key="sk-...", tools=list(toolset))
resp = client.ask("什么是 IsoBase？")
print(resp.content)
```

**多个知识库？** 为每个知识库使用不同的 `tool_name`：

```python
toolset.add_tool(create_knowledge_search_tool(service, kb_tech.id, tool_name="search_tech_docs"))
toolset.add_tool(create_knowledge_search_tool(service, kb_hr.id,   tool_name="search_hr_policies"))
```

**自定义描述？** 告诉模型每个知识库的内容：

```python
toolset.add_tool(create_knowledge_search_tool(
    service, kb.id,
    description="搜索内部产品 wiki 中的 API 参考资料和更新日志。",
))
```

高级检索模式（稀疏 / 混合 / 重排序）见 [Knowledge Base Quick Start](../knowledge/QUICKSTART.zh-cn.md)。

## 关键类型

| 类型              | 说明                                                                                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `LLMResponse`     | `.success: bool`、`.status_code: int`、`.content: str`、`.role: str`、`.tool_calls: list[ToolCall]`、`.reasoning_content: str`、`.usage: TokenUsage` |
| `ToolCall`        | 厂商中立的工具调用表示                                                                                                                               |
| `LLMMessage`      | 厂商中立的消息；支持 `system` / `user` / `assistant` / `tool` 角色                                                                                |
| `MessageContentBlock` | 厂商中立的内容块（`text` / `image` / `tool_use` / `tool_result` / `raw`）                                                                      |
| `LLMMessageHistory`  | 可序列化的中立消息列表；`to_list()` / `from_list()` 用于 JSON 持久化                                                                              |
| `FunctionTool`    | 将 Python 可调用对象包装为 LLM 工具；自动从签名和 Docstring 提取 schema                                                                              |
| `ToolSet`         | 工具集合；`.execute_tool_calls()` 执行工具调用                                                                                                       |
| `BaseLLMCallback` | 工具执行进度回调钩子（`on_tool_start` / `on_tool_end`）                                                                                              |

## 支持的提供商

| 类                  | API                                                              |
| ------------------- | ---------------------------------------------------------------- |
| `OpenAIChat`        | OpenAI Chat Completion（及任何 `/v1/chat/completions` 兼容端点） |
| `AnthropicMessages` | Anthropic Messages API                                           |

## 运行示例

```bash
# 单元测试（无需 API 密钥）
python -m pytest test/llm/ -v

# --- 以下联调测试需要 API 密钥 ---

# 复制 .env.example → .env，填入凭据后运行：
python -m test.llm.live.run_llm_basic
# 运行条件：.env 中配置 OpenAI 兼容 API 密钥

# 消息转换联调测试
python -m test.llm.live.run_message_conversion
# 运行条件：.env 中配置 OPENAI_CHAT_API_KEY 和 ANTHROPIC_MESSAGES_API_KEY

# 知识库 Agent 联调测试
python -m test.llm.live.run_agent_knowledge
# 运行条件：.env 中配置 EMBEDDING_API_KEY、EMBEDDING_BASE_URL、
#          LLM_API_KEY（OpenAI 兼容）、MODEL_ID，
#          且数据库中已经有知识库内容存入。
```
