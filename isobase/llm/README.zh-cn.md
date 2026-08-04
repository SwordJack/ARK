# LLM 模块

> For English documentation, see [README.md](./README.md)

## 简介

该 `isobase.llm` 模块是一个**厂商中立**的大语言模型（LLM）客户端层。它在不同厂商 API 之上暴露一套一致的接口（`ask` / `generate` / `generate_stream`），返回标准化的 `LLMResponse`，并支持多轮工具调用、多模态图像输入、流式输出与扩展思考（extended thinking）。目前已有两个可互换的 provider：OpenAI Chat Completion 兼容（`OpenAIChat`）与 Anthropic Messages 兼容（`AnthropicMessages`）。

设计目标是：**切换 provider 不需要改动调用代码**——工具以一种中立格式定义一次，按需渲染成各厂商的线上 schema；每个方法都返回同样的 `LLMResponse`。

## 功能特性

- **厂商中立的统一接口**：`OpenAIChat` 与 `AnthropicMessages` 共享相同的公开方法、返回相同的 `LLMResponse`，可直接互换。
- **厂商中立的消息历史**：`LLMMessage`、`MessageContentBlock` 和 `LLMMessageHistory` 允许你保存对话历史并在不同 provider 之间转移。每个 provider 都能通过 `from_neutral_message` / `to_neutral_message` 在中立格式和自身原生格式之间双向转换——与 `ToolCall` 的双向转换模式一致。
- **中立工具格式与自动 Schema 生成**：单个 `FunctionTool` 可直接从绑定的 Python 函数的签名（Signature）和文档字符串（Docstring）中**自动提取**其 `name`、`description` 和 JSON `parameters_schema`。在自动生成期间，它会优雅地跳过不定长参数（`*args`、`**kwargs`）以防止引发模型幻觉。不把任何一家的格式当作唯一标准，各提供商的客户端在发送请求前自动将中立的 `ToolCall` 即时翻译为自身需要的格式。
- **多轮工具调用编排**：`ask()` 运行一个有上限的递归工具调用循环（`max_tool_rounds`），自动执行 Python 可调用对象并把结果回传给模型。内置了严格的兜底机制，如果在循环触达上限时模型仍在尝试调用工具，框架将强制切断工具并要求模型进行最终的纯文本总结，彻底防止空回复和死循环黑洞。
- **严苛网关兼容与历史重建**：在重建对话历史时，完美适配要求极为严苛的企业级 API 代理中转站。包括显式保留空的 `content`（使用 `null` 占位），以及精确对齐并拼接 Anthropic 并发输出的 `tool_use` 与 `tool_result` 内容块。
- **流式与非流式**：每个 provider 都实现了两条路径；流式会先逐块产出文本/思考增量，最后再产出携带工具调用与用量的汇总块。
- **扩展思考**：通过 `thinking` 开启（如 Anthropic 的 `{"type": "adaptive"}`）；思考文本会进入 `LLMResponse.reasoning_content`。
- **多模态图像输入**：`build_user_message_content` 可附加 PIL 图像，并渲染为各厂商的图像块格式。
- **标准化响应与用量**：每次调用都返回 `LLMResponse`（success、status_code、content、tool_calls、reasoning_content、usage、raw_response），并用 `TokenUsage` 统计 token。
- **兼容网关健壮性**：Anthropic 流式路径能容忍发送非规范 `message_start.content: null` 的兼容网关（该字段会令 SDK 的高层流式助手崩溃）。

## 快速开始

`isobase.llm` **不会**在顶层 `isobase` 包中重新导出，请直接从 `isobase.llm` 导入。

```python
from isobase.llm import OpenAIChat, AnthropicMessages

# --- OpenAI Chat Completion 兼容 ---
client = OpenAIChat(api_key="sk-...", default_model="gpt-4o-mini")
resp = client.ask("Say pong.", stream=False)
print(resp.content, resp.usage.total_tokens)

# --- Anthropic Messages 兼容（可互换）---
client = AnthropicMessages(api_key="sk-ant-...", default_model="claude-opus-4-8")
resp = client.ask("Say pong.", stream=False)
print(resp.content, resp.usage.total_tokens)
```

### 流式

```python
for chunk in client.ask("Count from 1 to 5.", stream=True):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### 工具调用与联网搜索

工具以中立格式定义一次，对两个 provider 都适用。你可以把本地自定义的 Python 函数映射为工具，也可以直接使用内置的高级预置工具，比如**双引擎架构的联网搜索工具**。

#### 1. 联网搜索工具 (原生搜索 & 自定义兜底)

`IsoBase` 包含一个强大且鲁棒的 `SearchTool`。它会自动适应当前的 LLM 提供商。如果提供商（如 OpenAI、Qwen 百炼）原生支持联网搜索，它会将自身退化为原生的 `{"type": "web_search"}` 格式，由云端大厂代劳搜索；而对于不支持原生搜索的模型，你可以为其挂载 `TavilySearchProvider` 或 `BraveSearchProvider` 等具体实例，它将回退为本地的标准 Function Calling。

详细用法与自定义企业内网搜索接口配置，请参阅专用文档：[`isobase/llm/tools/search/README.zh-cn.md`](./tools/search/README.zh-cn.md)。

#### 2. 自定义本地函数工具

#### 选项 A：自定义工具的自动生成（默认行为）

```python
from isobase.llm.tools import FunctionTool

def get_weather(city: str) -> tuple[bool, str]:
    """Get the current weather for a city.

    Args:
        city: The name of the city.
    """
    return True, f"It is 22C and sunny in {city}."  # (是否成功, 内容)

# 自动从 Python 函数签名和 Docstring 中提取 name、description 和 parameters_schema
weather_tool = FunctionTool(get_weather)
```

#### 选项 B：自定义工具的手动显式覆盖（显式 Schema）

如果您需要精确控制向大模型暴露的元数据和参数 Schema，或者希望将 JSON-Schema 与 Python 函数签名解耦，可以显式提供 `name`、`description` 和 `parameters_schema` 参数：

```python
from isobase.llm.tools import FunctionTool

# 手动显式覆盖元数据和 schema
weather_tool = FunctionTool(
    mapped_callable=get_weather,
    name="get_weather_override",
    description="获取指定城市的当前天气状况与温度信息。",
    parameters_schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "目标城市名称（例如：'Paris', 'Tokyo'）"
            }
        },
        "required": ["city"]
    }
)
```

对于上述任一选项，都可以将工具注册到客户端：

```python
client = AnthropicMessages(
    api_key="sk-ant-...",
    tools=[weather_tool],
)
resp = client.ask("What's the weather in Paris? Use the tool.")
print(resp.content)                       # 工具轮之后的最终回答
print(client.latest_tool_call_result)     # {'get_weather': True}
```

#### 3. 执行回调与进度监听 (Callbacks)

如果你需要在前端 UI 实时显示工具的执行状态（例如“正在搜索网络…”、“找到 5 条结果”），你可以将实现了 `BaseLLMCallback` 协议的实例传入 `ask` 或 `generate_stream` 方法。

```python
from isobase.llm.callbacks import BaseLLMCallback
from typing import Any, Dict

class MyUIUpdater(BaseLLMCallback):
    def on_tool_start(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        if tool_name == "web_search":
            print(f"[UI 更新] 🔍 正在全网搜索：{arguments.get('query')}...")

    def on_tool_end(self, tool_name: str, result: Any) -> None:
        # 对于内置的搜索工具，result 是一个强类型的 SearchResult 对象
        if tool_name == "web_search" and getattr(result, "success", False):
            print(f"[UI 更新] ✨ 搜索完毕。共找到 {len(result.results)} 条结果。")

client.ask("东京的最新新闻", callbacks=[MyUIUpdater()])
```

### 4. 扩展思考（Anthropic）

```python
client = AnthropicMessages(api_key="sk-ant-...", thinking={"type": "adaptive"})
resp = client.ask("Solve 27 * 453 step by step.")
print(resp.reasoning_content)  # 思考文本
print(resp.content)            # 最终回答
```

### 5. 厂商中立的消息历史

对话历史可以保存为厂商中立格式，也可以在不同 provider 之间转移。这与 `ToolCall` 的双向转换模式一致——每个 provider 都能在中立格式和自身原生格式之间互转。

```python
from isobase.llm import (
    AnthropicMessages, OpenAIChat,
    LLMMessage, LLMMessageHistory, MessageContentBlock, ToolCall,
)

# --- 构建中立历史（任意 provider）---
history = LLMMessageHistory()
history.append(LLMMessage(role="user", content="巴黎的天气怎么样？"))
history.append(LLMMessage(role="assistant", content=[
    MessageContentBlock(type="text", text="让我查一下。"),
    MessageContentBlock(type="tool_use",
        tool_call=ToolCall(id="c1", name="get_weather", arguments='{"city":"Paris"}')),
]))

# --- 序列化为 JSON 以便持久保存 ---
saved = history.to_list()

# --- 在 OpenAI 上重新加载 ---
reloaded = LLMMessageHistory.from_list(saved)
openai = OpenAIChat(api_key="sk-...")
openai_msgs = [OpenAIChat.from_neutral_message(m) for m in reloaded.messages]
resp = openai.ask("结合刚才的工具调用结果回答我。", stream=False)

# --- 或切换到 Anthropic（system 自动提取至顶层参数）---
anthropic = AnthropicMessages(api_key="sk-ant-...")
anthropic_msgs, system_prompt = AnthropicMessages.from_neutral_history(reloaded)
anthropic.instructions = system_prompt
resp = anthropic.ask("结合刚才的工具调用结果回答我。", stream=False)
```

## 模块结构

- `entities.py` —— `LLMMessage`、`LLMMessageHistory`、`MessageContentBlock`、`LLMResponse`、`TokenUsage` 与 `ToolCall` 数据类；用作输入和每个客户端方法的标准化返回类型。
- `providers/base.py` —— `BaseLLMClient` 抽象接口（`ask`、`generate`、`generate_stream`、`from_neutral_message`、`to_neutral_message`、`from_neutral_history`）以及用于多模态消息的 `build_user_message_content`。
- `providers/openai_chat.py` —— `OpenAIChat`，OpenAI Chat Completion 兼容客户端。
- `providers/anthropic_messages.py` —— `AnthropicMessages`，Anthropic Messages 兼容客户端（按其对接的 Messages API 命名，与 `OpenAIChat` 按 Chat Completion API 命名同理）。
- `tools/base.py` —— `FunctionTool` 与 `ToolSet`；中立工具表示及执行引擎。
- `tools/search/` —— 包含双引擎架构的 `SearchTool`、`BaseSearchProvider` 抽象接口以及具体的联网搜索引擎实现。

图像辅助函数位于 `isobase/core/image_service.py`（`convert_image_to_data_url` 供 OpenAI，`convert_image_to_base64` 供 Anthropic）。

手动真实 API 冒烟测试（不由 pytest 收集）位于 `test/llm/live/` —— 把 `.env.example` 复制为 `.env`、填入凭据，运行 `python -m test.llm.live.run_llm_basic` 或 `python -m test.llm.live.run_message_conversion`。

## 进度

### 已完成

- `BaseLLMClient` 抽象接口与标准化的 `LLMResponse` / `TokenUsage`。
- `OpenAIChat` provider：非流式、流式、多轮工具调用、多模态图像输入、出错时保留历史。
- `AnthropicMessages` provider：非流式、流式、多轮工具调用、多模态图像输入、扩展思考、顶层 `system`、必填 `max_tokens`，以及对发送 `message_start.content: null` 的兼容网关的健壮处理。
- 中立 `FunctionTool`，含 OpenAI/Anthropic schema 双向转化与共享的执行核（`ToolSet.execute_tool_calls` / `execute_tool_calls_anthropic`）。
- 通过 `isobase.llm.tools.knowledge.create_knowledge_search_tool` 集成知识库检索工具。
- `SearchTool` 双引擎架构联网搜索（原生提供商搜索 + 自定义搜索引擎回退 `TavilySearchProvider` / `BraveSearchProvider`）。
- 厂商中立的消息历史（`LLMMessage` / `LLMMessageHistory` / `MessageContentBlock`），支持与 provider 原生格式的双向转换（`from_neutral_message` / `to_neutral_message` / `from_neutral_history`）、JSON 序列化以及跨 provider 的对话转移。
- `BaseLLMCallback` — 执行回调钩子，可实时追踪工具执行进度。
- 对各 SDK 打桩的单元测试（`test/llm/providers/`），消息转换测试（`test/llm/providers/test_message_conversion.py`），以及手动真实联调脚本（`test/llm/live/run_message_conversion.py`）。

### 尚未完成（以后做）

- **RAG 自动化**：知识库检索已经可以作为 `FunctionTool` 暴露；更高层的自动 RAG 编排仍留待后续实现。
- **MCP**：Model Context Protocol 集成尚未实现（在同样的工具出入口预留了空间）。
- **更多 provider**：例如 OpenAI Responses API（会是一个独立的类，而非 `OpenAIChat`）、Gemini 等。
- **结构化输出**：各厂商原生的 JSON-schema / 严格输出模式尚未通过中立层暴露。
- **异步客户端**：目前只有同步客户端。
- **Prompt 缓存 / 批处理**：厂商的成本优化特性尚未暴露。
- **自动化真实测试**：联调脚本是手动、人工检视的，未接入 pytest（接入需要带凭据守卫的 opt-in）。
