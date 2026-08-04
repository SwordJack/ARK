#! python3
# -*- encoding: utf-8 -*-
"""Core data structures for LLM responses and usage.

@File   :   entities.py
@Created:   2026/06/05 00:50
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union


MessageRole = Literal["system", "user", "assistant", "tool"]
ContentBlockType = Literal["text", "image", "tool_use", "tool_result", "raw"]


@dataclass
class MessageContentBlock:
    """Provider-neutral message content block.

    Attributes:
        type: The neutral content block type.
        text: Text content for text or tool_result blocks.
        source: Provider-neutral media source metadata for image blocks.
        tool_call: Tool call data for tool_use blocks.
        tool_call_id: Tool call ID for tool_result blocks.
        name: Optional tool name for tool_result blocks.
        is_error: Whether a tool_result block represents a failed tool call.
        raw: Optional provider-native block for data that has no neutral shape yet.
    """
    type: ContentBlockType
    text: str = ""
    source: Optional[Dict[str, Any]] = None
    tool_call: Optional[ToolCall] = None
    tool_call_id: str = ""
    name: str = ""
    is_error: bool = False
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the content block into plain JSON-compatible data."""
        data: Dict[str, Any] = {"type": self.type}
        if self.text:
            data["text"] = self.text
        if self.source is not None:
            data["source"] = self.source
        if self.tool_call is not None:
            data["tool_call"] = {
                "id": self.tool_call.id,
                "name": self.tool_call.name,
                "arguments": self.tool_call.arguments,
            }
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        if self.is_error:
            data["is_error"] = self.is_error
        if self.raw is not None:
            data["raw"] = self.raw
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MessageContentBlock":
        """Builds a content block from plain JSON-compatible data."""
        tool_call_data = data.get("tool_call")
        tool_call = None
        if tool_call_data:
            tool_call = ToolCall(
                id=tool_call_data["id"],
                name=tool_call_data["name"],
                arguments=tool_call_data["arguments"],
            )
        return cls(
            type=data["type"],
            text=data.get("text", ""),
            source=data.get("source"),
            tool_call=tool_call,
            tool_call_id=data.get("tool_call_id", ""),
            name=data.get("name", ""),
            is_error=data.get("is_error", False),
            raw=data.get("raw"),
        )


@dataclass
class LLMMessage:
    """Provider-neutral conversation message.

    Attributes:
        role: Neutral role for the message.
        content: Text content or a list of neutral content blocks.
    """
    role: MessageRole
    content: Union[str, List[MessageContentBlock]]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the message into plain JSON-compatible data."""
        if isinstance(self.content, str):
            content: Union[str, List[Dict[str, Any]]] = self.content
        else:
            content = [block.to_dict() for block in self.content]
        return {"role": self.role, "content": content}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMessage":
        """Builds a message from plain JSON-compatible data."""
        content = data["content"]
        if isinstance(content, list):
            content = [MessageContentBlock.from_dict(block) for block in content]
        return cls(role=data["role"], content=content)


@dataclass
class LLMMessageHistory:
    """Provider-neutral message history that can be persisted and reused.

    Attributes:
        messages: Ordered list of provider-neutral messages.
    """
    messages: List[LLMMessage] = field(default_factory=list)

    def append(self, message: LLMMessage) -> None:
        """Appends a neutral message to the history."""
        self.messages.append(message)

    def copy(self) -> "LLMMessageHistory":
        """Returns a shallow copy of the history."""
        return LLMMessageHistory(messages=self.messages.copy())

    def to_list(self) -> List[Dict[str, Any]]:
        """Serializes the history into a JSON-compatible list."""
        return [message.to_dict() for message in self.messages]

    @classmethod
    def from_list(cls, data: List[Dict[str, Any]]) -> "LLMMessageHistory":
        """Builds a neutral history from a JSON-compatible list."""
        return cls(messages=[LLMMessage.from_dict(message) for message in data])

@dataclass
class TokenUsage:
    """Represents token consumption for an LLM request.

    Attributes:
        input_tokens: Number of tokens in the input prompt.
        output_tokens: Number of tokens in the generated response.
        total_tokens: Sum of input and output tokens.
    """
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolCall:
    """A provider-neutral representation of a tool/function call requested by the LLM.

    Attributes:
        id: The unique identifier for this tool call.
        name: The name of the tool/function to call.
        arguments: A JSON-serialized string of the arguments.
    """
    id: str
    name: str
    arguments: str


@dataclass
class SearchResultItem:
    """Represents a single standardized search result from a web search provider.

    Attributes:
        title: The title of the search result page.
        url: The absolute URL of the page.
        snippet: A short summary or block of content.
        raw_content: Optional full-page parsed content or markdown.
    """
    title: str
    url: str
    snippet: str
    raw_content: Optional[str] = None


@dataclass
class SearchResult:
    """Represents the complete standardized outcome of a search provider query.

    Attributes:
        success: Whether the search operation succeeded.
        results: A list of SearchResultItem instances.
        error: Optional error message if the search failed.
    """
    success: bool
    results: List[SearchResultItem] = field(default_factory=list)
    error: Optional[str] = None

    def __str__(self) -> str:
        """Standardized string representation optimized for LLM prompt consumption."""
        if not self.success:
            return f"Search failed: {self.error}"

        if not self.results:
            return "No search results found."

        formatted_items = []
        for idx, item in enumerate(self.results, 1):
            block = f"[{idx}] {item.title}\nURL: {item.url}\nSnippet: {item.snippet}"
            if item.raw_content:
                block += f"\nFull Content:\n{item.raw_content}"
            formatted_items.append(block)

        return "\n\n---\n\n".join(formatted_items)


@dataclass
class LLMResponse:
    """Structured response from an LLM provider.

    Attributes:
        success: Whether the request was successful.
        status_code: The HTTP status code or internal error code.
        content: The main text content of the response.
        role: The role of the responder (usually 'assistant').
        usage: Detailed token consumption statistics.
        tool_calls: A list of tool calls requested by the model.
        raw_response: The original response object from the SDK/API.
        reasoning_content: Internal reasoning or chain-of-thought, if available.
    """
    success: bool
    status_code: int
    content: str = ""
    role: str = "assistant"
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_response: Any = None
    reasoning_content: str = ""

    def __str__(self) -> str:
        """Returns a string representation of the response summary."""
        return (f"LLMResponse(success={self.success}, status={self.status_code}, "
                f"content_len={len(self.content)})")
