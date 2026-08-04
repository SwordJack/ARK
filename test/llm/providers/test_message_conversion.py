#! python3
# -*- encoding: utf-8 -*-
"""Unit tests for provider-neutral message conversion.

Covers ``from_neutral_message`` / ``to_neutral_message`` round-trips, cross-
provider transfer, and ``LLMMessageHistory`` serialization, with and without
tool calls.

@File   :   test_message_conversion.py
@Created:   2026/08/05 04:19 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
import json

import pytest

from isobase.llm import (
    AnthropicMessages,
    LLMMessage,
    LLMMessageHistory,
    MessageContentBlock,
    OpenAIChat,
    ToolCall,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _roundtrip(provider_cls, msg: LLMMessage) -> LLMMessage:
    """Native → neutral → native → neutral single-provider round-trip."""
    native = provider_cls.from_neutral_message(msg)
    return provider_cls.to_neutral_message(native)


def _cross(from_provider, to_provider, msg: LLMMessage) -> LLMMessage:
    """Neutral → from-native → neutral → to-native → neutral."""
    native_a = from_provider.from_neutral_message(msg)
    neutral = from_provider.to_neutral_message(native_a)
    native_b = to_provider.from_neutral_message(neutral)
    return to_provider.to_neutral_message(native_b)


def _get_text_blocks(msg: LLMMessage):
    """Returns all text blocks from a message (assumes block-list content)."""
    return [b.text for b in msg.content if b.type == "text"]


def _get_tool_uses(msg: LLMMessage):
    """Returns tool_call dicts for every tool_use block."""
    return [
        {"id": b.tool_call.id, "name": b.tool_call.name, "arguments": b.tool_call.arguments}
        for b in msg.content
        if b.type == "tool_use" and b.tool_call is not None
    ]


def _get_tool_results(msg: LLMMessage):
    """Returns (tool_call_id, text, is_error) for every tool_result block."""
    return [
        (b.tool_call_id, b.text, b.is_error)
        for b in msg.content
        if b.type == "tool_result"
    ]


# ---------------------------------------------------------------------------
# plain-text round-trip (both providers)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("provider", [OpenAIChat, AnthropicMessages])
@pytest.mark.parametrize("role, content", [
    ("user", "Hello"),
    ("assistant", "Hi there"),
    ("system", "Be helpful."),
])
def test_plain_text_roundtrip(provider, role, content):
    """Plain-text messages survive native→neutral→native for every role."""
    msg = LLMMessage(role=role, content=content)
    back = _roundtrip(provider, msg)
    assert back.role == role
    assert back.content == content


# ---------------------------------------------------------------------------
# OpenAI-specific round-trips
# ---------------------------------------------------------------------------

def test_openai_tool_use_roundtrip():
    """Assistant message with tool_calls survives round-trip."""
    msg = LLMMessage(role="assistant", content=[
        MessageContentBlock(type="text", text="Let me look that up."),
        MessageContentBlock(type="tool_use",
            tool_call=ToolCall(id="c1", name="get_weather", arguments='{"city":"Paris"}')),
        MessageContentBlock(type="tool_use",
            tool_call=ToolCall(id="c2", name="get_time", arguments='{"tz":"CET"}')),
    ])
    native = OpenAIChat.from_neutral_message(msg)
    # Native must carry tool_calls with the OpenAI shape.
    assert native["role"] == "assistant"
    assert native["content"] == "Let me look that up."
    assert len(native["tool_calls"]) == 2
    assert native["tool_calls"][0]["type"] == "function"
    assert native["tool_calls"][0]["function"]["name"] == "get_weather"

    back = OpenAIChat.to_neutral_message(native)
    assert back.role == "assistant"
    assert _get_text_blocks(back) == ["Let me look that up."]
    tool_uses = _get_tool_uses(back)
    assert tool_uses == [
        {"id": "c1", "name": "get_weather", "arguments": '{"city":"Paris"}'},
        {"id": "c2", "name": "get_time", "arguments": '{"tz":"CET"}'},
    ]


def test_openai_tool_result_roundtrip():
    """OpenAI role:"tool" message becomes a tool_result block and back."""
    native = {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": "22°C"}
    neutral = OpenAIChat.to_neutral_message(native)
    assert neutral.role == "tool"
    results = _get_tool_results(neutral)
    assert results == [("c1", "22°C", False)]

    # Convert neutral back to native
    back = OpenAIChat.from_neutral_message(neutral)
    assert back["role"] == "tool"
    assert back["tool_call_id"] == "c1"
    assert back["content"] == "22°C"


def test_openai_assistant_no_tool_calls():
    """Assistant with only content (no tool_calls) round-trips correctly."""
    native = {"role": "assistant", "content": "The answer is 42."}
    neutral = OpenAIChat.to_neutral_message(native)
    assert neutral.role == "assistant"
    assert neutral.content == "The answer is 42."

    back = OpenAIChat.from_neutral_message(neutral)
    assert back["role"] == "assistant"
    assert back["content"] == "The answer is 42."


def test_openai_multimodal_user():
    """User message with text + image blocks round-trips."""
    msg = LLMMessage(role="user", content=[
        MessageContentBlock(type="text", text="Describe this"),
        MessageContentBlock(type="image",
            source={"type": "image_url", "image_url": {"url": "data:..."}}),
    ])
    native = OpenAIChat.from_neutral_message(msg)
    assert native["role"] == "user"
    assert isinstance(native["content"], list)
    assert native["content"][0]["type"] == "text"
    assert native["content"][1]["type"] == "image_url"

    back = OpenAIChat.to_neutral_message(native)
    assert back.role == "user"
    types = [b.type for b in back.content]
    assert types == ["text", "image"]


# ---------------------------------------------------------------------------
# Anthropic-specific round-trips
# ---------------------------------------------------------------------------

def test_anthropic_tool_use_roundtrip():
    """Anthropic assistant content-block list round-trips with tool_use."""
    msg = LLMMessage(role="assistant", content=[
        MessageContentBlock(type="text", text="Calling tools."),
        MessageContentBlock(type="tool_use",
            tool_call=ToolCall(id="tu1", name="search", arguments='{"q":"x"}')),
    ])
    native = AnthropicMessages.from_neutral_message(msg)
    assert native["role"] == "assistant"
    assert isinstance(native["content"], list)
    assert native["content"][0]["type"] == "text"
    assert native["content"][1]["type"] == "tool_use"
    # tool_use input should be the parsed dict (Anthropic wire shape)
    assert native["content"][1]["input"] == {"q": "x"}

    back = AnthropicMessages.to_neutral_message(native)
    assert back.role == "assistant"
    assert _get_text_blocks(back) == ["Calling tools."]
    tool_uses = _get_tool_uses(back)
    # arguments are re-serialised to JSON string in the neutral layer
    assert json.loads(tool_uses[0]["arguments"]) == {"q": "x"}


def test_anthropic_tool_result_roundtrip():
    """Anthropic tool_result in a user message round-trips."""
    native = {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "tu1", "content": "result", "is_error": False},
        ],
    }
    neutral = AnthropicMessages.to_neutral_message(native)
    assert neutral.role == "user"
    results = _get_tool_results(neutral)
    assert results == [("tu1", "result", False)]

    back = AnthropicMessages.from_neutral_message(neutral)
    assert back["role"] == "user"
    assert back["content"][0]["type"] == "tool_result"
    assert back["content"][0]["content"] == "result"


def test_anthropic_tool_result_error():
    """is_error flag survives round-trip."""
    msg = LLMMessage(role="user", content=[
        MessageContentBlock(type="tool_result", tool_call_id="tu2", text="boom", is_error=True),
    ])
    native = AnthropicMessages.from_neutral_message(msg)
    assert native["content"][0]["is_error"] is True

    back = AnthropicMessages.to_neutral_message(native)
    results = _get_tool_results(back)
    assert results == [("tu2", "boom", True)]


# ---------------------------------------------------------------------------
# cross-provider transfer
# ---------------------------------------------------------------------------

def test_cross_provider_user_text():
    """OpenAI user message → neutral → Anthropic and back."""
    msg = LLMMessage(role="user", content="What is the capital of France?")
    result = _cross(OpenAIChat, AnthropicMessages, msg)
    assert result.role == "user"
    assert result.content == "What is the capital of France?"


def test_cross_provider_assistant_with_tool_calls():
    """OpenAI assistant + tool_calls → neutral → Anthropic."""
    msg = LLMMessage(role="assistant", content=[
        MessageContentBlock(type="text", text="I'll check."),
        MessageContentBlock(type="tool_use",
            tool_call=ToolCall(id="c1", name="search", arguments='{"q":"Paris"}')),
    ])
    # OpenAI → Anthropic
    oai_native = OpenAIChat.from_neutral_message(msg)
    neutral = OpenAIChat.to_neutral_message(oai_native)
    ant_native = AnthropicMessages.from_neutral_message(neutral)
    assert ant_native["role"] == "assistant"
    assert isinstance(ant_native["content"], list)
    assert ant_native["content"][1]["type"] == "tool_use"

    # And back: Anthropic → OpenAI
    back_neutral = AnthropicMessages.to_neutral_message(ant_native)
    back_openai = OpenAIChat.from_neutral_message(back_neutral)
    assert back_openai["role"] == "assistant"
    assert back_openai["tool_calls"][0]["function"]["name"] == "search"


def test_cross_provider_tool_result():
    """Anthropic tool_result → neutral → OpenAI role:"tool"."""
    ant_native = {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "22°C"}],
    }
    neutral = AnthropicMessages.to_neutral_message(ant_native)
    oai_native = OpenAIChat.from_neutral_message(neutral)
    assert oai_native["role"] == "tool"
    assert oai_native["tool_call_id"] == "tu1"
    assert oai_native["content"] == "22°C"


def test_cross_provider_full_tool_round():
    """Simulates a complete tool-calling round across providers.

    The flow: user asks → assistant calls tool → tool result → assistant
    answers. Each turn is converted cross-provider.
    """
    turns = [
        LLMMessage(role="user", content="Weather in Paris?"),
        LLMMessage(role="assistant", content=[
            MessageContentBlock(type="tool_use",
                tool_call=ToolCall(id="c1", name="get_weather", arguments='{"city":"Paris"}')),
        ]),
        LLMMessage(role="tool", content=[
            MessageContentBlock(type="tool_result", tool_call_id="c1", text="22°C sunny"),
        ]),
        LLMMessage(role="assistant", content="It is 22°C and sunny in Paris."),
    ]

    # Convert every turn to both native formats — must not raise.
    # Note: Anthropic has no dedicated tool-message role; neutral
    # role:"tool" maps to a user message with tool_result blocks.
    for msg in turns:
        oai = OpenAIChat.from_neutral_message(msg)
        ant = AnthropicMessages.from_neutral_message(msg)
        # Round-trip back to neutral through each provider
        rt_oai = OpenAIChat.to_neutral_message(oai)
        rt_ant = AnthropicMessages.to_neutral_message(ant)
        # Roles must match for OpenAI; Anthropic remaps tool → user.
        assert rt_oai.role == msg.role
        expected_ant_role = "user" if msg.role == "tool" else msg.role
        assert rt_ant.role == expected_ant_role


# ---------------------------------------------------------------------------
# LLMMessageHistory persistence
# ---------------------------------------------------------------------------

def test_history_empty_roundtrip():
    """Empty history serialises and deserialises."""
    history = LLMMessageHistory()
    data = history.to_list()
    assert data == []
    restored = LLMMessageHistory.from_list(data)
    assert restored.messages == []


def test_history_plain_text_roundtrip():
    """History with plain-text messages survives JSON round-trip."""
    history = LLMMessageHistory()
    history.append(LLMMessage(role="user", content="Hello"))
    history.append(LLMMessage(role="assistant", content="Hi there"))
    history.append(LLMMessage(role="user", content="What's 2+2?"))

    data = history.to_list()
    restored = LLMMessageHistory.from_list(data)

    assert len(restored.messages) == 3
    for orig, rest in zip(history.messages, restored.messages):
        assert rest.role == orig.role
        assert rest.content == orig.content


def test_history_with_tool_calls_roundtrip():
    """History with tool-use and tool-result blocks survives JSON round-trip."""
    history = LLMMessageHistory()
    history.append(LLMMessage(role="user", content="Weather in Paris?"))
    history.append(LLMMessage(role="assistant", content=[
        MessageContentBlock(type="text", text="Checking..."),
        MessageContentBlock(type="tool_use",
            tool_call=ToolCall(id="c1", name="get_weather", arguments='{"city":"Paris"}')),
    ]))
    history.append(LLMMessage(role="tool", content=[
        MessageContentBlock(type="tool_result", tool_call_id="c1", text="22°C"),
    ]))
    history.append(LLMMessage(role="assistant", content="It is 22°C in Paris."))

    data = history.to_list()
    restored = LLMMessageHistory.from_list(data)

    assert len(restored.messages) == 4
    # Check the tool-call message
    tc_msg = restored.messages[1]
    assert tc_msg.role == "assistant"
    blocks = tc_msg.content
    assert blocks[0].type == "text"
    assert blocks[0].text == "Checking..."
    assert blocks[1].type == "tool_use"
    assert blocks[1].tool_call.id == "c1"
    assert blocks[1].tool_call.name == "get_weather"

    # Check the tool-result message
    tr_msg = restored.messages[2]
    assert tr_msg.role == "tool"
    tr_blocks = tr_msg.content
    assert tr_blocks[0].type == "tool_result"
    assert tr_blocks[0].tool_call_id == "c1"
    assert tr_blocks[0].text == "22°C"


def test_history_copy():
    """Shallow copy produces an independent history list."""
    history = LLMMessageHistory()
    history.append(LLMMessage(role="user", content="A"))
    copy = history.copy()
    copy.append(LLMMessage(role="assistant", content="B"))
    assert len(history.messages) == 1
    assert len(copy.messages) == 2
