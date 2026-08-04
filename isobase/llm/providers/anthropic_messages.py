#! python3
# -*- encoding: utf-8 -*-
"""Anthropic Messages API client with structured responses and tool support.

``AnthropicMessages`` is the concrete provider for the Anthropic Messages API
(``/v1/messages``) and any Anthropic-compatible gateway (via ``base_url``). It
mirrors the public surface of :class:`~isobase.llm.providers.openai_chat.OpenAIChat`
(``generate`` / ``generate_stream`` / ``ask`` /
``build_user_message_content``) and returns the standardized ``LLMResponse``,
so the two providers are interchangeable.

The class is named after the API it targets (the *Messages* API), matching the
convention behind ``OpenAIChat`` (the OpenAI *Chat Completion* API). Tool
calling, and the room left for RAG / MCP, route through the same ``ToolSet`` /
``LLMResponse`` entry points the OpenAI client uses.

Key Anthropic-specific handling versus OpenAI:
    - ``system`` is a top-level string, not a message role; ``self.messages``
      never contains a system entry.
    - ``max_tokens`` is mandatory.
    - Tool results are ``tool_result`` blocks inside a ``user`` message.
    - Tool schemas are flat ``{name, description, input_schema}``.
    - Images are ``{"type": "image", "source": {"type": "base64", ...}}``.
    - Extended thinking is opt-in via ``thinking`` (e.g. ``{"type": "adaptive"}``);
      on a tool round the assistant turn echoes back the raw content blocks so
      thinking signatures are preserved.

@File   :   anthropic_messages.py
@Created:   2026/06/06 21:13
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from inspect import signature
from json import dumps, loads as json_loads
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union, overload

from anthropic import Anthropic, BadRequestError
# Reuse the SDK's own streaming accumulator instead of hand-rolling block
# reassembly; see generate_stream for why we drive it ourselves.
from anthropic.lib.streaming._messages import accumulate_event
from PIL import Image as PILImage

from isobase.core.image_service import convert_image_to_base64
from isobase.core.logger import LOGGER

from .base import BaseLLMClient
from ..callbacks import BaseLLMCallback
from ..entities import LLMMessage, LLMMessageHistory, LLMResponse, MessageContentBlock, TokenUsage, ToolCall
from ..tools import FunctionTool, ToolSet


class AnthropicMessages(BaseLLMClient):
    """Client for the Anthropic Messages API and compatible endpoints.

    Supports structured ``LLMResponse``, safe history mutation, multi-turn tool
    calling, multimodal image input, streaming, and extended thinking.

    Attributes:
        client: The underlying Anthropic client instance.
        default_model: The default model ID to use for requests.
        instructions: System instructions defining the agent's behavior (sent as
            the top-level ``system`` parameter, never as a message).
        conversation_mode: Whether to maintain a message history.
        max_tool_rounds: Maximum number of recursive tool-calling rounds.
        max_tokens: The default ``max_tokens`` for requests (required by the API).
        thinking: Optional extended-thinking config (e.g. ``{"type": "adaptive"}``).
        messages: The current conversation history (never contains a system entry).
        tool_set: A collection of available tools.
        latest_tool_call_result: Results of the last tool executions.
        anthropic_args_dict: Filtered kwargs for the Anthropic SDK.
    """

    def __init__(self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "claude-opus-4-8",
        instructions: str = "",
        messages: List[Dict[str, Any]] = None,
        tools: Union[ToolSet, List[FunctionTool], None] = None,
        conversation_mode: bool = True,
        max_tool_rounds: int = 5,
        max_tokens: int = 4096,
        thinking: Optional[Dict[str, Any]] = None,
        **kwargs: Any):
        """Initializes an AnthropicMessages client.

        Args:
            api_key: The API key for authentication.
            base_url: The base URL for the API endpoint (for compatible gateways).
            default_model: The default model ID.
            instructions: System instructions for the model (top-level ``system``).
            messages: Initial list of conversation messages (no system entry).
            tools: An optional ToolSet or list of FunctionTool objects.
            conversation_mode: If True, history is preserved across asks.
            max_tool_rounds: Limits recursive tool calls per ask.
            max_tokens: Default maximum output tokens (required by the API).
            thinking: Optional extended-thinking config; omitted when None.
            **kwargs: Additional arguments for the Anthropic API.
        """
        self.client = Anthropic(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        self.instructions = instructions
        self.conversation_mode = conversation_mode
        self.max_tool_rounds = max_tool_rounds
        self.max_tokens = max_tokens
        self.thinking = thinking

        # System lives in the top-level `system` param, so it is never stored
        # as a message entry (unlike OpenAIChat).
        self.messages = messages or []

        if isinstance(tools, ToolSet):
            self.tool_set = tools
        else:
            self.tool_set = ToolSet(tools or [])
        self.latest_tool_call_result = {}

        create_params = signature(self.client.messages.create).parameters
        self.anthropic_args_dict = {
            k: v for k, v in kwargs.items() if k in create_params
        }

        LOGGER.info(f"AnthropicMessages initialized (model: {default_model})")

    # --- neutral / native message conversion ---------------------------------

    @classmethod
    def from_neutral_history(cls, history: LLMMessageHistory) -> List[Dict[str, Any]]:
        """Converts a full neutral history into an Anthropic-native messages list.

        Merges consecutive role:"tool" messages (neutral's convention for
        individual tool results from OpenAI) into a single user message
        containing all ``tool_result`` blocks, which is the shape the
        Anthropic Messages API requires.
        """
        result: List[Dict[str, Any]] = []
        pending_tool_results: List[Dict[str, Any]] = []
        for neutral in history.messages:
            if neutral.role == "tool":
                native = cls.__tool_message_to_tool_result_blocks(neutral)
                pending_tool_results.extend(native)
                continue
            # Flush any buffered tool results from prior messages.
            if pending_tool_results:
                result.append({"role": "user", "content": pending_tool_results})
                pending_tool_results = []
            result.append(cls.from_neutral_message(neutral))
        if pending_tool_results:
            result.append({"role": "user", "content": pending_tool_results})
        return result

    @classmethod
    def __tool_message_to_tool_result_blocks(cls,
        message: LLMMessage,
    ) -> List[Dict[str, Any]]:
        """Converts a neutral role:"tool" message into Anthropic tool_result blocks."""
        blocks: List[Dict[str, Any]] = []
        if isinstance(message.content, str):
            blocks.append({"type": "tool_result", "tool_use_id": "", "content": message.content})
            return blocks
        for block in message.content:
            if block.type == "tool_result":
                tr: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.tool_call_id,
                    "content": block.text or "",
                }
                if block.is_error:
                    tr["is_error"] = True
                blocks.append(tr)
        return blocks

    @classmethod
    def from_neutral_message(cls, message: LLMMessage) -> Dict[str, Any]:
        """Converts a provider-neutral message into an Anthropic-native dict.

        Anthropic never sends ``system`` inside ``messages`` — the caller is
        expected to extract that entry and pass it via the top-level
        ``system`` parameter instead.
        """
        if message.role == "system":
            return {"role": "system", "content": message.content if isinstance(message.content, str) else ""}

        if message.role == "user":
            if isinstance(message.content, str):
                return {"role": "user", "content": message.content}
            native_blocks: List[Dict[str, Any]] = []
            tool_results: List[Dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    native_blocks.append({"type": "text", "text": block.text})
                elif block.type == "image":
                    if block.source:
                        native_blocks.append(block.source)
                elif block.type == "tool_result":
                    tr: Dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": block.tool_call_id,
                        "content": block.text or "",
                    }
                    if block.is_error:
                        tr["is_error"] = True
                    tool_results.append(tr)
                elif block.type == "raw" and block.raw is not None:
                    native_blocks.append(block.raw)
            # Anthropic tool results live inside a user-message; normal text
            # and tool results are mutually exclusive blocks.
            if tool_results and not native_blocks:
                return {"role": "user", "content": tool_results}
            if tool_results:
                native_blocks.extend(tool_results)
            return {"role": "user", "content": native_blocks}

        if message.role == "assistant":
            if isinstance(message.content, str):
                return {"role": "assistant", "content": message.content}
            blocks: List[Dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    blocks.append({"type": "text", "text": block.text})
                elif block.type == "tool_use" and block.tool_call is not None:
                    blocks.append({
                        "type": "tool_use",
                        "id": block.tool_call.id,
                        "name": block.tool_call.name,
                        "input": json_loads(block.tool_call.arguments) if block.tool_call.arguments else {},
                    })
                elif block.type == "raw" and block.raw is not None:
                    blocks.append(block.raw)
            return {"role": "assistant", "content": blocks}

        # role == "tool" — translate to tool_result inside a user message
        if isinstance(message.content, str):
            return {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "",
                    "content": message.content,
                }],
            }
        blocks = []
        for block in (message.content if isinstance(message.content, list) else []):
            if block.type == "tool_result":
                tr_dict: Dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": block.tool_call_id,
                    "content": block.text or "",
                }
                if block.is_error:
                    tr_dict["is_error"] = True
                blocks.append(tr_dict)
            elif block.type == "text":
                blocks.append({"type": "text", "text": block.text})
            elif block.type == "raw" and block.raw is not None:
                blocks.append(block.raw)
        return {"role": "user", "content": blocks or message.content}

    @classmethod
    def to_neutral_message(cls, native_message: Dict[str, Any]) -> LLMMessage:
        """Converts an Anthropic-native message dict into a neutral LLMMessage.

        Handles the key differences from the OpenAI shape:

        - ``system`` entries in the message list denote operator-level context
          that was injected mid-conversation (a separate concept from the
          top-level ``system`` parameter).
        - Assistant content is a list of typed blocks (text / tool_use / thinking).
          Thinking blocks are captured as ``raw`` since the neutral layer
          does not yet model them natively.
        - Tool results are ``tool_result`` blocks inside a user message,
          never a standalone ``role:"tool"`` message.
        """
        role = native_message.get("role", "user")
        content = native_message.get("content", "")

        if role == "system":
            return LLMMessage(role="system", content=content if isinstance(content, str) else "")

        if role == "user":
            if isinstance(content, str):
                return LLMMessage(role="user", content=content)
            blocks: List[MessageContentBlock] = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "tool_result":
                            blocks.append(MessageContentBlock(
                                type="tool_result",
                                tool_call_id=item.get("tool_use_id", ""),
                                text=item.get("content", ""),
                                is_error=item.get("is_error", False),
                            ))
                        elif item_type == "text":
                            blocks.append(MessageContentBlock(type="text", text=item.get("text", "")))
                        elif item_type == "image":
                            blocks.append(MessageContentBlock(type="image", source=item))
                        else:
                            blocks.append(MessageContentBlock(type="raw", raw=item))
                    else:
                        blocks.append(MessageContentBlock(type="text", text=str(item)))
            return LLMMessage(role="user", content=blocks)

        if role == "assistant":
            if isinstance(content, str):
                return LLMMessage(role="assistant", content=content)
            blocks = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "text":
                            blocks.append(MessageContentBlock(type="text", text=item.get("text", "")))
                        elif item_type == "tool_use":
                            blocks.append(MessageContentBlock(
                                type="tool_use",
                                tool_call=ToolCall(
                                    id=item.get("id", ""),
                                    name=item.get("name", ""),
                                    arguments=dumps(item.get("input", {})),
                                ),
                            ))
                        elif item_type == "thinking":
                            blocks.append(MessageContentBlock(type="raw", raw=item))
                        else:
                            blocks.append(MessageContentBlock(type="raw", raw=item))
                    elif hasattr(item, "type"):
                        item_type = getattr(item, "type", "")
                        if item_type == "text":
                            blocks.append(MessageContentBlock(type="text",
                                text=getattr(item, "text", "")))
                        elif item_type == "tool_use":
                            blocks.append(MessageContentBlock(
                                type="tool_use",
                                tool_call=ToolCall(
                                    id=getattr(item, "id", ""),
                                    name=getattr(item, "name", ""),
                                    arguments=dumps(getattr(item, "input", {})),
                                ),
                            ))
                        else:
                            blocks.append(MessageContentBlock(type="raw", raw=item))
            return LLMMessage(role="assistant", content=blocks)

        # role == "tool" (unusual in Anthropic, but handle)
        if isinstance(content, str):
            return LLMMessage(role="tool", content=[
                MessageContentBlock(type="tool_result", text=content),
            ])
        return LLMMessage(role="user", content=content if isinstance(content, str) else str(content))

    @classmethod
    def build_user_message_content(
        cls, prompt: str,
        images: Optional[List[PILImage.Image]] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """Builds user message content in Anthropic format.

        Overrides the OpenAI-shaped base implementation to emit Anthropic
        content blocks: a ``text`` block plus base64 ``image`` blocks.

        Args:
            prompt: The text prompt provided by the user.
            images: An optional list of PIL Image objects to attach.

        Returns:
            A string if only text is provided, or a list of Anthropic content
            blocks for multimodal requests.

        Raises:
            TypeError: If any item in the images list is not a PIL Image.
        """
        if not images:
            return prompt

        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if not isinstance(img, PILImage.Image):
                raise TypeError("Each image must be a PIL.Image.Image instance.")
            media_type, image_base64 = convert_image_to_base64(img)
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64,
                },
            })
        return content

    def generate(self,
            messages: List[Dict[str, Any]],
            model: Optional[str] = None,
            max_tokens: Optional[int] = None,
            system: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            thinking: Optional[Dict[str, Any]] = None,
            **kwargs: Any
        ) -> LLMResponse:
        """Sends a single non-streaming Messages request.

        Top-level Anthropic parameters are passed explicitly (callers decide
        what to send) rather than auto-injected from ``self`` — this keeps the
        method a pure atomic call, matching ``OpenAIChat.generate``. The
        ask loops supply ``system`` / ``tools`` / ``thinking`` from the client's
        configuration.

        Args:
            messages: Conversation context (Anthropic message format).
            model: Specific model ID to use (defaults to the client default).
            max_tokens: Max output tokens (defaults to the client default).
            system: Top-level system instructions; omitted from the request
                when None.
            tools: Anthropic-format tool schemas; omitted when None.
            thinking: Extended-thinking config; omitted when None.
            **kwargs: Extra parameters for the API.

        Returns:
            An LLMResponse containing the model's output.
        """
        try:
            # Resolve client-level defaults here, at the public entry point, so
            # a direct call (no ask loop) still gets a model and the mandatory
            # max_tokens. __build_request_args assembles the rest; we stamp these
            # two in as authoritative (overriding any stray copy in kwargs).
            request_args = self.__build_request_args(
                system, tools, thinking, **kwargs)
            request_args["model"] = model or self.default_model
            request_args["max_tokens"] = (
                max_tokens if max_tokens is not None else self.max_tokens)
            message = self.client.messages.create(messages=messages,
                                                  **request_args)
            content, tool_calls, reasoning = self.__parse_content(
                message.content)
            usage = self.__extract_usage(message.usage)
            return LLMResponse(
                success=True, status_code=200,
                content=content, usage=usage,
                tool_calls=tool_calls, reasoning_content=reasoning,
                raw_response=message
            )
        except Exception as e:
            return LLMResponse(success=False, status_code=500, content=f"Error: {e}")

    def generate_stream(
            self, messages: List[Dict[str, Any]],
            model: Optional[str] = None,
            max_tokens: Optional[int] = None,
            system: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            thinking: Optional[Dict[str, Any]] = None,
            **kwargs: Any
        ) -> Iterator[LLMResponse]:
        """Sends a streaming Messages request.

        Top-level parameters are passed explicitly (see ``generate``).

        Args:
            messages: Conversation context (Anthropic message format).
            model: Specific model ID to use (defaults to the client default).
            max_tokens: Max output tokens (defaults to the client default).
            system: Top-level system instructions; omitted when None.
            tools: Anthropic-format tool schemas; omitted when None.
            thinking: Extended-thinking config; omitted when None.
            **kwargs: Extra parameters for the API.

        Yields:
            LLMResponse objects containing incremental chunks, then a final
            summary carrying aggregated tool calls and usage.
        """
        # Resolve client-level defaults here, at the public entry point (see
        # generate). __build_request_args assembles the rest; we stamp
        # model/max_tokens in as authoritative (overriding any stray in kwargs).
        request_args = self.__build_request_args(system, tools, thinking, **kwargs)
        request_args["model"] = model or self.default_model
        request_args["max_tokens"] = (
            max_tokens if max_tokens is not None else self.max_tokens)

        # We drive the RAW event iterator via ``messages.create(stream=True)``
        # rather than the SDK's high-level ``messages.stream()`` helper, but we
        # still reuse the SDK's own accumulator (``accumulate_event``) to build
        # the final message — we do NOT hand-roll block reassembly.
        #
        # Why not ``messages.stream()`` directly: it seeds its snapshot from the
        # ``message_start`` event's ``message.content``. The official API sends
        # ``content: []`` there, but some Anthropic-*compatible* gateways send
        # ``content: null``; the SDK then does ``snapshot.content.append(...)``
        # on the next ``content_block_start`` and raises ``'NoneType' object has
        # no attribute 'append'``. That crash is unreachable from the public
        # ``messages.stream()`` API (no hook to fix the event first).
        #
        # We could keep ``messages.stream()`` by monkeypatching the module-level
        # ``accumulate_event`` (the bare name it resolves at call time) to fix
        # the field — that does work. We don't, because it mutates a third-party
        # module attribute process-wide: every Anthropic stream in the process
        # (other code, tests) would be rerouted through our patch. Both
        # approaches couple to the same private ``accumulate_event``, so the
        # patch buys nothing extra while widening the blast radius from a local
        # call to a global override.
        #
        # So we take the raw events, normalize just that one offending field,
        # and feed each event through the SDK's accumulator. Live text/thinking
        # deltas are yielded as they arrive; the accumulated snapshot is parsed
        # with the same ``__parse_content`` / ``__extract_usage`` the
        # non-streaming path uses.
        stream = self.client.messages.create(messages=messages, stream=True,
                                             **request_args)

        snapshot = None
        for event in stream:
            # Normalize the lone non-conformant field before the SDK accumulator
            # sees it: a compatible gateway may send message_start.content=null.
            if getattr(event, "type", "") == "message_start":
                message = getattr(event, "message", None)
                if message is not None and getattr(message, "content", None) is None:
                    message.content = []

            snapshot = accumulate_event(event=event, current_snapshot=snapshot)

            # Surface incremental text / thinking to the caller as they stream.
            if getattr(event, "type", "") == "content_block_delta":
                delta = event.delta
                dtype = getattr(delta, "type", "")
                if dtype == "text_delta":
                    yield LLMResponse(
                        success=True, status_code=200,
                        content=delta.text, raw_response=event)
                elif dtype == "thinking_delta":
                    yield LLMResponse(
                        success=True, status_code=200,
                        content="", reasoning_content=delta.thinking,
                        raw_response=event)

        # Parse the SDK-accumulated snapshot with the shared (non-streaming)
        # helpers — tool_use inputs were reassembled from partial_json by the
        # accumulator, so there is nothing bespoke to rebuild here.
        _, tool_calls, _ = self.__parse_content(snapshot.content if snapshot else [])
        usage = self.__extract_usage(snapshot.usage if snapshot else None)

        # Final yield: empty content (text already streamed) carrying aggregated
        # tool calls + usage. ``raw_response`` is the accumulated Message snapshot
        # so __ask_loop_stream can echo the assistant turn verbatim on a tool
        # round (its content blocks preserve thinking signatures and tool_use).
        yield LLMResponse(
            success=True, status_code=200,
            content="", tool_calls=tool_calls,
            reasoning_content="", usage=usage,
            raw_response=snapshot
        )

    def _ask_loop(self,
                  prompt: str,
                  images: Optional[List[PILImage.Image]] = None,
                  callbacks: Optional[List[BaseLLMCallback]] = None,
                  **kwargs: Any) -> LLMResponse:
        """Internal loop for non-streaming interaction.

        Args:
            prompt: User prompt.
            images: Multimodal inputs.
            callbacks: Optional list of callback handlers.
            **kwargs: API arguments.

        Returns:
            Final LLMResponse after all tool calls.
        """
        try:
            user_content = self.build_user_message_content(prompt, images)
            current_messages = self.__prepare_messages(user_content)
            # Resolve the client's configured request shape once, then pass it
            # explicitly into each generate call (parity with OpenAIChat).
            tools_defs = self.__build_tools_schema()
            system = self.instructions or None

            total_usage = TokenUsage()
            final_content = ""
            final_reasoning = ""

            round_idx = 0
            while round_idx < self.max_tool_rounds:
                round_idx += 1
                response = self.generate(messages=current_messages,
                                               system=system,
                                               tools=tools_defs,
                                               thinking=self.thinking,
                                               **kwargs)
                if not response.success:
                    return response

                total_usage.input_tokens += response.usage.input_tokens
                total_usage.output_tokens += response.usage.output_tokens
                total_usage.total_tokens += response.usage.total_tokens
                final_reasoning += response.reasoning_content

                if not response.tool_calls:
                    final_content = response.content
                    break

                # Echo the assistant turn back verbatim (raw content blocks
                # preserve thinking signatures and tool_use blocks).
                current_messages.append({
                    "role": "assistant",
                    "content": response.raw_response.content
                })

                tool_outputs, tool_results = self.tool_set.execute_tool_calls(
                    response.tool_calls, callbacks=callbacks)

                tool_blocks = []
                for tc, output in zip(response.tool_calls, tool_outputs):
                    is_success = tool_results.get(tc.name, True)
                    block: Dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": str(output),
                    }
                    if not is_success:
                        block["is_error"] = True
                    tool_blocks.append(block)

                current_messages.append({
                    "role": "user",
                    "content": tool_blocks
                })
                self.latest_tool_call_result.update(tool_results)

                # Force a final generation turn without tools to summarize if we hit max rounds
                if round_idx == self.max_tool_rounds:
                    final_resp = self.generate(messages=current_messages, system=system, tools=None, thinking=self.thinking, **kwargs)
                    if final_resp.success:
                        total_usage.input_tokens += final_resp.usage.input_tokens
                        total_usage.output_tokens += final_resp.usage.output_tokens
                        total_usage.total_tokens += final_resp.usage.total_tokens
                        final_reasoning += (final_resp.reasoning_content or "")
                        final_content = final_resp.content

            if final_content:
                current_messages.append({
                    "role": "assistant",
                    "content": final_content
                })
            if self.conversation_mode:
                self.messages = current_messages

            return LLMResponse(
                success=True, status_code=200,
                content=final_content, usage=total_usage,
                reasoning_content=final_reasoning
            )

        except Exception as e:
            return self.__handle_ask_exception(e)

    def _ask_loop_stream(self,
                         prompt: str,
                         images: Optional[List[PILImage.Image]] = None,
                         callbacks: Optional[List[BaseLLMCallback]] = None,
                         **kwargs: Any) -> Iterator[LLMResponse]:
        """Internal loop for streaming interaction.

        Args:
            prompt: User prompt.
            images: Multimodal inputs.
            callbacks: Optional list of callback handlers.
            **kwargs: API arguments.

        Yields:
            Incremental chunks and a final summary per round.
        """
        try:
            user_content = self.build_user_message_content(prompt, images)
            current_messages = self.__prepare_messages(user_content)
            # Resolve the client's configured request shape once, then pass it
            # explicitly into each stream call (parity with OpenAIChat).
            tools_defs = self.__build_tools_schema()
            system = self.instructions or None

            total_usage = TokenUsage()
            final_content = ""
            final_reasoning = ""

            round_idx = 0
            while round_idx < self.max_tool_rounds:
                round_idx += 1
                tool_calls = []
                current_round_content = ""
                current_round_reasoning = ""
                final_message = None

                for chunk_resp in self.generate_stream(
                        messages=current_messages, system=system,
                        tools=tools_defs, thinking=self.thinking, **kwargs):

                    if chunk_resp.tool_calls:
                        tool_calls = chunk_resp.tool_calls
                    if chunk_resp.usage.total_tokens > 0:
                        total_usage.input_tokens += chunk_resp.usage.input_tokens
                        total_usage.output_tokens += chunk_resp.usage.output_tokens
                        total_usage.total_tokens += chunk_resp.usage.total_tokens
                    # The final summary chunk carries the SDK-accumulated Message
                    # snapshot in raw_response (delta chunks carry a stream event,
                    # which has no `.content`); detect it by that attribute.
                    if hasattr(chunk_resp.raw_response, "content"):
                        final_message = chunk_resp.raw_response

                    if chunk_resp.content or chunk_resp.reasoning_content:
                        current_round_content += chunk_resp.content
                        current_round_reasoning += chunk_resp.reasoning_content
                        yield chunk_resp

                final_reasoning += current_round_reasoning

                if not tool_calls:
                    final_content = current_round_content
                    break

                # Echo the assistant turn back verbatim from the accumulated
                # snapshot (its content blocks preserve thinking signatures and
                # tool_use blocks the API requires on the next round).
                current_messages.append({
                    "role": "assistant",
                    "content": final_message.content
                })

                tool_outputs, tool_results = self.tool_set.execute_tool_calls(
                    tool_calls, callbacks=callbacks)

                tool_blocks = []
                for tc, output in zip(tool_calls, tool_outputs):
                    is_success = tool_results.get(tc.name, True)
                    block: Dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": str(output),
                    }
                    if not is_success:
                        block["is_error"] = True
                    tool_blocks.append(block)

                current_messages.append({
                    "role": "user",
                    "content": tool_blocks
                })
                self.latest_tool_call_result.update(tool_results)

                # Force a final generation turn without tools to summarize if we hit max rounds
                if round_idx == self.max_tool_rounds:
                    final_chunk = None
                    for chunk_resp in self.generate_stream(messages=current_messages, system=system, tools=None, thinking=self.thinking, **kwargs):
                        if chunk_resp.usage and chunk_resp.usage.total_tokens > 0:
                            total_usage.input_tokens += chunk_resp.usage.input_tokens
                            total_usage.output_tokens += chunk_resp.usage.output_tokens
                            total_usage.total_tokens += chunk_resp.usage.total_tokens
                        if chunk_resp.content or chunk_resp.reasoning_content:
                            final_content += chunk_resp.content
                            final_reasoning += (chunk_resp.reasoning_content or "")
                            yield chunk_resp

            if final_content:
                current_messages.append({
                    "role": "assistant",
                    "content": final_content
                })
            if self.conversation_mode:
                self.messages = current_messages

        except Exception as e:
            yield self.__handle_ask_exception(e)

    def __build_tools_schema(self) -> Optional[List[Dict[str, Any]]]:
        """Builds tool schemas in Anthropic format.

        Returns:
            A list of Anthropic-format tool schemas, or None if no tools are defined.
        """
        if not self.tool_set or not self.tool_set.tools:
            return None
        tools_defs = []
        for t in self.tool_set.tools:
            # We treat SearchTool identically in Anthropic right now, but
            # intercepting here if Anthropic adds a specific "web_search" type
            # block similar to Google/OpenAI in the future.
            schema = t.parameters_schema.copy()
            if "type" not in schema:
                schema["type"] = "object"
            tools_defs.append({
                "name": t.name,
                "description": t.description,
                "input_schema": schema,
            })
        return tools_defs

    def __build_request_args(self, system: Optional[str] = None,
                             tools: Optional[List[Dict[str, Any]]] = None,
                             thinking: Optional[Dict[str, Any]] = None,
                             **kwargs: Any) -> Dict[str, Any]:
        """Assembles the optional/merged keyword arguments for a Messages call.

        ``model`` and ``max_tokens`` are resolved and stamped in by the public
        entry points (``generate`` / ``generate_stream``), so they
        are intentionally absent here. ``system`` / ``tools`` / ``thinking`` are
        included **only when not None**: the SDK forwards an explicit ``None`` as
        ``null`` in the request body (it strips its own ``NOT_GIVEN`` sentinel,
        not ``None``), and some compatible gateways reject a ``null`` here.

        Args:
            system: Top-level system instructions (omitted when None).
            tools: Anthropic-format tool schemas (omitted when None).
            thinking: Extended-thinking config (omitted when None).
            **kwargs: Per-call overrides merged over the client defaults.

        Returns:
            A dict of API arguments (without model/max_tokens, which the caller
            stamps in afterward).
        """
        args: Dict[str, Any] = {}
        if system is not None:
            args["system"] = system
        if tools is not None:
            args["tools"] = tools
        if thinking is not None:
            args["thinking"] = thinking
        args.update(self.anthropic_args_dict)
        args.update(kwargs)
        return args

    def __prepare_messages(
            self, user_content: Union[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Helper to prepare the message list for ask loops.

        Unlike OpenAIChat, the system prompt is never injected here; it is
        passed via the top-level ``system`` API parameter instead.
        """
        messages = self.messages.copy() if self.conversation_mode else []
        messages.append({"role": "user", "content": user_content})
        return messages

    def __parse_content(
            self, content_blocks: List[Any]
    ) -> Tuple[str, List[ToolCall], str]:
        """Parses Anthropic response content blocks.

        Args:
            content_blocks: The ``content`` list from a Message response.

        Returns:
            A tuple of ``(text, tool_calls, reasoning)``, where ``tool_calls``
            is a list of standardized ToolCall instances.
        """
        text = ""
        reasoning = ""
        tool_calls = []
        for block in (content_blocks or []):
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text += block.text
            elif block_type == "thinking":
                reasoning += getattr(block, "thinking", "") or ""
            elif block_type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dumps(block.input or {})
                    )
                )
        return text, tool_calls, reasoning

    def __extract_usage(self, usage: Any) -> TokenUsage:
        """Extracts TokenUsage from an Anthropic usage object.

        Args:
            usage: The usage object on a Message (or None).

        Returns:
            A populated TokenUsage instance. ``total_tokens`` is computed as
            input + output since Anthropic does not return a combined field.
        """
        if not usage:
            return TokenUsage()
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )

    def __handle_ask_exception(self, e: Exception) -> LLMResponse:
        """Helper to handle exceptions in ask loops."""
        LOGGER.error(f"ask failed: {e}")
        status = 400 if isinstance(e, BadRequestError) else 500
        return LLMResponse(success=False, status_code=status, content=f"Unexpected error: {e}")
