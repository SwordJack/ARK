#! python3
# -*- encoding: utf-8 -*-
"""OpenAI Chat Completion client with structured responses and multi-turn tool support.

@File   :   openai_chat.py
@Created:   2026/06/05 00:34
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from inspect import signature
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union, overload

from openai import (
    BadRequestError,
    OpenAI,
    Stream,
)
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionChunk, ChatCompletion
from PIL import Image as PILImage

from isobase.core.image_service import convert_image_to_data_url
from isobase.core.logger import LOGGER
from .base import BaseLLMClient
from ..callbacks import BaseLLMCallback
from ..entities import LLMMessage, LLMMessageHistory, LLMResponse, MessageContentBlock, TokenUsage, ToolCall
from ..tools import FunctionTool, SearchTool, ToolSet


class OpenAIChat(BaseLLMClient):
    """Optimized client for OpenAI Chat Completion compatible APIs.

    Supports structured LLMResponse, safe history mutation, and multi-turn
    tool calling via internal orchestration loops.

    Attributes:
        client: The underlying OpenAI client instance.
        default_model: The default model ID to use for requests.
        instructions: System instructions defining the agent's behavior.
        conversation_mode: Whether to maintain a message history.
        max_tool_rounds: Maximum number of recursive tool-calling rounds.
        messages: The current conversation history.
        tool_set: A collection of available tools.
        latest_tool_call_result: Results of the last tool executions.
        openai_args_dict: Filtered kwargs for the OpenAI SDK.
    """

    def __init__(self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-5.4-mini",
        instructions: str = "",
        messages: List[Dict[str, str]] = None,
        tools: Union[ToolSet, List[FunctionTool], None] = None,
        conversation_mode: bool = True,
        max_tool_rounds: int = 5,
        **kwargs: Any):
        """Initializes an OpenAIChat client.

        Args:
            api_key: The API key for authentication.
            base_url: The base URL for the API endpoint.
            default_model: The default model ID.
            instructions: System instructions for the model.
            messages: Initial list of conversation messages.
            tools: An optional ToolSet or list of FunctionTool objects.
            conversation_mode: If True, history is preserved across asks.
            max_tool_rounds: Limits recursive tool calls per ask.
            **kwargs: Additional arguments for the OpenAI client or API.
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        self.instructions = instructions
        self.conversation_mode = conversation_mode
        self.max_tool_rounds = max_tool_rounds

        self.messages = messages or []
        if not self.messages and self.instructions:
            self.messages = [{"role": "system", "content": self.instructions}]

        if isinstance(tools, ToolSet):
            self.tool_set = tools
        else:
            self.tool_set = ToolSet(tools or [])
        self.latest_tool_call_result = {}

        create_params = signature(self.client.chat.completions.create).parameters
        self.openai_args_dict = {
            k: v for k, v in kwargs.items() if k in create_params
        }

        LOGGER.info(f"OpenAIChat initialized (model: {default_model})")

    # --- neutral / native message conversion ---------------------------------

    @classmethod
    def from_neutral_history(cls, history: LLMMessageHistory) -> Tuple[List[Dict[str, Any]], str]:
        """Converts a neutral message history into OpenAI-native messages.

        OpenAI keeps ``system`` as a regular message, so the default
        extraction of the leading system prompt is applied simply to match
        the common ``(messages, system_prompt)`` return shape.
        """
        messages: List[Dict[str, Any]] = []
        system_prompt = ""
        for m in history.messages:
            if m.role == "system" and not system_prompt:
                system_prompt = m.content if isinstance(m.content, str) else ""
                messages.append({"role": "system", "content": system_prompt})
                continue
            messages.append(cls.from_neutral_message(m))
        return messages, system_prompt

    @classmethod
    def from_neutral_message(cls, message: LLMMessage) -> Dict[str, Any]:
        """Converts a provider-neutral message into OpenAI-native dict format.

        Follows the same shape the provider's ``generate`` / ``generate_stream``
        already accept so that existing callers are unaffected.
        """
        if message.role == "system":
            return {"role": "system", "content": message.content if isinstance(message.content, str) else ""}

        if message.role == "user":
            if isinstance(message.content, str):
                return {"role": "user", "content": message.content}
            # multimodal or tool-result user message
            content_blocks: List[Dict[str, Any]] = []
            for block in message.content:
                if block.type == "text":
                    content_blocks.append({"type": "text", "text": block.text})
                elif block.type == "image":
                    if block.source:
                        content_blocks.append(block.source)
                elif block.type == "tool_result":
                    item: Dict[str, Any] = {
                        "role": "tool",
                        "tool_call_id": block.tool_call_id,
                        "content": block.text,
                    }
                    if block.name:
                        item["name"] = block.name
                    return item  # tool_result in OpenAI is a standalone role:"tool" message
                elif block.type == "raw" and block.raw is not None:
                    content_blocks.append(block.raw)
            return {"role": "user", "content": content_blocks}

        if message.role == "assistant":
            if isinstance(message.content, str):
                return {"role": "assistant", "content": message.content}
            native: Dict[str, Any] = {"role": "assistant"}
            for block in message.content:
                if block.type == "text":
                    native["content"] = block.text
                elif block.type == "tool_use" and block.tool_call is not None:
                    native.setdefault("tool_calls", []).append({
                        "id": block.tool_call.id,
                        "type": "function",
                        "function": {
                            "name": block.tool_call.name,
                            "arguments": block.tool_call.arguments,
                        },
                    })
                elif block.type == "raw" and block.raw is not None:
                    native["content"] = block.raw
            if "content" not in native:
                native["content"] = None
            return native

        # role == "tool"
        if isinstance(message.content, str):
            return {"role": "tool", "content": message.content}
        tool_id, tool_name, tool_content = "", "", ""
        for block in message.content:
            if block.type == "tool_result":
                tool_id = block.tool_call_id
                tool_name = block.name
                tool_content = block.text
        native_tool: Dict[str, Any] = {"role": "tool", "content": tool_content}
        if tool_id:
            native_tool["tool_call_id"] = tool_id
        if tool_name:
            native_tool["name"] = tool_name
        return native_tool

    @classmethod
    def to_neutral_message(cls, native_message: Dict[str, Any]) -> LLMMessage:
        """Converts an OpenAI-native message dict into a neutral LLMMessage.

        Handles the three extra shapes that can appear in history after
        multi-turn tool loops:

        - system messages that use a standalone ``role:"system"`` dict
        - assistant messages carrying ``tool_calls`` alongside ``content``
        - ``role:"tool"`` messages for tool results
        """
        role = native_message.get("role", "user")
        content = native_message.get("content", "")

        if role == "system":
            return LLMMessage(role="system", content=content if isinstance(content, str) else "")

        if role == "user":
            if isinstance(content, str):
                return LLMMessage(role="user", content=content)
            blocks = []
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type == "text":
                            blocks.append(MessageContentBlock(type="text", text=item.get("text", "")))
                        elif item_type == "image_url":
                            blocks.append(MessageContentBlock(type="image", source=item))
                        else:
                            blocks.append(MessageContentBlock(type="raw", raw=item))
                    else:
                        blocks.append(MessageContentBlock(type="text", text=str(item)))
            return LLMMessage(role="user", content=blocks)

        if role == "assistant":
            tool_calls_raw = native_message.get("tool_calls") or []
            if not tool_calls_raw:
                return LLMMessage(role="assistant", content=content or "")

            blocks: List[MessageContentBlock] = []
            if content:
                blocks.append(MessageContentBlock(type="text", text=content))
            for tc in tool_calls_raw:
                fn = tc.get("function", {})
                blocks.append(MessageContentBlock(
                    type="tool_use",
                    tool_call=ToolCall(
                        id=tc.get("id", ""),
                        name=fn.get("name", ""),
                        arguments=fn.get("arguments", ""),
                    ),
                ))
            return LLMMessage(role="assistant", content=blocks)

        if role == "tool":
            return LLMMessage(role="tool", content=[
                MessageContentBlock(
                    type="tool_result",
                    tool_call_id=native_message.get("tool_call_id", ""),
                    name=native_message.get("name", ""),
                    text=content if isinstance(content, str) else str(content),
                )
            ])

        return LLMMessage(role="user", content=content if isinstance(content, str) else str(content))

    @classmethod
    def _build_user_message_content(cls, prompt: str,
        images: Optional[List["PILImage.Image"]] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """Builds a neutral image block for base class compatibility."""
        if not images:
            return prompt
        blocks = [{"type": "text", "text": prompt}]
        for img in images:
            if not isinstance(img, PILImage.Image):
                raise TypeError("Each image must be a PIL.Image.Image instance.")
            blocks.append({
                "type": "image_url",
                "image_url": {"url": convert_image_to_data_url(img)},
            })
        return blocks

    def generate(self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            **kwargs: Any
        ) -> LLMResponse:
        """Sends a single non-streaming chat completion request.

        Args:
            messages: Conversation context.
            model: Specific model ID to use.
            **kwargs: Extra parameters for the API.

        Returns:
            An LLMResponse containing the model's output.
        """
        try:
            target_model = model or self.default_model
            args = {**self.openai_args_dict, **kwargs}
            completion: ChatCompletion = self.client.chat.completions.create(
                model=target_model, messages=messages, stream=False, **args)
            content, tool_calls, reasoning, usage = self.__parse_completion(
                completion)
            return LLMResponse(
                success=True, status_code=200,
                content=content, usage=usage,
                tool_calls=tool_calls, reasoning_content=reasoning,
                raw_response=completion
            )
        except Exception as e:
            return LLMResponse(success=False, status_code=500, content=f"Error: {e}")

    def generate_stream(
            self, messages: List[Dict[str, str]],
            model: Optional[str] = None, **kwargs: Any
        ) -> Iterator[LLMResponse]:
        """Sends a streaming chat completion request.

        Args:
            messages: Conversation context.
            model: Specific model ID to use.
            **kwargs: Extra parameters for the API.

        Yields:
            LLMResponse objects containing chunks or final results.
        """
        target_model = model or self.default_model
        args = {**self.openai_args_dict, **kwargs}

        stream: Stream[ChatCompletionChunk] = self.client.chat.completions.create(
            model=target_model, messages=messages,
            stream=True, stream_options={"include_usage": True},
            **args
        )

        content = ""
        tool_calls = []
        reasoning = ""
        usage = TokenUsage()

        for chunk in stream:
            chunk_usage = self.__extract_usage(chunk.usage) if chunk.usage else None
            if chunk_usage:
                usage = chunk_usage

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            delta_content = delta.content or ""
            delta_reasoning = (delta.reasoning_content or "") if hasattr(delta, "reasoning_content") else ""

            if delta_content:
                content += delta_content
            if delta_reasoning:
                reasoning += delta_reasoning

            # Handle tool calls in stream.
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    while len(tool_calls) <= tc_chunk.index:
                        tool_calls.append({
                            "id": "",
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        })
                    tc = tool_calls[tc_chunk.index]
                    if tc_chunk.id:
                        tc["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name:
                            tc["function"]["name"] = tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            tc["function"]["arguments"] += tc_chunk.function.arguments

            # Yield incremental text chunks.
            if delta_content or delta_reasoning:
                yield LLMResponse(
                    success=True, status_code=200,
                    content=delta_content, reasoning_content=delta_reasoning,
                    raw_response=chunk
                )

        # Final yield with full aggregated data for logic/tools, but empty content
        # to avoid duplication in the ask loop.
        final_tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"]["arguments"]
            ) for tc in tool_calls
        ]

        yield LLMResponse(
            success=True, status_code=200,
            content="", tool_calls=final_tool_calls,
            reasoning_content="",
            usage=usage
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
            tools_defs = self.__build_tools_schema()

            total_usage = TokenUsage()
            final_content = ""
            final_reasoning = ""

            round_idx = 0
            while round_idx < self.max_tool_rounds:
                round_idx += 1
                response = self.generate(messages=current_messages,
                                               tools=tools_defs,
                                               **kwargs)
                if not response.success:
                    return response

                total_usage.input_tokens += response.usage.input_tokens
                total_usage.output_tokens += response.usage.output_tokens
                total_usage.total_tokens += response.usage.total_tokens
                final_reasoning += (response.reasoning_content or "")

                if not response.tool_calls:
                    final_content = response.content
                    break

                # Handle Tool Calls.
                assistant_msg = self.__build_assistant_message(
                    content=response.content,
                    tool_calls=response.raw_response.choices[0].message.tool_calls
                )
                current_messages.append(assistant_msg)

                tool_outputs, tool_results = self.tool_set.execute_tool_calls(
                    response.tool_calls, callbacks=callbacks)

                tool_messages = []
                for tc, output in zip(response.tool_calls, tool_outputs):
                    tool_messages.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "name": tc.name,
                        "content": str(output)
                    })
                current_messages.extend(tool_messages)
                self.latest_tool_call_result.update(tool_results)

                # Force a final generation turn without tools to summarize if we hit max rounds
                if round_idx == self.max_tool_rounds:
                    final_resp = self.generate(messages=current_messages, tools=None, **kwargs)
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
            Incremental chunks and final summary.
        """
        try:
            user_content = self.build_user_message_content(prompt, images)
            current_messages = self.__prepare_messages(user_content)
            tools_defs = self.__build_tools_schema()

            total_usage = TokenUsage()
            final_content = ""
            final_reasoning = ""

            round_idx = 0
            while round_idx < self.max_tool_rounds:
                round_idx += 1
                tool_calls = []
                current_round_content = ""
                current_round_reasoning = ""

                # Iterate through chunks from the streaming completion.
                for chunk_resp in self.generate_stream(
                        messages=current_messages, tools=tools_defs, **kwargs):
                    
                    if chunk_resp.tool_calls:
                        tool_calls = chunk_resp.tool_calls
                    if chunk_resp.usage:
                        total_usage.input_tokens += chunk_resp.usage.input_tokens
                        total_usage.output_tokens += chunk_resp.usage.output_tokens
                        total_usage.total_tokens += chunk_resp.usage.total_tokens
                    
                    # chunk_resp.content is incremental in chunks, empty in final.
                    if chunk_resp.content or chunk_resp.reasoning_content or chunk_resp.usage.total_tokens > 0:
                        current_round_content += chunk_resp.content
                        current_round_reasoning += chunk_resp.reasoning_content
                        # Forward the increment or final summary to the user.
                        yield chunk_resp

                final_reasoning += current_round_reasoning

                if not tool_calls:
                    final_content = current_round_content
                    break

                # Handle Tool Calls.
                assistant_msg = self.__build_assistant_message(
                    content=current_round_content,
                    tool_calls=tool_calls
                )
                current_messages.append(assistant_msg)

                tool_outputs, tool_results = self.tool_set.execute_tool_calls(
                    tool_calls, callbacks=callbacks)

                tool_messages = []
                for tc, output in zip(tool_calls, tool_outputs):
                    tool_messages.append({
                        "tool_call_id": tc.id,
                        "role": "tool",
                        "name": tc.name,
                        "content": str(output)
                    })
                current_messages.extend(tool_messages)
                self.latest_tool_call_result.update(tool_results)

                # Force a final generation turn without tools to summarize if we hit max rounds
                if round_idx == self.max_tool_rounds:
                    final_chunk = None
                    for chunk_resp in self.generate_stream(messages=current_messages, tools=None, **kwargs):
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

    def __extract_usage(self, usage: Optional[CompletionUsage]) -> TokenUsage:
        """Extracts TokenUsage from an OpenAI usage object.

        Args:
            usage (Optional[CompletionUsage]): The usage object returned by the OpenAI SDK.

        Returns:
            A populated TokenUsage instance.
        """
        if not usage:
            return TokenUsage()
        else:
            return TokenUsage(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens
            )

    def __parse_completion(
            self, completion: ChatCompletion
        ) -> Tuple[str, List[ToolCall], str, TokenUsage]:
        """Parses a standard ChatCompletion instance.

        Args:
            completion (ChatCompletion): The ChatCompletion instance from the OpenAI SDK.

        Returns:
            A tuple of (content, tool_calls, reasoning, usage).
        """
        choice = completion.choices[0]
        message = choice.message
        content = message.content or ""
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc.type == "function":
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=tc.function.arguments
                        )
                    )
        reasoning = message.reasoning_content if hasattr(message, "reasoning_content") else ""
        usage = self.__extract_usage(completion.usage)
        return content, tool_calls, reasoning, usage

    def __build_tools_schema(self) -> Optional[List[Dict[str, Any]]]:
        """Builds tool schemas in OpenAI function format.

        Returns:
            A list of OpenAI-format tool schemas, or None if no tools are defined.
        """
        if not self.tool_set or not self.tool_set.tools:
            return None

        schemas = []
        for tool in self.tool_set.tools:
            # Avoid direct import to prevent circular dependency if they're in same tree.
            if isinstance(tool, SearchTool) and not tool.force_external:
                schemas.append({"type": "web_search"})
            else:
                schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_schema,
                    }
                })
        return schemas

    def __prepare_messages(self, user_content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        """Helper to prepare the message list for ask loops."""
        messages = self.messages.copy() if self.conversation_mode else []
        if not messages and self.instructions:
            messages.append({"role": "system", "content": self.instructions})
        messages.append({"role": "user", "content": user_content})
        return messages

    def __build_assistant_message(
        self,
        content: Optional[str],
        tool_calls: List[Union[Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Constructs an assistant message containing content and tool calls.

        This method structures the assistant's turn into the exact format required
        by OpenAI's API (and strict enterprise proxies) so it can be appended to
        the conversation history for multi-turn routing.

        Due to the differences in how streaming and non-streaming responses are
        processed, this method acts as a universal compatibility layer that handles
        three distinct types of tool call representations:

        1. `dict`: Raw dictionaries typically generated during mid-stream accumulation.
        2. `ToolCall`: IsoBase's provider-neutral dataclass. Yielded at the end of
           `generate_stream` (in `_ask_loop_stream`). Since it is provider-neutral,
           it lacks OpenAI-specific fields like `type`, so we manually inject `"type": "function"`.
        3. `ChoiceMessageToolCall`: Native OpenAI SDK objects returned in non-streaming
           (`_ask_loop`). It uses `getattr` for safe access and `model_dump()` to
           preserve built-in native tools (like `web_search`) verbatim.

        Args:
            content: Text content from the assistant (or None).
            tool_calls: List of tool call objects (SDK or ToolCall) or raw dictionaries.

        Returns:
            The structured assistant message dictionary.
        """
        assistant_msg: Dict[str, Any] = {"role": "assistant", "tool_calls": []}

        for tc in tool_calls:
            # Case 1: Raw dictionary (e.g., from mid-stream accumulation)
            if isinstance(tc, dict):
                tc_type = tc.get("type", "function")
                if tc_type == "function":
                    assistant_msg["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    })
                else:
                    # Pass through built-in tools that were streamed as dicts
                    assistant_msg["tool_calls"].append(tc)

            # Case 2: IsoBase neutral dataclass (e.g., from generate_stream final yield)
            # Lacks the '.type' attribute, so we format it explicitly as a standard function.
            elif isinstance(tc, ToolCall):
                assistant_msg["tool_calls"].append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                    }
                })

            # Case 3: Native OpenAI SDK Object (e.g., from non-streaming completions)
            else:
                if getattr(tc, "type", "") == "function":
                    assistant_msg["tool_calls"].append({
                        "id": getattr(tc, "id", ""),
                        "type": "function",
                        "function": {
                            "name": getattr(tc.function, "name", ""),
                            "arguments": getattr(tc.function, "arguments", "")
                        }
                    })
                else:
                    # Pass through built-in tools (like web_search) verbatim using model_dump
                    if hasattr(tc, "model_dump"):
                        assistant_msg["tool_calls"].append(tc.model_dump())
                    else:
                        assistant_msg["tool_calls"].append(tc)

        if content:
            assistant_msg["content"] = content
        else:
            assistant_msg["content"] = None

        return assistant_msg

    def __handle_ask_exception(self, e: Exception) -> LLMResponse:
        """Helper to handle exceptions in ask loops."""
        LOGGER.error(f"ask failed: {e}")
        status = 400 if isinstance(e, BadRequestError) else 500
        return LLMResponse(success=False, status_code=status, content=f"Unexpected error: {e}")
