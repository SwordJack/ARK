#! python3
# -*- encoding: utf-8 -*-
"""Abstract base class for LLM providers.

@File   :   base.py
@Created:   2026/06/05 00:34
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union, overload

from PIL import Image as PILImage

from isobase.core.image_service import convert_image_to_data_url
from ..entities import LLMMessage, LLMMessageHistory, LLMResponse
from ..callbacks import BaseLLMCallback


class BaseLLMClient(ABC):
    """Abstract base class for all LLM provider clients.

    Ensures a consistent interface for interacting with various LLM APIs.
    """

    @abstractmethod
    def generate(self,
                 messages: List[Dict[str, str]],
                 model: Optional[str] = None,
                 **kwargs: Any) -> LLMResponse:
        """Sends a non-streaming generation request.

        Args:
            messages: A list of message objects representing the conversation.
            model: The specific model ID to use for this request.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An LLMResponse object containing the structured result.
        """
        pass

    @abstractmethod
    def generate_stream(self,
                        messages: List[Dict[str, str]],
                        model: Optional[str] = None,
                        **kwargs: Any) -> Iterator[LLMResponse]:
        """Sends a streaming generation request.

        Args:
            messages: A list of message objects representing the conversation.
            model: The specific model ID to use for this request.
            **kwargs: Additional provider-specific parameters.

        Yields:
            LLMResponse objects containing incremental content chunks.
        """
        pass

    @overload
    def ask(self,
            prompt: str,
            images: Optional[List[PILImage.Image]] = None,
            stream: Literal[False] = False,
            callbacks: Optional[List[BaseLLMCallback]] = None,
            **kwargs: Any) -> LLMResponse:
        ...

    @overload
    def ask(self,
            prompt: str,
            images: Optional[List[PILImage.Image]] = None,
            stream: Literal[True] = True,
            callbacks: Optional[List[BaseLLMCallback]] = None,
            **kwargs: Any) -> Iterator[LLMResponse]:
        ...

    def ask(self,
            prompt: str,
            images: Optional[List[PILImage.Image]] = None,
            stream: bool = False,
            callbacks: Optional[List[BaseLLMCallback]] = None,
            **kwargs: Any) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """Orchestrates a chat interaction, handling history and tool calls.

        Args:
            prompt: The user's input text.
            images: Optional list of images for multimodal input.
            stream: Whether to use streaming for the interaction.
            callbacks: Optional list of callback handlers.
            **kwargs: Additional provider-specific parameters.

        Returns:
            An LLMResponse (non-stream) or Iterator[LLMResponse] (stream).
        """
        if stream:
            return self._ask_loop_stream(prompt, images, callbacks=callbacks, **kwargs)
        else:
            return self._ask_loop(prompt, images, callbacks=callbacks, **kwargs)

    @classmethod
    @abstractmethod
    def from_neutral_message(cls, message: LLMMessage) -> Dict[str, Any]:
        """Converts a provider-neutral message into provider-native message format.

        Provider-native format is the dict shape accepted by this provider's
        ``generate`` / ``generate_stream`` ``messages`` parameter.

        Args:
            message: A provider-neutral LLMMessage.

        Returns:
            A dict representing one message in this provider's native shape.
        """
        pass

    @classmethod
    @abstractmethod
    def from_neutral_history(cls, history: LLMMessageHistory) -> Tuple[List[Dict[str, Any]], str]:
        """Converts a neutral message history into this provider's native messages.

        Subclasses must implement this method to handle provider-specific
        history-level transformations — for example, merging consecutive
        tool-result messages or extracting ``system``-role messages into a
        separate return value.

        Args:
            history: A provider-neutral ``LLMMessageHistory``.

        Returns:
            A tuple of ``(messages, system_prompt)``.
        """
        pass

    @classmethod
    @abstractmethod
    def to_neutral_message(cls, native_message: Dict[str, Any]) -> LLMMessage:
        """Converts a provider-native message dict into a neutral LLMMessage.

        Args:
            native_message: One provider-native message dict.

        Returns:
            An equivalent LLMMessage in the provider-neutral shape.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @classmethod
    def build_user_message_content(
        cls, prompt: str,
        images: Optional[List[PILImage.Image]] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """Builds user message content, supporting multimodal input.

        Standardizes the input into an OpenAI-compatible format if images
        are provided.

        Args:
            prompt: The text prompt provided by the user.
            images: An optional list of PIL Image objects to attach.

        Returns:
            A string if only text is provided, or a list of content dictionaries
            for multimodal requests.

        Raises:
            TypeError: If any item in the images list is not a PIL Image.
        """
        if not images:
            return prompt

        content = [{"type": "text", "text": prompt}]
        for img in images:
            if not isinstance(img, PILImage.Image):
                raise TypeError("Each image must be a PIL.Image.Image instance.")
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": convert_image_to_data_url(img)
                },
            })
        return content
