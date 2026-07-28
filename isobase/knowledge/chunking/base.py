#! python3
# -*- encoding: utf-8 -*-
"""Abstract base class for text chunking strategies.

@File   :   base.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    """Abstract chunker interface.

    A chunker splits text into smaller segments for embedding and retrieval.
    Different strategies balance chunk size, semantic coherence, and overlap.
    """

    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[str]:
        """Splits text into chunks.

        Args:
            text: Input text to split.
            **kwargs: Chunker-specific parameters.

        Returns:
            List of text chunks.

        Raises:
            ValueError: If text is invalid or parameters are out of range.
        """
        pass
