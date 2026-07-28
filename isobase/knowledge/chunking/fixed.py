#! python3
# -*- encoding: utf-8 -*-
"""Fixed-size text chunking with overlap.

@File   :   fixed.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import List
from .base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunker with overlap.

    Splits text into fixed-length chunks with configurable overlap.
    Simple character-based splitting that does not respect sentence
    or word boundaries.

    Attributes:
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters to overlap between chunks.

    Example:
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("Long text...")
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initializes the fixed-size chunker.

        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Number of characters to overlap between chunks.

        Raises:
            ValueError: If parameters are invalid.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, **kwargs) -> List[str]:
        """Splits text into fixed-size chunks with overlap.

        Args:
            text: Input text to split.
            **kwargs: Ignored (for API compatibility).

        Returns:
            List of text chunks. Empty list if text is empty.

        Example:
            >>> chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
            >>> chunker.chunk("0123456789abcdefghij")
            ['0123456789', '789abcdefg', 'efghij']
        """
        if not text:
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            if end >= text_len:
                break

            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap

        return chunks
