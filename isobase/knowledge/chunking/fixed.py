#! python3
# -*- encoding: utf-8 -*-
"""Fixed-size text chunking with overlap.

@File   :   fixed.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import Any, List, Optional

from .base import BaseChunker, ChunkSection


class FixedSizeChunker(BaseChunker):
    """Fixed-size chunker with overlap.

    Splits text into fixed-length chunks with configurable overlap.
    Simple character-based splitting that does not respect sentence
    or word boundaries.

    Attributes:
        chunk_size: Default maximum characters per chunk.
        chunk_overlap: Default number of characters to overlap between chunks.

    Example:
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk("Long text...")
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initializes the fixed-size chunker.

        Args:
            chunk_size: Default maximum characters per chunk.
            chunk_overlap: Default number of characters to overlap between chunks.

        Raises:
            ValueError: If chunk_overlap >= chunk_size, chunk_size <= 0, or
                chunk_overlap < 0.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap cannot be negative, got {chunk_overlap}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs: Any,
    ) -> List[ChunkSection]:
        """Splits text into fixed-size chunks with overlap.

        Args:
            text: Input text to split.
            chunk_size: Optional override for the default chunk size.
            chunk_overlap: Optional override for the default overlap.
            **kwargs: Ignored (for API compatibility with chunkers that
                accept additional parameters).

        Returns:
            List of chunk sections. Empty list if text is empty.

        Example:
            >>> chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
            >>> chunker.chunk("0123456789abcdefghij")
            [ChunkSection(content='0123456789', metadata={'chunk_strategy': 'fixed'}), ...]
        """
        if not text:
            return []

        size = chunk_size if chunk_size is not None else self.chunk_size
        overlap = chunk_overlap if chunk_overlap is not None else self.chunk_overlap

        # Validate overridden values
        if size <= 0:
            raise ValueError(f"chunk_size must be positive, got {size}")
        if overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {overlap}")
        if overlap >= size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be less than chunk_size ({size})"
            )

        chunks: List[ChunkSection] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + size
            chunks.append(
                ChunkSection(
                    content=text[start:end],
                    metadata={"chunk_strategy": "fixed"},
                )
            )

            if end >= text_len:
                break

            start = end - overlap

        return chunks
