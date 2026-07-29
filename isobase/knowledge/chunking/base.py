#! python3
# -*- encoding: utf-8 -*-
"""Abstract base class for text chunking strategies.

@File   :   base.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseChunker(ABC):
    """Abstract chunker interface.

    A chunker splits text into smaller segments for embedding and retrieval.
    Different strategies balance chunk size, semantic coherence, and overlap.

    All concrete chunkers must accept ``chunk_size`` and ``chunk_overlap``
    via keyword arguments so that a ``KnowledgeBase`` can override the
    instance defaults at runtime::

        chunker.chunk(text,
                      chunk_size=kb.chunk_size,
                      chunk_overlap=kb.chunk_overlap)
    """

    @abstractmethod
    def chunk(
        self,
        text: str,
        *,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Splits text into chunks.

        Args:
            text: Input text to split.
            chunk_size: Optional override for the chunker's default size.
            chunk_overlap: Optional override for the chunker's default overlap.
            **kwargs: Additional strategy-specific parameters (e.g.
                ``heading_path`` for semantic chunkers).

        Returns:
            List of text chunks.

        Raises:
            ValueError: If text is invalid or parameters are out of range.
        """
        pass
