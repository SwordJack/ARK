#! python3
# -*- encoding: utf-8 -*-
"""Markdown-aware text chunking.

@File   :   markdown.py
@Created:   2026/08/04 19:27 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import re
from typing import Any, List, Optional

from isobase.utils.markdown_fixer import MarkdownFixer

from .base import BaseChunker, ChunkSection
from .fixed import FixedSizeChunker


class MarkdownChunker(BaseChunker):
    """Markdown structure-aware chunker.

    Splits markdown by ATX headings first, then falls back to fixed-size
    character splitting for sections that exceed the configured chunk size.

    Attributes:
        chunk_size: Default maximum characters per chunk.
        chunk_overlap: Default number of characters to overlap between chunks.
    """

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initializes the markdown chunker.

        Args:
            chunk_size: Default maximum characters per chunk.
            chunk_overlap: Default number of characters to overlap between chunks.

        Raises:
            ValueError: If chunk_overlap >= chunk_size, chunk_size <= 0, or
                chunk_overlap < 0.
        """
        self._fixed_chunker = FixedSizeChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
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
        """Splits markdown text into heading-aware chunks.

        Args:
            text: Input markdown text to split.
            chunk_size: Optional override for the default chunk size.
            chunk_overlap: Optional override for the default overlap.
            **kwargs: Ignored (for API compatibility).

        Returns:
            List of chunk sections with ``heading_path`` metadata.
        """
        if not text:
            return []

        normalized = MarkdownFixer.normalize(text)
        sections = self._split_sections(normalized)
        chunks: List[ChunkSection] = []

        for section in sections:
            content = section.content.strip()
            if not content:
                continue

            fixed_chunks = self._fixed_chunker.chunk(
                content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for fixed_chunk in fixed_chunks:
                metadata = {
                    "chunk_strategy": "markdown",
                    "heading_path": section.metadata["heading_path"],
                }
                chunks.append(
                    ChunkSection(
                        content=fixed_chunk.content,
                        metadata=metadata,
                    )
                )

        return chunks

    @classmethod
    def _split_sections(cls, text: str) -> List[ChunkSection]:
        """Splits normalized markdown into sections by ATX headings.

        Args:
            text: Normalized markdown text.

        Returns:
            List of sections with their current heading path.
        """
        sections: List[ChunkSection] = []
        heading_path: List[str] = []
        current_lines: List[str] = []
        current_path: List[str] = []

        for line in text.splitlines():
            match = cls._HEADING_RE.match(line)
            if match:
                cls._append_section(sections, current_lines, current_path)
                level = len(match.group(1))
                heading = match.group(2).strip()
                heading_path = heading_path[:level - 1]
                heading_path.append(heading)
                current_path = heading_path.copy()
                current_lines = [line]
                continue

            current_lines.append(line)

        cls._append_section(sections, current_lines, current_path)
        return sections

    @staticmethod
    def _append_section(
        sections: List[ChunkSection],
        lines: List[str],
        heading_path: List[str],
    ) -> None:
        """Appends a section when it has non-blank content.

        Args:
            sections: Mutable section list to append to.
            lines: Lines in the current section.
            heading_path: Current heading path for this section.
        """
        content = "\n".join(lines).strip()
        if not content:
            return
        sections.append(
            ChunkSection(
                content=content,
                metadata={"heading_path": heading_path.copy()},
            )
        )
