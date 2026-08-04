#! python3
# -*- encoding: utf-8 -*-
"""Tests for markdown-aware chunking.

@File   :   test_markdown_chunking.py
@Created:   2026/08/04 19:29 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from isobase.knowledge.chunking import MarkdownChunker


def test_markdown_chunker_empty_text():
    """Empty text returns no chunks."""
    chunker = MarkdownChunker(chunk_size=100, chunk_overlap=0)
    assert chunker.chunk("") == []


def test_markdown_chunker_heading_path():
    """Chunks include heading path metadata from markdown headings."""
    chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=0)
    text = """# Guide

Intro.

## Install

Run pip.

## Usage

Call API.
"""

    chunks = chunker.chunk(text)

    assert [chunk.metadata["heading_path"] for chunk in chunks] == [
        ["Guide"],
        ["Guide", "Install"],
        ["Guide", "Usage"],
    ]
    assert chunks[0].content.startswith("# Guide")
    assert chunks[1].content.startswith("## Install")
    assert chunks[2].content.startswith("## Usage")
    assert all(chunk.metadata["chunk_strategy"] == "markdown" for chunk in chunks)


def test_markdown_chunker_preamble_has_empty_heading_path():
    """Text before the first heading is preserved with an empty heading path."""
    chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=0)
    chunks = chunker.chunk("Preamble.\n\n# Title\n\nBody.")

    assert chunks[0].content == "Preamble."
    assert chunks[0].metadata["heading_path"] == []
    assert chunks[1].metadata["heading_path"] == ["Title"]


def test_markdown_chunker_normalizes_input():
    """Markdown input is normalized before chunking."""
    chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=0)
    chunks = chunker.chunk("##Heading\r\n\r\n* item  \r\n")

    assert chunks[0].content == "## Heading\n\n- item"
    assert chunks[0].metadata["heading_path"] == ["Heading"]


def test_markdown_chunker_splits_large_section_with_overlap():
    """Large markdown sections fall back to fixed-size splitting."""
    chunker = MarkdownChunker(chunk_size=20, chunk_overlap=5)
    chunks = chunker.chunk("# Title\n\n" + "A" * 50)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 20 for chunk in chunks)
    assert all(chunk.metadata["heading_path"] == ["Title"] for chunk in chunks)
    assert chunks[0].content[-5:] == chunks[1].content[:5]


def test_markdown_chunker_ignores_headings_inside_fenced_code_blocks():
    """Headings inside fenced code blocks stay in the section body."""
    chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=0)
    text = """# Guide

```python
# not a heading
print('hello')
```

After.
"""

    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0].metadata["heading_path"] == ["Guide"]
    assert "# not a heading" in chunks[0].content


def test_markdown_chunker_override_size():
    """chunk_size can be overridden at call time."""
    chunker = MarkdownChunker(chunk_size=1000, chunk_overlap=0)
    chunks = chunker.chunk("# Title\n\n0123456789abcdef", chunk_size=10)

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 10 for chunk in chunks)
