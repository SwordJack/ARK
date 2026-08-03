#! python3
# -*- encoding: utf-8 -*-
"""Tests for chunking strategies.

@File   :   test_chunking.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest
from isobase.knowledge.chunking import FixedSizeChunker


def test_fixed_chunker_no_overlap():
    """Test fixed-size chunker without overlap."""
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
    text = "0123456789abcdefghij"
    chunks = chunker.chunk(text)

    assert len(chunks) == 2
    assert chunks[0] == "0123456789"
    assert chunks[1] == "abcdefghij"


def test_fixed_chunker_with_overlap():
    """Test fixed-size chunker with overlap."""
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
    text = "0123456789abcdefghij"
    chunks = chunker.chunk(text)

    assert len(chunks) == 3
    assert chunks[0] == "0123456789"
    assert chunks[1] == "789abcdefg"  # Overlaps with previous
    assert chunks[2] == "efghij"


def test_fixed_chunker_empty_text():
    """Test chunker with empty text."""
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
    chunks = chunker.chunk("")

    assert chunks == []


def test_fixed_chunker_text_shorter_than_chunk_size():
    """Test chunker when text is shorter than chunk size."""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    text = "short text"
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_fixed_chunker_exact_chunk_size():
    """Test chunker when text is exactly chunk size."""
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
    text = "0123456789"
    chunks = chunker.chunk(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_fixed_chunker_invalid_parameters():
    """Test chunker initialization with invalid parameters."""
    # overlap >= chunk_size
    with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
        FixedSizeChunker(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
        FixedSizeChunker(chunk_size=10, chunk_overlap=15)

    # negative chunk_size
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedSizeChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        FixedSizeChunker(chunk_size=-5, chunk_overlap=0)

    # negative overlap
    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        FixedSizeChunker(chunk_size=10, chunk_overlap=-1)


def test_fixed_chunker_long_text():
    """Test chunker with long text."""
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    text = "A" * 200
    chunks = chunker.chunk(text)

    # Expected chunks: ceil(200 / (50 - 10)) = ceil(200 / 40) = 5
    # Actually: 1st: 0-50, 2nd: 40-90, 3rd: 80-130, 4th: 120-170, 5th: 160-200
    assert len(chunks) == 5
    assert all(len(chunk) <= 50 for chunk in chunks)

    # Check overlap
    for i in range(len(chunks) - 1):
        overlap_start = chunks[i][-10:]
        overlap_end = chunks[i + 1][:10]
        assert overlap_start == overlap_end


def test_fixed_chunker_realistic_text():
    """Test chunker with realistic text content."""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
    text = (
        "RAG stands for Retrieval-Augmented Generation. "
        "It is a technique that combines information retrieval with text generation. "
        "RAG systems first retrieve relevant documents from a knowledge base."
    )

    chunks = chunker.chunk(text)

    assert len(chunks) >= 2
    assert all(len(chunk) <= 100 for chunk in chunks)

    # Verify all text is covered
    reconstructed = chunks[0]
    for i in range(1, len(chunks)):
        # Remove overlapping part
        reconstructed += chunks[i][20:]

    assert len(reconstructed) >= len(text)


def test_fixed_chunker_override_size():
    """chunk_size can be overridden at call time."""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
    chunks = chunker.chunk("0123456789abcdef", chunk_size=5)
    assert chunks == ["01234", "56789", "abcde", "f"]


def test_fixed_chunker_override_overlap():
    """chunk_overlap can be overridden at call time."""
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
    chunks = chunker.chunk(
        "0123456789abcdefghij", chunk_size=10, chunk_overlap=3
    )
    assert len(chunks) == 3
    assert chunks[1] == "789abcdefg"


def test_fixed_chunker_override_validation():
    """Override values are still validated."""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunker.chunk("text", chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
        chunker.chunk("text", chunk_overlap=-1)

    with pytest.raises(ValueError, match="chunk_overlap.*must be less than"):
        chunker.chunk("text", chunk_size=10, chunk_overlap=10)


def test_fixed_chunker_overrides_do_not_mutate_instance():
    """Override call does not change the chunker's own defaults."""
    chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=10)
    chunker.chunk("text", chunk_size=50, chunk_overlap=5)
    assert chunker.chunk_size == 100
    assert chunker.chunk_overlap == 10


def test_fixed_chunker_accepts_kwargs():
    """Extra keyword args are accepted (future-proofing for other strategies)."""
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
    chunks = chunker.chunk(
        "0123456789", heading_path=["Ch1", "1.1"], separator="\n"
    )
    assert chunks == ["0123456789"]
