#! python3
# -*- encoding: utf-8 -*-
"""Tests for RetrievalOption and RetrievalPipeline.

@File   :   test_pipeline.py
@Created:   2026/08/04 21:50 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from isobase.knowledge.entities import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalOption,
)
from isobase.knowledge.retrieval import RetrievalPipeline
from isobase.knowledge.stores import MemoryKnowledgeStore


def _seed_store():
    """Creates a store with one KB and 5 chunks."""
    store = MemoryKnowledgeStore()
    kb = KnowledgeBase(id="kb1", name="Test KB")
    store.create_knowledge_base(kb)
    doc = KnowledgeDocument(id="d1", knowledge_base_id="kb1", title="Doc")
    store.add_document(doc)

    # Embedding vectors: [val, 0.0] — score is proportional to val.
    chunks = []
    embeddings = []
    for i, val in enumerate([10.0, 8.0, 6.0, 4.0, 2.0]):
        chunk = KnowledgeChunk(
            id=f"c{i}",
            document_id="d1",
            knowledge_base_id="kb1",
            content=f"chunk {i}",
            index=i,
            metadata={"key": f"val{i}"},
        )
        chunks.append(chunk)
        embeddings.append([val, 0.0])

    store.add_chunks(chunks, embeddings)
    return store


# ---- RetrievalOption -----------------------------------------------------

def test_retrieval_options_defaults():
    """Default options: top_k=5, candidate_k=None, no metadata filter."""
    option = RetrievalOption()
    assert option.top_k == 5
    assert option.effective_candidate_k() == 5
    assert option.metadata_filter == {}


def test_retrieval_options_top_k_trumps():
    """candidate_k defaults to top_k when not set."""
    option = RetrievalOption(top_k=3)
    assert option.effective_candidate_k() == 3


def test_effective_candidate_k_max_rule():
    """candidate_k cannot be less than top_k."""
    option = RetrievalOption(top_k=3, candidate_k=2)
    assert option.effective_candidate_k() == 3


def test_effective_candidate_k_larger_than_top_k():
    """candidate_k can be larger than top_k."""
    option = RetrievalOption(top_k=3, candidate_k=10)
    assert option.effective_candidate_k() == 10


# ---- RetrievalPipeline.search --------------------------------------------

def test_pipeline_default_behavior_matches_store():
    """Pipeline without options behaves like store.search (dense top_k)."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)
    query_embedding = [10.0, 0.0]  # aligned with c0

    results = pipeline.search("kb1", query_embedding, RetrievalOption(top_k=3))

    assert len(results) == 3
    # First result is closest to query
    assert results[0].chunk.id == "c0"
    assert results[0].score_source == "dense"
    # Sorted by descending score
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_pipeline_empty_kb():
    """Pipeline returns empty list for unknown KB."""
    store = MemoryKnowledgeStore()
    pipeline = RetrievalPipeline(store)
    results = pipeline.search(
        "nonexistent", [10.0, 0.0], RetrievalOption(top_k=5)
    )
    assert results == []


def test_pipeline_candidate_k_vs_final_k():
    """candidate_k controls candidates; result is trimmed to top_k."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)
    query_embedding = [10.0, 0.0]

    # candidate_k=5, top_k=2 — final list has exactly 2 items
    results = pipeline.search(
        "kb1",
        query_embedding,
        RetrievalOption(top_k=2, candidate_k=5),
    )
    assert len(results) == 2
    assert results[0].chunk.id == "c0"
    assert results[1].chunk.id == "c1"


def test_pipeline_metadata_filter_matches():
    """Metadata filter keeps only chunks matching the filter."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)
    query_embedding = [10.0, 0.0]

    results = pipeline.search(
        "kb1",
        query_embedding,
        RetrievalOption(top_k=5, metadata_filter={"key": "val0"}),
    )
    assert len(results) == 1
    assert results[0].chunk.id == "c0"


def test_pipeline_metadata_filter_no_match_returns_empty():
    """Metadata filter with no matching chunks returns empty."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)
    query_embedding = [10.0, 0.0]

    results = pipeline.search(
        "kb1",
        query_embedding,
        RetrievalOption(top_k=5, metadata_filter={"key": "nonexistent"}),
    )
    assert results == []
