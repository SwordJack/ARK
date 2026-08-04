#! python3
# -*- encoding: utf-8 -*-
"""Tests for RetrievalOption, RetrievalPipeline, reranker hooks, and sparse retrieval.

@File   :   test_pipeline.py
@Created:   2026/08/04 21:50 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest

from isobase.knowledge.entities import (
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeDocument,
    RetrievalOption,
)
from isobase.knowledge.retrieval import BaseReranker, NoOpReranker, RetrievalPipeline
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


# ---- Reranker ------------------------------------------------------------

def test_noop_reranker_preserves_order():
    """NoOpReranker preserves ordering and trims to top_k."""
    from isobase.knowledge.entities import RetrievalResult

    c0 = KnowledgeChunk(id="c0", document_id="d1", knowledge_base_id="kb1", content="a", index=0)
    c1 = KnowledgeChunk(id="c1", document_id="d1", knowledge_base_id="kb1", content="b", index=1)
    results = [
        RetrievalResult(chunk=c0, score=0.9, score_source="dense"),
        RetrievalResult(chunk=c1, score=0.3, score_source="dense"),
    ]
    reranker = NoOpReranker()
    out = reranker.rerank(None, results, 2)
    assert len(out) == 2
    assert out[0].chunk.id == "c0"
    assert out[1].chunk.id == "c1"


def test_noop_reranker_trims():
    """NoOpReranker trims to top_k."""
    from isobase.knowledge.entities import RetrievalResult

    chunks = [
        KnowledgeChunk(id=f"c{i}", document_id="d1", knowledge_base_id="kb1", content=str(i), index=i)
        for i in range(5)
    ]
    results = [
        RetrievalResult(chunk=c, score=1.0 - i * 0.1, score_source="dense")
        for i, c in enumerate(chunks)
    ]
    reranker = NoOpReranker()
    out = reranker.rerank(None, results, 3)
    assert len(out) == 3
    assert [r.chunk.id for r in out] == ["c0", "c1", "c2"]


def test_pipeline_with_default_reranker_is_noop():
    """Default pipeline uses NoOpReranker — behavior unchanged."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)  # no explicit reranker
    query_embedding = [10.0, 0.0]

    results = pipeline.search("kb1", query_embedding, RetrievalOption(top_k=3))
    assert len(results) == 3
    assert results[0].chunk.id == "c0"
    assert results[0].score_source == "dense"


def test_pipeline_with_custom_reranker():
    """A custom reranker can reorder and its output is final-trimmed."""
    store = _seed_store()
    query_embedding = [10.0, 0.0]

    class _ReverseReranker(BaseReranker):
        def rerank(self, query, results, top_k):
            return list(reversed(results[:top_k]))

    pipeline = RetrievalPipeline(store, reranker=_ReverseReranker())
    results = pipeline.search("kb1", query_embedding, RetrievalOption(top_k=3))

    assert len(results) == 3
    # Dense order: c0 (score 1.0), c1 (0.8), c2 (0.6)
    # Reverse: c2, c1, c0
    assert [r.chunk.id for r in results] == ["c2", "c1", "c0"]


# ---- Sparse retrieval -----------------------------------------------------

def test_pipeline_sparse_basic():
    """Pipeline with use_sparse=True returns sparse results for text query."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)
    query_embedding = [10.0, 0.0]  # ignored when sparse

    results = pipeline.search(
        "kb1",
        query_embedding,
        RetrievalOption(top_k=3, use_sparse=True),
        query_text="chunk 0",
    )
    assert len(results) >= 1
    assert results[0].score_source == "sparse"


def test_pipeline_sparse_no_query_text_raises():
    """Sparse mode requires query_text."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)

    with pytest.raises(ValueError, match="query_text is required"):
        pipeline.search(
            "kb1",
            [10.0, 0.0],
            RetrievalOption(use_sparse=True),
        )


def test_pipeline_sparse_metadata_filter():
    """Metadata filter works with sparse retrieval."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)

    results = pipeline.search(
        "kb1",
        [10.0, 0.0],
        RetrievalOption(top_k=5, use_sparse=True, metadata_filter={"key": "val0"}),
        query_text="chunk",
    )
    assert len(results) == 1
    assert results[0].chunk.id == "c0"
    assert results[0].score_source == "sparse"


def test_pipeline_sparse_empty_kb():
    """Sparse retrieval on empty KB returns empty list."""
    store = MemoryKnowledgeStore()
    pipeline = RetrievalPipeline(store)

    results = pipeline.search(
        "nonexistent",
        [10.0, 0.0],
        RetrievalOption(use_sparse=True),
        query_text="hello",
    )
    assert results == []


def test_pipeline_default_dense_unchanged():
    """Default pipeline without use_sparse keeps dense behavior."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)
    query_embedding = [10.0, 0.0]

    results = pipeline.search("kb1", query_embedding, RetrievalOption(top_k=3))
    assert len(results) == 3
    assert results[0].chunk.id == "c0"
    assert results[0].score_source == "dense"


# ---- Hybrid retrieval (RRF fusion) ----------------------------------------

def test_pipeline_hybrid_basic():
    """Hybrid retrieval fuses dense + sparse via RRF."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)
    query_embedding = [10.0, 0.0]  # dense: c0 top

    results = pipeline.search(
        "kb1",
        query_embedding,
        RetrievalOption(top_k=3, use_hybrid=True),
        query_text="chunk 1",
    )
    assert len(results) >= 1
    assert results[0].score_source == "fused"
    # Scores are RRF values — all > 0
    for r in results:
        assert r.score > 0.0


def test_pipeline_hybrid_no_query_text_raises():
    """Hybrid mode requires query_text."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)

    with pytest.raises(ValueError, match="query_text is required"):
        pipeline.search(
            "kb1",
            [10.0, 0.0],
            RetrievalOption(use_hybrid=True),
        )


def test_pipeline_hybrid_empty_kb():
    """Hybrid retrieval on empty KB returns empty list."""
    store = MemoryKnowledgeStore()
    pipeline = RetrievalPipeline(store)

    results = pipeline.search(
        "nonexistent",
        [10.0, 0.0],
        RetrievalOption(use_hybrid=True),
        query_text="hello",
    )
    assert results == []


def test_pipeline_hybrid_metadata_filter():
    """Metadata filter works with hybrid retrieval."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)

    results = pipeline.search(
        "kb1",
        [10.0, 0.0],
        RetrievalOption(
            top_k=5, use_hybrid=True, metadata_filter={"key": "val0"}
        ),
        query_text="chunk",
    )
    assert len(results) == 1
    assert results[0].chunk.id == "c0"
    assert results[0].score_source == "fused"


def test_pipeline_hybrid_consistency():
    """A chunk in both dense and sparse legs gets a higher RRF score."""
    store = _seed_store()
    pipeline = RetrievalPipeline(store)
    # Query embedding aligns with c0 (val=10.0) — dense rank 1.
    # Query text "chunk 0" matches c0 content — sparse rank 1.
    # c0 should appear in both legs and therefore gets the best RRF score.
    query_embedding = [10.0, 0.0]

    results = pipeline.search(
        "kb1",
        query_embedding,
        RetrievalOption(top_k=5, use_hybrid=True),
        query_text="chunk 0",
    )
    assert len(results) >= 1
    assert results[0].chunk.id == "c0"
    assert results[0].score_source == "fused"
