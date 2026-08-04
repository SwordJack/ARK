#! python3
# -*- encoding: utf-8 -*-
"""Tests for Reciprocal Rank Fusion (RRF).

@File   :   test_rank_fusion.py
@Created:   2026/08/05 01:53 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from isobase.knowledge.entities import KnowledgeChunk, RetrievalResult
from isobase.knowledge.retrieval import RankFusion


def _make_result(chunk_id: str, score: float, source: str = "dense") -> RetrievalResult:
    """Factory for test RetrievalResult objects."""
    chunk = KnowledgeChunk(
        id=chunk_id,
        document_id="d1",
        knowledge_base_id="kb1",
        content=f"content {chunk_id}",
        index=0,
    )
    return RetrievalResult(chunk=chunk, score=score, score_source=source)


def test_rrf_fuse_both_empty():
    """Empty lists produce empty fused output."""
    fusion = RankFusion(k=60)
    result = fusion.fuse([], [])
    assert result == []


def test_rrf_fuse_dense_only():
    """Single-leg fusion preserves dense ordering."""
    fusion = RankFusion(k=60)
    dense = [
        _make_result("a", 0.9),
        _make_result("b", 0.5),
        _make_result("c", 0.3),
    ]
    result = fusion.fuse(dense, [])
    assert len(result) == 3
    assert [r.chunk.id for r in result] == ["a", "b", "c"]
    for r in result:
        assert r.score_source == "fused"


def test_rrf_fuse_sparse_only():
    """Single-leg fusion preserves sparse ordering."""
    fusion = RankFusion(k=60)
    sparse = [
        _make_result("x", 0.8, "sparse"),
        _make_result("y", 0.4, "sparse"),
    ]
    result = fusion.fuse([], sparse)
    assert len(result) == 2
    assert [r.chunk.id for r in result] == ["x", "y"]
    for r in result:
        assert r.score_source == "fused"


def test_rrf_fuse_top_k_trims():
    """top_k limits the fused result count."""
    fusion = RankFusion(k=60)
    dense = [_make_result(f"d{i}", 1.0 - i * 0.1) for i in range(10)]
    sparse = [_make_result(f"s{i}", 1.0 - i * 0.1, "sparse") for i in range(5)]
    result = fusion.fuse(dense, sparse, top_k=5)
    assert len(result) == 5


def test_rrf_fuse_overlap_boosts():
    """A chunk appearing in both lists gets a higher RRF score."""
    fusion = RankFusion(k=60)
    dense = [
        _make_result("c1", 0.9),
        _make_result("c2", 0.5),
        _make_result("c3", 0.3),
    ]
    sparse = [
        _make_result("c2", 0.8, "sparse"),  # c2 overlaps — rank 1 in sparse
        _make_result("c4", 0.4, "sparse"),
    ]

    result = fusion.fuse(dense, sparse)
    # c2: dense rank 2 (1/(60+2)=0.0161) + sparse rank 1 (1/(60+1)=0.0164) = 0.0325
    # c1: dense rank 1 (1/(60+1)=0.0164) + not in sparse = 0.0164
    # So c2 should rank first due to the boost from both lists.
    assert result[0].chunk.id == "c2"


def test_rrf_fuse_preserves_document():
    """Fused results inherit document provenance from the original."""
    from isobase.knowledge.entities import KnowledgeDocument

    doc = KnowledgeDocument(id="d1", knowledge_base_id="kb1", title="Test")

    fusion = RankFusion(k=60)
    chunk = KnowledgeChunk(
        id="c1", document_id="d1", knowledge_base_id="kb1",
        content="test", index=0,
    )
    dense_result = RetrievalResult(
        chunk=chunk, score=0.9, document=doc, score_source="dense",
    )

    result = fusion.fuse([dense_result], [])
    assert len(result) == 1
    assert result[0].document is doc
    assert result[0].document.title == "Test"


def test_rrf_fuse_score_is_rrf_value():
    """Fused score equals the computed RRF formula value."""
    fusion = RankFusion(k=60)
    # Single result at rank 1 → 1/(60+1) ≈ 0.01639
    dense = [_make_result("a", 0.9)]
    result = fusion.fuse(dense, [])
    assert len(result) == 1
    expected = 1.0 / (60.0 + 1.0)
    assert abs(result[0].score - expected) < 1e-6


def test_rrf_fuse_custom_k():
    """Custom k parameter affects RRF scores."""
    fusion_k1 = RankFusion(k=1)
    fusion_k100 = RankFusion(k=100)

    dense = [_make_result("a", 0.9)]
    r1 = fusion_k1.fuse(dense, [])
    r2 = fusion_k100.fuse(dense, [])

    # Score should differ: 1/(1+1)=0.5 vs 1/(100+1)≈0.0099
    assert r1[0].score != r2[0].score
    assert r1[0].score > r2[0].score
