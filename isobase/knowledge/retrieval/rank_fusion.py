#! python3
# -*- encoding: utf-8 -*-
"""Reciprocal Rank Fusion (RRF) for dense + sparse result merging.

@File   :   rank_fusion.py
@Created:   2026/08/05 01:52 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..entities import RetrievalResult


@dataclass
class FusedResult:
    """A single chunk result after RRF fusion.

    Attributes:
        chunk_id: Identifier of the fused chunk.
        score: RRF score (higher is better).
        dense_result: The original dense retrieval result when present.
        sparse_result: The original sparse retrieval result when present.
    """

    chunk_id: str
    score: float
    dense_result: Optional[RetrievalResult] = None
    sparse_result: Optional[RetrievalResult] = None


class RankFusion:
    """Reciprocal Rank Fusion (RRF) combiner.

    Fuses two ranked result lists (dense + sparse) by their rank positions
    rather than raw scores, producing a single consolidated ranking.

    RRF formula: ``score(doc) = sum(1 / (k + rank_i))`` over all lists
    where the document appears.

    Attributes:
        k: Smoothing constant (default ``60``, per the literature).
    """

    def __init__(self, k: int = 60) -> None:
        """Initialises RRF fusion.

        Args:
            k: Smoothing constant that dampens the influence of minor rank
                differences.  Standard value is ``60``.
        """
        self.k = k

    # -- public API ---------------------------------------------------------

    def fuse(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
        top_k: int = 20,
    ) -> List[RetrievalResult]:
        """Fuses dense and sparse result lists via RRF.

        Args:
            dense_results: Dense retrieval results (ordered by score).
            sparse_results: Sparse retrieval results (ordered by score).
            top_k: Maximum number of fused results to return.  Defaults to
                ``20``.

        Returns:
            A single list of ``RetrievalResult`` objects sorted by descending
            RRF score.  The ``score`` field holds the fused RRF value and
            ``score_source`` is set to ``"fused"``.
        """
        # Build rank mappings: chunk_id -> 1-based rank.
        dense_ranks: Dict[str, int] = {
            r.chunk.id: idx + 1 for idx, r in enumerate(dense_results)
        }
        sparse_ranks: Dict[str, int] = {
            r.chunk.id: idx + 1 for idx, r in enumerate(sparse_results)
        }

        # Collect all unique chunk_ids.
        all_chunk_ids: set[str] = set()
        chunk_map: Dict[str, RetrievalResult] = {}
        for r in dense_results:
            all_chunk_ids.add(r.chunk.id)
            chunk_map[r.chunk.id] = r
        for r in sparse_results:
            all_chunk_ids.add(r.chunk.id)
            if r.chunk.id not in chunk_map:
                chunk_map[r.chunk.id] = r

        # Compute RRF scores.
        rrf_scores: Dict[str, float] = {}
        for chunk_id in all_chunk_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1.0 / (self.k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += 1.0 / (self.k + sparse_ranks[chunk_id])
            rrf_scores[chunk_id] = score

        # Sort by descending RRF score.
        sorted_ids = sorted(
            rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
        )[:top_k]

        # Build output — reuse the original RetrievalResult for chunk+document
        # provenance, but replace score and score_source.
        fused: List[RetrievalResult] = []
        for chunk_id in sorted_ids:
            original = chunk_map[chunk_id]
            fused.append(
                RetrievalResult(
                    chunk=original.chunk,
                    score=rrf_scores[chunk_id],
                    document=original.document,
                    score_source="fused",
                )
            )
        return fused
