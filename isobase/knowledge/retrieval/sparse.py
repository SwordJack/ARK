#! python3
# -*- encoding: utf-8 -*-
"""Sparse text retrieval implementation.

@File   :   sparse.py
@Created:   2026/08/04 23:34 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from ..entities import KnowledgeChunk, KnowledgeDocument, RetrievalResult


@dataclass
class SparseRetrievalItem:
    """A stored chunk available for sparse text retrieval.

    Args:
        chunk: Indexed chunk to score.
        document: Optional parent document for result provenance.
    """

    chunk: KnowledgeChunk
    document: Optional[KnowledgeDocument] = None


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize_text(text: str) -> List[str]:
    """Tokenizes text for lightweight sparse retrieval.

    Args:
        text: Text to tokenize.

    Returns:
        Lower-cased word tokens.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class SparseRetriever:
    """Ranks chunks by BM25-style sparse text matching."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initializes BM25 parameters.

        Args:
            k1: Term-frequency saturation parameter.
            b: Document-length normalization parameter.
        """
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query: str,
        items: List[SparseRetrievalItem],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Ranks candidate chunks for a text query.

        Args:
            query: Query text.
            items: Candidate chunks.
            top_k: Maximum number of results to return.

        Returns:
            Retrieval results sorted by descending BM25 score. Chunks with no
            matching query terms are omitted.
        """
        query_terms = tokenize_text(query)
        if not query_terms or not items:
            return []

        tokenized_chunks = [tokenize_text(item.chunk.content) for item in items]
        avg_length = sum(len(tokens) for tokens in tokenized_chunks) / len(items)
        doc_freq = Counter()
        for tokens in tokenized_chunks:
            doc_freq.update(set(tokens))

        results = []
        for item, tokens in zip(items, tokenized_chunks):
            score = self._score(query_terms, tokens, doc_freq, len(items), avg_length)
            if score <= 0.0:
                continue
            results.append(RetrievalResult(
                chunk=item.chunk,
                score=score,
                document=item.document,
                score_source="sparse",
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _score(
        self,
        query_terms: List[str],
        document_terms: List[str],
        doc_freq: Counter,
        document_count: int,
        avg_length: float,
    ) -> float:
        """Computes BM25 score for one tokenized document.

        Args:
            query_terms: Tokenized query terms.
            document_terms: Tokenized document terms.
            doc_freq: Number of documents containing each term.
            document_count: Total number of documents.
            avg_length: Average token length across documents.

        Returns:
            BM25 relevance score.
        """
        if not document_terms or avg_length == 0.0:
            return 0.0

        term_freq = Counter(document_terms)
        score = 0.0
        for term in query_terms:
            frequency = term_freq.get(term, 0)
            if frequency == 0:
                continue

            idf = math.log(1.0 + (document_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * len(document_terms) / avg_length
            )
            score += idf * frequency * (self.k1 + 1.0) / denominator

        return score
