#! python3
# -*- encoding: utf-8 -*-
"""Sparse text retrieval implementation.

@File   :   sparse.py
@Created:   2026/08/04 23:34 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from ..entities import KnowledgeChunk, KnowledgeDocument, RetrievalResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SparseRetrievalItem:
    """A stored chunk available for sparse text retrieval.

    Args:
        chunk: Indexed chunk to score.
        document: Optional parent document for result provenance.
    """

    chunk: KnowledgeChunk
    document: Optional[KnowledgeDocument] = None


#: Type alias for a pluggable tokenizer — ``(str) -> list[str]``.
Tokenizer = Callable[[str], List[str]]


# ---------------------------------------------------------------------------
# Module-private tokenizer helpers
# ---------------------------------------------------------------------------

#: Compiled regex that matches Unicode word-char runs.  Suitable for English
#: and languages without morphological segmentation requirements.
_WORD_PATTERN: re.Pattern = re.compile(r"\w+", re.UNICODE)


def _default_tokenizer(text: str) -> List[str]:
    """Tokenizes text into lower-cased word tokens."""
    return _WORD_PATTERN.findall(text.lower())


# ---------------------------------------------------------------------------
# Module-private stopword helpers
# ---------------------------------------------------------------------------

_DEFAULT_STOPWORDS: frozenset = frozenset()


#: Filesystem path to the curated stopword list shipped with IsoBase.
_CURATED_STOPWORDS_PATH: str = os.path.join(
    os.path.dirname(__file__), "providers", "stopwords.txt"
)

#: Lazily-initialised cache for :func:`curated_stopwords`.
_CURATED_STOPWORDS_CACHE: Optional[frozenset] = None


# ---------------------------------------------------------------------------
# Public stopword API
# ---------------------------------------------------------------------------

def load_stopwords(path: str) -> frozenset:
    """Loads stopwords from a text file (one word per line).

    Blank lines and lines whose first non-whitespace character is ``#`` are
    treated as comments and skipped.

    Args:
        path: Path to the stopwords file.

    Returns:
        A ``frozenset`` of lower-cased stopwords suitable for passing to
        :class:`SparseRetriever`.
    """
    with Path(path).open(encoding="utf-8") as fh:
        words: Set[str] = set()
        for raw in fh:
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            words.add(stripped.lower())
    return frozenset(words)


def curated_stopwords() -> frozenset:
    """Returns the curated stopword set shipped with IsoBase.

    Loaded lazily on first call; subsequent calls return the cached
    ``frozenset``.

    Returns:
        A ``frozenset`` of lower-cased stopwords covering English and
        Chinese function words.
    """
    global _CURATED_STOPWORDS_CACHE
    if _CURATED_STOPWORDS_CACHE is None:
        _CURATED_STOPWORDS_CACHE = load_stopwords(_CURATED_STOPWORDS_PATH)
    return _CURATED_STOPWORDS_CACHE


# ---------------------------------------------------------------------------
# Sparse retriever
# ---------------------------------------------------------------------------

class SparseRetriever:
    """Ranks chunks by BM25-style sparse text matching.

    The retriever is provider-neutral — it works on any in-memory
    :class:`SparseRetrievalItem` list regardless of the backing store.

    Attributes:
        k1: Term-frequency saturation parameter.
        b: Document-length normalization parameter.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[Set[str]] = None,
        tokenizer: Tokenizer = _default_tokenizer,
    ) -> None:
        """Initialises BM25 parameters and tokenizer.

        Args:
            k1: Term-frequency saturation parameter (default ``1.5``).
            b: Document-length normalisation parameter (default ``0.75``).
            stopwords: Stopwords to exclude during tokenization. When
                ``None`` (the default), no stopword filtering is applied.
            tokenizer: Tokenizer callable ``(str) -> list[str]``.  Defaults
                to a simple Unicode word splitter.  Inject ``jieba.cut`` or a
                custom tokenizer for CJK support.
        """
        self.k1 = k1
        self.b = b
        self.stopwords: Set[str] = stopwords or _DEFAULT_STOPWORDS
        self.tokenizer = tokenizer

    # -- public API ---------------------------------------------------------

    def retrieve(
        self,
        query: str,
        items: List[SparseRetrievalItem],
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """Ranks candidate chunks for a text query.

        Args:
            query: Query text.
            items: Candidate chunks to score.
            top_k: Maximum number of results to return. Defaults to ``5``.

        Returns:
            Retrieval results sorted by descending BM25 score.  Chunks with
            no matching query terms are omitted.  Returns an empty list when
            ``items`` is empty or the query has no content-bearing tokens.
        """
        query_terms = self.__tokenize_and_filter(query)
        if not query_terms or not items:
            return []

        tokenized_chunks = [
            self.__tokenize_and_filter(item.chunk.content) for item in items
        ]
        avg_length = sum(len(tokens) for tokens in tokenized_chunks) / len(items)
        doc_freq: Counter[str] = Counter()
        for tokens in tokenized_chunks:
            doc_freq.update(set(tokens))

        results: List[RetrievalResult] = []
        for item, tokens in zip(items, tokenized_chunks):
            score = self.__score(
                query_terms, tokens, doc_freq, len(items), avg_length
            )
            if score <= 0.0:
                continue
            results.append(
                RetrievalResult(
                    chunk=item.chunk,
                    score=score,
                    document=item.document,
                    score_source="sparse",
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # -- private helpers ----------------------------------------------------

    def __tokenize_and_filter(self, text: str) -> List[str]:
        """Tokenizes ``text`` then removes any configured stopwords."""
        tokens = self.tokenizer(text)
        if not self.stopwords:
            return tokens
        return [t for t in tokens if t not in self.stopwords]

    def __score(
        self,
        query_terms: List[str],
        document_terms: List[str],
        doc_freq: Counter[str],
        document_count: int,
        avg_length: float,
    ) -> float:
        """Computes BM25 score for a single tokenized document.

        Args:
            query_terms: Tokenized query terms.
            document_terms: Tokenized document terms.
            doc_freq: Number of documents containing each term.
            document_count: Total number of documents in the corpus.
            avg_length: Average token length across documents.

        Returns:
            BM25 relevance score.  Returns ``0.0`` when ``document_terms``
            is empty or ``avg_length`` is zero.
        """
        if not document_terms or avg_length == 0.0:
            return 0.0

        term_freq = Counter(document_terms)
        score = 0.0
        for term in query_terms:
            frequency = term_freq.get(term, 0)
            if frequency == 0:
                continue

            # Smooth IDF: log(1 + (N - df + 0.5) / (df + 0.5))
            idf = math.log(
                1.0
                + (document_count - doc_freq[term] + 0.5)
                / (doc_freq[term] + 0.5),
            )
            denominator = frequency + self.k1 * (
                1.0 - self.b + self.b * len(document_terms) / avg_length
            )
            score += idf * frequency * (self.k1 + 1.0) / denominator

        return score


# ---------------------------------------------------------------------------
# Convenience re-export
# ---------------------------------------------------------------------------

def tokenize_text(text: str) -> List[str]:
    """Tokenizes text using the default Unicode word splitter.

    This is a module-level convenience for users who only need tokenization
    without the full BM25 scorer.

    Args:
        text: Text to tokenize.

    Returns:
        Lower-cased word tokens.
    """
    return _default_tokenizer(text)
