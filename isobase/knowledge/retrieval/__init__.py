#! python3
# -*- encoding: utf-8 -*-
"""Retrieval strategies for knowledge base search.

@File   :   __init__.py
@Created:   2026/08/03 19:55 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from .base import BaseRetriever, DenseRetrievalItem
from .dense import DenseRetriever, cosine_similarity
from .pipeline import RetrievalPipeline
from .reranker import BaseReranker, NoOpReranker
from .sparse import SparseRetrievalItem, SparseRetriever, tokenize_text

__all__ = [
    "BaseRetriever",
    "DenseRetrievalItem",
    "DenseRetriever",
    "cosine_similarity",
    "RetrievalPipeline",
    "BaseReranker",
    "NoOpReranker",
    "SparseRetrievalItem",
    "SparseRetriever",
    "tokenize_text",
]
