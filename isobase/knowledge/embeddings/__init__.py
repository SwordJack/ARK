#! python3
# -*- encoding: utf-8 -*-
"""Text embedding clients for vector representation.

@File   :   __init__.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from .base import BaseEmbeddingClient
from .openai import OpenAIEmbeddingClient

__all__ = [
    "BaseEmbeddingClient",
    "OpenAIEmbeddingClient",
]
