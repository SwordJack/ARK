#! python3
# -*- encoding: utf-8 -*-
"""Text chunking strategies for document splitting.

@File   :   __init__.py
@Created:   2026/07/29 00:20
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from .base import BaseChunker
from .fixed import FixedSizeChunker

__all__ = [
    "BaseChunker",
    "FixedSizeChunker",
]
