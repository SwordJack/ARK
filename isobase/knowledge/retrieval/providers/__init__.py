#! python3
# -*- encoding: utf-8 -*-
"""Provider-specific retrieval implementations.

@File   :   __init__.py
@Created:   2026/08/04 22:43 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from .openai_reranker import OpenAIReranker

__all__ = [
    "OpenAIReranker",
]
