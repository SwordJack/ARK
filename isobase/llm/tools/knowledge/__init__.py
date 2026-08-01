#! python3
# -*- encoding: utf-8 -*-
"""Knowledge-base tool integration.

Thin wrappers that turn ``isobase.knowledge.Service`` calls into
``FunctionTool`` instances compatible with ``ToolSet``.

@File   :   __init__.py
@Created:   2026/08/01 20:40
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from .tools import create_knowledge_search_tool

__all__ = [
    "create_knowledge_search_tool",
]
