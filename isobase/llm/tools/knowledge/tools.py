#! python3
# -*- encoding: utf-8 -*-
"""Knowledge-base tool factory.

@File   :   tools.py
@Created:   2026/08/01 20:42
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from typing import Optional
from isobase.llm.tools.base import FunctionTool
from isobase.knowledge import KnowledgeBaseService


def create_knowledge_search_tool(
    service: KnowledgeBaseService,
    knowledge_base_id: str,
    top_k: int = 5,
    tool_name: str = "search_knowledge_base",
) -> FunctionTool:
    """Creates a knowledge base search tool for LLM integration.

    The returned FunctionTool can be added to a ToolSet and exposed
    to LLMs for automatic knowledge retrieval during conversations.

    Args:
        service: Knowledge base service instance.
        knowledge_base_id: Target knowledge base ID to search.
        top_k: Default number of results to return.
        tool_name: Name of the tool (for LLM tool calling).

    Returns:
        A FunctionTool that can be added to a ToolSet.

    Example usage:
        from isobase.llm.tools.knowledge import create_knowledge_search_tool
        from isobase.llm.tools import ToolSet

        kb_tool = create_knowledge_search_tool(service, kb.id, top_k=3)
        toolset = ToolSet()
        toolset.add_tool(kb_tool)
    """

    def search_knowledge_base(query: str, limit: Optional[int] = None) -> str:
        """Searches indexed private knowledge for relevant context.

        Use this tool when the user asks about uploaded documents, project notes,
        or indexed factual context that may not be in your training data.

        Args:
            query: Concise search query describing the information needed.
            limit: Maximum number of results to return (default: 5).

        Returns:
            Formatted search results with source attribution.
            Returns "No relevant knowledge found" if no results.
        """
        k = limit if limit is not None else top_k
        context = service.retrieve_as_context(query, knowledge_base_id, top_k=k)

        if not context:
            return "No relevant knowledge found for this query."

        return context

    return FunctionTool(
        mapped_callable=search_knowledge_base,
        name=tool_name,
        description=(
            "Search indexed private knowledge. Use when the user asks about "
            "uploaded documents, project notes, or domain-specific information."
        ),
    )
