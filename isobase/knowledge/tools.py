#! python3
# -*- encoding: utf-8 -*-
"""LLM tool integration for knowledge base search.

@File   :   tools.py
@Created:   2026/07/29 00:16
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from typing import Optional
from isobase.llm.tools.base import FunctionTool
from .service import KnowledgeBaseService


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
        # Create tool
        kb_tool = create_knowledge_search_tool(
            service=service,
            knowledge_base_id=kb.id,
            top_k=3
        )

        # Add to toolset
        from isobase.llm.tools.base import ToolSet
        toolset = ToolSet()
        toolset.add_tool(kb_tool)

        # Use with LLM
        from isobase.llm.providers.openai_chat import OpenAIChat
        llm = OpenAIChat(api_key="...")
        response = llm.generate(
            messages=[{"role": "user", "content": "What is RAG?"}],
            tools=toolset.to_openai_schema()
        )

        # Execute tool calls if any
        if response.tool_calls:
            results, _ = toolset.execute_tool_calls(response.tool_calls)
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

        Example queries:
            - "RAG architecture overview"
            - "authentication implementation details"
            - "API rate limits for embedding endpoints"
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
