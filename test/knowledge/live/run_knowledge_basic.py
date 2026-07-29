#! python3
# -*- encoding: utf-8 -*-
"""Manual live smoke test for the knowledge base module against real APIs.

This script is intentionally **not** collected by pytest (filename does not
start with ``test_``) and never runs in CI.  It reads credentials from a
git-ignored ``.env`` file sitting next to this file.

Usage:
    1. Copy the template and fill in real values (the real .env is git-ignored):
         cp test/knowledge/live/.env.example test/knowledge/live/.env
       Then edit test/knowledge/live/.env.
    2. Run from the repo root (so the ``isobase`` package is importable):
         python -m test.knowledge.live.run_knowledge_basic [openai|dashscope|aimlapi]

Each provider section: creates an embedded KB, indexes several documents, then
runs retrieval queries.  Leave a provider's API_KEY blank/absent to skip it.

Expected .env variables (see .env.example):
    OPENAI_EMBEDDING_BASE_URL / OPENAI_EMBEDDING_API_KEY / OPENAI_EMBEDDING_MODEL / OPENAI_EMBEDDING_DIMENSIONS (optional)
    DASHSCOPE_EMBEDDING_BASE_URL / DASHSCOPE_EMBEDDING_API_KEY / DASHSCOPE_EMBEDDING_MODEL / DASHSCOPE_EMBEDDING_DIMENSIONS (optional)
    AIMLAPI_EMBEDDING_BASE_URL / AIMLAPI_EMBEDDING_API_KEY / AIMLAPI_EMBEDDING_MODEL / AIMLAPI_EMBEDDING_DIMENSIONS (optional)

@File   :   run_knowledge_basic.py
@Created:   2026/07/29 00:28
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import sys
from os import path
from typing import Any, Dict, Optional

from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.stores import MemoryKnowledgeStore

ENV_PATH = path.join(path.dirname(__file__), ".env")


def _resolve_dimensions(client: OpenAIEmbeddingClient) -> Optional[int]:
    """Return dimension count via explicit setting or a probe request."""
    try:
        return client.dimensions
    except RuntimeError:
        pass
    try:
        return client.fetch_dimensions()
    except Exception:
        return None


def _load_env() -> Dict[str, str]:
    """Parse the git-ignored .env file into a dict, or exit with guidance."""
    if not path.exists(ENV_PATH):
        sys.exit(
            f"Missing {ENV_PATH}.\n"
            "Copy the template and fill in real values:\n"
            f"  cp {ENV_PATH}.example {ENV_PATH}"
        )

    env: Dict[str, str] = {}
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                env[key] = value
    return env


def _client_kwargs(env: Dict[str, str], prefix: str) -> Optional[Dict[str, Any]]:
    """Build kwargs for OpenAIEmbeddingClient from {prefix}_* env vars.

    Returns None when the API key is missing or still a placeholder.
    """
    api_key = env.get(f"{prefix}_API_KEY", "").strip()
    if not api_key or api_key.endswith("XXXXXX"):
        return None

    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": env.get(f"{prefix}_BASE_URL", "https://api.openai.com/v1"),
        "model": env.get(f"{prefix}_MODEL", "text-embedding-3-small"),
    }

    dims = env.get(f"{prefix}_DIMENSIONS", "").strip()
    if dims:
        kwargs["dimensions"] = int(dims)

    return kwargs


# ---------------------------------------------------------------------------
# Test documents
# ---------------------------------------------------------------------------

_DOCUMENTS = [
    {
        "title": "RAG Overview",
        "source_uri": "docs/rag.md",
        "text": (
            "RAG stands for Retrieval-Augmented Generation. It is a technique "
            "that combines information retrieval with text generation. RAG systems "
            "first retrieve relevant documents from a knowledge base, then use "
            "those documents as context for generating responses. This approach "
            "helps LLMs provide more accurate and up-to-date information."
        ),
        "metadata": {"author": "Alice", "date": "2024-01-15"},
    },
    {
        "title": "Vector Databases",
        "source_uri": "docs/vector-db.md",
        "text": (
            "Vector databases are specialized storage systems designed for "
            "high-dimensional embeddings. They enable fast similarity search using "
            "techniques like approximate nearest neighbor (ANN) search. Popular "
            "vector databases include FAISS, Pinecone, Qdrant, and Milvus. These "
            "systems are essential for RAG applications."
        ),
        "metadata": {"author": "Bob", "date": "2024-01-16"},
    },
    {
        "title": "Text Embeddings",
        "source_uri": "docs/embeddings.md",
        "text": (
            "Embeddings are dense vector representations of text that capture "
            "semantic meaning. They are created by neural networks trained to place "
            "semantically similar texts close together in vector space. Common "
            "embedding models include OpenAI's text-embedding-3, Alibaba's Qwen "
            "embedding models, and open-source models like sentence-transformers."
        ),
        "metadata": {"author": "Alice", "date": "2024-01-17"},
    },
]

_QUERIES = [
    "What is RAG?",
    "Tell me about vector databases",
    "How do embeddings work?",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_provider(label: str, kwargs: Dict[str, Any]) -> None:
    """Index documents and run retrieval queries using one embedding provider."""
    print(f"\n{'='*60}")
    print(f"  Provider: {label}")
    print(f"{'='*60}")

    client = OpenAIEmbeddingClient(**kwargs)
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=30)
    store = MemoryKnowledgeStore()

    service = KnowledgeBaseService(
        embedding_client=client,
        store=store,
        chunker=chunker,
    )

    # Create knowledge base
    try:
        dims = client.dimensions
    except RuntimeError:
        dims = client.fetch_dimensions()
    kb = service.create_knowledge_base(
        name=f"AI Docs ({label})",
        description="E2E smoke-test knowledge base",
    )
    print(f"  KB created: {kb.name} ({kb.id[:8]}…)  [dim={dims}]")

    # Index
    for doc in _DOCUMENTS:
        indexed = service.index_text(
            knowledge_base_id=kb.id,
            text=doc["text"],
            title=doc["title"],
            source_uri=doc["source_uri"],
            metadata=doc["metadata"],
        )
        print(f"  Indexed: {indexed.title}")

    # Retrieve
    for query in _QUERIES:
        results = service.retrieve(query=query, knowledge_base_id=kb.id, top_k=3)
        print(f"\n  Query: \"{query}\"")
        if not results:
            print("    → No results")
            continue
        for i, r in enumerate(results, 1):
            src = r.document.title if r.document else "?"
            print(f"    [{i}] score={r.score:.4f}  src={src}  "
                  f"text={r.chunk.content[:80]!r}")

    # Formatted context
    context = service.retrieve_as_context(
        query="Explain RAG systems", knowledge_base_id=kb.id, top_k=2
    )
    print(f"\n  Formatted context length: {len(context)} chars")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not path.exists(ENV_PATH):
        sys.exit(
            f"Missing {ENV_PATH}.\n"
            "Copy the template and fill in real values:\n"
            f"  cp {ENV_PATH}.example {ENV_PATH}"
        )

    env = _load_env()

    providers = [
        ("OpenAI", "OPENAI_EMBEDDING"),
        ("DashScope", "DASHSCOPE_EMBEDDING"),
        ("AIMLAPI", "AIMLAPI_EMBEDDING"),
    ]

    # Optional command-line filter: "openai", "dashscope", or "aimlapi"
    if len(sys.argv) > 1:
        wanted = sys.argv[1].lower()
        providers = [
            (label, prefix)
            for label, prefix in providers
            if label.lower().startswith(wanted)
        ]
        if not providers:
            sys.exit(f"Unknown provider '{wanted}'. "
                     f"Valid values: openai, dashscope, aimlapi")

    any_run = False
    for label, prefix in providers:
        kwargs = _client_kwargs(env, prefix)
        if kwargs is None:
            print(f"[SKIP] {label} — no API key configured")
            continue
        run_provider(label, kwargs)
        any_run = True

    if not any_run:
        print(
            "No provider configured. "
            f"Edit {ENV_PATH} to set at least one API key."
        )


if __name__ == "__main__":
    main()
