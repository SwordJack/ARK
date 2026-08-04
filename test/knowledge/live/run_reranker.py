#! python3
# -*- encoding: utf-8 -*-
"""Live smoke test for the OpenAIReranker against the Alibaba DashScope rerank API.

Compares dense-only retrieval with reranked retrieval side-by-side on
the same documents / queries, so you can judge whether the rerank model
actually improves ranking quality.

Usage
-----
1. Copy the template and fill in real values (the real .env is git-ignored)::

       cp test/knowledge/live/.env.example test/knowledge/live/.env

   Then edit ``test/knowledge/live/.env`` — ensure both
   ``DASHSCOPE_EMBEDDING_*`` **and** ``DASHSCOPE_RERANK_*`` are set.

2. Run from the repo root::

       python -m test.knowledge.live.run_reranker

@File   :   run_reranker.py
@Created:   2026/08/04 23:10 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.entities import RetrievalOption, RetrievalResult
from isobase.knowledge.retrieval import RetrievalPipeline
from isobase.knowledge.retrieval.providers import OpenAIReranker
from isobase.knowledge.stores import MemoryKnowledgeStore

HERE = os.path.dirname(__file__)
ENV_PATH = os.path.join(HERE, ".env")

# ---------------------------------------------------------------------------
# Test documents — semantic coverage matters for rerank evaluation
# ---------------------------------------------------------------------------

_DOCUMENTS = [
    {
        "title": "Python AsyncIO Guide",
        "source_uri": "docs/async.md",
        "text": (
            "Python's asyncio module provides infrastructure for writing "
            "single-threaded concurrent code using coroutines. The async/await "
            "syntax, introduced in Python 3.5, enables non-blocking I/O without "
            "callbacks. Key primitives include Task, Future, and Event Loop. "
            "asyncio.run() is the recommended entry point since Python 3.7."
        ),
    },
    {
        "title": "Introduction to Threads",
        "source_uri": "docs/threads.md",
        "text": (
            "Threads are the smallest unit of execution within a process. "
            "Python's threading module provides a high-level interface for "
            "creating and managing threads. However, due to the Global "
            "Interpreter Lock (GIL), CPU-bound Python threads do not achieve "
            "true parallelism on CPython."
        ),
    },
    {
        "title": "Django Web Framework",
        "source_uri": "docs/django.md",
        "text": (
            "Django is a high-level Python web framework that encourages rapid "
            "development and clean, pragmatic design. It follows the model-template-"
            "views (MTV) pattern and includes an ORM, admin interface, and "
            "authentication system out of the box."
        ),
    },
    {
        "title": "Concurrency Patterns in Go",
        "source_uri": "docs/go-concurrency.md",
        "text": (
            "Go's concurrency model is built around goroutines and channels. "
            "Goroutines are lightweight threads managed by the Go runtime. "
            "Channels provide a way for goroutines to communicate with each "
            "other and synchronize execution. The select statement lets a "
            "goroutine wait on multiple communication operations."
        ),
    },
    {
        "title": "FastAPI — Modern Python Web Framework",
        "source_uri": "docs/fastapi.md",
        "text": (
            "FastAPI is a modern, high-performance web framework for building "
            "APIs with Python 3.7+ based on standard Python type hints. It is "
            "built on top of Starlette for the web parts and Pydantic for the "
            "data parts. FastAPI supports async endpoints natively using asyncio."
        ),
    },
]

_QUERIES = [
    "How does async work in Python?",
    "What is the best Python web framework?",
    "Explain goroutines and channels",
    "What is the GIL in Python?",
]


# ---------------------------------------------------------------------------
# .env helpers (same conventions as run_knowledge_basic.py)
# ---------------------------------------------------------------------------


def _load_env() -> Dict[str, str]:
    """Parse the git-ignored .env file into a dict, or exit with guidance."""
    if not os.path.exists(ENV_PATH):
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


def _embedding_kwargs(
    env: Dict[str, str], prefix: str = "DASHSCOPE_EMBEDDING"
) -> Optional[Dict[str, Any]]:
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


def _reranker_kwargs(
    env: Dict[str, str], prefix: str = "DASHSCOPE_RERANK"
) -> Optional[Dict[str, Any]]:
    """Build kwargs for OpenAIReranker from {prefix}_* env vars."""
    api_key = env.get(f"{prefix}_API_KEY", "").strip()
    if not api_key or api_key.endswith("XXXXXX"):
        return None

    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": env.get(f"{prefix}_BASE_URL", ""),
        "model": env.get(f"{prefix}_MODEL", "qwen3-rerank"),
    }

    endpoint = env.get(f"{prefix}_ENDPOINT", "").strip()
    if endpoint:
        kwargs["endpoint"] = endpoint

    return kwargs


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _h1(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}")


def _h2(text: str) -> None:
    print(f"\n--- {text} ---")


def _print_results(label: str, results: List[RetrievalResult]) -> None:
    """Pretty-print a result list."""
    print(f"\n  [{label}]  ({len(results)} result(s))")
    print(f"  {'-' * 50}")
    if not results:
        print("    (empty)")
        return
    for i, r in enumerate(results, 1):
        src = r.document.title if r.document else "?"
        preview = r.chunk.content[:90].replace("\n", "\\n")
        print(
            f"    [{i}] score={r.score:.4f}  src={src}  "
            f"source={r.score_source}\n"
            f"        {preview}..."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    env = _load_env()

    # -- Embedding client ---------------------------------------------------
    emb_kwargs = _embedding_kwargs(env)
    if emb_kwargs is None:
        sys.exit(
            "DashScope embedding not configured. "
            f"Edit {ENV_PATH} to set DASHSCOPE_EMBEDDING_API_KEY."
        )
    embedding_client = OpenAIEmbeddingClient(**emb_kwargs)
    dims = embedding_client.fetch_dimensions()
    print(f"  Embedding: {embedding_client.model}  dim={dims}")

    # -- Index documents ----------------------------------------------------
    chunker = FixedSizeChunker(chunk_size=300, chunk_overlap=50)
    store = MemoryKnowledgeStore()
    service = KnowledgeBaseService(
        embedding_client=embedding_client,
        store=store,
        chunker=chunker,
    )

    kb = service.create_knowledge_base(
        name="Rerank Smoke Test",
        description="Documents for comparing dense-only vs reranked retrieval.",
    )
    print(f"  KB: [{kb.id[-8:]}] {kb.name!r}")

    for doc in _DOCUMENTS:
        indexed = service.index_text(
            knowledge_base_id=kb.id,
            text=doc["text"],
            title=doc["title"],
            source_uri=doc["source_uri"],
        )
        print(f"  Indexed: {indexed.title}")

    # -- Reranker -----------------------------------------------------------
    rerank_kwargs = _reranker_kwargs(env)
    if rerank_kwargs is None:
        print(
            "\n  ⚠  DashScope rerank not configured — "
            f"set DASHSCOPE_RERANK_API_KEY in {ENV_PATH}"
        )
        print("  Running dense-only comparison baseline only.\n")
        reranker = None
    else:
        reranker = OpenAIReranker(**rerank_kwargs)
        print(f"\n  Reranker: {reranker.model}  endpoint={reranker.endpoint}")

    # -- Compare dense-only vs reranked -------------------------------------
    pipeline_noop = RetrievalPipeline(store)  # default NoOpReranker

    for query in _QUERIES:
        _h2(f'Query: "{query}"')

        # Dense-only
        dense_results = pipeline_noop.search(
            kb.id,
            embedding_client.embed_query(query),
            RetrievalOption(top_k=5),
            query_text=query,
        )

        if reranker is None:
            _print_results("dense-only (baseline)", dense_results)
            continue

        # Reranked
        pipeline_rerank = RetrievalPipeline(store, reranker=reranker)
        reranked_results = pipeline_rerank.search(
            kb.id,
            embedding_client.embed_query(query),
            RetrievalOption(top_k=5),
            query_text=query,
        )

        # Side-by-side
        _print_results("dense-only", dense_results)
        _print_results("reranked", reranked_results)

        # Diff summary
        dense_ids = [r.chunk.document_id for r in dense_results]
        rerank_ids = [r.chunk.document_id for r in reranked_results]
        if dense_ids != rerank_ids:
            print(
                "\n  ⚡ Ordering changed! "
                f"dense={[r[-6:] for r in dense_ids]}  "
                f"reranked={[r[-6:] for r in rerank_ids]}"
            )
        else:
            print("  → Ordering unchanged (coincidental)")

    # -- Formatted context (reranked) ---------------------------------------
    if reranker is not None:
        _h2("Formatted context (via service, reranker as default pipeline)")

        # Re-create service with reranker wired in
        service2 = KnowledgeBaseService(
            embedding_client=embedding_client,
            store=store,
            chunker=chunker,
        )
        # Sneak it in — the service creates its pipeline in __init__, so we
        # overwrite with a pipeline that uses the reranker.
        service2._pipeline = RetrievalPipeline(store, reranker=reranker)

        context = service2.retrieve_as_context(
            query="How does async await work in Python?",
            knowledge_base_id=kb.id,
            top_k=2,
        )
        print(f"  Length: {len(context)} chars")
        print(context)

    print("\n  ✅ Reranker live test 完成。")


if __name__ == "__main__":
    main()
