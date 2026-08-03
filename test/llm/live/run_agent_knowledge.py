#! python3
# -*- encoding: utf-8 -*-
"""Manual live smoke test: LLM answering with and without knowledge-base RAG.

This script is intentionally **not** collected by pytest.

It reads an existing knowledge base from MongoDB (populated by
``test/knowledge/live/run_lifecycle_mongo.py``) and answers the same
set of domain-specific questions *without* and *with* RAG retrieval,
so you can visually compare whether the knowledge base makes a
meaningful difference.

Usage
-----
1. Ensure MongoDB is running on localhost:27017 and the knowledge-base
   data from ``run_lifecycle_mongo.py`` is still present.

2. Both ``test/llm/live/.env`` (LLM credentials) and
   ``test/knowledge/live/.env`` (embedding credentials) must be
   configured.

3. Run from the repo root::

       python -m test.llm.live.run_agent_knowledge

   Or pick a provider::

       python -m test.llm.live.run_agent_knowledge openai
       python -m test.llm.live.run_agent_knowledge anthropic

@File   :   run_agent_knowledge.py
@Created:   2026/08/03 23:29 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from isobase.database.mongo import MongoDbService
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.entities import KnowledgeBase
from isobase.knowledge.stores import MongoKnowledgeStore
from isobase.llm import LLMClient, AnthropicMessages, OpenAIChat
from isobase.llm.tools.knowledge import create_knowledge_search_tool

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LLM_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
KB_ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "knowledge", "live", ".env",
)

MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "isobase"

# ---------------------------------------------------------------------------
# Queries that probe domain knowledge from the indexed documents.
# The LLM without RAG should lack this information; with RAG it should
# produce grounded, source-attributed answers.
# ---------------------------------------------------------------------------
_QUERIES = [
    u"工程控制论第三版保留和修订了哪些内容？",
    u"钱学森在序中如何看待技术革命和控制论的关系？",
    u"电子数字计算机为什么会推动自动控制技术革命？",
    "What technical revolutions does Qian Xuesen discuss in the preface?",
    "How does the author describe the relationship between cybernetics and systems engineering?",
    "What role did electronic digital computers play in automation according to the text?",
]


# ---------------------------------------------------------------------------
# Helpers — .env loading (same convention as existing live scripts)
# ---------------------------------------------------------------------------

def _load_env(path: str) -> Dict[str, str]:
    """Parse a git-ignored .env file into a dict, or exit with guidance."""
    if not os.path.exists(path):
        sys.exit(
            f"Missing {path}.\n"
            f"Copy the template and fill in real values:\n"
            f"  cp {path}.example {path}"
        )

    env: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
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
    """Build LLM client kwargs from ``{prefix}_BASE_URL / API_KEY / MODEL``."""
    api_key = env.get(f"{prefix}_API_KEY", "").strip()
    if not api_key or api_key.endswith("XXXXXX"):
        return None

    kwargs: Dict[str, Any] = {"api_key": api_key}
    base_url = env.get(f"{prefix}_BASE_URL", "").strip()
    if base_url:
        kwargs["base_url"] = base_url
    model = env.get(f"{prefix}_MODEL", "").strip()
    if model:
        kwargs["default_model"] = model
    return kwargs


def _embedding_kwargs(env: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Build embedding client kwargs from DashScope env vars."""
    prefix = "DASHSCOPE_EMBEDDING"
    api_key = env.get(f"{prefix}_API_KEY", "").strip()
    if not api_key or api_key.endswith("XXXXXX"):
        return None

    kwargs: Dict[str, Any] = {
        "api_key": api_key,
        "base_url": env.get(
            f"{prefix}_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        "model": env.get(f"{prefix}_MODEL", "text-embedding-v4"),
    }
    dims = env.get(f"{prefix}_DIMENSIONS", "").strip()
    if dims:
        kwargs["dimensions"] = int(dims)
    return kwargs


# ---------------------------------------------------------------------------
# Bootstrap — connect to the *existing* MongoDB knowledge base
# ---------------------------------------------------------------------------

def _build_kb_service() -> KnowledgeBaseService:
    """Build a KnowledgeBaseService pointed at the existing MongoDB data."""
    kb_env = _load_env(KB_ENV_PATH)
    emb_kwargs = _embedding_kwargs(kb_env)
    if emb_kwargs is None:
        sys.exit(
            "DashScope API key not configured. "
            f"Edit {KB_ENV_PATH} to set DASHSCOPE_EMBEDDING_API_KEY."
        )
    embedding_client = OpenAIEmbeddingClient(
        dimensions=emb_kwargs.pop("dimensions", 1024), **emb_kwargs
    )
    print(f"  Embedding: {embedding_client.model}  dim={embedding_client.dimensions}")

    mongo_service = MongoDbService(uri=MONGO_URI, database_name=MONGO_DB)
    store = MongoKnowledgeStore(mongo_service)
    # Chunker is only needed for indexing; retrieval reads what's already stored.
    chunker = FixedSizeChunker(chunk_size=800, chunk_overlap=100)

    return KnowledgeBaseService(
        embedding_client=embedding_client,
        store=store,
        chunker=chunker,
    )


def _find_existing_kb(service: KnowledgeBaseService) -> KnowledgeBase:
    """Find an existing knowledge base in MongoDB; exit if none found."""
    kbs = service.list_knowledge_bases()
    if not kbs:
        sys.exit(
            "No knowledge bases found in MongoDB.\n"
            "Run `python -m test.knowledge.live.run_lifecycle_mongo` first "
            "to populate data."
        )
    kb = kbs[0]
    doc_count = len(service.list_documents(kb.id))
    print(f"  Knowledge base : [{kb.id[-8:]}] {kb.name!r}")
    print(f"  Documents      : {doc_count}")
    return kb


# ---------------------------------------------------------------------------
# Core scenario — with / without RAG
# ---------------------------------------------------------------------------

def _run_rag_scenario(
    llm_client: LLMClient,
    kb_service: KnowledgeBaseService,
    kb_id: str,
    llm_label: str,
    query: str,
) -> None:
    """Answer *query* without and with knowledge-base retrieval."""
    sep = "─" * 70

    # --- Round 1: without RAG -------------------------------------------------
    print(f"\n{sep}")
    print(f"  [{llm_label}] WITHOUT RAG")
    print(f"  Query: {query!r}\n")
    llm_client.messages = []
    resp_without = llm_client.ask(query, stream=False)
    print(f"  Response:\n{resp_without.content}\n")

    # --- Round 2: with RAG ----------------------------------------------------
    print(f"\n{sep}")
    print(f"  [{llm_label}] WITH RAG (search_knowledge_base tool)")
    print(f"  Query: {query!r}\n")
    llm_client.messages = []

    kb_tool = create_knowledge_search_tool(kb_service, kb_id, top_k=3)
    llm_client.tool_set.tools = [kb_tool]

    resp_with = llm_client.ask(query, stream=False)
    print(f"  Response:\n{resp_with.content}\n")

    # Report whether the tool was actually called
    if llm_client.latest_tool_call_result:
        called = list(llm_client.latest_tool_call_result.keys())
        print(f"  [Tool log] Model called: {called}")
    else:
        print("  [Tool log] Model did NOT call the knowledge search tool.")


# ---------------------------------------------------------------------------
# Entry points per provider
# ---------------------------------------------------------------------------

def _run_provider(
    llm_kwargs: Dict[str, Any],
    factory_name: str,
    kb_service: KnowledgeBaseService,
    kb_id: str,
) -> None:
    """Run the RAG scenario battery for one LLM provider."""
    llm_label = f"{factory_name} ({llm_kwargs.get('default_model', '?')})"

    if factory_name == "OpenAIChat":
        client: LLMClient = OpenAIChat(**llm_kwargs)
    else:
        client = AnthropicMessages(**llm_kwargs)

    for query in _QUERIES:
        _run_rag_scenario(client, kb_service, kb_id, llm_label, query)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(only: Optional[str] = None) -> None:
    """Entry point.

    Args:
        only: ``"openai"`` or ``"anthropic"`` to run a single provider,
            or None for every provider whose API key is present.
    """
    llm_env = _load_env(LLM_ENV_PATH)

    print("=" * 70)
    print("  LLM + Knowledge Base RAG — Live Smoke Test")
    print("=" * 70)

    # Connect to existing MongoDB knowledge base
    kb_service = _build_kb_service()
    kb = _find_existing_kb(kb_service)

    # Resolve which providers to run
    provider_map: Dict[str, tuple[str, Optional[Dict[str, Any]]]] = {
        "openai": ("OpenAIChat", _client_kwargs(llm_env, "OPENAI_CHAT")),
        "anthropic": (
            "AnthropicMessages",
            _client_kwargs(llm_env, "ANTHROPIC_MESSAGES"),
        ),
    }

    if only and only not in provider_map:
        sys.exit(f"Unknown provider '{only}'. Choose from: {list(provider_map)}")

    executed = False
    for name, (factory_name, kwargs) in provider_map.items():
        if only and name != only:
            continue
        if kwargs is None:
            print(f"\n  [skip] {factory_name} — no API key in {LLM_ENV_PATH}")
            continue
        print(f"\n  🚀 Running {factory_name} ...")
        _run_provider(kwargs, factory_name, kb_service, kb.id)
        executed = True

    if not executed:
        print("\n  No providers configured. Check your .env files.")

    # Quiet MongoDB cleanup (ignore if not open)
    try:
        kb_service.store.mongo_service.close()
    except Exception:
        pass

    print("\n" + "=" * 70)
    print("  ✅ RAG live test complete.")
    print("=" * 70)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
