#! python3
# -*- encoding: utf-8 -*-
"""Live smoke test for markdown-aware chunking with MongoDB.

Uses ``MarkdownChunker`` against the real markdown chapter file to
demonstrate heading-aware splitting.  Inspect the database with
mongosh / Compass between steps.

Usage
-----
1. Make sure MongoDB is running on localhost:27017.
2. Make sure the DashScope API key is configured in the ``.env`` file
   sitting next to this script (see ``.env.example``).
3. Run from the repo root::

       python -m test.knowledge.live.run_markdown_chunking

The script pauses after every major step so you can inspect the
database contents with an external tool (mongosh, Compass, etc.).
Press Enter to continue or Ctrl-C to abort.

@File   :   run_markdown_chunking.py
@Created:   2026/08/04 19:56 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

from isobase.database.mongo import MongoDbService
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import OpenAIEmbeddingClient
from isobase.knowledge.chunking import MarkdownChunker
from isobase.knowledge.entities import (
    KnowledgeBase,
    KnowledgeDocument,
    KnowledgeChunk,
    RetrievalResult,
)
from isobase.knowledge.stores import MongoKnowledgeStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB = "isobase"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
CHAPTER_FILE = os.path.join(DATA_DIR, "chapter_000.md")
ENV_PATH = os.path.join(HERE, ".env")


# ---------------------------------------------------------------------------
# Helpers — .env loading (same convention as run_lifecycle_mongo.py)
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


def _dashscope_client_kwargs(env: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Build kwargs for OpenAIEmbeddingClient from DASHSCOPE_EMBEDDING_* env vars."""
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
# Helpers — output & pause
# ---------------------------------------------------------------------------


def _pause(step: str, next_desc: str = "") -> None:
    """Pause execution so the user can inspect the database.

    In an interactive terminal this waits for Enter; in a non-TTY
    environment (CI / automated run) it prints a separator and continues.
    """
    if not sys.stdin.isatty():
        print(f"\n  ⏭  [{step}] (自动继续 — 非交互环境)")
        return
    prompt = f"\n  ⏸  [{step}] — press Enter"
    if next_desc:
        prompt += f" (next: {next_desc})"
    prompt += ", Ctrl-C to abort: "
    try:
        input(prompt)
    except KeyboardInterrupt:
        print("\n\n  Aborted by user.")
        sys.exit(0)


def _h1(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print("=" * 60)


def _h2(text: str) -> None:
    print(f"\n--- {text} ---")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # -- Bootstrap ----------------------------------------------------------
    print("  Markdown Chunker Live Test (MongoDB)")
    print(f"  MongoDB URI: {MONGO_URI}")
    print(f"  MongoDB DB : {MONGO_DB}")
    print(f"  chunk_size  : {CHUNK_SIZE}")
    print(f"  chunk_overlap: {CHUNK_OVERLAP}")

    # Embedding client
    env = _load_env()
    kwargs = _dashscope_client_kwargs(env)
    if kwargs is None:
        sys.exit(
            "DashScope API key not configured. "
            f"Edit {ENV_PATH} to set DASHSCOPE_EMBEDDING_API_KEY."
        )
    embedding_client = OpenAIEmbeddingClient(
        dimensions=kwargs.pop("dimensions", 1024), **kwargs
    )
    print(f"  Embedding: {embedding_client.model}  dim={embedding_client.dimensions}")

    # Store + chunker
    mongo_service = MongoDbService(uri=MONGO_URI, database_name=MONGO_DB)
    store = MongoKnowledgeStore(mongo_service)
    chunker = MarkdownChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    service = KnowledgeBaseService(
        embedding_client=embedding_client,
        store=store,
        chunker=chunker,
        embed_batch_size=10,  # DashScope 单批次上限为 10
    )

    # =====================================================================
    # Step 1: Dry-run — chunk without storing, inspect heading structure
    # =====================================================================
    _h1("Step 1: Dry-run — 仅切分不入库，观察 heading 结构")

    with open(CHAPTER_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"  源文件: {CHAPTER_FILE}")
    print(f"  原始文本长度: {len(raw_text)} chars")

    dry_chunks = chunker.chunk(raw_text)
    print(f"  切分后总 chunk 数: {len(dry_chunks)}")

    # Group by heading path
    by_heading: Dict[str, int] = defaultdict(int)
    for chunk in dry_chunks:
        path_key = " / ".join(chunk.metadata.get("heading_path", [])) or "(preamble)"
        by_heading[path_key] += 1

    print(f"\n  标题路径分布 ({len(by_heading)} 个 section):")
    for path_key, count in sorted(by_heading.items()):
        print(f"    [{count:2d} chunk(s)]  {path_key}")

    # Show a few chunks with metadata
    print(f"\n  示例 chunk（前 5 个）:")
    for i, chunk in enumerate(dry_chunks[:5]):
        heading = " / ".join(chunk.metadata.get("heading_path", [])) or "(preamble)"
        preview = chunk.content[:80].replace("\n", "\\n")
        print(
            f"    [{i}] heading_path: [{heading}]\n"
            f"        size={len(chunk.content):4d}  "
            f"strategy={chunk.metadata.get('chunk_strategy')}\n"
            f"        content={preview}..."
        )

    _pause("切分完成（未入库）", "创建知识库并索引")

    # =====================================================================
    # Step 2: Create KB & index with MarkdownChunker
    # =====================================================================
    _h1("Step 2: 创建知识库并索引文档（Markdown 切分）")

    kb = service.create_knowledge_base(
        name="工程控制论 (Markdown)",
        description="使用 MarkdownChunker 分块的真实 markdown 文档",
        metadata={"chunker": "MarkdownChunker", "env": "live-test"},
    )
    print(f"  KB: [{kb.id[-8:]}] {kb.name!r}")
    print(f"  chunk_size={kb.chunk_size}, chunk_overlap={kb.chunk_overlap}")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
    source_uri = os.path.relpath(CHAPTER_FILE, repo_root)

    doc = service.index_text(
        knowledge_base_id=kb.id,
        text=raw_text,
        title="工程控制论（上册）前言",
        source_uri=source_uri,
        metadata={
            "book": "工程控制论（上册）",
            "authors": ["钱学森", "宋健"],
            "kind": "book-chapter",
        },
    )
    print(f"  文档: [{doc.id[-8:]}] {doc.title!r}")
    print(f"  源文件: {source_uri}")
    print(f"  内容长度: {len(raw_text)} chars")

    # Count chunks per heading path from stored data
    _h2("从数据库读取 chunk，验证 heading_path 分布")
    stored_docs = service.list_documents(kb.id)
    print(f"  kb 文档数: {len(stored_docs)}")

    _pause("索引入库完成", "检索验证")

    # =====================================================================
    # Step 3: Retrieval — compare with heading metadata
    # =====================================================================
    _h1("Step 3: 检索 — 观察 chunk metadata 在结果中的表现")

    queries = [
        "钱学森在序中如何看待技术革命和控制论的关系？",
        "工程控制论第三版保留了哪些内容？",
        "电子数字计算机在自动化中起了什么作用？",
        "What does Qian Xuesen say about technological revolution?",
    ]

    for query in queries:
        _h2(f'检索: "{query}"')
        results = service.retrieve(query=query, knowledge_base_id=kb.id, top_k=3)

        if not results:
            print("    → 无结果")
            continue

        for i, r in enumerate(results, 1):
            heading = " / ".join(
                r.chunk.metadata.get("heading_path", [])
            ) or "(preamble)"
            strategy = r.chunk.metadata.get("chunk_strategy", "?")
            print(
                f"    [{i}] score={r.score:.4f}  "
                f"heading_path=[{heading}]  "
                f"strategy={strategy}"
            )
            print(
                f"        chunk_size={len(r.chunk.content)}  "
                f"token_count={r.chunk.token_count}"
            )
            preview = r.chunk.content[:100].replace("\n", "\\n")
            print(f"        content={preview}...")

    _pause("检索验证完成", "cleanup")

    # =====================================================================
    # Step 4: Cleanup
    # =====================================================================
    _h1("Step 4: 清理")
    service.delete_knowledge_base(kb.id)
    remaining = service.list_knowledge_bases()
    print(f"  剩余知识库数: {len(remaining)}  (应为 0)")

    mongo_service.close()
    print("\n  ✅ Markdown Chunker live test 完成。")


if __name__ == "__main__":
    main()
