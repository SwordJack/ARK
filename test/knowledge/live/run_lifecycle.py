#! python3
# -*- encoding: utf-8 -*-
"""Live smoke test covering the full knowledge-base lifecycle with PostgreSQL.

This script is intentionally **not** collected by pytest (it doesn't start
with ``test_``) and never runs in CI.

It uses a deterministic ``FakeEmbeddingClient`` so no API keys are needed.
The only external dependency is a running PostgreSQL instance.

Usage
-----
1. Make sure PostgreSQL is running and the target database exists::

       createdb isobase

2. Run from the repo root::

       python -m test.knowledge.live.run_lifecycle

The script pauses after every major step so you can inspect the database
contents with an external tool (psql, DBeaver, etc.).  Press Enter to
continue or Ctrl-C to abort.

@File   :   run_lifecycle.py
@Created:   2026/08/02 03:16 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

from __future__ import annotations

import sys
from typing import List

from isobase.database.sql import SqlDbService
from isobase.knowledge import KnowledgeBaseService
from isobase.knowledge.embeddings import BaseEmbeddingClient
from isobase.knowledge.chunking import FixedSizeChunker
from isobase.knowledge.entities import (
    KnowledgeBase,
    KnowledgeDocument,
    KnowledgeChunk,
    RetrievalResult,
)
from isobase.knowledge.stores import SqlKnowledgeStore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PG_URI = "postgresql://postgres:postgres@localhost:5432/isobase"

# ---------------------------------------------------------------------------
# Fake embedding client (no API calls)
# ---------------------------------------------------------------------------


class FakeEmbeddingClient(BaseEmbeddingClient):
    """Deterministic fake embeddings so no API key is ever needed."""

    def __init__(self, dimensions: int = 128):
        super().__init__(dimensions=dimensions)

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        embeddings = []
        for text in texts:
            vec = []
            for i in range(self._dimensions):
                val = (len(text) + sum(ord(c) for c in text[:10]) + i) % 100 / 100.0
                vec.append(val)
            embeddings.append(vec)
        return embeddings

    def embed_query(self, query: str, **kwargs) -> List[float]:
        return self.embed_texts([query], **kwargs)[0]

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def fetch_dimensions(self) -> int:
        return self._dimensions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pause(step: str, next_desc: str = "") -> None:
    """Pause execution so the user can inspect the database.

    Press Enter to continue, Ctrl-C to abort.
    """
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


def _dump_kb(kb: KnowledgeBase) -> None:
    print(f"      id               = {kb.id}")
    print(f"      name             = {kb.name!r}")
    print(f"      description      = {kb.description!r}")
    print(f"      embedding_model  = {kb.embedding_model_id!r}")
    print(f"      dimensions       = {kb.dimensions}")
    print(f"      chunk_size       = {kb.chunk_size}")
    print(f"      chunk_overlap    = {kb.chunk_overlap}")
    print(f"      metadata         = {kb.metadata}")
    print(f"      created_time     = {kb.created_time}")
    print(f"      updated_time     = {kb.updated_time}")


def _dump_doc(doc: KnowledgeDocument, indent: str = "      ") -> None:
    print(f"{indent}id               = {doc.id}")
    print(f"{indent}title            = {doc.title!r}")
    print(f"{indent}source_uri       = {doc.source_uri!r}")
    print(f"{indent}knowledge_base_id= {doc.knowledge_base_id}")
    print(f"{indent}content          = {doc.content[:100]}{'...' if len(doc.content) > 100 else ''}")
    print(f"{indent}metadata         = {doc.metadata}")
    print(f"{indent}created_time     = {doc.created_time}")


def _dump_result(r: RetrievalResult, i: int) -> None:
    doc_title = r.document.title if r.document else "?"
    print(f"    [{i}] score = {r.score:.4f}")
    print(f"         chunk.id        = {r.chunk.id}")
    print(f"         chunk.doc_id    = {r.chunk.document_id}")
    print(f"         chunk.index     = {r.chunk.index}")
    print(f"         chunk.token_cnt = {r.chunk.token_count}")
    print(f"         chunk.content   = {r.chunk.content[:80]!r}")
    print(f"         document        = {doc_title}")


# ---------------------------------------------------------------------------
# Test documents
# ---------------------------------------------------------------------------

_DOCUMENTS = [
    {
        "title": "RAG 概述",
        "source_uri": "docs/rag.md",
        "text": (
            "RAG 全称 Retrieval-Augmented Generation（检索增强生成），是一种将信息检索与文本生成结合的技术。"
            "RAG 系统首先从知识库中检索相关文档，然后将这些文档作为上下文用于生成回答。"
            "这种方法可以帮助大语言模型提供更准确、更及时的信息，有效缓解模型幻觉问题。"
            "RAG 的核心组件包括：文档解析器、文本分块器、嵌入模型、向量存储和检索引擎。"
        ),
        "metadata": {"author": "张三", "date": "2024-01-15", "tags": ["RAG", "入门"]},
    },
    {
        "title": "向量数据库简介",
        "source_uri": "docs/vector-db.md",
        "text": (
            "向量数据库是专为高维嵌入向量设计的存储系统。它们利用近似最近邻（ANN）搜索等技术实现快速相似度检索。"
            "主流向量数据库包括 FAISS、Pinecone、Qdrant 和 Milvus。"
            "这些系统是 RAG 应用的关键基础设施，直接影响检索质量与延迟。"
            "选择向量数据库时应考虑：查询性能、可扩展性、过滤能力以及运维复杂度。"
        ),
        "metadata": {"author": "李四", "date": "2024-01-16", "tags": ["向量数据库", "基础设施"]},
    },
    {
        "title": "文本嵌入技术",
        "source_uri": "docs/embeddings.md",
        "text": (
            "嵌入（Embedding）是文本的稠密向量表示，能捕获语义含义。"
            "它们由神经网络训练生成，将语义相似的文本在向量空间中放置得更近。"
            "常见的嵌入模型包括 OpenAI 的 text-embedding-3、阿里云的 Qwen 嵌入模型，"
            "以及开源模型如 sentence-transformers。嵌入质量直接影响 RAG 系统的检索准确率。"
        ),
        "metadata": {"author": "张三", "date": "2024-01-17", "tags": ["嵌入", "NLP"]},
    },
]

_QUERIES = [
    "什么是 RAG？",
    "向量数据库有哪些？",
    "嵌入是怎么工作的？",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # -- Bootstrap ----------------------------------------------------------
    print("  PostgreSQL URI:", PG_URI)
    sql_service = SqlDbService(PG_URI)
    store = SqlKnowledgeStore(sql_service)
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=30)
    embedding_client = FakeEmbeddingClient(dimensions=16)

    service = KnowledgeBaseService(
        embedding_client=embedding_client,
        store=store,
        chunker=chunker,
    )

    # =======================================================================
    # Step 1: Create knowledge bases
    # =======================================================================
    _h1("Step 1: 创建知识库")
    _h2("创建 kb1 — 科技文档库")
    kb1 = service.create_knowledge_base(
        name="科技文档库",
        description="存放 RAG、向量数据库等技术文档",
        metadata={"owner": "dev-team", "env": "live-test"},
    )
    _dump_kb(kb1)

    _h2("创建 kb2 — 空知识库（用于验证空列表/空搜索）")
    kb2 = service.create_knowledge_base(
        name="空知识库",
        description="这个库没有任何文档",
    )
    _dump_kb(kb2)

    _pause("创建知识库完成", "list_knowledge_bases")

    # =======================================================================
    # Step 2: List knowledge bases
    # =======================================================================
    _h1("Step 2: 列出所有知识库")
    kbs = service.list_knowledge_bases()
    print(f"  共 {len(kbs)} 个知识库:")
    for kb in kbs:
        print(f"    - [{kb.id[-8:]}] {kb.name!r}  ({kb.description!r})")
    _pause("list_knowledge_bases 完成", "index_text × 3")

    # =======================================================================
    # Step 3: Index documents
    # =======================================================================
    _h1("Step 3: 索引文档")
    doc_ids: list[str] = []
    for i, doc_meta in enumerate(_DOCUMENTS, 1):
        _h2(f"索引文档 {i}/{len(_DOCUMENTS)}: {doc_meta['title']}")
        doc = service.index_text(
            knowledge_base_id=kb1.id,
            text=doc_meta["text"],
            title=doc_meta["title"],
            source_uri=doc_meta["source_uri"],
            metadata=doc_meta["metadata"],
        )
        doc_ids.append(doc.id)
        _dump_doc(doc)
        print(f"      (document 已存储，chunks 已嵌入)")

    # Verify kb2 stays empty
    _h2("验证 kb2（空知识库）仍然为空")
    empty_docs = service.list_documents(kb2.id)
    print(f"  kb2 文档数: {len(empty_docs)}  (应为 0)")

    _pause("索引文档完成", "list_documents + get_document")

    # =======================================================================
    # Step 4: List & get documents
    # =======================================================================
    _h1("Step 4: 查看文档列表与详情")

    _h2("list_documents(kb1) — 列出 kb1 下所有文档")
    docs = service.list_documents(kb1.id)
    print(f"  kb1 共 {len(docs)} 个文档:")
    for doc in docs:
        print(f"    - [{doc.id[-8:]}] {doc.title!r}  (source: {doc.source_uri})")

    _h2("get_document — 逐个获取文档详情")
    for doc_id in doc_ids:
        doc = service.get_document(doc_id)
        print(f"\n  --- 文档 [{doc.id[-8:]}] ---")
        _dump_doc(doc)

    _pause("文档列表查看完成", "retrieve × 3")

    # =======================================================================
    # Step 5: Search / retrieve
    # =======================================================================
    _h1("Step 5: 检索查询")
    for query in _QUERIES:
        _h2(f'检索: "{query}"')
        results = service.retrieve(query=query, knowledge_base_id=kb1.id, top_k=3)
        if not results:
            print("    → 无结果")
        else:
            for i, r in enumerate(results, 1):
                _dump_result(r, i)

    _h2("检索空知识库 kb2 — 应返回空列表")
    empty_results = service.retrieve(query="测试", knowledge_base_id=kb2.id)
    print(f"  结果数: {len(empty_results)}  (应为 0)")

    _h2("retrieve_as_context — 格式化上下文")
    context = service.retrieve_as_context(query="什么是 RAG？", knowledge_base_id=kb1.id, top_k=2)
    print(f"  上下文长度: {len(context)} chars")
    if context:
        print(f"  预览:\n{context[:300]}...")

    _pause("检索验证完成", "delete_document (doc1)")

    # =======================================================================
    # Step 6: Delete a document
    # =======================================================================
    _h1("Step 6: 删除文档")

    _h2(f"删除前: kb1 有 {len(service.list_documents(kb1.id))} 个文档")
    target_doc_id = doc_ids[0]
    target_doc = service.get_document(target_doc_id)
    print(f"  即将删除: [{target_doc.id[-8:]}] {target_doc.title!r}")

    _pause("确认删除前", "执行 delete_document")

    service.delete_document(target_doc_id)
    print(f"  ✅ 已删除文档 [{target_doc_id[-8:]}]")

    _h2("删除后验证")
    docs_after = service.list_documents(kb1.id)
    print(f"  kb1 文档数: {len(docs_after)}  (应为 {len(doc_ids) - 1})")
    for doc in docs_after:
        print(f"    - [{doc.id[-8:]}] {doc.title!r}")

    # Verify deleted doc is truly gone
    try:
        service.get_document(target_doc_id)
        print("  ❌ BUG: 已删除的文档仍然可以 get_document!")
    except KeyError:
        print(f"  ✅ get_document 正确抛出 KeyError")

    # Verify search no longer returns chunks from deleted doc
    _h2("检索验证: 删除文档的 chunks 不应再出现")
    results_after = service.retrieve(query="RAG", knowledge_base_id=kb1.id, top_k=5)
    deleted_chunks = [r for r in results_after if r.chunk.document_id == target_doc_id]
    if deleted_chunks:
        print(f"  ❌ BUG: 检索仍返回 {len(deleted_chunks)} 个已删除文档的 chunk!")
    else:
        print(f"  ✅ 检索结果中无已删除文档的 chunk")

    _pause("文档删除验证完成", "delete_knowledge_base (kb2)")

    # =======================================================================
    # Step 7: Delete a knowledge base
    # =======================================================================
    _h1("Step 7: 删除知识库")

    _h2("删除前状态")
    print(f"  知识库数: {len(service.list_knowledge_bases())}  (应为 2)")
    print(f"  即将删除: kb2 ({kb2.name!r})")

    _pause("确认删除前", "执行 delete_knowledge_base")

    service.delete_knowledge_base(kb2.id)
    print(f"  ✅ 已删除知识库 [{kb2.id[-8:]}]")

    _h2("删除后验证")
    kbs_after = service.list_knowledge_bases()
    print(f"  知识库数: {len(kbs_after)}  (应为 1)")
    for kb in kbs_after:
        print(f"    - [{kb.id[-8:]}] {kb.name!r}")

    try:
        service.store.get_knowledge_base(kb2.id)
        print("  ❌ BUG: 已删除的知识库仍然存在!")
    except KeyError:
        print(f"  ✅ get_knowledge_base 正确抛出 KeyError")

    _pause("知识库删除验证完成", "清理并退出")

    # =======================================================================
    # Step 8: Cleanup
    # =======================================================================
    _h1("Step 8: 清理 — 删除剩余 kb1")
    service.delete_knowledge_base(kb1.id)
    remaining = service.list_knowledge_bases()
    print(f"  剩余知识库数: {len(remaining)}  (应为 0)")

    sql_service.close()

    _h1("✅ 全生命周期 live test 完成")
    print("  验证了: create → list → index → get → search → delete_doc → delete_kb")
    print("  数据库已清空，可以重复运行。")


if __name__ == "__main__":
    main()
