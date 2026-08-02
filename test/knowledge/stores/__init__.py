#! python3
# -*- encoding: utf-8 -*-
"""Tests for knowledge store backends.

@File   :   __init__.py
@Created:   2026/08/02 03:16 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# ---------------------------------------------------------------------------
# Store Contract Test onboarding checklist for new backends
# ---------------------------------------------------------------------------
#
# ``test_store_contract.py`` defines the shared behaviour contract that every
# :class:`~isobase.knowledge.stores.BaseKnowledgeStore` implementation must
# satisfy.  A new backend that passes this suite is guaranteed to behave
# identically to Memory / SQL / Mongo stores from the perspective of
# :class:`~isobase.knowledge.KnowledgeBaseService`.
#
# How to onboard a new backend (e.g. ``PgVectorKnowledgeStore``):
#
# 1.  Implement :class:`~isobase.knowledge.stores.BaseKnowledgeStore`.
# 2.  Add a fixture branch in ``test_store_contract.py``'s ``store_factory``
#     that creates an isolated instance of your store.
# 3.  Add your param value (e.g. ``"pgvector"``) to the ``params`` list of
#     the ``store_factory`` fixture.
# 4.  Run the contract suite::
#
#         python -m pytest test/knowledge/stores/test_store_contract.py -v
#
# 5.  Every test that passes for ``[pgvector]`` as well as for
#     ``[memory]`` and ``[sql]`` confirms one more behaviour is aligned.
#
# Contract tests are the minimum bar.  You are encouraged to keep additional
# backend-specific tests (persistence across restarts, index creation,
# connection handling, etc.) in a separate test file — just like
# ``test_sql.py`` supplements the shared contract for ``SqlKnowledgeStore``.
