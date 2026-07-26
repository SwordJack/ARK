# Database Module

> This module is under active development.

## Introduction

The `isobase.database` module provides database access utilities for IsoBase. Its long-term goal is to define a unified interface for working with relational databases (such as SQLite and PostgreSQL) and non-relational databases (such as MongoDB).

This shared database layer provides general-purpose storage for any applications built on top of IsoBase. While advanced RAG (Retrieval-Augmented Generation) and knowledge-base features specifically rely on vector-capable storage (such as dedicated vector databases or PostgreSQL with `pgvector`), standard relational and non-relational backends remain first-class components for application state, document models, and execution records.

## Design Goals

- **Framework-agnostic access**: database services should work in IsoBase as a Python library, without depending on Flask, Django, or application context globals.
- **Unified model operations**: common operations such as insert, update, delete, find, count, aggregate, and index management should follow predictable patterns across backends where possible.
- **Dependency injection friendly**: callers should be able to configure database clients explicitly, pass existing connections, and test with mock or temporary databases.
- **General-purpose storage**: support application state, document models, and standard transactional needs for IsoBase users.
- **LLM knowledge-base support**: the module should provide the storage foundation for future knowledge ingestion and document retrieval. While document/metadata storage uses general backends, embeddings and vector indexes will target specialized vector databases or PostgreSQL with `pgvector`.
- **Backend separation**: backend-specific details should stay inside their service/model implementations instead of leaking into LLM or workflow code.

## Current Status

The module currently includes MongoDB support through [`mongodb.py`](./mongodb.py):

- `MongoDbService`: a small framework-neutral holder for a PyMongo client, database, time zone, and transaction settings.
- `MongoDbModel`: a base class for MongoDB-backed models with common CRUD and query operations.
- `configure_mongodb`: a convenience function for configuring the module-level default MongoDB service.
- `mongo`: the module-level default `MongoDbService` instance.

MongoDB is fully supported for general-purpose application storage, document models, and execution records. However, note that it is **not** intended for embeddings or vector similarity search. Vector backends, such as a dedicated vector database or PostgreSQL with `pgvector`, are planned to handle the knowledge-base vector layer separately.

Relational database abstractions for SQLite and PostgreSQL are planned but not yet implemented in this directory. The existing SQLite persistence used by the workflow engine lives in `isobase.workflow.database` and is currently separate from this module.

## Quick Start

`isobase.database` is not re-exported at the top-level `isobase` package, so import from this subpackage directly.

```python
from zoneinfo import ZoneInfo

from isobase.database import MongoDbModel, configure_mongodb

configure_mongodb(
    uri="mongodb://localhost:27017",
    database_name="isobase",
    time_zone=ZoneInfo("UTC"),
)


class KnowledgeDocument(MongoDbModel):
    """Example MongoDB-backed model."""

    collection_name = "knowledge_documents"


document = KnowledgeDocument(
    title="IsoBase database notes",
    content="Database services will support LLM knowledge-base storage.",
    metadata={"source": "README"},
)
document.insert()

loaded = KnowledgeDocument.find_by_id(document.id)
print(loaded.to_dict())
```

## Using an Existing MongoClient

Applications that already manage their own PyMongo client can inject it instead of passing a URI:

```python
from pymongo import MongoClient

from isobase.database import configure_mongodb

client = MongoClient("mongodb://localhost:27017")
configure_mongodb(client=client, database_name="isobase")
```

For model-specific configuration, assign a dedicated service to a model class:

```python
from isobase.database import MongoDbModel, MongoDbService

service = MongoDbService(
    uri="mongodb://localhost:27017",
    database_name="tenant_a",
)


class TenantDocument(MongoDbModel):
    collection_name = "documents"


TenantDocument.use_mongo_service(service)
```

## MongoDB Model Operations

`MongoDbModel` provides common operations for MongoDB-backed models:

```python
item = KnowledgeDocument(title="Example")
item.insert()

item.update_attributes({"title": "Updated example"})
item.inc_attributes({"view_count": 1})

same_item = KnowledgeDocument.find_by_id(item.id)
items = KnowledgeDocument.find_many({"title": "Updated example"})
count = KnowledgeDocument.count({"title": "Updated example"})

item.delete()
```

Index and aggregation helpers are also available:

```python
KnowledgeDocument.create_index("title")
KnowledgeDocument.list_indexes()
KnowledgeDocument.aggregate([
    {"$match": {"metadata.source": "README"}},
    {"$count": "total"},
])
```

## Transaction Support

`MongoDbModel.execute_atomic` runs a callback inside a MongoDB transaction when the configured MongoDB deployment supports transactions. If the deployment does not support transactions, such as a standalone MongoDB server, the callback is executed without a session.

```python
def create_document(session=None):
    document = KnowledgeDocument(title="Atomic insert")
    document.insert(session=session)
    return document

created = KnowledgeDocument.execute_atomic(create_document)
```

You can explicitly control transaction behavior during configuration:

```python
configure_mongodb(
    uri="mongodb://localhost:27017",
    database_name="isobase",
    transactions_enabled=False,
)
```

## Intended Direction

The database module is intended to be the unified persistence boundary for IsoBase applications. Future interfaces may cover:

- backend-neutral repository or collection interfaces;
- relational backends such as SQLite and PostgreSQL for structured application data;
- non-relational backends such as MongoDB for unstructured application state and document records;
- vector-capable backends (dedicated vector databases or PostgreSQL with `pgvector`) specifically serving LLM embeddings and retrieval indexes;
- document and chunk storage, with cross-backend metadata filtering.

Until those abstractions are introduced, code should depend only on the concrete backend service it needs and avoid adding Flask-style global context dependencies.

## Notes for Contributors

- Keep database code framework-neutral.
- Prefer explicit configuration and dependency injection over hidden global application state.
- Do not import database services from the top-level `isobase` package unless that becomes an intentional public API decision.
- Keep backend-specific behavior isolated to backend-specific files.
- Add tests under `test/database/` when adding or changing database behavior.
