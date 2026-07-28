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

The module currently provides production-ready support for both non-relational and relational paradigms:

**1. MongoDB (`mongodb.py`)**

- `MongoDbService`: a framework-neutral holder for a PyMongo client, database, time zone, and transaction settings.
- `MongoDbModel`: a base class for MongoDB-backed models with common CRUD and query operations.
- MongoDB is fully supported for general-purpose application storage, document models, and execution records. However, note that it is **not** intended for embeddings or vector similarity search.

**2. SQL Relational Databases (`sql.py`)**

- `SqlDbService`: a framework-neutral SQLAlchemy connection holder (Engine & Session factory). Compatible with SQLite (auto-configured for thread-safety) and PostgreSQL.
- `SqlDbModelMixin`: a mixin providing the same Repository-pattern methods (`insert`, `find_many`, `count`, etc.) for SQLAlchemy models.
- Natively supports both simple keyword matching and complex SQLAlchemy filter expressions.

## Quick Start

`isobase.database` is not re-exported at the top-level `isobase` package, so import from this subpackage directly.

### MongoDB Example

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

document = KnowledgeDocument(title="IsoBase database notes")
document.insert()

loaded = KnowledgeDocument.find_by_id(document.id)
```

### SQL Relational Example

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import DeclarativeBase
from isobase.database import SqlDbModelMixin, configure_sql_db, sql_db

configure_sql_db(uri="sqlite:///app.db")

class Base(DeclarativeBase):
    pass

class User(Base, SqlDbModelMixin):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)

# Create tables
sql_db.create_all(Base)

user = User(name="Alice")
user.insert()

loaded = User.find_by_id(user.id)
```

## Custom Service Instances

Applications that manage their own client/engine can inject it directly, or assign a dedicated service to a specific model class (e.g. for multi-tenant or multi-database setups):

```python
from isobase.database import MongoDbModel, MongoDbService

service = MongoDbService(uri="mongodb://localhost:27017", database_name="tenant_a")

class TenantDocument(MongoDbModel):
    collection_name = "documents"

TenantDocument.use_mongo_service(service)
```

_(The same `use_sql_service(service)` approach works for `SqlDbModelMixin` and `SqlDbService`.)_

## Model Operations & Querying

Both database backends share similar naming conventions (`insert`, `update`, `delete`, `find_one`, `find_many`, `count`, `execute_atomic`), but their querying paradigms natively match their respective technologies.

**For MongoDB**, queries use MongoDB filter dictionaries:

```python
# Dictionary filter
items = KnowledgeDocument.find_many({"title": "Example"})
count = KnowledgeDocument.count({"view_count": {"$gt": 10}})

# MongoDB-specific operations
KnowledgeDocument.update_attributes({"title": "Updated"})
KnowledgeDocument.inc_attributes({"view_count": 1})
```

**For SQL Databases**, queries support standard simple kwargs alongside powerful SQLAlchemy expressions:

```python
# Simple keyword match
items = User.find_many(name="Alice")

# SQLAlchemy expression match (complex conditions)
recent_users = User.find_many(User.id > 10, name="Alice")
count = User.count(User.name.like("A%"))
```

## Schema Management

While global schema initialization is supported (e.g., `sql_db.create_all(Base)`), models in both paradigms also provide explicit, single-entity schema management.

**For SQL Databases**, you can explicitly create or drop the table for a specific model. This relies on SQLAlchemy's `checkfirst` behavior by default to safely skip existing tables:

```python
# Creates the table safely (if it does not exist)
User.create_table(checkfirst=True)

# Drops the table safely (if it exists)
User.drop_table(checkfirst=True)
```

**For MongoDB**, while collections are automatically implicitly created upon the first insertion, you can explicitly create a collection to specify options such as capping, size limits, or schema validators:

```python
# Explicitly create a capped collection with size limits
KnowledgeDocument.create_collection(
    capped=True,
    size=1024 * 1024, # 1 MB
    max=5000          # Max 5000 documents
)

# Drop the collection
KnowledgeDocument.drop_collection()
```

## Transaction Support

Both backends support atomic transactions via the `execute_atomic` callback wrapper.

```python
def create_records(session):
    user = User(name="Bob")
    user.insert(session=session)
    return user

# Executes within a managed SQL transaction or MongoDB transaction.
# Automatically commits on success and rolls back on unhandled exceptions.
created = User.execute_atomic(create_records)
````

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
