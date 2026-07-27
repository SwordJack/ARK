import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from bson import ObjectId

from isobase.database import configure_mongodb, MongoDbModel

# Need to setup mongodb without relying on the unstarted local server,
# or use mongomock. Since the user asked to test against the real MongoDB,
# I will attempt connecting to it.

@pytest.fixture(scope="module")
def mongo_service():
    """Setup MongoDB test service"""
    mongo_uri = "mongodb://localhost:27017"

    svc = configure_mongodb(
        uri=mongo_uri,
        database_name="isobase",
        time_zone=ZoneInfo("UTC"),
        transactions_enabled=False # Force false for standalone dev server testing to avoid transaction aborts
    )

    yield svc

    # Cleanup test collections
    svc.db.drop_collection(IntegrationDoc.collection_name)
    svc.close()

class IntegrationDoc(MongoDbModel):
    collection_name = "integration_docs"

def test_mongo_create_and_drop_collection(mongo_service):
    class TempDoc(MongoDbModel):
        collection_name = "temp_integration_docs"

    # Ensure it's dropped first
    TempDoc.drop_collection()

    # Explicitly create with options (e.g. capped)
    # Using small size just for testing
    coll = TempDoc.create_collection(capped=True, size=1024, max=5)
    assert coll.name == "temp_integration_docs"

    # Insert 6 docs, but because max=5, it should only keep 5
    for i in range(6):
        d = TempDoc(title=f"Doc {i}")
        d.insert()

    assert TempDoc.count() == 5

    TempDoc.drop_collection()
    assert TempDoc.count() == 0

def test_mongo_insert_and_find(mongo_service):
    doc = IntegrationDoc(title="Alice Document", value=100)
    doc.insert()

    assert doc.id is not None
    assert getattr(doc, "created_time", None) is not None

    found = IntegrationDoc.find_by_id(doc.id)
    assert found is not None
    assert found.title == "Alice Document"
    assert found.value == 100

def test_mongo_find_many_and_count(mongo_service):
    doc1 = IntegrationDoc(title="Bob Document", value=200)
    doc1.insert()

    doc2 = IntegrationDoc(title="Bob Document", value=300)
    doc2.insert()

    bobs = IntegrationDoc.find_many({"title": "Bob Document"})
    assert len(bobs) >= 2

    # Query using mongodb dict operators
    high_value = IntegrationDoc.find_many({"value": {"$gt": 250}, "title": "Bob Document"})
    assert len(high_value) >= 1

    count = IntegrationDoc.count({"title": "Bob Document"})
    assert count >= 2

def test_mongo_update_and_delete(mongo_service):
    doc = IntegrationDoc(title="Charlie", value=400)
    doc.insert()

    doc.update_attributes({"value": 450})
    doc.inc_attributes({"value": 10})

    found = IntegrationDoc.find_by_id(doc.id)
    assert found.value == 460

    doc_id = doc.id
    doc.delete()

    assert IntegrationDoc.find_by_id(doc_id) is None
