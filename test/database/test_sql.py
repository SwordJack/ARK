import pytest
from datetime import datetime, timezone
import os

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import DeclarativeBase

from isobase.database import configure_sql_db, SqlDbModelMixin, sql_db

class Base(DeclarativeBase):
    pass

class TestUserModel(Base, SqlDbModelMixin):
    __tablename__ = "test_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    created_time = Column(DateTime, nullable=True)

@pytest.fixture
def sqlite_db(tmp_path):
    db_path = tmp_path / "test.db"
    # sqlite requires check_same_thread=False which is handled by configure
    svc = configure_sql_db(f"sqlite:///{db_path}")
    svc.create_all(Base)
    yield svc
    svc.close()
    if os.path.exists(db_path):
        os.remove(db_path)

def test_insert_and_find(sqlite_db):
    user = TestUserModel(name="Alice")
    user.insert()

    assert user.id is not None

    found = TestUserModel.find_by_id(user.id)
    assert found is not None
    assert found.name == "Alice"

def test_find_many_and_count(sqlite_db):
    user1 = TestUserModel(name="Bob")
    user1.created_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    user1.insert()
    user2 = TestUserModel(name="Bob")
    user2.created_time = datetime(2021, 1, 1, tzinfo=timezone.utc)
    user2.insert()

    # Exact match keyword
    bobs = TestUserModel.find_many(name="Bob")
    assert len(bobs) == 2

    # Expression match (greater than)
    recent = TestUserModel.find_many(TestUserModel.created_time > datetime(2020, 6, 1, tzinfo=timezone.utc))
    assert len(recent) == 1

    count = TestUserModel.count(name="Bob")
    assert count == 2

def test_update(sqlite_db):
    user = TestUserModel(name="Charlie")
    user.insert()

    user.name = "Charlie Updated"
    user.update()

    found = TestUserModel.find_by_id(user.id)
    assert found.name == "Charlie Updated"

def test_delete(sqlite_db):
    user = TestUserModel(name="Dave")
    user.insert()

    user_id = user.id
    user.delete()

    found = TestUserModel.find_by_id(user_id)
    assert found is None

def test_execute_atomic(sqlite_db):
    def success_tx(session):
        u = TestUserModel(name="TxUser")
        u.insert(session)
        return u.name

    res = TestUserModel.execute_atomic(success_tx)
    assert res == "TxUser"
    assert TestUserModel.count(name="TxUser") == 1

    def fail_tx(session):
        u = TestUserModel(name="FailUser")
        u.insert(session)
        raise ValueError("Rollback")

    with pytest.raises(ValueError):
        TestUserModel.execute_atomic(fail_tx)

    assert TestUserModel.count(name="FailUser") == 0
