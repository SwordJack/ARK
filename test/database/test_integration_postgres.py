#! python3
# -*- coding: utf-8 -*-
"""
@File   : test_integration_postgres.py
@Created: 2026/07/28 03:45 (UTC+08:00)
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

# Here put the import lib.
import os
import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import DeclarativeBase

from isobase.database import configure_sql_db, SqlDbModelMixin

class Base(DeclarativeBase):
    pass

class IntegrationUser(Base, SqlDbModelMixin):
    __tablename__ = "integration_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False)
    created_time = Column(DateTime, nullable=True)

@pytest.fixture(scope="module")
def pg_service():
    """Setup PostgreSQL test service"""
    pg_uri = "postgresql://postgres:postgres@localhost:5432/isobase"

    svc = configure_sql_db(
        uri=pg_uri,
        time_zone=ZoneInfo("UTC")
    )

    # Drop table if exists, then create
    Base.metadata.drop_all(svc.engine)
    svc.create_all(Base)

    yield svc

    Base.metadata.drop_all(svc.engine)
    svc.close()

def test_pg_insert_and_find(pg_service):
    user = IntegrationUser(name="Alice")
    user.insert()

    assert user.id is not None

    found = IntegrationUser.find_by_id(user.id)
    assert found is not None
    assert found.name == "Alice"
    assert found.created_time is not None

def test_pg_find_many_and_count(pg_service):
    user1 = IntegrationUser(name="Bob")
    user1.created_time = datetime(2022, 1, 1, tzinfo=timezone.utc)
    user1.insert()

    user2 = IntegrationUser(name="Bob")
    user2.created_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
    user2.insert()

    # Exact match keyword
    bobs = IntegrationUser.find_many(name="Bob")
    assert len(bobs) >= 2

    # Expression match (greater than)
    recent = IntegrationUser.find_many(
        IntegrationUser.created_time > datetime(2022, 6, 1, tzinfo=timezone.utc),
        name="Bob"
    )
    assert len(recent) >= 1

    count = IntegrationUser.count(name="Bob")
    assert count >= 2

def test_pg_create_and_drop_table(pg_service):
    class TempUser(Base, SqlDbModelMixin):
        __tablename__ = "temp_integration_users"

        id = Column(Integer, primary_key=True)
        name = Column(String(50))

    # Ensure it's dropped first (in case of left-over from previous crash)
    TempUser.drop_table(checkfirst=True)

    # Use the single-table method
    TempUser.create_table(checkfirst=True)

    # Verify it works
    u = TempUser(id=1, name="Temp PG")
    u.insert()
    assert TempUser.count() == 1

    # Drop it
    TempUser.drop_table()

    from sqlalchemy.exc import ProgrammingError
    with pytest.raises(ProgrammingError):
        # counting a dropped table should raise an error in SQL
        TempUser.count()

def test_pg_execute_transaction(pg_service):

    def success_tx(session):
        u = IntegrationUser(name="TxUserPG")
        u.insert(session)
        return u.name

    res = IntegrationUser.execute_transaction(success_tx)
    assert res == "TxUserPG"
    assert IntegrationUser.count(name="TxUserPG") == 1

    def fail_tx(session):
        u = IntegrationUser(name="FailUserPG")
        u.insert(session)
        raise ValueError("Rollback PG")

    with pytest.raises(ValueError):
        IntegrationUser.execute_transaction(fail_tx)

    assert IntegrationUser.count(name="FailUserPG") == 0
