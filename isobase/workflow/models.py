#! python3
# -*- coding: utf-8 -*-
"""
This page is used to define the table structure/model of the database
that records execution information for each WorkUnit and WorkFlow instance.

@File   : models.py
@Created: 2025/04/02 22:41
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from typing import Optional
from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import DeclarativeBase, Session

from isobase.database.sql import SqlDbModelMixin


class Base(DeclarativeBase):
    pass


class WorkflowExecutionRecord(Base, SqlDbModelMixin):
    """
    A SQLAlchemy ORM model for storing and managing execution entities in the database.

    Attributes:
        identifier (str): A unique identifier for the execution entity.
        status (int): The execution status of the entity.
        updated_time (datetime): The timestamp of the last update to the entity.

    Methods:
        save_status: Inserts a new entity or updates an existing one in the database.
    """
    __tablename__ = "execution_entities"

    identifier = Column(String(255), primary_key=True, unique=True)
    status = Column(Integer, nullable=False)
    updated_time = Column(DateTime, nullable=False)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(identifier={self.identifier}, status={self.status}, updated_time={self.updated_time})"

    @classmethod
    def save_status(cls, identifier: str, status: int, session: Optional[Session] = None) -> None:
        """
        Inserts a new entity into the database or updates an existing one based on the identifier.

        Args:
            identifier (str): The unique identifier for the entity.
            status (int): The execution status of the entity.
            session (Session, optional): The database session.
        """
        existing = cls.find_by_id(identifier, session=session)
        now = datetime.now(timezone.utc)

        if existing:
            existing.status = status
            existing.updated_time = now
            existing.update(session=session)
        else:
            new_record = cls(identifier=identifier, status=status)
            # update method or SqlDbModelMixin insert sets created_time/updated_time,
            # but we define updated_time directly.
            new_record.updated_time = now
            new_record.insert(session=session)
