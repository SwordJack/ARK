#! python3
# -*- coding: utf-8 -*-
"""
This file defines the parent mixin and service for SQLAlchemy-backed models,
providing a unified interface compatible with SQLite, PostgreSQL, and other
SQL databases.

@File   : sql.py
@Created: 2026/07/28 10:00
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

# Here put the import lib.
from __future__ import annotations
from datetime import datetime, timezone, tzinfo
from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, TypeVar

from sqlalchemy import Engine, create_engine, delete as sa_delete, select, update as sa_update
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError


T = TypeVar("T", bound="SqlDbModelMixin")


class SqlDbService(object):
    """A connection holder and session manager for SQLAlchemy-based SQL databases.

    Compatible with both SQLite and PostgreSQL by wrapping SQLAlchemy Engine and sessionmaker.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        time_zone: tzinfo = timezone.utc,
    ) -> None:
        """Initializes the SQL database service.

        Args:
            uri (str, optional): Database connection URI (e.g., 'sqlite:///app.db', 'postgresql://user:pass@localhost/db').
            engine_kwargs (dict, optional): Additional arguments passed to sqlalchemy.create_engine.
            time_zone (tzinfo, optional): Time zone used for model timestamps.
        """
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self.time_zone = time_zone

        if uri is not None:
            self.configure(uri=uri, engine_kwargs=engine_kwargs, time_zone=time_zone)
        return

    def configure(
        self,
        uri: str,
        engine_kwargs: Optional[Dict[str, Any]] = None,
        time_zone: tzinfo = timezone.utc,
    ) -> SqlDbService:
        """Configures the SQLAlchemy engine and sessionmaker.

        Args:
            uri (str): Database connection URI.
            engine_kwargs (dict, optional): Additional arguments for create_engine.
            time_zone (tzinfo, optional): Time zone used for model timestamps.

        Returns:
            SqlDbService: This service instance.
        """
        kwargs = engine_kwargs or {}

        # Optimize for SQLite concurrency by default
        if uri.startswith("sqlite"):
            if "connect_args" not in kwargs:
                kwargs["connect_args"] = {"check_same_thread": False}

        self._engine = create_engine(uri, **kwargs)
        self._session_factory = sessionmaker(bind=self._engine)
        self.time_zone = time_zone
        return self

    def close(self) -> None:
        """Disposes the configured SQLAlchemy engine."""
        if self._engine is not None:
            self._engine.dispose()
        self._engine = None
        self._session_factory = None
        return

    @property
    def engine(self) -> Engine:
        """Returns the configured SQLAlchemy engine.

        Returns:
            Engine: The SQLAlchemy engine.

        Raises:
            RuntimeError: If the service has not been configured.
        """
        if self._engine is None:
            raise RuntimeError("SqlDbService is not configured. Call configure_sql_db() first.")
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Returns the configured SQLAlchemy sessionmaker.

        Raises:
            RuntimeError: If the service has not been configured.
        """
        if self._session_factory is None:
            raise RuntimeError("SqlDbService is not configured.")
        return self._session_factory

    def create_session(self) -> Session:
        """Creates a new SQLAlchemy session.

        Returns:
            Session: A newly created database session.
        """
        return self.session_factory()

    def create_all(self, base_class: type[DeclarativeBase]) -> None:
        """Creates all tables defined in the base class metadata.

        Args:
            base_class (type[DeclarativeBase]): The declarative base class.
        """
        base_class.metadata.create_all(self.engine, checkfirst=True)
        return


sql_db = SqlDbService()


def configure_sql_db(
    uri: str,
    engine_kwargs: Optional[Dict[str, Any]] = None,
    time_zone: tzinfo = timezone.utc,
) -> SqlDbService:
    """Configures IsoBase's default SQL database service.

    Args:
        uri (str): Database connection URI.
        engine_kwargs (dict, optional): Additional arguments for create_engine.
        time_zone (tzinfo, optional): Time zone used for model timestamps.

    Returns:
        SqlDbService: The configured default SQL service.
    """
    return sql_db.configure(
        uri=uri,
        engine_kwargs=engine_kwargs,
        time_zone=time_zone,
    )


class SqlDbModelMixin:
    """Mixin providing common CRUD and query operations for SQLAlchemy models."""

    sql_service: ClassVar[SqlDbService] = sql_db

    @classmethod
    def use_sql_service(cls, service: SqlDbService) -> None:
        """Assigns a SQL database service to this model class.

        Args:
            service (SqlDbService): Service used by this model class.
        """
        cls.sql_service = service
        return

    @classmethod
    def _as_service_timezone(cls, value: datetime) -> datetime:
        """Converts a datetime value to the model service time zone."""
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(cls.sql_service.time_zone)

    def to_dict(self) -> dict:
        """Serializes this instance into a dictionary."""
        data = {}
        for column in self.__table__.columns:
            val = getattr(self, column.name)
            if isinstance(val, datetime):
                val = self._as_service_timezone(val).isoformat()
            data[column.name] = val
        return data

    @classmethod
    def find_by_id(cls: type[T], id_value: Any, session: Optional[Session] = None) -> Optional[T]:
        """Finds a model instance by its primary key.

        Args:
            id_value (Any): The primary key value.
            session (Session, optional): Existing session to use.

        Returns:
            SqlDbModelMixin: The matching document as a model instance, if found.
        """
        if session is not None:
            return session.get(cls, id_value)

        with cls.sql_service.create_session() as s:
            return s.get(cls, id_value)

    @classmethod
    def find_one(
        cls: type[T],
        *expressions: Any,
        session: Optional[Session] = None,
        **filter_by: Any,
    ) -> Optional[T]:
        """Finds a single instance matching the provided criteria.

        Args:
            *expressions: SQLAlchemy filter expressions (e.g., Model.age > 18).
            session (Session, optional): Existing session to use.
            **filter_by: Exact match keyword arguments (e.g., name="Alice").
        """
        def _execute(s: Session) -> Optional[T]:
            stmt = select(cls).filter(*expressions).filter_by(**filter_by)
            return s.execute(stmt).scalars().first()

        if session is not None:
            return _execute(session)
        with cls.sql_service.create_session() as s:
            return _execute(s)

    @classmethod
    def find_many(
        cls: type[T],
        *expressions: Any,
        session: Optional[Session] = None,
        **filter_by: Any,
    ) -> Sequence[T]:
        """Finds multiple instances matching the provided criteria.
        
        Args:
            *expressions: SQLAlchemy filter expressions (e.g., Model.age > 18).
            session (Session, optional): Existing session to use.
            **filter_by: Exact match keyword arguments.
        """
        def _execute(s: Session) -> Sequence[T]:
            stmt = select(cls).filter(*expressions).filter_by(**filter_by)
            return s.execute(stmt).scalars().all()

        if session is not None:
            return _execute(session)
        with cls.sql_service.create_session() as s:
            return _execute(s)

    @classmethod
    def count(
        cls,
        *expressions: Any,
        session: Optional[Session] = None,
        **filter_by: Any,
    ) -> int:
        """Counts instances matching the provided criteria."""
        from sqlalchemy import func
        def _execute(s: Session) -> int:
            stmt = select(func.count()).select_from(cls).filter(*expressions).filter_by(**filter_by)
            return s.execute(stmt).scalar() or 0

        if session is not None:
            return _execute(session)
        with cls.sql_service.create_session() as s:
            return _execute(s)

    def insert(self, session: Optional[Session] = None) -> None:
        """Inserts this instance as a new record in the database."""
        now = datetime.now(tz=self.sql_service.time_zone)
        if hasattr(self, "created_time") and getattr(self, "created_time") is None:
            setattr(self, "created_time", now)
        if hasattr(self, "updated_time") and getattr(self, "updated_time") is None:
            setattr(self, "updated_time", now)

        if session is not None:
            session.add(self)
            session.flush()
        else:
            with self.sql_service.create_session() as s:
                s.add(self)
                s.commit()
                s.refresh(self)
        return

    def update(self, session: Optional[Session] = None) -> None:
        """Updates the database record corresponding to this instance."""
        now = datetime.now(tz=self.sql_service.time_zone)
        if hasattr(self, "updated_time"):
            setattr(self, "updated_time", now)

        if session is not None:
            session.add(self)
            session.flush()
        else:
            with self.sql_service.create_session() as s:
                # Merge current instance into the session state and commit
                s.merge(self)
                s.commit()
        return

    def delete(self, session: Optional[Session] = None) -> None:
        """Deletes the record corresponding to this instance."""
        if session is not None:
            session.delete(self)
            session.flush()
        else:
            with self.sql_service.create_session() as s:
                s.delete(self)
                s.commit()
        return

    @classmethod
    def delete_many(
        cls,
        *expressions: Any,
        session: Optional[Session] = None,
        **filter_by: Any,
    ) -> int:
        """Deletes multiple records matching the criteria.
        
        Note: Requires at least one expression or filter_by kwarg to prevent accidental wipe.
        """
        if not expressions and not filter_by:
            raise ValueError("At least one expression or filter_by arg is required to prevent mass deletion.")

        def _execute(s: Session) -> int:
            stmt = sa_delete(cls).filter(*expressions).filter_by(**filter_by)
            result = s.execute(stmt)
            return result.rowcount

        if session is not None:
            return _execute(session)

        with cls.sql_service.create_session() as s:
            count = _execute(s)
            s.commit()
            return count

    @classmethod
    def execute_atomic(cls, callback: Callable[..., Any], **kwargs: Any) -> Any:
        """Executes a callback within a database transaction session.

        Args:
            callback (callable): A function that accepts a ``session`` keyword argument.
            **kwargs: Additional arguments to pass to the callback.

        Returns:
            Any: The return value of the callback.
        """
        with cls.sql_service.create_session() as session:
            try:
                result = callback(session=session, **kwargs)
                session.commit()
                return result
            except Exception:
                session.rollback()
                raise
