#! python3
# -*- coding: utf-8 -*-
"""
This file defines the parent class for MongoDB-backed models.

@File   : mongodb.py
@Created: 2025/04/09 16:17
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

# Here put the import lib.
from __future__ import annotations
from abc import ABC
from datetime import datetime, timezone, tzinfo
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, TypeVar
from uuid import uuid4

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import OperationFailure


T = TypeVar("T", bound="MongoDbModel")


class MongoDbService(object):
    """A small MongoDB connection holder for IsoBase library users.

    The service is intentionally framework-agnostic. Applications may configure
    it with a MongoDB URI and database name, or pass an existing ``MongoClient``.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        database_name: Optional[str] = None,
        client: Optional[MongoClient] = None,
        time_zone: tzinfo = timezone.utc,
        transactions_enabled: Optional[bool] = None,
    ) -> None:
        """Initializes the MongoDB service.

        Args:
            uri (str, optional): MongoDB connection URI.
            database_name (str, optional): Database name to use.
            client (MongoClient, optional): Existing PyMongo client.
            time_zone (tzinfo, optional): Time zone used for model timestamps.
            transactions_enabled (bool, optional): Explicit transaction support flag.
        """
        self._client = client
        self._database_name = database_name
        self._db = client[database_name] if client is not None and database_name else None
        self.time_zone = time_zone
        self.transactions_enabled = transactions_enabled

        if uri is not None:
            self.configure(
                uri=uri,
                database_name=database_name,
                time_zone=time_zone,
                transactions_enabled=transactions_enabled,
            )
        return

    def configure(
        self,
        uri: Optional[str] = None,
        database_name: Optional[str] = None,
        client: Optional[MongoClient] = None,
        time_zone: tzinfo = timezone.utc,
        transactions_enabled: Optional[bool] = None,
    ) -> MongoDbService:
        """Configures the MongoDB service.

        Args:
            uri (str, optional): MongoDB connection URI.
            database_name (str, optional): Database name to use.
            client (MongoClient, optional): Existing PyMongo client.
            time_zone (tzinfo, optional): Time zone used for model timestamps.
            transactions_enabled (bool, optional): Explicit transaction support flag.

        Returns:
            MongoDbService: This service instance.
        """
        if client is None:
            if uri is None:
                raise ValueError("Either uri or client is required to configure MongoDbService.")
            client = MongoClient(uri)

        self._client = client
        self._database_name = database_name
        self._db = client[database_name] if database_name else None
        self.time_zone = time_zone
        self.transactions_enabled = transactions_enabled
        return self

    def close(self) -> None:
        """Closes the configured MongoDB client."""
        if self._client is not None:
            self._client.close()
        self._client = None
        self._db = None
        return

    @property
    def client(self) -> MongoClient:
        """Returns the configured MongoDB client.

        Returns:
            MongoClient: The configured PyMongo client.

        Raises:
            RuntimeError: If the service has not been configured.
        """
        if self._client is None:
            raise RuntimeError("MongoDbService is not configured. Call configure_mongodb() first.")
        return self._client

    @property
    def db(self) -> Database:
        """Returns the configured MongoDB database.

        Returns:
            Database: The configured PyMongo database.

        Raises:
            RuntimeError: If the service has not been configured with a database.
        """
        if self._db is None:
            raise RuntimeError("MongoDbService is not configured with a database name.")
        return self._db


mongo_db = MongoDbService()


def configure_mongo_db(
    uri: Optional[str] = None,
    database_name: Optional[str] = None,
    client: Optional[MongoClient] = None,
    time_zone: tzinfo = timezone.utc,
    transactions_enabled: Optional[bool] = None,
) -> MongoDbService:
    """Configures IsoBase's default MongoDB service.

    Args:
        uri (str, optional): MongoDB connection URI.
        database_name (str, optional): Database name to use.
        client (MongoClient, optional): Existing PyMongo client.
        time_zone (tzinfo, optional): Time zone used for model timestamps.
        transactions_enabled (bool, optional): Explicit transaction support flag.

    Returns:
        MongoDbService: The configured default MongoDB service.
    """
    return mongo_db.configure(
        uri=uri,
        database_name=database_name,
        client=client,
        time_zone=time_zone,
        transactions_enabled=transactions_enabled,
    )


class MongoDbModel(ABC):
    """Abstract base model class with common MongoDB operations using PyMongo."""

    collection_name: ClassVar[Optional[str]] = None
    mongo_service: ClassVar[MongoDbService] = mongo_db
    _supports_transactions: ClassVar[Optional[bool]] = None

    def __init__(
        self,
        created_time: Optional[datetime] = None,
        updated_time: Optional[datetime] = None,
        **kwargs: Any,
    ) -> None:
        """Initializes instance attributes from keyword arguments.

        Args:
            created_time (datetime, optional): Creation timestamp.
            updated_time (datetime, optional): Update timestamp.
            **kwargs: Additional attributes to assign to the instance.
        """
        if created_time is not None:
            self.created_time = self._as_service_timezone(created_time)
        if updated_time is not None:
            self.updated_time = self._as_service_timezone(updated_time)

        if "_id" not in kwargs:
            kwargs["_id"] = str(uuid4())

        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)
        return

    @classmethod
    def use_mongo_service(cls, service: MongoDbService) -> None:
        """Assigns a MongoDB service to this model class.

        Args:
            service (MongoDbService): Service used by this model class.
        """
        cls.mongo_service = service
        cls._supports_transactions = None
        return

    @classmethod
    def _as_service_timezone(cls, value: datetime) -> datetime:
        """Converts a datetime value to the model service time zone.

        Args:
            value (datetime): Datetime value to convert.

        Returns:
            datetime: Time-zone-aware datetime in the configured time zone.
        """
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(cls.mongo_service.time_zone)

    @property
    def id(self) -> str:
        """Returns the string version of the document's ``_id``.

        Returns:
            str: The document's ``_id``.
        """
        return str(self._id)

    @id.setter
    def id(self, value: Any) -> None:
        """Sets the document's ``_id``.

        Args:
            value (Any): New document id.
        """
        self._id = value
        return

    def to_dict(self, to_database: bool = False) -> dict:
        """Serializes this instance into a dictionary.

        Args:
            to_database (bool, optional): Whether the instance is being serialized for MongoDB.

        Returns:
            dict: The instance's data as a dictionary.
        """
        data = self.__dict__.copy()

        if not to_database:
            data["id"] = self.id
            data.pop("_id", None)

            for key, value in data.items():
                if isinstance(value, datetime):
                    data[key] = self._as_service_timezone(value).isoformat()
        return data

    @classmethod
    def from_dict(cls: type[T], data: dict) -> T:
        """Deserializes a dictionary into a model instance.

        Args:
            data (dict): The source dictionary.

        Returns:
            MongoDbModel: An instance of the class.
        """
        parsed_data = data.copy()
        for key, value in parsed_data.items():
            if isinstance(value, str):
                try:
                    parsed_data[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass

        instance = cls(**parsed_data)
        if "_id" in parsed_data:
            setattr(instance, "_id", parsed_data["_id"])
        return instance

    @classmethod
    def _derive_collection_name(cls) -> str:
        """Derives a collection name from the class name.

        Returns:
            str: The pluralized lowercase form of the class name.
        """
        name = cls.__name__.lower()
        return name if name.endswith("s") else name + "s"


    @classmethod
    def get_collection(cls) -> Collection:
        """Gets the MongoDB collection object.

        Returns:
            Collection: The PyMongo collection for this model.
        """
        name = cls.collection_name or cls._derive_collection_name()
        return cls.mongo_service.db[name]

    @classmethod
    def create_collection(
        cls,
        capped: bool = False,
        size: Optional[int] = None,
        max: Optional[int] = None,
        validator: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Collection:
        """Explicitly creates the collection for this model.
        
        Note: MongoDB auto-creates collections on first insert. This method is 
        useful when you need to specify collection options (e.g., capped collections, 
        schema validators).

        Args:
            capped (bool, optional): To create a capped collection. Default False.
            size (int, optional): Maximum size in bytes for a capped collection.
            max (int, optional): Maximum number of documents in a capped collection.
            validator (dict, optional): JSON Schema validator rules for the collection.
            **kwargs: Additional options passed to PyMongo's create_collection.
            
        Returns:
            Collection: The created collection.
        """
        name = cls.collection_name or cls._derive_collection_name()
        
        if capped:
            kwargs["capped"] = True
        if size is not None:
            kwargs["size"] = size
        if max is not None:
            kwargs["max"] = max
        if validator is not None:
            kwargs["validator"] = validator
            
        return cls.mongo_service.db.create_collection(name, **kwargs)

    @classmethod
    def drop_collection(cls) -> None:
        """Drops the collection for this model if it exists."""
        name = cls.collection_name or cls._derive_collection_name()
        cls.mongo_service.db.drop_collection(name)

    @classmethod
    def create_index(cls, keys: Any, **kwargs: Any) -> str:
        """Creates an index on the collection.

        Args:
            keys (Any): Index specification.
            **kwargs: Additional options for index creation.

        Returns:
            str: The name of the created index.
        """
        return cls.get_collection().create_index(keys, **kwargs)

    @classmethod
    def list_indexes(cls) -> Any:
        """Lists all indexes on the collection.

        Returns:
            CommandCursor: A cursor over the index documents.
        """
        return cls.get_collection().list_indexes()

    @classmethod
    def drop_index(cls, index_name: str) -> Any:
        """Drops an index by name.

        Args:
            index_name (str): The name of the index to drop.

        Returns:
            Any: The PyMongo drop index result.
        """
        return cls.get_collection().drop_index(index_name)

    @classmethod
    def find_by_id(cls: type[T], id: Any) -> Optional[T]:
        """Finds a document by its id.

        Args:
            id (Any): The ID of the document.

        Returns:
            MongoDbModel: The matching document as a model instance, if found.
        """
        try:
            oid = ObjectId(id) if not isinstance(id, ObjectId) else id
            data = cls.get_collection().find_one({"_id": oid})
        except Exception:
            data = cls.get_collection().find_one({"_id": id})
        return cls.from_dict(data) if data else None

    @classmethod
    def find_one(cls: type[T], filter: dict) -> Optional[T]:
        """Finds a single document by filter.

        Args:
            filter (dict): A MongoDB filter query.

        Returns:
            MongoDbModel: The matching document as a model instance, if found.
        """
        data = cls.get_collection().find_one(filter)
        return cls.from_dict(data) if data else None

    @classmethod
    def find_many(
        cls: type[T],
        filter: Optional[dict] = None,
        sort: Optional[List[Tuple[str, int]]] = None,
        skip: int = 0,
        limit: int = 0,
    ) -> List[T]:
        """Finds multiple documents by filter.

        Args:
            filter (dict, optional): A MongoDB filter query.
            sort (List[Tuple[str, int]], optional): A list of (key, direction) pairs.
            skip (int, optional): Number of documents to skip.
            limit (int, optional): Maximum number of documents to return.

        Returns:
            list: A list of model instances.
        """
        cursor = cls.get_collection().find(filter or {})
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)
        return [cls.from_dict(doc) for doc in cursor]

    @classmethod
    def count(cls, filter: Optional[dict] = None) -> int:
        """Counts documents by filter.

        Args:
            filter (dict, optional): A MongoDB filter query.

        Returns:
            int: The number of matching documents.
        """
        return cls.get_collection().count_documents(filter or {})

    @classmethod
    def aggregate(cls, pipeline: List[Dict]) -> list:
        """Performs an aggregation query on the model's collection.

        Args:
            pipeline (List[Dict]): A list of aggregation pipeline stages.

        Returns:
            list: The result of the aggregation pipeline.
        """
        return list(cls.get_collection().aggregate(pipeline))

    def insert(self, session: Any = None) -> Any:
        """Inserts this instance as a new document in the database.

        Args:
            session (ClientSession, optional): A MongoDB session for transactions.

        Returns:
            Any: The inserted document id.
        """
        self.created_time = datetime.now(tz=self.mongo_service.time_zone)
        self.updated_time = self.created_time
        data = self.to_dict(to_database=True)
        custom_id = data.pop("_id", None)
        data.pop("id", None)
        if custom_id:
            data["_id"] = custom_id
        result = self.get_collection().insert_one(data, session=session)
        if not custom_id:
            self._id = result.inserted_id
        return result.inserted_id

    def update(self, session: Any = None) -> int:
        """Updates the document corresponding to this instance.

        Args:
            session (ClientSession, optional): A MongoDB session for transactions.

        Returns:
            int: The number of documents modified.
        """
        if not hasattr(self, "_id"):
            raise ValueError("This instance must have _id to update.")
        self.updated_time = datetime.now(tz=self.mongo_service.time_zone)
        data = self.to_dict(to_database=True)
        data.pop("_id", None)
        result = self.get_collection().update_one({"_id": self._id}, {"$set": data}, session=session)
        return result.modified_count

    def delete(self, session: Any = None) -> int:
        """Deletes the document corresponding to this instance.

        Args:
            session (ClientSession, optional): A MongoDB session for transactions.

        Returns:
            int: The number of documents deleted.
        """
        if not hasattr(self, "_id"):
            raise ValueError("This instance must have _id to delete.")
        result = self.get_collection().delete_one({"_id": self._id}, session=session)
        return result.deleted_count

    @classmethod
    def delete_many(cls, filter: dict, session: Any = None) -> int:
        """Deletes multiple documents matching the filter.

        Args:
            filter (dict): A non-empty MongoDB filter query document.
            session (ClientSession, optional): A MongoDB session for transactions.

        Returns:
            int: The number of documents deleted.
        """
        if not isinstance(filter, dict) or not filter:
            raise ValueError("A non-empty filter dictionary is required for delete_many to prevent accidental mass deletion.")

        result = cls.get_collection().delete_many(filter, session=session)
        return result.deleted_count

    def update_attributes(
        self,
        mapper: Dict[str, Any],
        editable_attrs: Optional[List[str]] = None,
        update_timestamp: bool = True,
        timestamp_field: str = "updated_time",
        skip_none_or_empty: bool = False,
        add_nonexistent_attrs: bool = False,
        session: Any = None,
    ) -> Tuple[List[str], List[str], List[str], str]:
        """Updates selected fields of this instance and syncs with MongoDB.

        Args:
            mapper (Dict[str, Any]): Dictionary of field names and new values.
            editable_attrs (List[str], optional): List of allowed fields to edit.
            update_timestamp (bool, optional): Whether to update a timestamp field.
            timestamp_field (str, optional): Name of the timestamp field.
            skip_none_or_empty (bool, optional): Whether None or empty values should be skipped.
            add_nonexistent_attrs (bool, optional): Whether nonexistent attributes can be declared.
            session (ClientSession, optional): A MongoDB session for transactions.

        Returns:
            Tuple[List[str], List[str], List[str], str]: Updated, blocked, nonexistent fields, and summary.
        """
        editable_attrs = editable_attrs or []
        updated_attrs = []
        blocked_attrs = []
        nonexistent_attrs = []
        update_data = {}

        for field_name, field_value in mapper.items():
            if skip_none_or_empty and not field_name:
                continue
            has_attr = hasattr(self, field_name)
            if has_attr or add_nonexistent_attrs:
                if field_name in editable_attrs or not editable_attrs:
                    if add_nonexistent_attrs or (has_attr and getattr(self, field_name) != field_value):
                        if field_value or not skip_none_or_empty:
                            setattr(self, field_name, field_value)
                            update_data[field_name] = field_value
                else:
                    blocked_attrs.append(field_name)
            else:
                nonexistent_attrs.append(field_name)

        if update_data:
            if update_timestamp:
                now = datetime.now(self.mongo_service.time_zone)
                setattr(self, timestamp_field, now)
                update_data[timestamp_field] = now

            try:
                result = self.get_collection().update_one(
                    {"_id": self._id}, {"$set": update_data}, session=session
                )
                if result.modified_count > 0:
                    updated_attrs = list(update_data.keys())
            except Exception as e:
                raise RuntimeError(f"Update failed: {e}") from e

        message_parts = []
        if updated_attrs:
            message_parts.append(f"Updated: {', '.join(updated_attrs)}.")
        if blocked_attrs:
            message_parts.append(f"Blocked: {', '.join(blocked_attrs)}.")
        if nonexistent_attrs:
            message_parts.append(f"Nonexistent: {', '.join(nonexistent_attrs)}.")
        return updated_attrs, blocked_attrs, nonexistent_attrs, " ".join(message_parts)

    def unset_attributes(
        self,
        field_names: List[str],
        protected_fields: Optional[List[str]] = None,
        session: Any = None,
    ) -> int:
        """Unsets fields from the MongoDB document and this model instance.

        Args:
            field_names (List[str]): A list of field names to delete from the document.
            protected_fields (List[str], optional): Additional fields that should not be deleted.
            session (ClientSession, optional): A MongoDB session for transactions.

        Returns:
            int: The number of documents modified.
        """
        if not hasattr(self, "_id"):
            raise ValueError("This instance must have _id to unset attributes.")

        all_protected = {"_id"}
        if protected_fields:
            all_protected.update(protected_fields)

        fields_to_unset = [field for field in field_names if field not in all_protected]
        if not fields_to_unset:
            return 0

        unset_payload = {"$unset": {field: "" for field in fields_to_unset}}
        result = self.get_collection().update_one({"_id": self._id}, unset_payload, session=session)

        if result.modified_count > 0:
            for field in fields_to_unset:
                if hasattr(self, field):
                    delattr(self, field)
        return result.modified_count

    def unset_attribute(
        self,
        field_name: str,
        protected_fields: Optional[List[str]] = None,
        session: Any = None,
    ) -> int:
        """Unsets a single field from the MongoDB document.

        Args:
            field_name (str): The name of the field to delete.
            protected_fields (List[str], optional): Additional fields that should not be deleted.
            session (ClientSession, optional): A MongoDB session for transactions.

        Returns:
            int: The number of documents modified.
        """
        return self.unset_attributes([field_name], protected_fields=protected_fields, session=session)

    def inc_attributes(self, mapper: Dict[str, int], session: Any = None) -> int:
        """Atomically increments one or more fields.

        Args:
            mapper (Dict[str, int]): Dictionary of field names and increment values.
            session (ClientSession, optional): A MongoDB session for transactions.

        Returns:
            int: The number of documents modified.
        """
        if not hasattr(self, "_id"):
            raise ValueError("This instance must have _id to increment attributes.")
        if not mapper:
            return 0

        try:
            result = self.get_collection().update_one({"_id": self._id}, {"$inc": mapper}, session=session)
            if result.modified_count > 0:
                for key, value in mapper.items():
                    current_val = getattr(self, key, 0)
                    if not isinstance(current_val, (int, float)):
                        current_val = 0
                    setattr(self, key, current_val + value)
            return result.modified_count
        except Exception as e:
            raise RuntimeError(f"Increment failed: {e}") from e

    @classmethod
    def execute_transaction(cls, callback: Callable[..., Any], **kwargs: Any) -> Any:
        """Executes a callback within a MongoDB transaction if supported.

        Args:
            callback (callable): A function that accepts a ``session`` keyword argument.
            **kwargs: Additional arguments to pass to the callback.

        Returns:
            Any: The return value of the callback.
        """
        if cls._supports_transactions is None:
            cls.init_transaction_support()

        if cls._supports_transactions is False:
            return callback(session=None, **kwargs)

        try:
            with cls.mongo_service.client.start_session() as session:
                with session.start_transaction():
                    return callback(session=session, **kwargs)
        except OperationFailure as e:
            if e.code == 20:
                cls._supports_transactions = False
                return callback(session=None, **kwargs)
            raise e

    @classmethod
    def init_transaction_support(cls) -> Optional[bool]:
        """Initializes and returns whether the configured MongoDB supports transactions.

        Returns:
            bool: Whether transactions are supported, or the explicitly configured value.
        """
        if cls.mongo_service.transactions_enabled is not None:
            cls._supports_transactions = cls.mongo_service.transactions_enabled
            return cls._supports_transactions
        if cls._supports_transactions is not None:
            return cls._supports_transactions

        try:
            if cls.mongo_service.client.topology_description.topology_type_name == "Unknown":
                cls.mongo_service.client.admin.command("ping")

            topology = cls.mongo_service.client.topology_description.topology_type_name
            cls._supports_transactions = topology != "Single"
        except Exception:
            cls._supports_transactions = True
        return cls._supports_transactions

    def synchronize_from(
        self,
        source_obj: MongoDbModel,
        synchronize_attributes: Optional[List[str]] = None,
        skip_none_or_empty: bool = False,
    ) -> int:
        """Synchronizes specific attributes from another model to current model.

        Args:
            source_obj (MongoDbModel): The object to synchronize from.
            synchronize_attributes (List[str], optional): The attributes to synchronize.
            skip_none_or_empty (bool, optional): Whether None or empty values should be skipped.

        Returns:
            int: The number of documents modified.
        """
        modified_count = 0
        attr_mapper = {}
        for attr_key in synchronize_attributes or []:
            if hasattr(self, attr_key):
                attr_mapper[attr_key] = getattr(source_obj, attr_key)
        updated_fields, _, _, _ = self.update_attributes(
            mapper=attr_mapper,
            skip_none_or_empty=skip_none_or_empty,
        )
        if updated_fields:
            modified_count += 1
        return modified_count
