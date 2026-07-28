#! python3
# -*- coding: utf-8 -*-
"""This module supports database operations.

@File   : __init__.py
@Created: 2026/07/17 02:04 (UTC+08:00)
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

# Here put the import lib.
from .mongo import MongoDbModel, MongoDbService, configure_mongo_db, mongo_db
from .sql import SqlDbModelMixin, SqlDbService, configure_sql_db, sql_db


