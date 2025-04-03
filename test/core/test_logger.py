#! python3
# -*- encoding: utf-8 -*-
"""
@File   : test_logger.py
@Created: 2025/04/01 16:17
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""
from ark import LOGGER

# Here put the import lib.

def test_logger():
    LOGGER.info("Test ARK logger info.")
    LOGGER.warning("Test ARK logger warning.")
    LOGGER.error("Test ARK logger error.")
    LOGGER.exception("Test ARK logger exception.")
    LOGGER.critical("Test ARK logger critical.")
