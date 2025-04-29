#! python3
# -*- encoding: utf-8 -*-
"""
Format logger output using python logging and colorlog module.

Reference: https://github.com/jyesselm/dreem/blob/main/dreem/logger.py

@File   :   logger.py
@Created:   2025/04/01 16:16
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
import os
import logging
import colorlog

class ArkError(Exception):
    pass

def init_logger(name, log_outfile=None, testing_mode=False, start=False) -> logging.Logger:
    """Initialize a logger instance to """
    log_format = (
        "[%(asctime)s " "%(name)s " "%(funcName)s] " "%(levelname)s " "%(message)s"
    )
    # bold_seq = "\033[1m"
    # colorlog_format = f"{bold_seq}" "%(log_color)s" f"{log_format}"
    colorlog_format = "%(log_color)s" f"{log_format}"
    logger = logging.getLogger(name)
    # colorlog.basicConfig(format=colorlog_format, datefmt="%H:%M")
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            colorlog_format,
            datefmt="%Y-%m-%d %H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    logger.addHandler(handler)

    if log_outfile is not None:
        if start:
            if os.path.isfile(log_outfile):
                os.remove(log_outfile)
        fileHandler = logging.FileHandler(log_outfile)
        fileHandler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(fileHandler)

    if testing_mode:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger


def log_error_and_exit(log: logging.Logger, msg):
    log.error(msg)
    raise ArkError(msg)

def str_to_log_level(s: str):
    s = s.rstrip().lstrip().lower()
    if s == "info":
        return logging.INFO
    elif s == "debug":
        return logging.DEBUG
    elif s == "warn":
        return logging.WARN
    elif s == "warning":
        return logging.WARNING
    elif s == "error":
        return logging.ERROR
    elif s == "critical":
        return logging.CRITICAL
    else:
        raise ValueError("unknown log level: {}".format(s))

LOGGER = init_logger("ARK", None, start=True)
