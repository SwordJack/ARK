#! python3
# -*- coding: utf-8 -*-
"""
@File   : file_system.py
@Created: 2025/04/02 22:45
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

import shutil
import os
from os import path

from .logger import LOGGER

# region Directory.

def make_directory(directory_path: str) -> None:
    """Create a leaf directory and all intermediate ones.
    
    Args:
        directory_path (str): The path to the directory.
    """
    if not path.isdir(directory_path):
        os.makedirs(directory_path, exist_ok=True)
    else:
        LOGGER.debug(f"`{directory_path}` already exists. Nothing to make.")
    return

def is_empty_directory(dir_path: str) -> bool:
    """Is the supplied directory empty?"""
    if path.isdir(dir_path):
        return not os.listdir(dir_path)
    else:
        LOGGER.debug(f"`{dir_path}` doesn't exist.")
        return False

def remove_directory(directory_path: str, empty_only: bool = True) -> None:
    """Removes an existing directory.
    
    Args:
        directory_path (str): The path to the directory.
        empty_only (bool, optional): Indicate if the directory can only be removed when it's empty. Default True.
    """
    if path.isdir(directory_path):
        if empty_only and not is_empty_directory(directory_path):
            LOGGER.debug(f"`{directory_path}` is not empty. Nothing to remove unless setting `empty_only=False`.")
        else:
            shutil.rmtree(directory_path)
    else:
        LOGGER.debug(f"`{directory_path}` doesn't exist. Nothing to remove")
    return

# endregion

# region File.

def remove_file(filepath: str) -> None:
    """Remove a file if it exists. A warning is raised but nothing will be done if the `filepath` provided is a path to a directory.
    
    Args:
        filepath (str): The path to a file.
    """
    if (path.isdir(filepath)):
        LOGGER.warning(f"`{filepath}` is a directory. Nothing to remove.")
        return
    if path.lexists(filepath):
        os.remove(filepath)
    else:
        LOGGER.debug(f"`{filepath}` cannot be found. Nothing to remove.")
    return


def move_file(source: str, destination: str) -> str:
    """Move a file if it exists.
    
    Args:
        source (str): The current path of the file.
        destination (str): The destination path or directory of the file.

    Returns:
        destination_filepath (str): The path to the new file.
    """
    if (path.isdir(source)):
        LOGGER.warning(f"Source path `{source}` is a directory. Nothing to move.")
        return None
    if (destination.endswith("/") or destination.endswith("/.")):
        make_directory(destination)
        destination = path.join(destination, path.basename(source))
    destination_filepath = str(shutil.move(source, destination))
    return destination_filepath

def copy_file(source: str, destination: str) -> str:
    """Copy a file if it exists.
    
    Args:
        source (str): The current path of the file.
        destination (str): The destination path or directory of the file.

    Returns:
        destination_filepath (str): The path to the new file.
    """
    if (path.isdir(source)):
        LOGGER.warning(f"Source path `{source}` is a directory. Nothing to move.")
        return None
    if (destination.endswith("/") or destination.endswith("/.")):
        make_directory(destination)
        destination = path.join(destination, path.basename(source))
    destination_filepath = str(shutil.copy2(source, destination))
    return destination_filepath

# endregion
