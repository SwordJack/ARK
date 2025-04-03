#! python3
# -*- coding: utf-8 -*-
"""
@File   : test_file_system.py
@Created: 2025/04/02 22:53
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from os import path
from ark.core.file_system import (
    make_directory,
    remove_directory,
    copy_file,
    move_file,
    remove_file,
)

current_file_directory = path.dirname(__file__)
data_directory = path.join(current_file_directory, "data")
work_directory = path.join(current_file_directory, "work")

def test_make_and_remove_directory():
    remove_directory(work_directory)
    make_directory(work_directory)
    assert path.isdir(work_directory)
    remove_directory(work_directory)
    assert not path.exists(work_directory)

def test_copy_move_and_remove_file():
    test_file = path.join(data_directory, "test_file.md")
    dest_1 = path.join(work_directory, ".")
    dest_1 = copy_file(test_file, dest_1)
    assert path.isfile(dest_1)

    dest_2 = move_file(dest_1, path.join(work_directory, "test_file_move.md"))
    assert path.isfile(dest_2)
    assert not path.exists(dest_1)

    remove_file(dest_2)
    assert not path.exists(dest_2)
    remove_directory(work_directory)
    assert not path.exists(work_directory)
    