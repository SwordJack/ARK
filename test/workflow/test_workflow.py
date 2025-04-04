#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   test_workflow.py
@Created:   2025/04/01 16:18
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
from os import path
from datetime import datetime
from random import randint
import pytest

from ark import LOGGER
from ark.workflow import ExecutionEntity, WorkFlow, WorkUnit, GeneralWorkUnit
from ark.workflow import UNIT_API_MAPPER, StatusCode

current_file_directory = path.dirname(__file__)
DATA_DIR = path.join(current_file_directory, "data")
WORK_DIR = path.join(current_file_directory, "work")

class PseudoAPI():
    """This class acts as a container to save some static pseudo SCIENCE APIs for test."""

    @classmethod
    def test_kwargs(cls, x: str, **kwargs):
        '''This function is to test kwargs inspection only.'''
        result = x
        for key, value in kwargs.items():
            result += f'\n {key} = {value}'
        return result
    
    @classmethod
    def union_values(cls, **kwargs) -> str:
        """This function prints out all the keyword argument values and return the printed value."""
        unioned_value = list(map(str, kwargs.values()))
        LOGGER.info(f"Unioned Value is: {unioned_value}")
        return unioned_value
    
    @classmethod
    def print_value(cls, value) -> None:
        print(value)
        return
    
    @classmethod
    def checkpoint_fruit_producer(cls, fruit_name: str) -> None:
        """This function produce input value for checkpoint functions.
        
        Args:
            fruit_name (str): The name of the fruit you want to produce.
        
        Returns:
            A generated string value containing the `fruit_name` you enter.
        """
        brands = ["Zhang, San", "Li, Si", "Wang, Wu", "Ma, Liu"]
        current_time = datetime.now()
        result = f"{brands[randint(0, len(brands)-1)]}'s {fruit_name}, produced at {current_time.strftime('%Y-%m-%d_%H:%M')}."
        return result
        
    @classmethod
    def checkpoint_apple_eater_error(cls, test_str: str) -> None:
        """This function act as a checkpoint to test if the reload (continue computing) functions normally.
        
        Args:
            test_str (str): A string value as input for checkpoint.

        Raises:
            ValueError: If the `test_str` does not contain `apple`.
        """
        LOGGER.info(f"I received {test_str}...")
        if ("apple" in test_str):
            return
        else:
            raise ValueError("I need apples!")

UNIT_API_MAPPER.update({
    "test_kwargs": PseudoAPI.test_kwargs,
    "print": PseudoAPI.print_value,
    "union_values": PseudoAPI.union_values,
    "checkpoint_producer": PseudoAPI.checkpoint_fruit_producer,
    "checkpoint_error": PseudoAPI.checkpoint_apple_eater_error
})

def test_workunit_checkpoint_producer(caplog):
    """A test for workunit."""
    fruit = "apple"
    unit_dict = {
        "api" : "checkpoint_producer",
        "store_as" : "producer",
        "args" : {
            "fruit_name" : fruit,
        }
    }
    workunit = WorkUnit.from_dict(unit_dict=unit_dict, debug=True)
    key, return_value = workunit.execute()
    assert key == "producer"
    assert fruit in return_value

def test_workunit_self_inspection_unrecognized_api(caplog):
    '''A test for unrecognized API.'''
    unit_dict = {
        "api" : "missing_api",
        "store_as" : "missing",
        "args" : {
            "url" : "https://www.linux.org/",
        }
    }
    with pytest.raises(KeyError) as e:    # The KeyError is expected to be raised.
        workunit = WorkUnit.from_dict(unit_dict=unit_dict, debug=True)
        assert workunit.status == StatusCode.FAILED_INITIALIZATION
    assert 'cannot be mapped' in caplog.text.lower()

def test_workunit_self_inspection_missing_args(caplog):
    '''A test for missing argument(s).'''
    unit_dict = {
        "api" : "checkpoint_producer",
        "store_as" : "producer",
        "args" : {
            "url" : "https://www.linux.org/",
        }
    }
    with pytest.raises(ValueError) as e:    # The ValueError is expected to be raised.
        workunit = WorkUnit.from_dict(unit_dict=unit_dict, debug=True)
        assert workunit.status == StatusCode.FAILED_INITIALIZATION
    assert 'missing required argument' in caplog.text.lower()