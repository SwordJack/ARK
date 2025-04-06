#! python3
# -*- coding: utf-8 -*-
"""
@File   : pseudo_api.py
@Created: 2025/04/04 21:40
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""
from datetime import datetime
from random import randint

from ark import LOGGER
from ark.workflow import UNIT_API_MAPPER

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
        brands = ["ZHANG San", "LI Si", "WANG Wu", "MA Liu"]
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