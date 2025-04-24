#! python3
# -*- coding: utf-8 -*-
"""
@File   : test_phone.py
@Created: 2025/04/24 03:14
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from ark import phone

def test_parse_phone_number():
    phone_obj_1 = phone.parse_phone_number("13524678980")
    assert phone_obj_1.country_code == 86
    phone_obj_2 = phone.parse_phone_number("+13106159797")
    assert phone_obj_2.country_code == 1
    phone_obj_2 = phone.parse_phone_number("+1 310 615 9797")
    assert phone_obj_2.country_code == 1
