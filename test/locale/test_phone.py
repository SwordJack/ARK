#! python3
# -*- coding: utf-8 -*-
"""
@File   : test_phone.py
@Created: 2025/04/24 03:14
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from ark.locale.phone import (
    parse_phone_number,
    format_e164,
    format_international,
    format_national,
    format_rfc3966,
)

def test_parse_phone_number():
    phone_obj_1 = parse_phone_number("13524678980")
    assert phone_obj_1.country_code == 86
    phone_obj_2 = parse_phone_number("+13106159797")
    assert phone_obj_2.country_code == 1
    phone_obj_2 = parse_phone_number("+1 (310) 615-9797")
    assert phone_obj_2.country_code == 1
    phone_obj_3 = parse_phone_number("135246789830")
    assert phone_obj_3 is None

def test_format():
    phone_obj_1 = parse_phone_number("13524678980")
    assert " " not in format_e164(phone_obj_1)
    assert " " in format_international(phone_obj_1) and "+" in format_international(phone_obj_1)
    assert " " in format_national(phone_obj_1) and "+" not in format_national(phone_obj_1)
    assert " " not in format_rfc3966(phone_obj_1) and "-" in format_rfc3966(phone_obj_1)
    phone_obj_2 = parse_phone_number("02128923111")
    assert " " not in format_e164(phone_obj_2)
