#! python3
# -*- coding: utf-8 -*-
"""
@File   : test_country.py
@Created: 2025/04/24 02:05
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from ark.locale.country import (
    get_country_by_alpha2,
    get_language_by_alpha2,
)

def test_get_country():
    country_1 = get_country_by_alpha2("cn")
    country_2 = get_country_by_alpha2("CN")
    assert country_1.name == "China"
    assert country_1 == country_2
    country_3 = get_country_by_alpha2("us")
    assert country_1 != country_3
    country_4 = get_country_by_alpha2("gb")
    country_5 = get_country_by_alpha2("uk")
    assert country_4.name == "United Kingdom"
    assert country_5 is None

def test_get_language():
    lang_1 = get_language_by_alpha2("zh")
    lang_2 = get_language_by_alpha2("zh")
    assert lang_1.name == "Chinese"
    assert lang_1 == lang_2
    lang_3 = get_language_by_alpha2("en")
    assert lang_3.name == "English"
    lang_4 = get_language_by_alpha2("sss")
    assert lang_4 is None
