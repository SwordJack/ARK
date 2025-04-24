#! python3
# -*- coding: utf-8 -*-
"""
@File   : country.py
@Created: 2025/04/23 15:48
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from typing import Optional
import pycountry
from pycountry.db import Country

def get_country_by_alpha2(code: str) -> Optional[Country]:
    return pycountry.countries.get(alpha_2=code)

def get_language_by_alpha2(code: str) -> Optional["pycountry.db.Language"]:
    return pycountry.languages.get(alpha_2=code)
