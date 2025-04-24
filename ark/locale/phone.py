#! python3
# -*- coding: utf-8 -*-
"""
@File   : phone.py
@Created: 2025/04/24 01:42
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

import phonenumbers
from typing import Optional

def parse_phone_number(raw_phone: str, region_hint: str = "CN") -> Optional[phonenumbers.PhoneNumber]:
    """Parses and validates a phone number.

    Args:
        raw_phone (str): A phone number string.
        region_hint (str): Default region to assume if no country code is provided.

    Returns:
        PhoneNumber object if valid, else None.
    """
    try:
        number = phonenumbers.parse(raw_phone, region_hint)
        if phonenumbers.is_valid_number(number):
            return number
        return None
    except phonenumbers.NumberParseException:
        return None

def format_e164(number_obj: phonenumbers.PhoneNumber) -> str:
    if (number_obj is None):
        return None
    else:
        return phonenumbers.format_number(number_obj, phonenumbers.PhoneNumberFormat.E164)
