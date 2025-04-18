#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   locale_resolver.py
@Created:   2025/04/18 14:57
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
import phonenumbers
import pycountry
from typing import Optional


class LocaleResolver:
    """Utility class to resolve locale/language from phone number and validate locale codes."""

    # You can expand this map as needed
    DEFAULT_LOCALE_MAP = {
        "US": "en_us",
        "GB": "en_uk",
        "CN": "zh_cn",
        "TW": "zh_tw",
        "HK": "zh_hk",
        "JP": "ja_jp",
        "KR": "ko_kr",
        "FR": "fr_fr",
        "DE": "de_de",
        "ES": "es_es",
    }

    @staticmethod
    def parse_phone_number(raw_phone: str, region_hint: str = "US") -> Optional[phonenumbers.PhoneNumber]:
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

    @classmethod
    def resolve_locale_from_phone(cls, raw_phone: str) -> Optional[str]:
        """Resolves a language-locale code (like 'en_us') from a phone number.

        Args:
            raw_phone (str): Raw phone number string.

        Returns:
            str: A locale code like 'en_us', or None if not resolvable.
        """
        number = cls.parse_phone_number(raw_phone)
        if number is None:
            return None

        country_code = phonenumbers.region_code_for_number(number)  # e.g., "CN", "US"
        locale_code = cls.DEFAULT_LOCALE_MAP.get(country_code)

        if locale_code and cls.is_valid_locale(locale_code):
            return locale_code
        return None

    @staticmethod
    def is_valid_locale(locale_code: str) -> bool:
        """Checks whether a given locale code (e.g., 'en_us') is valid.

        Args:
            locale_code (str): Locale code in 'xx_yy' format.

        Returns:
            bool: True if valid, else False.
        """
        if "_" not in locale_code:
            return False
        lang, country = locale_code.lower().split("_", 1)
        lang_match = pycountry.languages.get(alpha_2=lang)
        country_match = pycountry.countries.get(alpha_2=country.upper())
        return lang_match is not None and country_match is not None
