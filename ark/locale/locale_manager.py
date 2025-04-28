#! python3
# -*- encoding: utf-8 -*-
"""
@File   : locale_manager.py
@Created: 2025/04/18 14:57
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

# Here put the import lib.
from typing import Optional
from .phone import parse_phone_number, region_code_for_number, format_e164
from .country import get_country_by_alpha2, get_language_by_alpha2


class LocaleManager:
    """Utility class to resolve locale/language from phone number and validate locale codes."""

    # Expand this map as needed
    DEFAULT_LOCALE_MAP = {
        "CN": "zh_cn",
        "HK": "zh_hk",
        "MO": "zh_hk",
        "TW": "zh_tw",
        "SG": "zh_sg",
        "CA": "en_ca",
        "DE": "de_de",
        "ES": "es_es",
        "FR": "fr_fr",
        "GB": "en_uk",
        "GR": "el_gr",
        "JP": "ja_jp",
        "KR": "ko_kr",
        "US": "en_us",
    }

    # Expand this map as needed
    COUNTRY_CODE_ALIASES = {
        "UK": "GB",
        "EL": "GR",  # 希腊 Greece
    }

    @classmethod
    def is_valid_locale(cls, locale_code: str) -> bool:
        """Checks whether a given locale code (e.g., 'en_us') is valid.

        Args:
            locale_code (str): Locale code in 'xx_yy' or 'xx-yy' format, e.g. 'zh_cn', 'en_us', etc.

        Returns:
            bool: True if valid, else False.
        """
        locale_code = locale_code.replace("-", "_")
        if "_" not in locale_code:
            return False
        lang, country = locale_code.lower().split("_", 1)
        lang_match = get_language_by_alpha2(code=lang)
        country_code = cls.COUNTRY_CODE_ALIASES.get(country.upper(), country.upper())
        country_match = get_country_by_alpha2(code=country_code)
        return lang_match is not None and country_match is not None

    @classmethod
    def resolve_locale_from_phone(cls, raw_phone: str) -> Optional[str]:
        """Resolves a language-locale code (like 'en_us') from a phone number.

        Args:
            raw_phone (str): Raw phone number string, which will be parsed as a Chinese phone number if country code is not included.

        Returns:
            str: A locale code like 'en_us', or None if not resolvable.
        """
        number = parse_phone_number(raw_phone)
        if number is None:
            return None

        region_code = region_code_for_number(number)
        locale_code = cls.DEFAULT_LOCALE_MAP.get(region_code, "en_us")

        if locale_code and cls.is_valid_locale(locale_code):
            return locale_code
        return None
