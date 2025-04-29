#! python3
# -*- coding: utf-8 -*-
"""
@File   : test_locale_manager.py
@Created: 2025/04/24 16:15
@Author : SwordJack
@Contact: https://github.com/SwordJack/
"""

from ark import LocaleManager

class TestLocalManager:

    def test_is_valid_locale(self):
        assert LocaleManager.is_valid_locale(locale_code="zh_cn")
        assert LocaleManager.is_valid_locale(locale_code="zh-cn")
        assert LocaleManager.is_valid_locale(locale_code="en_uk")
        assert LocaleManager.is_valid_locale(locale_code="en_gb")
        assert not LocaleManager.is_valid_locale(locale_code="ee_uu")
    
    def test_resolve_locale_from_phone(self):
        assert LocaleManager.resolve_locale_from_phone(raw_phone="+1 (615) 818-0310") == "en_us"
        assert LocaleManager.resolve_locale_from_phone(raw_phone="+1 (226) 818-0310") == "en_ca"
        # assert LocaleManager.resolve_locale_from_phone("+1 (615) 111-0310") == "en_us"
        # assert LocaleManager.resolve_locale_from_phone("(615) 111-0310") == "en_us"
        assert LocaleManager.resolve_locale_from_phone("13366889900") == "zh_cn"
