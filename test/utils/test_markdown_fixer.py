#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   test_markdown_fixer.py
@Created:   2026/08/04 17:33 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

import pytest
from isobase.utils.markdown_fixer import MarkdownFixer


# ======================== normalize – basic ===================================

def test_normalize_empty():
    assert MarkdownFixer.normalize("") == ""


def test_normalize_line_endings():
    text = "line1\r\nline2\rline3"
    assert MarkdownFixer.normalize(text) == "line1\nline2\nline3\n"


def test_normalize_bom_removal():
    text = "﻿line1\nline2"
    assert MarkdownFixer.normalize(text) == "line1\nline2\n"


# ======================== normalize – trailing whitespace =====================

def test_normalize_trailing_spaces():
    text = "hello   \nworld  \t  \n"
    assert MarkdownFixer.normalize(text) == "hello\nworld\n"


def test_normalize_trailing_tabs():
    text = "text\t\t\n"
    assert MarkdownFixer.normalize(text) == "text\n"


def test_normalize_no_trailing_whitespace_idempotent():
    text = "clean line\nanother line\n"
    result = MarkdownFixer.normalize(text)
    assert result == "clean line\nanother line\n"
    # idempotent
    assert MarkdownFixer.normalize(result) == result


# ======================== normalize – blank line collapse =====================

def test_normalize_blank_line_collapse():
    text = "a\n\n\n\nb\n"
    assert MarkdownFixer.normalize(text) == "a\n\nb\n"


def test_normalize_single_blank_line_preserved():
    text = "a\n\nb\n"
    assert MarkdownFixer.normalize(text) == "a\n\nb\n"


def test_normalize_no_blank_lines_preserved():
    text = "a\nb\nc\n"
    assert MarkdownFixer.normalize(text) == "a\nb\nc\n"


# ======================== normalize – EOF single newline ======================

def test_normalize_eof_single_newline():
    text = "line1\nline2"
    assert MarkdownFixer.normalize(text) == "line1\nline2\n"


def test_normalize_eof_many_newlines():
    text = "line1\n\n\n\n"
    assert MarkdownFixer.normalize(text) == "line1\n"


# ======================== normalize – ATX heading spacing =====================

def test_normalize_heading_missing_space():
    text = "##Heading"
    assert MarkdownFixer.normalize(text) == "## Heading\n"


def test_normalize_heading_has_space_unchanged():
    text = "### Already spaced\n"
    assert MarkdownFixer.normalize(text) == "### Already spaced\n"


def test_normalize_heading_closing_hash_unchanged():
    text = "## Heading ##\n"
    assert MarkdownFixer.normalize(text) == "## Heading ##\n"


def test_normalize_setext_heading_unchanged():
    text = "Title\n=====\n"
    assert MarkdownFixer.normalize(text) == "Title\n=====\n"


# ======================== normalize – unordered list markers ==================

def test_normalize_list_asterisk_to_dash():
    text = "* item\n"
    assert MarkdownFixer.normalize(text) == "- item\n"


def test_normalize_list_plus_to_dash():
    text = "+ item\n"
    assert MarkdownFixer.normalize(text) == "- item\n"


def test_normalize_list_dash_unchanged():
    text = "- item\n"
    assert MarkdownFixer.normalize(text) == "- item\n"


def test_normalize_list_nested_indented():
    text = "  * nested\n    + deep\n"
    assert MarkdownFixer.normalize(text) == "  - nested\n    - deep\n"


# ======================== normalize – emphasis (bold) =========================

def test_normalize_underscore_bold_to_asterisk():
    text = "this is __bold__ text\n"
    assert MarkdownFixer.normalize(text) == "this is **bold** text\n"


def test_normalize_asterisk_bold_unchanged():
    text = "already **bold**\n"
    assert MarkdownFixer.normalize(text) == "already **bold**\n"


def test_normalize_underscore_bold_must_have_word_boundary():
    """``variable __init__ method`` has spaces on both sides of ``__``,
    so the double-underscore IS a valid left/right-flanking delimiter
    per CommonMark — it converts to asterisks correctly."""
    text = "variable __init__ method\n"
    assert MarkdownFixer.normalize(text) == "variable **init** method\n"


# ======================== normalize – emphasis (italic) =======================

def test_normalize_underscore_italic_to_asterisk():
    text = "some _italic_ word\n"
    assert MarkdownFixer.normalize(text) == "some *italic* word\n"


def test_normalize_asterisk_italic_unchanged():
    text = "already *italic*\n"
    assert MarkdownFixer.normalize(text) == "already *italic*\n"


def test_normalize_underscore_in_identifiers_unchanged():
    text = "file_name_not_italic\n"
    assert MarkdownFixer.normalize(text) == "file_name_not_italic\n"


# ======================== normalize – fenced code blocks ======================

def test_normalize_tilde_fence_to_backtick():
    text = "~~~\ncode\n~~~\n"
    assert MarkdownFixer.normalize(text) == "```\ncode\n```\n"


def test_normalize_tilde_fence_with_language():
    text = "~~~JS\ncode\n~~~\n"
    assert MarkdownFixer.normalize(text) == "```js\ncode\n```\n"


def test_normalize_backtick_fence_unchanged():
    text = "```python\ncode\n```\n"
    assert MarkdownFixer.normalize(text) == "```python\ncode\n```\n"


def test_normalize_fence_language_lowercase():
    text = "```JSON\n{}\n```\n"
    assert MarkdownFixer.normalize(text) == "```json\n{}\n```\n"


# ======================== normalize – compound scenarios ======================

def test_normalize_compound():
    text = "﻿___BOLD___  \r\n\r\n\r\n* item1\r\n+ item2\r\n```JSON\r\n{}\r\n~~~"
    expected = "___BOLD___\n\n- item1\n- item2\n```json\n{}\n```\n"
    assert MarkdownFixer.normalize(text) == expected


# ======================== extract_front_matter ================================

def test_extract_front_matter_no_front_matter():
    text = "# Document Title\n\nThis is a standard markdown document."
    meta, body = MarkdownFixer.extract_front_matter(text)
    assert meta == {}
    assert body == "# Document Title\n\nThis is a standard markdown document.\n"


def test_extract_front_matter_valid():
    text = """---
title: "Project Milestone"
version: 1.0
tags:
  - doc
  - milestone
---
# Document Title

This is a standard markdown document."""

    meta, body = MarkdownFixer.extract_front_matter(text)
    assert meta == {
        "title": "Project Milestone",
        "version": 1.0,
        "tags": ["doc", "milestone"],
    }
    assert body == "# Document Title\n\nThis is a standard markdown document.\n"


def test_extract_front_matter_empty_front_matter():
    text = """---
---
# Document Title

This is a standard markdown document."""

    meta, body = MarkdownFixer.extract_front_matter(text)
    assert meta == {}
    assert body == "# Document Title\n\nThis is a standard markdown document.\n"


def test_extract_front_matter_whitespace_and_bom():
    text = "﻿---\ntitle: Test\n---\n\n# Title"
    meta, body = MarkdownFixer.extract_front_matter(text)
    assert meta == {"title": "Test"}
    assert body == "\n# Title\n"


def test_extract_front_matter_invalid_yaml():
    text = """---
invalid_yaml: [this
---
# Title"""
    with pytest.raises(ValueError, match="Failed to parse front matter YAML"):
        MarkdownFixer.extract_front_matter(text)


def test_extract_front_matter_not_dict():
    text = """---
- item1
- item2
---
# Title"""
    with pytest.raises(ValueError, match="Front matter is not a valid YAML dictionary"):
        MarkdownFixer.extract_front_matter(text)
