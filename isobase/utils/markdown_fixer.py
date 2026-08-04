#! python3
# -*- encoding: utf-8 -*-
"""
@File   :   markdown_fixer.py
@Created:   2026/08/04 17:45 (UTC+08:00)
@Author :   SwordJack
@Contact:   https://github.com/SwordJack/
"""

# Here put the import lib.
import re
from typing import Any, Dict, Tuple

import yaml


class MarkdownFixer:
    """Utility methods for normalizing Markdown text before chunking.

    Formatting rules are modelled after Prettier's Markdown defaults:

    * Remove BOM and trailing whitespace (including tabs) from every line.
    * Collapse 3+ consecutive blank lines into a single blank line; ensure
        exactly one trailing newline at EOF.
    * ATX headings: insert a space after the closing '#' if omitted
        (``##Heading`` → ``## Heading``).
    * Unordered list markers: unify ``*`` / ``+`` → ``-``.
    * Emphasis markers: unify ``_italic_`` → ``*italic*`` and
        ``__bold__`` → ``**bold**`` (word-boundary aware to avoid
        mangling snake_case identifiers).
    * Fenced code blocks: normalise ``~~~`` → ``` ``` ``` and lowercase
        the info-string (language tag) on opening fences.
    """

    # ------ precompiled regexes ------------------------------------------------

    _ATX_MISSING_SPACE_RE = re.compile(r"^(#{1,6})([^ \n#])", flags=re.M)
    _UNORDERED_LIST_MARKER_RE = re.compile(r"^(\s*)[*+](\s)", flags=re.M)

    # Bold: __text__  (word-boundary: no leading/trailing _)
    _UNDERSCORE_BOLD_RE = re.compile(r"(?<![_\w])__([^_\n]+)__(?![_\w])")

    # Italic: _text_  (word-boundary: not preceded/followed by a word char or _)
    _UNDERSCORE_ITALIC_RE = re.compile(r"(?<![_\w])_([^_\n]+)_(?![_\w])")

    _TILDE_FENCE_RE = re.compile(r"^(```+|~~~+)", flags=re.M)

    _TRIM_LINE_RE = re.compile(r"[ \t]+$", flags=re.M)
    _COLLAPSE_BLANKS_RE = re.compile(r"\n{3,}")

    # ======================== public API ========================================

    @classmethod
    def normalize(cls, text: str) -> str:
        """Normalize markdown text to a canonical form (Prettier-style)."""
        if not text:
            return ""

        # 1.  Strip BOM and unify line endings (CRLF / CR → LF).
        result = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")

        # 2.  Trim trailing whitespace (spaces + tabs) from every line.
        result = cls._TRIM_LINE_RE.sub("", result)

        # 3.  Collapse 3+ consecutive blank lines into a single blank line.
        result = cls._COLLAPSE_BLANKS_RE.sub("\n\n", result)

        # 4.  Exactly one trailing newline at EOF.
        result = result.rstrip("\n") + "\n"

        # 5.  ATX heading spacing: "###Text" → "### Text".
        result = cls._ATX_MISSING_SPACE_RE.sub(r"\1 \2", result)

        # 6.  Unordered list markers: * / + → -
        result = cls._UNORDERED_LIST_MARKER_RE.sub(r"\1-\2", result)

        # 7.  Emphasis: __bold__ → **bold** (before italic to avoid
        #     double-processing).
        result = cls._UNDERSCORE_BOLD_RE.sub(r"**\1**", result)
        result = cls._UNDERSCORE_ITALIC_RE.sub(r"*\1*", result)

        # 8.  Fenced code blocks: ~~~ → ```, lowercase info-string.
        result = cls._normalize_fences(result)

        return result

    @classmethod
    def is_fence_line(cls, line: str) -> bool:
        """Return whether a line starts a fenced code block delimiter."""
        return cls._TILDE_FENCE_RE.match(line) is not None

    @classmethod
    def _normalize_fences(cls, text: str) -> str:
        """Convert tildes to backticks and lowercase info-strings on opening fences."""
        lines = text.split("\n")
        output: list[str] = []
        in_fence = False

        for line in lines:
            m = cls._TILDE_FENCE_RE.match(line)
            if not m:
                output.append(line)
                continue

            fence = m.group(1)
            if not in_fence:
                # Opening fence: normalise marker + lowercase info-string.
                rest = line[m.end():].strip()
                in_fence = True

                # Pick the longer of original backtick count vs 3 (minimum).
                tick_count = max(3, len(fence)) if fence[0] == "`" else 3
                marker = "`" * tick_count
                output.append(marker + (rest.lower() if rest else ""))
                # If the original used tildes with no info-string we keep the
                # "in_fence" flag; if closing a tilde fence (which is also a
                # fence line) we'll flip on the next match.
            else:
                # Possible closing fence — use same tick width as last opener.
                in_fence = False
                rest = line[m.end():].strip()
                # Conservatively reconstruct a backtick fence; the original
                # opening width has already been "lost" so we use 3.
                output.append("`" * 3 + rest)

        return "\n".join(output)

    @classmethod
    def extract_front_matter(cls, text: str) -> Tuple[Dict[str, Any], str]:
        """Extract YAML front matter if present; return (metadata, body)."""
        normalized = cls.normalize(text)
        lines = normalized.split("\n")

        if not lines or lines[0].strip() != "---":
            return {}, normalized

        end_index = None
        for index, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_index = index
                break

        if end_index is None:
            return {}, normalized

        front_matter_text = "\n".join(lines[1:end_index])
        body = "\n".join(lines[end_index + 1:])

        if not front_matter_text.strip():
            return {}, body

        try:
            metadata = yaml.safe_load(front_matter_text)
        except yaml.YAMLError as exc:
            raise ValueError("Failed to parse front matter YAML") from exc

        if metadata is None:
            return {}, body
        if not isinstance(metadata, dict):
            raise ValueError("Front matter is not a valid YAML dictionary")

        return metadata, body
