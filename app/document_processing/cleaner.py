"""
Text normalization utilities applied to raw extracted PDF text.
"""
import re


class TextCleaner:
    """Normalizes whitespace and strips noisy artifacts without destroying readable content."""

    _MULTI_SPACE_RE = re.compile(r"[ \t]+")
    _MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
    _HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

    def clean(self, text: str) -> str:
        if not text:
            return ""

        # Reattach words that were split across a line break with a hyphen, e.g. "infor-\nmation"
        text = self._HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

        # Collapse excessive horizontal whitespace
        text = self._MULTI_SPACE_RE.sub(" ", text)

        # Collapse excessive blank lines but keep paragraph breaks
        text = self._MULTI_NEWLINE_RE.sub("\n\n", text)

        return text.strip()
