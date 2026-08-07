"""Token counting.

Chunk sizes and the context budget are both measured with the *embedding model's own*
tokenizer rather than a generic one. A "512 token" chunk measured with the wrong
tokenizer can silently overflow the encoder's window and get truncated, which loses
the tail of every long chunk without raising anything.

If the tokenizer cannot be loaded (no network on first run, no cached weights) the
module degrades to a character-ratio estimate and says so loudly, so the numbers in an
ablation table are never quietly based on a different measurement than they claim.
"""

from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)

# Empirical ratio for English biomedical prose on WordPiece/BPE vocabularies.
# Only used when the real tokenizer is unavailable.
_FALLBACK_CHARS_PER_TOKEN = 4.0


class Tokenizer:
    """Thin wrapper with a hard-fallback path and an honest ``is_exact`` flag."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tok = None
        self.is_exact = False
        try:
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(model_name)
            self.is_exact = True
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            log.warning(
                "tokenizer for %s unavailable (%s); falling back to a ~%.0f chars/token "
                "estimate. Token counts in this run are approximate.",
                model_name,
                exc,
                _FALLBACK_CHARS_PER_TOKEN,
            )

    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False))
        return max(1, int(len(text) / _FALLBACK_CHARS_PER_TOKEN))

    def encode(self, text: str) -> list[int]:
        if self._tok is not None:
            return self._tok.encode(text, add_special_tokens=False)
        raise RuntimeError("exact encoding requires a real tokenizer")

    def truncate(self, text: str, max_tokens: int) -> str:
        """Cut text to at most ``max_tokens``, preferring a sentence boundary."""
        if self.count(text) <= max_tokens:
            return text
        if self._tok is not None:
            ids = self._tok.encode(text, add_special_tokens=False)[:max_tokens]
            out = self._tok.decode(ids, skip_special_tokens=True)
        else:
            out = text[: int(max_tokens * _FALLBACK_CHARS_PER_TOKEN)]
        # Trailing partial sentences read as corruption in a citation; drop them
        # when there is enough text left to be worth keeping.
        cut = max(out.rfind(". "), out.rfind("\n"))
        if cut > len(out) * 0.6:
            out = out[: cut + 1]
        return out.strip()


@lru_cache(maxsize=8)
def get_tokenizer(model_name: str) -> Tokenizer:
    return Tokenizer(model_name)
