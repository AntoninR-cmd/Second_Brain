from __future__ import annotations

from hashlib import sha256

SEMANTIC_TEXT_VERSION = "title-content-v1"


def build_semantic_text(*, title: str, content: str) -> str:
    """Build the only text representation embedded by Phase 4."""

    return f"{title.strip()}\n\n{content.strip()}"


def semantic_text_fingerprint(*, title: str, content: str) -> str:
    """Return the stable SHA-256 fingerprint of the semantic text."""

    semantic_text = build_semantic_text(title=title, content=content)
    return sha256(semantic_text.encode("utf-8")).hexdigest()
