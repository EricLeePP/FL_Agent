"""Content hashes that do not depend on file location, line endings or JSON layout."""

import hashlib
import json
from pathlib import Path

from .trusted_http import strict_json


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def canonical_json_sha256(value) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def normalized_text_sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def content_sha256(raw: bytes) -> str:
    """Hash JSON by its parsed value and anything else by its newline-normalized text."""
    try:
        return canonical_json_sha256(strict_json(raw.decode("utf-8-sig")))
    except (UnicodeError, ValueError):
        return normalized_text_sha256(raw.decode("utf-8", errors="replace"))


def files_sha256(paths) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(normalized_text_sha256(Path(path).read_text(encoding="utf-8")).encode("ascii"))
    return digest.hexdigest()
