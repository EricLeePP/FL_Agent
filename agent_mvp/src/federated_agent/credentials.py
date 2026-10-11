"""Load operator credentials locally, without putting them in research context."""

import os
import re
from pathlib import Path


class CredentialError(ValueError):
    """A local credential source is unreadable or malformed."""


def load_deepseek_api_key(
    key_file: str | Path | None = None,
    key_env: str = "DEEPSEEK_API_KEY",
) -> str | None:
    """Explicit file > file environment variable > key environment variable.

    A configured but invalid file must never fall back to another credential.
    File contents are a bare key, not configuration, JSON, or instructions.
    """
    if key_file is None:
        key_file = os.environ.get("DEEPSEEK_API_KEY_FILE")
    if key_file is not None:
        if not str(key_file).strip():
            raise CredentialError("DeepSeek API key file path is empty.")
        try:
            with Path(key_file).expanduser().open(encoding="utf-8-sig") as source:
                value = source.read(4097)
        except (OSError, UnicodeError, ValueError):
            raise CredentialError("Cannot read the DeepSeek API key file as UTF-8 text.") from None
        if len(value) > 4096 or not re.fullmatch(r"sk-[A-Za-z0-9_-]+", value.strip()):
            raise CredentialError(
                "DeepSeek API key file must contain only one bare sk- key; "
                "remove labels, quotes, and other text."
            )
        return value.strip()
    return os.environ.get(key_env, "").strip() or None
