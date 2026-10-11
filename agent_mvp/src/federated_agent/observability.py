"""Logging with secret redaction and incident IDs for unexpected failures."""

import logging
import re
import sys
import uuid
from pathlib import Path

LOGGER_NAME = "federated_agent"
_HANDLER_MARK = "_federated_agent_handler"

_SECRET_PATTERNS = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "sk-***"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer ***"),
    (re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://[^\s:/@]+):[^\s/@]+@"), r"\1:***@"),
    (re.compile(r"(?i)\b(password|passwd|api[_-]?key|token|secret)(\s*[=:]\s*|\"\s*:\s*\")([^\s,;\"']+)"), r"\1\2***"),
)


def scrub(text: str) -> str:
    """Remove common credential forms from text that may reach logs or researchers."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class _CurrentStderrHandler(logging.StreamHandler):
    """Write to whatever sys.stderr is at emit time, so redirection is honoured."""

    def __init__(self):
        super().__init__(sys.stderr)

    @property
    def stream(self):
        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass


class RedactingFormatter(logging.Formatter):
    """Format a record, including any traceback, and then remove credentials."""

    def format(self, record):
        return scrub(super().format(record))


def configure_logging(verbosity: int = 0, log_file: str | Path | None = None, *, console: bool = True) -> logging.Logger:
    """Configure the package logger. Calling it again replaces earlier handlers."""
    level = logging.WARNING if verbosity <= 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logger = logging.getLogger(LOGGER_NAME)
    for handler in [h for h in logger.handlers if getattr(h, _HANDLER_MARK, False)]:
        logger.removeHandler(handler)
        handler.close()
    formatter = RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = []
    if console:
        stream = _CurrentStderrHandler()
        stream.setLevel(level)
        handlers.append(stream)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    for handler in handlers:
        handler.setFormatter(formatter)
        setattr(handler, _HANDLER_MARK, True)
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if log_file is not None else level)
    logger.propagate = False
    return logger


def new_incident_id() -> str:
    return uuid.uuid4().hex[:12]


def log_incident(logger: logging.Logger, message: str, exc: BaseException | None = None) -> str:
    """Log an unexpected failure with its traceback and return a correlation ID."""
    incident = new_incident_id()
    logger.error("%s (incident %s)", message, incident, exc_info=exc)
    return incident
