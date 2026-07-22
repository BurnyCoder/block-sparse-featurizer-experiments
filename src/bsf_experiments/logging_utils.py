"""Timestamped dual-destination logging with defense-in-depth secret redaction.

Python logging handlers provide immediate terminal and UTF-8 file output; see
https://docs.python.org/3/howto/logging.html#logging-to-multiple-destinations.
No length limit is applied, which preserves complete model prompts and outputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


REDACTED = "[REDACTED]"
_KEY_BOUNDARY = re.compile(
    r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|[^A-Za-z0-9]+"
)
_SECRET_KEY_WORDS = frozenset(
    {"authorization", "credential", "password", "secret", "token"}
)
_SECRET_KEY_SUFFIXES = ("apikey", "credential", "password", "secret", "token")
_SECRET_ASSIGNMENT = re.compile(
    r"(?P<quote>[\"']?)(?P<key>\b[A-Za-z_][A-Za-z0-9_.-]*)(?P=quote)"
    r"\s*[:=]\s*"
    r"(?P<value>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|[^\s,;]+)"
)
_EMBEDDED_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bhf_[A-Za-z0-9_-]{6,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{6,}\b"),
)


def is_secret_key(key: object) -> bool:
    """Return whether a mapping/environment key conventionally carries a secret."""

    text = str(key).strip()
    words = tuple(word.lower() for word in _KEY_BOUNDARY.split(text) if word)
    if any(word in _SECRET_KEY_WORDS for word in words):
        return True
    if any(left == "api" and right == "key" for left, right in zip(words, words[1:])):
        return True

    # A suffix check covers all-lowercase compound environment keys without
    # treating lookalikes such as ``tokenizer`` or ``secretary`` as secrets.
    compact = re.sub(r"[^A-Za-z0-9]+", "", text).lower()
    return compact.startswith("authorization") or compact.endswith(_SECRET_KEY_SUFFIXES)


def _redact_secret_assignment(match: re.Match[str]) -> str:
    """Redact only assignments whose identifier is classified as credential-like.

    Passing a callable to ``re.sub`` is the documented way to make replacement
    conditional on named capture groups:
    https://docs.python.org/3/library/re.html#re.sub.
    """

    return REDACTED if is_secret_key(match.group("key")) else match.group(0)


def _known_secret_values() -> tuple[str, ...]:
    """Collect nontrivial secret values for literal replacement, never for output."""

    return tuple(
        value
        for key, value in os.environ.items()
        if is_secret_key(key) and len(value) >= 4
    )


def redact_text(text: str, *, secret_values: Sequence[str] | None = None) -> str:
    """Remove known literal secrets and common credential shapes from free text."""

    redacted = text
    for secret in (
        secret_values if secret_values is not None else _known_secret_values()
    ):
        if secret:
            redacted = redacted.replace(secret, REDACTED)
    for pattern in _EMBEDDED_SECRET_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    return _SECRET_ASSIGNMENT.sub(_redact_secret_assignment, redacted)


def sanitize_for_logging(
    value: Any, *, secret_values: Sequence[str] | None = None
) -> Any:
    """Recursively sanitize structured values while preserving all safe content."""

    secrets = (
        tuple(secret_values) if secret_values is not None else _known_secret_values()
    )
    if is_dataclass(value) and not isinstance(value, type):
        return sanitize_for_logging(asdict(value), secret_values=secrets)
    if isinstance(value, Mapping):
        return {
            key: REDACTED
            if is_secret_key(key)
            else sanitize_for_logging(item, secret_values=secrets)
            for key, item in value.items()
        }
    if isinstance(value, str):
        return redact_text(value, secret_values=secrets)
    if isinstance(value, tuple):
        return tuple(
            sanitize_for_logging(item, secret_values=secrets) for item in value
        )
    if isinstance(value, list):
        return [sanitize_for_logging(item, secret_values=secrets) for item in value]
    if isinstance(value, set):
        return {sanitize_for_logging(item, secret_values=secrets) for item in value}
    return value


class RedactingFilter(logging.Filter):
    """Sanitize message templates and interpolation arguments before formatting."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Mutate a private LogRecord copy owned by this logger's handlers."""

        secrets = _known_secret_values()
        record.msg = sanitize_for_logging(record.msg, secret_values=secrets)
        record.args = sanitize_for_logging(record.args, secret_values=secrets)
        return True


class RedactingFormatter(logging.Formatter):
    """Redact the final rendered record, including exception trace messages."""

    converter = time.gmtime

    def format(self, record: logging.LogRecord) -> str:
        """Apply final free-text redaction after standard logging interpolation."""

        return redact_text(super().format(record))


def create_run_logger(
    run_name: str,
    log_dir: str | Path,
    *,
    level: int | str = logging.INFO,
) -> tuple[logging.Logger, Path]:
    """Create an isolated timestamped logger writing to a file and stdout."""

    destination = Path(log_dir)
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-.") or "run"
    log_path = destination / f"{timestamp}-{safe_run_name}.log"
    logger = logging.getLogger(f"bsf_experiments.{safe_run_name}.{timestamp}")
    logger.setLevel(level)
    logger.propagate = False

    formatter = RedactingFormatter(
        fmt="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for handler in (
        logging.FileHandler(log_path, mode="x", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setLevel(level)
        handler.addFilter(RedactingFilter())
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger, log_path


def log_event(logger: logging.Logger, event: str, payload: Any) -> None:
    """Write one complete structured event as compact, searchable JSON."""

    sanitized = sanitize_for_logging(payload)
    serialized = json.dumps(
        sanitized, ensure_ascii=False, default=str, separators=(",", ":")
    )
    logger.info("event=%s payload=%s", event, serialized)


__all__ = [
    "REDACTED",
    "RedactingFilter",
    "RedactingFormatter",
    "create_run_logger",
    "is_secret_key",
    "log_event",
    "redact_text",
    "sanitize_for_logging",
]
