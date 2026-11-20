"""Canonical JSON encoding and SHA-256 fingerprints.

The brief pins the encoding precisely: SHA-256 over UTF-8 JSON with recursively
sorted keys, explicit nulls, stable array ordering, string codes and decimal
strings. The backend must reproduce these byte-for-byte from the same values, so
every rule here is deliberate and none of it may drift without bumping
`fingerprintSchemaVersion`:

  * keys sorted recursively, at every depth
  * nulls kept, never dropped — an absent field and a null field must not hash
    the same, otherwise "product deleted" and "product never linked" collide
  * separators without whitespace, so no pretty-printing sneaks into the bytes
  * ensure_ascii=False, so the hash is over real UTF-8 (Cyrillic and Uzbek Latin
    text hash as themselves, not as \\uXXXX escapes)
  * Decimal / float -> a normalized decimal string; binary floats are never
    hashed, because 0.1 has no stable textual form across languages
  * datetime -> RFC 3339 in UTC with a trailing Z
  * arrays keep caller order; callers sort before handing them over

`canonical_json` is exported on purpose: the shared test vectors the backend
needs are just its output for a fixed input.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any
from uuid import UUID

# Bump only when a rule above changes. Mirrored by RunConfig.fingerprint_schema_version.
FINGERPRINT_ALGORITHM = "sha256"


def _normalize(value: Any) -> Any:
    """Map a database value onto the small set of JSON types we hash."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        return _decimal_to_string(value)
    if isinstance(value, float):
        # Binary floats have no stable textual representation across platforms
        # and languages; money and quantities arrive as Decimal from psycopg, so
        # a float here means someone introduced one. Convert through the exact
        # decimal form so the value at least hashes deterministically.
        return _decimal_to_string(Decimal(repr(value)))
    if isinstance(value, datetime):
        return _datetime_to_string(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, dict):
        # sort_keys on json.dumps only sorts what it sees; nested dicts inside
        # lists are handled here so the ordering really is recursive.
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (set, frozenset)):
        # A set has no order of its own; sorting the rendered members gives one.
        return sorted((_normalize(v) for v in value), key=lambda v: json.dumps(v, ensure_ascii=False, sort_keys=True))
    return str(value)


def _decimal_to_string(value: Decimal) -> str:
    """A normalized, exponent-free decimal string: 1.500 and 1.5 hash alike."""
    if value != value or value.is_infinite():  # NaN / Infinity
        return str(value)
    normalized = value.normalize()
    sign, digits, exponent = normalized.as_tuple()
    if isinstance(exponent, int) and exponent > 0:
        # normalize() renders 1500 as 1.5E+3; expand it back to plain digits.
        normalized = normalized.quantize(Decimal(1))
    text = format(normalized, "f")
    if text in ("-0", "-0.0"):
        return "0"
    return text


def _datetime_to_string(value: datetime) -> str:
    """RFC 3339 in UTC. A naive timestamp is read as UTC, not as local time."""
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return aware.isoformat(timespec="microseconds").replace("+00:00", "Z")


def canonical_json(payload: Any) -> str:
    """The exact string that gets hashed. Share this with the backend as-is."""
    return json.dumps(
        _normalize(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def fingerprint(payload: Any) -> str:
    """SHA-256 hex digest of the canonical JSON encoding of `payload`."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def stable_key(*parts: Any) -> str:
    """Short hash for internal keys (evidence dedupe), not part of the contract."""
    return hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).hexdigest()[:32]


__all__ = ["FINGERPRINT_ALGORITHM", "canonical_json", "fingerprint", "stable_key"]
