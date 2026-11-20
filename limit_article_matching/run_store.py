"""Durable storage for runs and their results.

SQLite, in the AI service's own storage — the brief permits the service to
persist jobs and results locally, while its UzAssets credential stays read-only.
Persisting rather than holding runs in memory matters for a real reason: results
must survive a process restart between the run completing and the backend paging
through it, and the backend is required to fetch every page.

Concurrency is handled with WAL plus a process-wide lock. Runs are written from a
background task while the API reads them, and SQLite's default rollback journal
would make those readers block on the writer.

Cursors are opaque to the caller and stable under insertion: they encode the last
`productId` returned, and paging resumes strictly after it. An offset would
silently skip or repeat rows when rows arrive mid-page; a keyset cursor cannot.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from limit_article_matching.errors import IdempotencyConflictError, RunNotFoundError
from limit_article_matching.fingerprint import fingerprint
from limit_article_matching.schemas import (
    MatchStatus,
    ProductMatchItem,
    RunCounts,
    RunStatus,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                     TEXT PRIMARY KEY,
    status                     TEXT NOT NULL,
    organization_id            INTEGER NOT NULL,
    product_ids_json           TEXT NOT NULL,
    idempotency_key            TEXT,
    request_hash               TEXT NOT NULL,
    created_at                 TEXT NOT NULL,
    started_at                 TEXT,
    completed_at               TEXT,
    expires_at                 TEXT,
    source_snapshot_at         TEXT,
    model_version              TEXT,
    rules_version              TEXT NOT NULL,
    fingerprint_schema_version TEXT NOT NULL,
    catalog_fingerprint        TEXT,
    total_products             INTEGER NOT NULL DEFAULT 0,
    processed_products         INTEGER NOT NULL DEFAULT 0,
    matched                    INTEGER NOT NULL DEFAULT 0,
    ambiguous                  INTEGER NOT NULL DEFAULT 0,
    no_match                   INTEGER NOT NULL DEFAULT 0,
    error                      INTEGER NOT NULL DEFAULT 0,
    error_json                 TEXT
);

-- One live run per idempotency key. Enforced here rather than in application
-- code so two concurrent POSTs with the same key cannot both win.
CREATE UNIQUE INDEX IF NOT EXISTS runs_idempotency_key
    ON runs (idempotency_key) WHERE idempotency_key IS NOT NULL;

CREATE INDEX IF NOT EXISTS runs_expires_at ON runs (expires_at);

CREATE TABLE IF NOT EXISTS run_results (
    run_id     TEXT NOT NULL,
    product_id INTEGER NOT NULL,
    item_json  TEXT NOT NULL,
    PRIMARY KEY (run_id, product_id)
) WITHOUT ROWID;
"""


@dataclass(slots=True)
class RunRecord:
    run_id: str
    status: RunStatus
    organization_id: int
    product_ids: list[int]
    idempotency_key: str | None
    request_hash: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    expires_at: datetime | None
    source_snapshot_at: datetime | None
    model_version: str | None
    rules_version: str
    fingerprint_schema_version: str
    catalog_fingerprint: str | None
    counts: RunCounts
    error: dict[str, Any] | None


class RunStore:
    def __init__(self, path: str, *, retention_hours: int) -> None:
        self.path = path
        self.retention = timedelta(hours=retention_hours)
        self._lock = threading.RLock()
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ── run lifecycle ───────────────────────────────────────────────────────

    def create_run(
        self,
        *,
        run_id: str,
        organization_id: int,
        product_ids: list[int],
        idempotency_key: str | None,
        rules_version: str,
        fingerprint_schema_version: str,
    ) -> RunRecord:
        """Insert a QUEUED run, honouring idempotency.

        Same key and same normalized body returns the original run. Same key with
        a different body is a 409 — the brief is explicit, and silently running
        the new body under an old key would hide a backend bug.

        Ids are sorted and deduplicated here as well as in the request model, so
        the hash reflects intent rather than the order the caller happened to
        send — and so it still does when something other than the HTTP layer
        creates a run.
        """
        product_ids = sorted(set(product_ids))
        request_hash = fingerprint({"organizationId": organization_id, "productIds": product_ids})
        now = _utcnow()

        with self._tx() as conn:
            if idempotency_key:
                row = conn.execute(
                    "SELECT * FROM runs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if row is not None:
                    existing = _record_from_row(row)
                    if existing.request_hash != request_hash:
                        raise IdempotencyConflictError(
                            "Idempotency-Key was already used with a different request body"
                        )
                    return existing

            conn.execute(
                """
                INSERT INTO runs (
                    run_id, status, organization_id, product_ids_json, idempotency_key,
                    request_hash, created_at, rules_version, fingerprint_schema_version,
                    total_products
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(RunStatus.QUEUED),
                    organization_id,
                    json.dumps(product_ids),
                    idempotency_key,
                    request_hash,
                    _iso(now),
                    rules_version,
                    fingerprint_schema_version,
                    len(product_ids),
                ),
            )

        return RunRecord(
            run_id=run_id,
            status=RunStatus.QUEUED,
            organization_id=organization_id,
            product_ids=product_ids,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            created_at=now,
            started_at=None,
            completed_at=None,
            expires_at=None,
            source_snapshot_at=None,
            model_version=None,
            rules_version=rules_version,
            fingerprint_schema_version=fingerprint_schema_version,
            catalog_fingerprint=None,
            counts=RunCounts(totalProducts=len(product_ids), processedProducts=0),
            error=None,
        )

    def mark_running(
        self,
        run_id: str,
        *,
        source_snapshot_at: datetime,
        catalog_fingerprint: str,
        total_products: int,
        model_version: str | None,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE runs
                   SET status = ?, started_at = COALESCE(started_at, ?),
                       source_snapshot_at = ?, catalog_fingerprint = ?,
                       total_products = ?, model_version = ?
                 WHERE run_id = ?
                """,
                (
                    str(RunStatus.RUNNING),
                    _iso(_utcnow()),
                    _iso(source_snapshot_at),
                    catalog_fingerprint,
                    total_products,
                    model_version,
                    run_id,
                ),
            )

    def append_results(self, run_id: str, items: list[ProductMatchItem]) -> None:
        """Persist a batch and advance the counters in the same transaction.

        Counts and rows move together so a status read can never report more
        processed products than are actually retrievable.
        """
        if not items:
            return
        payload = [
            (run_id, item.product_id, json.dumps(item.model_dump(by_alias=True, mode="json"), ensure_ascii=False))
            for item in items
        ]
        buckets = {status: 0 for status in MatchStatus}
        for item in items:
            buckets[item.match_status] += 1

        with self._tx() as conn:
            before = conn.execute(
                "SELECT count(*) AS n FROM run_results WHERE run_id = ?", (run_id,)
            ).fetchone()["n"]
            # INSERT OR REPLACE keeps a retried batch idempotent: the same product
            # written twice must not double-count.
            conn.executemany(
                "INSERT OR REPLACE INTO run_results (run_id, product_id, item_json) VALUES (?, ?, ?)",
                payload,
            )
            after = conn.execute(
                "SELECT count(*) AS n FROM run_results WHERE run_id = ?", (run_id,)
            ).fetchone()["n"]
            conn.execute(
                """
                UPDATE runs
                   SET processed_products = ?,
                       matched   = matched   + ?,
                       ambiguous = ambiguous + ?,
                       no_match  = no_match  + ?,
                       error     = error     + ?
                 WHERE run_id = ?
                """,
                (
                    after,
                    buckets[MatchStatus.MATCHED],
                    buckets[MatchStatus.AMBIGUOUS],
                    buckets[MatchStatus.NO_MATCH],
                    buckets[MatchStatus.ERROR],
                    run_id,
                ),
            )
            if after - before != len(items):
                logger.warning(
                    "Run %s: %d items written but row count moved by %d (duplicate product ids?)",
                    run_id,
                    len(items),
                    after - before,
                )

    def mark_completed(self, run_id: str) -> None:
        completed = _utcnow()
        with self._tx() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, completed_at = ?, expires_at = ? WHERE run_id = ?",
                (str(RunStatus.COMPLETED), _iso(completed), _iso(completed + self.retention), run_id),
            )

    def mark_failed(self, run_id: str, *, code: str, message: str) -> None:
        failed = _utcnow()
        with self._tx() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, completed_at = ?, expires_at = ?, error_json = ? WHERE run_id = ?",
                (
                    str(RunStatus.FAILED),
                    _iso(failed),
                    _iso(failed + self.retention),
                    json.dumps({"code": code, "message": message}, ensure_ascii=False),
                    run_id,
                ),
            )

    # ── reads ───────────────────────────────────────────────────────────────

    def get_run(self, run_id: str, *, organization_id: int | None = None) -> RunRecord:
        """One run, optionally scoped to the caller's organization.

        The brief requires status and results access to be bound to the
        authorized caller and organization. A run belonging to another
        organization is reported as not found rather than as forbidden, so its
        existence is not confirmed.
        """
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(f"Run {run_id} was not found or has expired")
        record = _record_from_row(row)
        if organization_id is not None and record.organization_id != organization_id:
            raise RunNotFoundError(f"Run {run_id} was not found or has expired")
        if record.expires_at is not None and record.expires_at <= _utcnow():
            self.delete_run(run_id)
            raise RunNotFoundError(f"Run {run_id} was not found or has expired")
        return record

    def get_results_page(
        self, run_id: str, *, cursor: str | None, limit: int
    ) -> tuple[list[ProductMatchItem], str | None]:
        after = _decode_cursor(cursor)
        params: list[Any] = [run_id]
        keyset = ""
        if after is not None:
            keyset = " AND product_id > ?"
            params.append(after)
        params.append(limit + 1)  # one extra row tells us whether more remain

        with self._lock:
            rows = self._conn.execute(
                "SELECT product_id, item_json FROM run_results "
                f"WHERE run_id = ?{keyset} ORDER BY product_id LIMIT ?",
                params,
            ).fetchall()

        has_more = len(rows) > limit
        rows = rows[:limit]
        items = [ProductMatchItem.model_validate(json.loads(row["item_json"])) for row in rows]
        next_cursor = _encode_cursor(rows[-1]["product_id"]) if has_more and rows else None
        return items, next_cursor

    def delete_run(self, run_id: str) -> None:
        with self._tx() as conn:
            conn.execute("DELETE FROM run_results WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    def purge_expired(self) -> int:
        now = _iso(_utcnow())
        with self._tx() as conn:
            expired = [
                row["run_id"]
                for row in conn.execute(
                    "SELECT run_id FROM runs WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now,),
                ).fetchall()
            ]
            for run_id in expired:
                conn.execute("DELETE FROM run_results WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        return len(expired)

    def reset_orphaned_runs(self) -> int:
        """Fail runs left mid-flight by a restart.

        A QUEUED or RUNNING row after startup belongs to a process that no longer
        exists; leaving it would make the backend poll a run that will never
        finish.
        """
        now = _utcnow()
        with self._tx() as conn:
            cursor = conn.execute(
                "UPDATE runs SET status = ?, completed_at = ?, expires_at = ?, error_json = ? "
                "WHERE status IN (?, ?)",
                (
                    str(RunStatus.FAILED),
                    _iso(now),
                    _iso(now + self.retention),
                    json.dumps(
                        {"code": "service_restarted", "message": "Run was interrupted by a service restart"},
                        ensure_ascii=False,
                    ),
                    str(RunStatus.QUEUED),
                    str(RunStatus.RUNNING),
                ),
            )
            return cursor.rowcount or 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# ── helpers ─────────────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _encode_cursor(product_id: int) -> str:
    return base64.urlsafe_b64encode(str(product_id).encode("ascii")).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str | None) -> int | None:
    if not cursor:
        return None
    from limit_article_matching.errors import InvalidRequestError

    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        return int(base64.urlsafe_b64decode(padded).decode("ascii"))
    except (ValueError, TypeError) as exc:
        raise InvalidRequestError("cursor is not a valid pagination cursor") from exc


def _record_from_row(row: sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=row["run_id"],
        status=RunStatus(row["status"]),
        organization_id=row["organization_id"],
        product_ids=json.loads(row["product_ids_json"]),
        idempotency_key=row["idempotency_key"],
        request_hash=row["request_hash"],
        created_at=_parse_dt(row["created_at"]) or _utcnow(),
        started_at=_parse_dt(row["started_at"]),
        completed_at=_parse_dt(row["completed_at"]),
        expires_at=_parse_dt(row["expires_at"]),
        source_snapshot_at=_parse_dt(row["source_snapshot_at"]),
        model_version=row["model_version"],
        rules_version=row["rules_version"],
        fingerprint_schema_version=row["fingerprint_schema_version"],
        catalog_fingerprint=row["catalog_fingerprint"],
        counts=RunCounts(
            totalProducts=row["total_products"],
            processedProducts=row["processed_products"],
            matched=row["matched"],
            ambiguous=row["ambiguous"],
            noMatch=row["no_match"],
            error=row["error"],
        ),
        error=json.loads(row["error_json"]) if row["error_json"] else None,
    )


__all__ = ["RunRecord", "RunStore"]
