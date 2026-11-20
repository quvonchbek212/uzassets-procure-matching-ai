"""Read-only extraction of products, their MXIK evidence and the candidate catalog.

Four things drive the design.

**One consistent snapshot.** The brief requires products, MXIK evidence and
candidate articles to come from a consistent read-only snapshot, with the
transaction released before inference. So the whole extraction happens inside one
REPEATABLE READ, READ ONLY transaction on one connection, and that transaction
closes before a single token is generated.

**Read-only, belt and braces.** The credential is already `read-user`, but the
session additionally sets `default_transaction_read_only = on` and opens the
transaction `READ ONLY`, so even a future coding mistake cannot write. Every
statement is bounded by a server-side `statement_timeout`.

**Organization isolation is a filter, not a check.** `fetch_products` selects on
`organization_id` rather than fetching by id and comparing afterwards. A product
belonging to another organization therefore never enters this process at all,
and is indistinguishable from one that does not exist — which is what the brief
requires: "Do not expose another organization's product details."

**The mapping is resolved, not assumed.** Every logical field declares the column
names it may appear under, `introspect()` resolves them against
`information_schema`, and the SELECT is built from what actually exists. This is
not theoretical: six of the first brief's mappings differed from the live schema
and `treasury_limit_article_parent_link` does not exist at all. The names listed
first below are the ones verified live on 2026-09-08; the alternates remain so a
differently-named deployment still works.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from limit_article_matching.errors import DependencyUnavailableError
from limit_article_matching.settings import DatabaseConfig, get_config

logger = logging.getLogger(__name__)


# ── logical schema ──────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """A logical field and the physical column names it may carry."""

    logical: str
    candidates: tuple[str, ...]
    required: bool = False

    def resolve(self, available: set[str]) -> str | None:
        for name in self.candidates:
            if name in available:
                return name
        return None


def _col(logical: str, *candidates: str, required: bool = False) -> ColumnSpec:
    return ColumnSpec(logical, (logical, *candidates), required=required)


PRODUCT_COLUMNS: tuple[ColumnSpec, ...] = (
    _col("id", required=True),
    _col("organization_id", "org_id", required=True),
    _col("product_name", "name", required=True),
    _col("product_kod", "product_code", "kod", "code"),
    _col("deleted", "is_deleted"),
    _col("mxik_id",),
    _col("updated_at", "modified_at", "last_modified_date"),
)

CLASSIFICATION_COLUMNS: tuple[ColumnSpec, ...] = (
    _col("id", required=True),
    _col("product_id", required=True),
    _col("mxik_id", required=True),
    _col("deleted", "is_deleted"),
    _col("updated_at", "modified_at", "last_modified_date"),
)

MXIK_COLUMNS: tuple[ColumnSpec, ...] = (
    _col("id", required=True),
    _col("kod", "code", "mxik_code", required=True),
    _col("name", "nomi", "title"),
    _col("name_ru", "nomi_ru"),
    _col("name_uz_latn", "name_latn"),
    _col("group_name", "group"),
    _col("class_name", "class"),
    _col("position_name", "position"),
    _col("sub_position_name", "subposition_name", "subposition", "sub_position"),
    _col("brand_name", "brand"),
    _col("attribute_name", "attribute"),
    # 'INTERNAL_PRODUCT_CODE' marks a placeholder row that carries no taxonomy.
    _col("type", "mxik_type", "kind"),
)

ARTICLE_COLUMNS: tuple[ColumnSpec, ...] = (
    _col("id", required=True),
    _col("article_code", "code", required=True),
    _col("article_name", "name", required=True),
    _col("organization_id", required=True),
    _col("is_active", "active"),
    _col("department_id", "dep_id"),
    _col("parent_article_id", "parent_id"),
    _col("external_1c_id", "external1c_id", "external_id"),
    _col("is_deleted", "deleted"),
    _col("updated_at", "modified_at", "last_modified_date"),
)

PARENT_LINK_COLUMNS: tuple[ColumnSpec, ...] = (
    _col("id",),
    _col("article_id", "limit_article_id", "child_article_id", required=True),
    _col("parent_article_id", "parent_limit_article_id", required=True),
    _col("is_deleted", "deleted"),
)

TABLE_SPECS: dict[str, tuple[ColumnSpec, ...]] = {
    "organization_product": PRODUCT_COLUMNS,
    "uz_goods_classification": CLASSIFICATION_COLUMNS,
    "mxik": MXIK_COLUMNS,
    "treasury_limit_article": ARTICLE_COLUMNS,
    "treasury_limit_article_parent_link": PARENT_LINK_COLUMNS,
}

# The parent-link table is optional context: the brief says to include hierarchy
# from it *and* from parent_article_id, and the live schema has only the latter.
REQUIRED_TABLES = frozenset(TABLE_SPECS) - {"treasury_limit_article_parent_link"}


@dataclass(slots=True)
class TableMap:
    """What actually exists for one table."""

    table: str
    exists: bool
    columns: dict[str, str] = field(default_factory=dict)  # logical -> physical
    missing_required: list[str] = field(default_factory=list)
    missing_optional: list[str] = field(default_factory=list)

    def has(self, logical: str) -> bool:
        return logical in self.columns

    def physical(self, logical: str) -> str | None:
        return self.columns.get(logical)


@dataclass(slots=True)
class SchemaMap:
    schema: str
    tables: dict[str, TableMap]

    @property
    def usable(self) -> bool:
        return not self.problems()

    def problems(self) -> list[str]:
        issues: list[str] = []
        for name in sorted(REQUIRED_TABLES):
            table = self.tables.get(name)
            if table is None or not table.exists:
                issues.append(f"table {self.schema}.{name} not found")
                continue
            for logical in table.missing_required:
                spec = next(s for s in TABLE_SPECS[name] if s.logical == logical)
                issues.append(
                    f"{name}: required field '{logical}' not found "
                    f"(tried: {', '.join(spec.candidates)})"
                )
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "usable": self.usable,
            "problems": self.problems(),
            "tables": {
                name: {
                    "exists": t.exists,
                    "resolvedColumns": dict(sorted(t.columns.items())),
                    "missingRequired": t.missing_required,
                    "missingOptional": t.missing_optional,
                }
                for name, t in sorted(self.tables.items())
            },
        }


class SchemaMismatchError(DependencyUnavailableError):
    code = "schema_mismatch"


# ── connection ──────────────────────────────────────────────────────────────


@contextmanager
def _tunnel(cfg: DatabaseConfig) -> Iterator[tuple[str, int]]:
    """Yield (host, port) to connect to, opening an SSH tunnel when configured."""
    if not cfg.use_ssh_tunnel:
        yield cfg.host, cfg.port
        return

    if not os.path.isfile(cfg.ssh_private_key_path):
        raise DependencyUnavailableError(
            "SSH private key not found at DB_SSH_PRIVATE_KEY_PATH "
            f"({cfg.ssh_private_key_path!r}); the database is only reachable through the tunnel",
            code="ssh_key_missing",
        )

    from sshtunnel import BaseSSHTunnelForwarderError, SSHTunnelForwarder

    try:
        with SSHTunnelForwarder(
            (cfg.ssh_host, cfg.ssh_port),
            ssh_username=cfg.ssh_username,
            ssh_pkey=cfg.ssh_private_key_path,
            ssh_private_key_password=cfg.ssh_passphrase or None,
            remote_bind_address=(cfg.host, cfg.port),
            # DB_LOCAL_BIND_PORT=0 lets the OS pick a free port. A fixed port
            # looks like it works — two forwarders bind it without error — but
            # they then share one listener and corrupt each other's connections.
            local_bind_address=("127.0.0.1", cfg.local_bind_port),
        ) as forwarder:
            yield "127.0.0.1", forwarder.local_bind_port
    except BaseSSHTunnelForwarderError as exc:
        raise DependencyUnavailableError(
            f"SSH tunnel to {cfg.ssh_host} failed: {exc}", code="ssh_tunnel_failed"
        ) from exc


class SnapshotSession:
    """One read-only connection held open for the whole extraction."""

    def __init__(self, cfg: DatabaseConfig | None = None) -> None:
        self.cfg = cfg or get_config().database
        self._conn: psycopg.Connection | None = None
        self.snapshot_at: datetime | None = None

    @contextmanager
    def open(self) -> Iterator["SnapshotSession"]:
        cfg = self.cfg
        missing = cfg.missing_keys()
        if missing:
            raise DependencyUnavailableError(
                "Database is not configured; missing " + ", ".join(missing),
                code="db_not_configured",
            )

        with _tunnel(cfg) as (host, port):
            try:
                conn = psycopg.connect(
                    host=host,
                    port=port,
                    user=cfg.user,
                    password=cfg.password,
                    dbname=cfg.name,
                    connect_timeout=cfg.connect_timeout_sec,
                    row_factory=dict_row,
                    autocommit=True,
                )
            except psycopg.Error as exc:
                raise DependencyUnavailableError(
                    f"Could not connect to {cfg.name}: {_safe_db_error(exc)}",
                    code="db_unavailable",
                ) from exc

            try:
                with conn:
                    self._conn = conn
                    self._prepare(conn)
                    # REPEATABLE READ gives every query below the same snapshot;
                    # READ ONLY makes a stray write impossible rather than merely
                    # unlikely.
                    with conn.transaction():
                        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                        self.snapshot_at = self._server_now(conn)
                        yield self
            finally:
                self._conn = None

    def _prepare(self, conn: psycopg.Connection) -> None:
        conn.execute(
            sql.SQL("SET search_path TO {}, public").format(sql.Identifier(self.cfg.schema))
        )
        conn.execute(sql.SQL("SET statement_timeout = {}").format(sql.Literal(self.cfg.statement_timeout_ms)))
        conn.execute("SET default_transaction_read_only = on")
        conn.execute("SET idle_in_transaction_session_timeout = 300000")

    @staticmethod
    def _server_now(conn: psycopg.Connection) -> datetime:
        row = conn.execute("SELECT now() AT TIME ZONE 'UTC' AS now").fetchone()
        value = row["now"] if row else None
        if isinstance(value, datetime):
            return value.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc)

    # ── queries ─────────────────────────────────────────────────────────────

    def _query(
        self,
        statement: sql.Composed | sql.SQL | str,
        params: Sequence[Any] | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        conn = self._conn
        if conn is None:
            raise RuntimeError("SnapshotSession used outside its open() block")
        try:
            with conn.cursor() as cur:
                cur.execute(statement, params)
                return cur.fetchall()
        except psycopg.Error as exc:
            raise DependencyUnavailableError(
                f"Snapshot query failed: {_safe_db_error(exc)}", code="db_query_failed"
            ) from exc

    def introspect(self) -> SchemaMap:
        """Resolve every logical field against what the live schema really has."""
        rows = self._query(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = ANY(%s)
            """,
            (self.cfg.schema, list(TABLE_SPECS)),
        )
        by_table: dict[str, set[str]] = {}
        for row in rows:
            by_table.setdefault(row["table_name"], set()).add(row["column_name"])

        tables: dict[str, TableMap] = {}
        for table, specs in TABLE_SPECS.items():
            available = by_table.get(table)
            if available is None:
                tables[table] = TableMap(
                    table=table,
                    exists=False,
                    missing_required=[s.logical for s in specs if s.required],
                )
                continue
            resolved: dict[str, str] = {}
            missing_required: list[str] = []
            missing_optional: list[str] = []
            for spec in specs:
                physical = spec.resolve(available)
                if physical is not None:
                    resolved[spec.logical] = physical
                elif spec.required:
                    missing_required.append(spec.logical)
                else:
                    missing_optional.append(spec.logical)
            tables[table] = TableMap(
                table=table,
                exists=True,
                columns=resolved,
                missing_required=missing_required,
                missing_optional=missing_optional,
            )
        return SchemaMap(schema=self.cfg.schema, tables=tables)

    def fetch_products(
        self, schema_map: SchemaMap, *, organization_id: int, product_ids: list[int]
    ) -> list[dict[str, Any]]:
        """The requested products that belong to this organization.

        Filtering on `organization_id` inside the query — rather than fetching by
        id and comparing afterwards — is what keeps another organization's
        products invisible. Anything missing from the result is simply
        unavailable to this caller, with no way to tell why.

        Deleted products are returned rather than filtered out: the brief wants
        missing or unusable evidence reported, and a silently absent product
        would look identical to one that was never requested.
        """
        prod = schema_map.tables["organization_product"]
        selects = _select_columns(prod, "p", "", PRODUCT_COLUMNS)
        statement = sql.SQL(
            "SELECT {selects} FROM {table} p WHERE p.{org} = %(organization_id)s "
            "AND p.{id} = ANY(%(product_ids)s) ORDER BY p.{id}"
        ).format(
            selects=sql.SQL(", ").join(selects),
            table=sql.Identifier("organization_product"),
            org=sql.Identifier(prod.physical("organization_id")),
            id=sql.Identifier(prod.physical("id")),
        )
        return self._query(
            statement, {"organization_id": organization_id, "product_ids": product_ids}
        )

    def fetch_mxik_links(
        self, schema_map: SchemaMap, *, product_ids: list[int]
    ) -> list[dict[str, Any]]:
        """Every non-deleted classification link and its MXIK record.

        A product may have several. The brief asks for all valid links gathered
        and deduplicated rather than the first one taken, so this returns them
        all and `Product.from_rows` does the deduplication by `mxik_id`.
        """
        if not product_ids:
            return []

        cls = schema_map.tables["uz_goods_classification"]
        mxik = schema_map.tables["mxik"]

        selects = [
            sql.SQL("c.{} AS {}").format(
                sql.Identifier(cls.physical("id")), sql.Identifier("classification_id")
            ),
            sql.SQL("c.{} AS {}").format(
                sql.Identifier(cls.physical("product_id")), sql.Identifier("product_id")
            ),
        ]
        selects += _select_columns(mxik, "m", "mxik_", MXIK_COLUMNS)

        where = [
            sql.SQL("c.{} = ANY(%(product_ids)s)").format(sql.Identifier(cls.physical("product_id")))
        ]
        if cls.has("deleted"):
            where.append(sql.SQL("c.{} = false").format(sql.Identifier(cls.physical("deleted"))))

        statement = sql.SQL(
            "SELECT {selects} FROM {cls} c JOIN {mxik} m ON m.{mid} = c.{cmid} "
            "WHERE {where} ORDER BY c.{product_id}, m.{mid}"
        ).format(
            selects=sql.SQL(", ").join(selects),
            cls=sql.Identifier("uz_goods_classification"),
            mxik=sql.Identifier("mxik"),
            mid=sql.Identifier(mxik.physical("id")),
            cmid=sql.Identifier(cls.physical("mxik_id")),
            where=sql.SQL(" AND ").join(where),
            product_id=sql.Identifier(cls.physical("product_id")),
        )
        return self._query(statement, {"product_ids": product_ids})

    def fetch_direct_mxik(
        self, schema_map: SchemaMap, *, product_ids: list[int]
    ) -> list[dict[str, Any]]:
        """MXIK reached straight from `organization_product.mxik_id`.

        A second, independent path to the same evidence. "Gather all valid links"
        includes this one, and every row is tagged with its source so the brief's
        requirement to document where each attribute came from is satisfiable.
        Returns nothing when the column does not exist.
        """
        prod = schema_map.tables["organization_product"]
        mxik = schema_map.tables["mxik"]
        if not product_ids or not prod.has("mxik_id"):
            return []

        selects = [
            sql.SQL("p.{} AS {}").format(
                sql.Identifier(prod.physical("id")), sql.Identifier("product_id")
            ),
            sql.SQL("NULL AS {}").format(sql.Identifier("classification_id")),
        ]
        selects += _select_columns(mxik, "m", "mxik_", MXIK_COLUMNS)

        statement = sql.SQL(
            "SELECT {selects} FROM {prod} p JOIN {mxik} m ON m.{mid} = p.{pmid} "
            "WHERE p.{id} = ANY(%(product_ids)s) ORDER BY p.{id}, m.{mid}"
        ).format(
            selects=sql.SQL(", ").join(selects),
            prod=sql.Identifier("organization_product"),
            mxik=sql.Identifier("mxik"),
            mid=sql.Identifier(mxik.physical("id")),
            pmid=sql.Identifier(prod.physical("mxik_id")),
            id=sql.Identifier(prod.physical("id")),
        )
        return self._query(statement, {"product_ids": product_ids})

    def fetch_product_ids_for_requests(
        self, *, organization_id: int, request_ids: list[int] | None = None, limit: int = 500
    ) -> list[int]:
        """Distinct product ids reachable from an organization's PLAN requests.

        **Not part of the service.** No API path calls this; it exists so the
        command-line runner can find real ids to test with. The brief is explicit
        that the service "must not discover additional products from plan
        requests" — that rule is about never analyzing more than the caller asked
        for, which this cannot cause: the ids it returns still have to be passed
        back in as an explicit `productIds` request.

        Deliberately written against the plan tables directly rather than through
        the resolved schema map, because those tables are not part of this
        service's mapping and should not become part of it.
        """
        conditions = [
            "r.organization_id = %(organization_id)s",
            "r.kind = 'PLAN'",
            "r.is_deleted = false",
            "l.is_deleted = false",
            "c.deleted = false",
        ]
        params: dict[str, Any] = {"organization_id": organization_id, "limit": limit}
        if request_ids:
            conditions.append("r.id = ANY(%(request_ids)s)")
            params["request_ids"] = request_ids

        statement = (
            "SELECT DISTINCT c.product_id AS product_id "
            "FROM treasury_purchase_request r "
            "JOIN treasury_purchase_request_line l ON l.request_id = r.id "
            "JOIN uz_goods_classification c ON c.id = l.corporate_item_id "
            f"WHERE {' AND '.join(conditions)} "
            "ORDER BY c.product_id LIMIT %(limit)s"
        )
        return [int(row["product_id"]) for row in self._query(statement, params)]

    def fetch_catalog(self, schema_map: SchemaMap, *, organization_id: int) -> list[dict[str, Any]]:
        """Active limit articles of this organization — the only legal targets."""
        art = schema_map.tables["treasury_limit_article"]
        selects = _select_columns(art, "a", "", ARTICLE_COLUMNS)
        where = [
            sql.SQL("a.{} = %(organization_id)s").format(
                sql.Identifier(art.physical("organization_id"))
            )
        ]
        if art.has("is_active"):
            where.append(sql.SQL("a.{} = true").format(sql.Identifier(art.physical("is_active"))))
        if art.has("is_deleted"):
            where.append(sql.SQL("a.{} = false").format(sql.Identifier(art.physical("is_deleted"))))

        statement = sql.SQL("SELECT {selects} FROM {table} a WHERE {where} ORDER BY a.{id}").format(
            selects=sql.SQL(", ").join(selects),
            table=sql.Identifier("treasury_limit_article"),
            where=sql.SQL(" AND ").join(where),
            id=sql.Identifier(art.physical("id")),
        )
        return self._query(statement, {"organization_id": organization_id})

    def fetch_parent_links(
        self, schema_map: SchemaMap, *, article_ids: list[int]
    ) -> list[dict[str, Any]]:
        """Extra parent paths. Absent table or empty catalog simply yields none."""
        link = schema_map.tables.get("treasury_limit_article_parent_link")
        if link is None or not link.exists or link.missing_required or not article_ids:
            return []
        selects = _select_columns(link, "k", "", PARENT_LINK_COLUMNS)
        where = [
            sql.SQL("k.{} = ANY(%(article_ids)s)").format(sql.Identifier(link.physical("article_id")))
        ]
        if link.has("is_deleted"):
            where.append(sql.SQL("k.{} = false").format(sql.Identifier(link.physical("is_deleted"))))
        statement = sql.SQL("SELECT {selects} FROM {table} k WHERE {where}").format(
            selects=sql.SQL(", ").join(selects),
            table=sql.Identifier("treasury_limit_article_parent_link"),
            where=sql.SQL(" AND ").join(where),
        )
        return self._query(statement, {"article_ids": article_ids})


def _select_columns(
    table: TableMap, alias: str, prefix: str, specs: Sequence[ColumnSpec]
) -> list[sql.Composed]:
    """`alias.physical AS prefix_logical` for resolved columns, NULL for the rest.

    Emitting a typed NULL for unresolved optional columns keeps every row shape
    identical, so downstream code reads `row["mxik_brand_name"]` without caring
    whether the column exists in this deployment.
    """
    out: list[sql.Composed] = []
    for spec in specs:
        label = f"{prefix}{spec.logical}"
        physical = table.physical(spec.logical) if table.exists else None
        if physical is None:
            out.append(sql.SQL("NULL AS {}").format(sql.Identifier(label)))
        else:
            out.append(
                sql.SQL("{}.{} AS {}").format(
                    sql.Identifier(alias), sql.Identifier(physical), sql.Identifier(label)
                )
            )
    return out


def _safe_db_error(exc: psycopg.Error) -> str:
    """Error text without credentials or SQL internals, per the brief."""
    diag = getattr(exc, "diag", None)
    primary = getattr(diag, "message_primary", None) if diag else None
    sqlstate = getattr(exc, "sqlstate", None)
    text = primary or (str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__)
    return f"[{sqlstate}] {text}" if sqlstate else text


__all__ = [
    "ARTICLE_COLUMNS",
    "ColumnSpec",
    "SchemaMap",
    "SchemaMismatchError",
    "SnapshotSession",
    "TABLE_SPECS",
    "TableMap",
]
