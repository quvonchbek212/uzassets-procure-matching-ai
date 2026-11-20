"""Run orchestration: validate, extract one snapshot, infer, persist.

The sequencing follows the brief exactly. Products, their MXIK evidence and the
candidate catalog are extracted inside a single read-only REPEATABLE READ
transaction, which then closes. Only afterwards does inference start — no model
call ever holds a database transaction open.

Three properties are load-bearing and easy to lose:

**One outcome per requested distinct product, always.** Every failure path
produces an ERROR item rather than dropping the product, and the run is only
COMPLETED when `processedProducts == totalProducts`. A product that vanished
between extraction and results would leave the backend believing it had an answer
for something it never got.

**Availability is decided before the job is accepted.** Products that do not
belong to the organization — whether because they are another organization's or
because they do not exist — are rejected at POST time with 404, not discovered
halfway through a run. The two cases are deliberately indistinguishable.

**Nothing request-shaped leaks in.** The extraction reads products directly by
id. It never walks plan requests, never looks at which article a line currently
sits on, and never lets a request's department or budget year reach the model.
That is what makes a stored mapping reusable rather than an accident of the
request it happened to be discovered through.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from limit_article_matching.db_reader import SchemaMap, SnapshotSession
from limit_article_matching.errors import (
    DependencyUnavailableError,
    ErrorCode,
    InvalidRequestError,
    ProductMatchingError,
    ProductsUnavailableError,
    RunNotCompletedError,
)
from limit_article_matching.matcher import (
    Decision,
    ProductMatcher,
    build_error_item,
    build_result_item,
)
from limit_article_matching.models import Catalog, Product
from limit_article_matching.run_store import RunStore
from limit_article_matching.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    ProductMatchItem,
    RunResultsResponse,
    RunStatus,
    RunStatusResponse,
)
from limit_article_matching.settings import LimitArticleMatchingConfig, get_config

logger = logging.getLogger(__name__)

RESULT_FLUSH_BATCH = 100


@dataclass(slots=True)
class Snapshot:
    """Everything one run reads from the database, read once."""

    products: list[Product]
    catalog: Catalog
    snapshot_at: datetime
    schema_map: SchemaMap | None = None
    extraction_warnings: list[str] = field(default_factory=list)


class ProductMatchingService:
    def __init__(
        self,
        *,
        config: LimitArticleMatchingConfig | None = None,
        store: RunStore | None = None,
        ai_client_factory: Any = None,
    ) -> None:
        self.config = config or get_config()
        self.store = store or RunStore(
            self.config.run.storage_path, retention_hours=self.config.run.retention_hours
        )
        self._ai_client_factory = ai_client_factory or _default_ai_client_factory
        self._semaphore = asyncio.Semaphore(self.config.matching.max_concurrency)
        self._tasks: set[asyncio.Task] = set()

    # ── public API ──────────────────────────────────────────────────────────

    async def create_run(
        self, request: CreateRunRequest, *, idempotency_key: str
    ) -> CreateRunResponse:
        product_ids = request.product_ids  # already sorted and deduplicated
        if len(product_ids) > self.config.run.max_products_per_run:
            raise InvalidRequestError(
                f"productIds holds {len(product_ids)} distinct ids, above this service's "
                f"limit of {self.config.run.max_products_per_run} per call"
            )

        # Ownership and availability are settled before the 202, so the backend
        # learns about a bad id immediately rather than from a half-empty result.
        await asyncio.to_thread(self._validate_products, request.organization_id, product_ids)

        run = self.store.create_run(
            run_id=f"product-run-{uuid.uuid4().hex[:20]}",
            organization_id=request.organization_id,
            product_ids=product_ids,
            idempotency_key=idempotency_key,
            rules_version=self.config.run.rules_version,
            fingerprint_schema_version=self.config.run.fingerprint_schema_version,
        )

        if run.status is RunStatus.QUEUED and run.started_at is None:
            self._spawn(run.run_id, run.organization_id, run.product_ids)

        return CreateRunResponse(
            runId=run.run_id,
            status=run.status,
            organizationId=run.organization_id,
            productIds=run.product_ids,
        )

    def get_status(self, run_id: str, *, organization_id: int | None = None) -> RunStatusResponse:
        record = self.store.get_run(run_id, organization_id=organization_id)
        return RunStatusResponse(
            runId=record.run_id,
            status=record.status,
            organizationId=record.organization_id,
            productIds=record.product_ids,
            sourceSnapshotAt=record.source_snapshot_at,
            startedAt=record.started_at,
            completedAt=record.completed_at,
            expiresAt=record.expires_at,
            modelVersion=record.model_version,
            rulesVersion=record.rules_version,
            fingerprintSchemaVersion=record.fingerprint_schema_version,
            catalogFingerprint=record.catalog_fingerprint,
            counts=record.counts,
            error=record.error,
        )

    def get_results(
        self,
        run_id: str,
        *,
        cursor: str | None,
        limit: int | None,
        organization_id: int | None = None,
    ) -> RunResultsResponse:
        record = self.store.get_run(run_id, organization_id=organization_id)
        if record.status is not RunStatus.COMPLETED:
            # Serving partial pages would let the backend store mappings from a
            # run that is still changing and believe it had seen all of it.
            raise RunNotCompletedError(
                f"Run {run_id} is {record.status}; results are served only after COMPLETED"
            )

        page_size = self._page_size(limit)
        items, next_cursor = self.store.get_results_page(run_id, cursor=cursor, limit=page_size)
        return RunResultsResponse(
            runId=run_id,
            organizationId=record.organization_id,
            items=items,
            nextCursor=next_cursor,
        )

    def _page_size(self, limit: int | None) -> int:
        if limit is None:
            return self.config.run.default_page_size
        if limit < 1 or limit > self.config.run.max_page_size:
            raise InvalidRequestError(f"limit must be between 1 and {self.config.run.max_page_size}")
        return limit

    # ── validation ──────────────────────────────────────────────────────────

    def _validate_products(self, organization_id: int, product_ids: list[int]) -> None:
        """Every requested product must be visible to this organization.

        The query filters on `organization_id`, so a product belonging to someone
        else simply does not come back — identical to one that does not exist.
        The error lists the ids the caller already sent, which reveals nothing new
        about anyone else's data.
        """
        session = SnapshotSession(self.config.database)
        with session.open() as snapshot:
            schema_map = snapshot.introspect()
            self._require_usable_schema(schema_map)
            rows = snapshot.fetch_products(
                schema_map, organization_id=organization_id, product_ids=product_ids
            )

        found = {int(row["id"]) for row in rows}
        missing = sorted(set(product_ids) - found)
        if missing:
            shown = missing[:20]
            suffix = "" if len(missing) == len(shown) else f" (+{len(missing) - len(shown)} more)"
            raise ProductsUnavailableError(
                f"{len(missing)} of the requested products are not available in organization "
                f"{organization_id}: {shown}{suffix}"
            )

    @staticmethod
    def _require_usable_schema(schema_map: SchemaMap) -> None:
        problems = schema_map.problems()
        if problems:
            raise DependencyUnavailableError(
                "Live schema does not match the expected mapping: " + "; ".join(problems[:5]),
                code="schema_mismatch",
            )

    # ── execution ───────────────────────────────────────────────────────────

    def _spawn(self, run_id: str, organization_id: int, product_ids: list[int]) -> None:
        task = asyncio.create_task(
            self._execute(run_id, organization_id, product_ids), name=f"pm-{run_id}"
        )
        # Without a reference the loop may garbage-collect a running task.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _execute(self, run_id: str, organization_id: int, product_ids: list[int]) -> None:
        try:
            snapshot = await asyncio.to_thread(self._extract, organization_id, product_ids)
        except ProductMatchingError as exc:
            logger.warning("Run %s extraction failed: %s", run_id, exc)
            self.store.mark_failed(run_id, code=exc.code, message=exc.message)
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("Run %s extraction crashed", run_id)
            self.store.mark_failed(run_id, code="extraction_failed", message=str(exc)[:500])
            return

        client = self._ai_client_factory(self.config.matching.ai_model)
        self.store.mark_running(
            run_id,
            source_snapshot_at=snapshot.snapshot_at,
            catalog_fingerprint=snapshot.catalog.catalog_fingerprint,
            total_products=len(snapshot.products),
            model_version=getattr(client, "model_name", self.config.matching.ai_model),
        )

        if not snapshot.products:
            self.store.mark_completed(run_id)
            return

        try:
            await self._infer_all(run_id, snapshot, client)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Run %s inference crashed", run_id)
            self.store.mark_failed(run_id, code="inference_failed", message=str(exc)[:500])
            return

        record = self.store.get_run(run_id)
        if record.counts.processed_products != record.counts.total_products:
            self.store.mark_failed(
                run_id,
                code="incomplete_results",
                message=(
                    f"processed {record.counts.processed_products} of "
                    f"{record.counts.total_products} products; the run is not complete"
                ),
            )
            return
        self.store.mark_completed(run_id)

    def _extract(self, organization_id: int, product_ids: list[int]) -> Snapshot:
        """One read-only snapshot: products, MXIK evidence and the catalog."""
        session = SnapshotSession(self.config.database)
        with session.open() as snapshot:
            schema_map = snapshot.introspect()
            self._require_usable_schema(schema_map)

            product_rows = snapshot.fetch_products(
                schema_map, organization_id=organization_id, product_ids=product_ids
            )
            owned_ids = [int(row["id"]) for row in product_rows]
            link_rows = snapshot.fetch_mxik_links(schema_map, product_ids=owned_ids)
            direct_rows = snapshot.fetch_direct_mxik(schema_map, product_ids=owned_ids)

            catalog_rows = snapshot.fetch_catalog(schema_map, organization_id=organization_id)
            parent_links = snapshot.fetch_parent_links(
                schema_map, article_ids=[int(r["id"]) for r in catalog_rows]
            )
            snapshot_at = snapshot.snapshot_at

        # Classification links first so that, when both paths reach the same
        # MXIK, the one with a classification id is the one kept.
        by_product: dict[int, list[tuple[dict[str, Any], str]]] = {}
        for row in link_rows:
            by_product.setdefault(int(row["product_id"]), []).append((row, "uz_goods_classification"))
        for row in direct_rows:
            by_product.setdefault(int(row["product_id"]), []).append((row, "organization_product"))

        products = [
            Product.from_rows(row, by_product.get(int(row["id"]), [])) for row in product_rows
        ]
        catalog = Catalog.build(organization_id, catalog_rows, parent_links)

        for table_name, table in schema_map.tables.items():
            if table.exists and table.missing_optional:
                logger.info(
                    "Schema: %s has no column for optional fields %s; they will be null",
                    table_name,
                    ", ".join(table.missing_optional),
                )

        assert snapshot_at is not None
        return Snapshot(
            products=products,
            catalog=catalog,
            snapshot_at=snapshot_at,
            schema_map=schema_map,
        )

    async def _infer_all(self, run_id: str, snapshot: Snapshot, client: Any) -> None:
        matcher = ProductMatcher(client, snapshot.catalog, self.config.matching)
        decisions: dict[int, Decision | BaseException] = {}
        work = list(snapshot.products)

        async def resolve_one(product: Product) -> None:
            async with self._semaphore:
                try:
                    decisions[product.id] = await asyncio.to_thread(matcher.decide, product)
                except BaseException as exc:  # noqa: BLE001 — recorded per product below
                    decisions[product.id] = exc

        async def resolve_batch(batch: Sequence[Product]) -> list[Product]:
            """Answer a batch; hand back whatever it declined to answer."""
            async with self._semaphore:
                try:
                    outcomes = await asyncio.to_thread(matcher.decide_batch, list(batch))
                except BaseException as exc:  # noqa: BLE001
                    # A failed batch must not fail its products: each one still
                    # gets its own attempt before anything is recorded as an error.
                    logger.info("Batch of %d failed (%s); retrying individually", len(batch), exc)
                    return list(batch)

            leftovers: list[Product] = []
            for product, outcome in zip(batch, outcomes):
                if outcome is None:
                    leftovers.append(product)
                else:
                    decisions[product.id] = outcome
            return leftovers

        batch_size = max(1, self.config.matching.batch_size)
        if batch_size == 1:
            await asyncio.gather(*(resolve_one(p) for p in work))
        else:
            batches = [work[i : i + batch_size] for i in range(0, len(work), batch_size)]
            leftover_lists = await asyncio.gather(*(resolve_batch(b) for b in batches))
            leftovers = [p for group in leftover_lists for p in group]
            if leftovers:
                logger.info(
                    "Run %s: %d of %d products were not answered in a batch; asking individually",
                    run_id,
                    len(leftovers),
                    len(work),
                )
                await asyncio.gather(*(resolve_one(p) for p in leftovers))

        pending: list[ProductMatchItem] = []
        for product in snapshot.products:
            pending.append(self._to_item(product, decisions.get(product.id), snapshot))
            if len(pending) >= RESULT_FLUSH_BATCH:
                self.store.append_results(run_id, pending)
                pending = []
        if pending:
            self.store.append_results(run_id, pending)

    def _to_item(
        self, product: Product, outcome: Decision | BaseException | None, snapshot: Snapshot
    ) -> ProductMatchItem:
        if outcome is None:
            return build_error_item(
                product, ErrorCode.INTERNAL_ERROR, "Bu mahsulot uchun natija hosil bo'lmadi."
            )
        if isinstance(outcome, BaseException):
            return build_error_item(product, _error_code_for(outcome), _error_message(outcome))
        return build_result_item(
            product, outcome.with_warnings(snapshot.extraction_warnings), snapshot.catalog
        )

    # ── diagnostics ─────────────────────────────────────────────────────────

    def preview(self, organization_id: int, product_ids: list[int]) -> dict[str, Any]:
        """Extraction and candidate scoring only — no inference, no cost."""
        from limit_article_matching.candidates import shortlist

        snapshot = self._extract(organization_id, product_ids)
        rows = []
        for product in snapshot.products:
            fits_whole = (
                len(snapshot.catalog.articles) <= self.config.matching.full_catalog_threshold
            )
            picks = shortlist(
                product,
                snapshot.catalog,
                size=len(snapshot.catalog.articles)
                if fits_whole
                else self.config.matching.candidate_shortlist_size,
                keep_zero_score=fits_whole,
            )
            rows.append(
                {
                    "productId": product.id,
                    "productName": product.name,
                    "mxikLinks": len(product.mxik_links),
                    "usableMxikLinks": len(product.usable_mxik_links),
                    "warnings": product.warnings,
                    "sourceFingerprint": product.source_fingerprint,
                    "candidates": [
                        {
                            "limitArticleId": c.id,
                            "articleCode": c.article.code,
                            "articleName": c.article.name,
                            "score": round(c.score, 4),
                            "signals": list(c.signals),
                        }
                        for c in picks[:5]
                    ],
                }
            )
        return {
            "sourceSnapshotAt": snapshot.snapshot_at,
            "catalogFingerprint": snapshot.catalog.catalog_fingerprint,
            "catalogSize": len(snapshot.catalog.articles),
            "duplicateNameGroups": len(snapshot.catalog.duplicate_name_ids),
            "products": rows,
        }

    def schema_report(self) -> dict[str, Any]:
        """Resolve the live schema without running anything. Diagnostics only."""
        session = SnapshotSession(self.config.database)
        with session.open() as snapshot:
            return snapshot.introspect().to_dict()


# ── helpers ─────────────────────────────────────────────────────────────────


def _error_code_for(exc: BaseException) -> ErrorCode:
    name = exc.__class__.__name__.lower()
    text = str(exc).lower()
    if "timeout" in name or "timed out" in text:
        return ErrorCode.INFERENCE_TIMEOUT
    if "json" in name or "malformed" in text or "json" in text:
        return ErrorCode.MALFORMED_MODEL_OUTPUT
    if isinstance(exc, (asyncio.CancelledError, MemoryError, KeyboardInterrupt)):
        return ErrorCode.INTERNAL_ERROR
    return ErrorCode.INFERENCE_FAILED


def _error_message(exc: BaseException) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return f"Bu mahsulotni tahlil qilib bo'lmadi: {detail[:300]}"


def _default_ai_client_factory(ai_model: str) -> Any:
    from limit_article_matching.ai_client import get_ai_client

    return get_ai_client(ai_model)


__all__ = ["ProductMatchingService", "Snapshot"]
