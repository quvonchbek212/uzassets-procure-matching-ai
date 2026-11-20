"""Drive the product limit-article matching service from the command line.

Why this exists rather than just calling the HTTP endpoint: a run over thousands
of products takes hours, nothing resumes a run after a restart (the service fails
any run left RUNNING by a dead process, on purpose, so the backend never polls a
run that will never finish), and reviewing the output is far easier as one CSV
than as paged JSON.

So this splits a product set into chunks, runs them one at a time, and records
what finished in a state file. Re-running with the same state file skips what is
already done, so an interrupted session continues instead of starting over.

Usage:
    # which products the 2026 plan cohort actually uses
    python scripts/run_product_matching.py --list --from-requests

    # a small quality sample first (recommended before committing hours)
    python scripts/run_product_matching.py --from-requests --sample 20 --out var/pm/sample

    # named products
    python scripts/run_product_matching.py --products 183373,183374 --out var/pm/test

    # everything the plan cohort touches, resumable
    python scripts/run_product_matching.py --from-requests --all --out var/pm/full
    #   ...interrupt with Ctrl-C, then re-run the same command to continue

    # extraction and candidate scoring only, no inference and no cost
    python scripts/run_product_matching.py --from-requests --sample 5 --dry-run

--from-requests resolves product ids from the organization's PLAN request lines.
That is a convenience for finding real ids to test with; the service itself never
walks plan requests, it only ever analyzes the ids it is handed.

Output per chunk lands in --out: one JSON file with the full result items, plus a
single results.csv across all chunks for business review.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from limit_article_matching.db_reader import SnapshotSession
from limit_article_matching.errors import ProductMatchingError
from limit_article_matching.schemas import CreateRunRequest, RunStatus
from limit_article_matching.service import ProductMatchingService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logging.getLogger("paramiko").setLevel(logging.WARNING)
logging.getLogger("sshtunnel").setLevel(logging.WARNING)
logger = logging.getLogger("product-matcher")

CSV_COLUMNS = [
    "productId",
    "productCode",
    "productName",
    "mxikCodes",
    "matchStatus",
    "storable",
    "matchedLimitArticleId",
    "matchedArticleCode",
    "matchedArticleName",
    "confidence",
    "explanation",
    "alternatives",
    "warnings",
    "errorCode",
    "sourceFingerprint",
]

_stop_requested = False


def _handle_interrupt(_signum: int, _frame: Any) -> None:
    """Finish the chunk in flight, then stop.

    Killing mid-chunk would leave that run RUNNING and the service would fail it
    on the next start, throwing away work that was nearly done. Waiting for the
    chunk boundary costs at most one chunk and keeps the state file honest.
    """
    global _stop_requested
    if _stop_requested:
        logger.warning("Second interrupt — exiting now; the current chunk will be marked FAILED")
        raise KeyboardInterrupt
    _stop_requested = True
    logger.warning("Interrupt received — finishing the current chunk, then stopping")


# ── state ───────────────────────────────────────────────────────────────────


@dataclass
class RunnerState:
    """What has already been done, so a re-run continues instead of repeating."""

    path: Path
    organization_id: int
    chunks: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, organization_id: int) -> "RunnerState":
        if not path.exists():
            return cls(path=path, organization_id=organization_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        stored = data.get("organizationId")
        if stored is not None and stored != organization_id:
            # Reusing a state file across organizations would skip chunks that
            # were never run for *this* one.
            raise SystemExit(
                f"{path} holds state for organization {stored}; use a different --state file"
            )
        return cls(path=path, organization_id=organization_id, chunks=data.get("chunks", {}))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "organizationId": self.organization_id,
                    "updatedAt": _now(),
                    "chunks": self.chunks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def completed(self, key: str) -> bool:
        return self.chunks.get(key, {}).get("status") == str(RunStatus.COMPLETED)

    def record(self, key: str, payload: dict[str, Any]) -> None:
        self.chunks[key] = {**payload, "updatedAt": _now()}
        self.save()

    def totals(self) -> dict[str, int]:
        out = {"products": 0, "matched": 0, "ambiguous": 0, "noMatch": 0, "error": 0}
        for chunk in self.chunks.values():
            counts = chunk.get("counts") or {}
            out["products"] += counts.get("processedProducts", 0)
            out["matched"] += counts.get("matched", 0)
            out["ambiguous"] += counts.get("ambiguous", 0)
            out["noMatch"] += counts.get("noMatch", 0)
            out["error"] += counts.get("error", 0)
        return out


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── output ──────────────────────────────────────────────────────────────────


def write_chunk_json(out_dir: Path, key: str, run_id: str, items: list[Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chunk-{key}.json"
    path.write_text(
        json.dumps(
            {"runId": run_id, "items": [i.model_dump(by_alias=True, mode="json") for i in items]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def append_csv(out_dir: Path, items: list[Any]) -> Path:
    """One flat file a reviewer can sort and filter, appended chunk by chunk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "results.csv"
    exists = path.exists()
    # utf-8-sig: Excel on Windows reads plain UTF-8 CSV as cp1251 and mangles
    # every Cyrillic article name; the BOM is what makes it open correctly.
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "productId": item.product_id,
                    "productCode": item.product_code,
                    "productName": item.product_name,
                    "mxikCodes": " | ".join(
                        link.code or "" for link in item.input.mxik_links if link.code
                    ),
                    "matchStatus": str(item.match_status),
                    # The column a reviewer cares about most: whether the backend
                    # may persist this as a reusable mapping.
                    "storable": "yes" if item.storable else "no",
                    "matchedLimitArticleId": item.matched_limit_article_id,
                    "matchedArticleCode": item.matched_article_code,
                    "matchedArticleName": item.matched_article_name,
                    "confidence": item.confidence,
                    "explanation": item.explanation,
                    "alternatives": " | ".join(
                        f"{a.limit_article_id}:{a.score}:{a.article_name}" for a in item.alternatives
                    ),
                    "warnings": ",".join(item.warnings),
                    "errorCode": item.error_code,
                    "sourceFingerprint": item.source_fingerprint,
                }
            )
    return path


# ── the run loop ────────────────────────────────────────────────────────────


async def run_chunk(
    service: ProductMatchingService,
    organization_id: int,
    product_ids: list[int],
    *,
    poll_sec: float,
) -> tuple[str, Any, list[Any]]:
    request = CreateRunRequest(organizationId=organization_id, productIds=product_ids)
    # A fresh idempotency key per attempt: a deliberate re-run of a chunk (after a
    # failure, or with a changed prompt) must actually re-run rather than hand
    # back the old result. The state file is what prevents accidental repeats.
    key = f"cli-{organization_id}-{product_ids[0]}-{int(time.time() * 1000)}"
    response = await service.create_run(request, idempotency_key=key)

    while True:
        status = service.get_status(response.run_id)
        if status.status in (RunStatus.COMPLETED, RunStatus.FAILED):
            break
        await asyncio.sleep(poll_sec)

    items: list[Any] = []
    if status.status is RunStatus.COMPLETED:
        cursor = None
        while True:
            page = service.get_results(response.run_id, cursor=cursor, limit=1000)
            items.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break
    return response.run_id, status, items


def resolve_product_ids(args: argparse.Namespace) -> list[int]:
    """Where the ids come from: explicit, or read off the plan cohort."""
    if args.products:
        return sorted({int(x) for x in args.products.split(",") if x.strip()})

    logger.info("Resolving product ids from PLAN requests ...")
    request_ids = (
        [int(x) for x in args.from_requests.split(",") if x.strip()]
        if isinstance(args.from_requests, str) and args.from_requests
        else None
    )
    session = SnapshotSession()
    with session.open() as snapshot:
        return snapshot.fetch_product_ids_for_requests(
            organization_id=args.org, request_ids=request_ids, limit=args.max_products
        )


async def drive(args: argparse.Namespace) -> int:
    service = ProductMatchingService()

    if not args.products and args.from_requests is None:
        logger.error("Choose --products or --from-requests")
        return 2

    product_ids = await asyncio.to_thread(resolve_product_ids, args)
    if not product_ids:
        logger.warning("No products resolved for organization %s", args.org)
        return 0
    logger.info("Resolved %d distinct product(s)", len(product_ids))

    if args.list:
        for product_id in product_ids:
            print(product_id)
        print(f"\nTOTAL {len(product_ids)}")
        return 0

    if args.sample:
        product_ids = product_ids[: args.sample]
        logger.info("Sampling the first %d", len(product_ids))

    if args.dry_run:
        report = await asyncio.to_thread(service.preview, args.org, product_ids[: args.preview_limit])
        print(f"\ncatalog: {report['catalogSize']} articles, "
              f"{report['duplicateNameGroups']} duplicate-name group(s)")
        print(f"catalogFingerprint: {report['catalogFingerprint']}\n")
        for row in report["products"]:
            print(f"[{row['productId']}] {row['productName']}")
            print(f"    mxik links: {row['usableMxikLinks']} usable of {row['mxikLinks']}")
            if row["warnings"]:
                print(f"    warnings: {', '.join(row['warnings'])}")
            for candidate in row["candidates"]:
                print(
                    f"    {candidate['score']:.3f}  id={candidate['limitArticleId']:<6}"
                    f" {str(candidate['articleName'])[:60]}"
                )
            print()
        return 0

    out_dir = Path(args.out)
    state = RunnerState.load(Path(args.state) if args.state else out_dir / "state.json", args.org)

    chunks = [
        product_ids[i : i + args.chunk_size] for i in range(0, len(product_ids), args.chunk_size)
    ]
    pending = [c for c in chunks if not state.completed(str(c[0]))]
    skipped = len(chunks) - len(pending)

    logger.info(
        "Plan: %d chunk(s) of up to %d product(s)%s",
        len(chunks),
        args.chunk_size,
        f"; {skipped} already completed, skipping" if skipped else "",
    )
    if not pending:
        logger.info("Nothing left to do.")
        summarize(state, out_dir)
        return 0

    signal.signal(signal.SIGINT, _handle_interrupt)

    pending_products = sum(len(c) for c in pending)
    started = time.time()
    done = 0

    for index, ids in enumerate(pending, start=1):
        key = str(ids[0])
        logger.info("[%d/%d] chunk %s — %d product(s) ...", index, len(pending), key, len(ids))
        chunk_started = time.time()

        try:
            run_id, status, items = await run_chunk(service, args.org, ids, poll_sec=args.poll_sec)
        except ProductMatchingError as exc:
            logger.error("[%s] rejected: %s %s", key, exc.code, exc.message)
            state.record(key, {"status": "REJECTED", "code": exc.code, "message": exc.message})
            if args.stop_on_error:
                return 1
            continue

        elapsed = time.time() - chunk_started
        counts = status.counts.model_dump(by_alias=True)

        if status.status is RunStatus.FAILED:
            logger.error("[%s] run %s FAILED: %s", key, run_id, status.error)
            state.record(key, {"runId": run_id, "status": "FAILED", "error": status.error})
            if args.stop_on_error:
                return 1
            continue

        json_path = write_chunk_json(out_dir, key, run_id, items)
        append_csv(out_dir, items)
        state.record(
            key,
            {
                "runId": run_id,
                "status": str(status.status),
                "counts": counts,
                "file": json_path.name,
                "seconds": round(elapsed, 1),
            },
        )

        done += counts["processedProducts"]
        rate = done / max(time.time() - started, 1e-6)
        remaining = max(pending_products - done, 0)
        logger.info(
            "[%d/%d] %s done in %.0fs — MATCHED=%d AMBIGUOUS=%d NO_MATCH=%d ERROR=%d%s",
            index,
            len(pending),
            key,
            elapsed,
            counts["matched"],
            counts["ambiguous"],
            counts["noMatch"],
            counts["error"],
            f"  | ~{remaining / rate / 60:.0f} min left" if rate > 0 and remaining else "",
        )

        if _stop_requested:
            logger.warning("Stopping after chunk %s. Re-run the same command to continue.", key)
            break

    summarize(state, out_dir)
    return 0


def summarize(state: RunnerState, out_dir: Path) -> None:
    totals = state.totals()
    products = totals["products"] or 1
    logger.info("─" * 60)
    logger.info(
        "Total: %d product(s) — MATCHED %d (%.1f%%), AMBIGUOUS %d (%.1f%%), "
        "NO_MATCH %d (%.1f%%), ERROR %d",
        totals["products"],
        totals["matched"], 100 * totals["matched"] / products,
        totals["ambiguous"], 100 * totals["ambiguous"] / products,
        totals["noMatch"], 100 * totals["noMatch"] / products,
        totals["error"],
    )
    logger.info(
        "Storable mappings: %d of %d (%.1f%%)",
        totals["matched"],
        totals["products"],
        100 * totals["matched"] / products,
    )
    failed = [k for k, v in state.chunks.items() if v.get("status") != str(RunStatus.COMPLETED)]
    if failed:
        logger.warning("Chunks not completed: %s", ", ".join(sorted(failed)))
    logger.info("Results: %s", (out_dir / "results.csv").resolve())
    logger.info("State:   %s", state.path.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run product limit-article matching in resumable chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--org", type=int, default=1, help="organizationId (default 1)")

    selection = parser.add_argument_group("selection")
    selection.add_argument("--products", help="comma-separated organization_product.id values")
    selection.add_argument(
        "--from-requests",
        nargs="?",
        const="",
        default=None,
        metavar="REQUEST_IDS",
        help="resolve product ids from PLAN requests; optionally limit to given request ids",
    )
    selection.add_argument("--list", action="store_true", help="print the resolved ids and exit")
    selection.add_argument("--sample", type=int, help="keep only the first N products")
    selection.add_argument("--all", action="store_true", help="run every resolved product")
    selection.add_argument(
        "--max-products", type=int, default=20000, help="ceiling when resolving from requests"
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--chunk-size", type=int, default=100, help="products per run (default 100, max 500)"
    )
    execution.add_argument("--out", default="var/pm", help="output directory (default var/pm)")
    execution.add_argument("--state", help="state file (default <out>/state.json)")
    execution.add_argument("--poll-sec", type=float, default=5.0, help="status poll interval")
    execution.add_argument("--stop-on-error", action="store_true", help="stop at the first failure")
    execution.add_argument("--dry-run", action="store_true", help="extract and score candidates only")
    execution.add_argument("--preview-limit", type=int, default=10, help="products shown in --dry-run")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.chunk_size <= 500:
        logger.error("--chunk-size must be between 1 and 500")
        return 2
    try:
        return asyncio.run(drive(args))
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130
    except ProductMatchingError as exc:
        logger.error("%s: %s", exc.code, exc.message)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
