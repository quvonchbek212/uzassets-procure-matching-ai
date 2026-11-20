"""FastAPI surface for the product limit-article matching service.

Endpoints follow the brief's proposed contract:

    POST /v1/product-limit-article-matching/runs
    GET  /v1/product-limit-article-matching/runs/{runId}
    GET  /v1/product-limit-article-matching/runs/{runId}/results

These three are the whole contract; nothing else is exposed. Schema verification
and candidate review live in scripts/run_product_matching.py instead, where they
cost nobody an endpoint to maintain.

Errors carry `code`, `message` and `traceId` and never leak SQL or credentials —
`_safe_db_error` in db_reader strips the driver's detail before it reaches here,
and the unexpected-exception handler replies with a generic message while the
real traceback goes to the log under the same traceId.

**There is no authentication here, by decision.** These endpoints are open, so
whatever protects them is network-level: the service must stay on the internal
network and must not be exposed publicly. Anyone who can reach the URL can start
a run and read product names and the budget catalog.

That also changes what the `organizationId` query parameter means. It still
scopes a status or results lookup, which keeps one organization's runs from
leaking into another's UI by accident — but it is self-declared and verifies
nothing. The brief asks for status/results access to be bound to the authorized
caller; that binding cannot exist until a credential does.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Path, Query, Request
from fastapi.responses import JSONResponse

from limit_article_matching.errors import InvalidRequestError, ProductMatchingError
from limit_article_matching.schemas import (
    CreateRunRequest,
    CreateRunResponse,
    RunResultsResponse,
    RunStatusResponse,
)
from limit_article_matching.service import ProductMatchingService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/v1/product-limit-article-matching",
    tags=["Product Limit Article Matching"],
)

MAX_IDEMPOTENCY_KEY_CHARS = 200


@lru_cache(maxsize=1)
def get_service() -> ProductMatchingService:
    """One service per process: it owns the SQLite store and the run semaphore."""
    service = ProductMatchingService()
    orphaned = service.store.reset_orphaned_runs()
    if orphaned:
        logger.warning("Failed %d run(s) left QUEUED/RUNNING by a previous process", orphaned)
    service.store.purge_expired()
    return service


ServiceDep = Annotated[ProductMatchingService, Depends(get_service)]
OrgScope = Annotated[
    int | None,
    Query(
        alias="organizationId",
        ge=1,
        description=(
            "Scopes the lookup to one organization. Self-declared — it prevents an "
            "accidental cross-organization read, not a deliberate one."
        ),
    ),
]


def _trace_id(request: Request) -> str:
    """Reuse the caller's correlation id when it sent one."""
    incoming = request.headers.get("X-Request-Id") or request.headers.get("X-Trace-Id")
    return incoming.strip()[:64] if incoming else uuid.uuid4().hex


def _error_response(exc: ProductMatchingError, trace_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message, "traceId": trace_id},
    )


def _internal_error(trace_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"code": "internal_error", "message": "Unexpected server error", "traceId": trace_id},
    )


@router.post(
    "/runs",
    response_model=CreateRunResponse,
    status_code=202,
    summary="Start a product matching run",
    response_model_by_alias=True,
    responses={
        400: {"description": "Malformed or oversized input"},
        404: {"description": "Requested products are not available to this organization"},
        409: {"description": "Idempotency conflict"},
        503: {"description": "Database or schema unavailable"},
    },
)
async def create_run(
    request: Request,
    payload: CreateRunRequest,
    service: ServiceDep,
    idempotency_key: Annotated[
        str | None,
        Header(
            alias="Idempotency-Key",
            description=(
                "Required. Same caller, key and normalized body returns the original run; "
                "a changed body returns 409."
            ),
        ),
    ] = None,
) -> JSONResponse:
    trace_id = _trace_id(request)

    # Required by the brief, unlike the line-matching contract where it was
    # optional. A product run is expensive and its result gets stored, so an
    # accidental repeat is worth preventing rather than tolerating.
    key = (idempotency_key or "").strip()
    if not key:
        return _error_response(InvalidRequestError("Idempotency-Key header is required"), trace_id)
    if len(key) > MAX_IDEMPOTENCY_KEY_CHARS:
        return _error_response(InvalidRequestError("Idempotency-Key is too long"), trace_id)

    try:
        response = await service.create_run(payload, idempotency_key=key)
    except ProductMatchingError as exc:
        logger.warning("[%s] create_run rejected: %s %s", trace_id, exc.code, exc.message)
        return _error_response(exc, trace_id)
    except Exception:
        logger.exception("[%s] create_run failed", trace_id)
        return _internal_error(trace_id)

    return JSONResponse(
        status_code=202,
        content=response.model_dump(by_alias=True, mode="json"),
        headers={"X-Trace-Id": trace_id},
    )


@router.get(
    "/runs/{runId}",
    response_model=RunStatusResponse,
    response_model_by_alias=True,
    summary="Run status and counts",
    responses={404: {"description": "Run not found or expired"}},
)
async def get_run_status(
    request: Request,
    service: ServiceDep,
    run_id: Annotated[str, Path(alias="runId", max_length=64)],
    organization_id: OrgScope = None,
) -> JSONResponse:
    trace_id = _trace_id(request)
    try:
        status = service.get_status(run_id, organization_id=organization_id)
    except ProductMatchingError as exc:
        return _error_response(exc, trace_id)
    except Exception:
        logger.exception("[%s] get_run_status failed for %s", trace_id, run_id)
        return _internal_error(trace_id)
    return JSONResponse(
        status_code=200,
        content=status.model_dump(by_alias=True, mode="json"),
        headers={"X-Trace-Id": trace_id},
    )


@router.get(
    "/runs/{runId}/results",
    response_model=RunResultsResponse,
    response_model_by_alias=True,
    summary="Paginated results — served only after COMPLETED",
    responses={
        404: {"description": "Run not found or expired"},
        409: {"description": "Run has not completed"},
    },
)
async def get_run_results(
    request: Request,
    service: ServiceDep,
    run_id: Annotated[str, Path(alias="runId", max_length=64)],
    cursor: Annotated[str | None, Query(max_length=512)] = None,
    limit: Annotated[int | None, Query(ge=1, le=1000)] = None,
    organization_id: OrgScope = None,
) -> JSONResponse:
    trace_id = _trace_id(request)
    try:
        results = await asyncio.to_thread(
            service.get_results,
            run_id,
            cursor=cursor,
            limit=limit,
            organization_id=organization_id,
        )
    except ProductMatchingError as exc:
        return _error_response(exc, trace_id)
    except Exception:
        logger.exception("[%s] get_run_results failed for %s", trace_id, run_id)
        return _internal_error(trace_id)
    return JSONResponse(
        status_code=200,
        content=results.model_dump(by_alias=True, mode="json"),
        headers={"X-Trace-Id": trace_id},
    )


__all__ = ["get_service", "router"]
