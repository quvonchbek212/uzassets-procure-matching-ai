"""Structured error and warning codes.

The brief requires machine-readable codes on ERROR results and structured
warning codes on every result, so both vocabularies live here rather than as
string literals scattered through the pipeline. Adding a code is additive; the
backend treats an unknown code as an opaque string.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Per-product failures. A product always comes back — never dropped."""

    INFERENCE_FAILED = "INFERENCE_FAILED"
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    MALFORMED_MODEL_OUTPUT = "MALFORMED_MODEL_OUTPUT"
    EVIDENCE_EXTRACTION_FAILED = "EVIDENCE_EXTRACTION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class WarningCode(StrEnum):
    """Data-quality, hierarchy and reusability signals attached to a result."""

    # Evidence gaps
    MISSING_MXIK = "MISSING_MXIK"
    EMPTY_PRODUCT_NAME = "EMPTY_PRODUCT_NAME"
    # Every MXIK link is an internal placeholder: it carries no taxonomy and its
    # name merely repeats the product name.
    MXIK_NOT_CLASSIFIED = "MXIK_NOT_CLASSIFIED"
    PRODUCT_NAME_ONLY_EVIDENCE = "PRODUCT_NAME_ONLY_EVIDENCE"
    # Several MXIK links whose taxonomies disagree — the product is classified
    # two different ways and neither can be assumed.
    CONFLICTING_MXIK_LINKS = "CONFLICTING_MXIK_LINKS"

    # Unusable evidence — present but not trustworthy
    DELETED_PRODUCT = "DELETED_PRODUCT"

    # Catalog / hierarchy
    EMPTY_CANDIDATE_CATALOG = "EMPTY_CANDIDATE_CATALOG"
    DUPLICATE_ARTICLE_NAME = "DUPLICATE_ARTICLE_NAME"
    AMBIGUOUS_PARENT_PATH = "AMBIGUOUS_PARENT_PATH"
    PARENT_PATH_UNRESOLVED = "PARENT_PATH_UNRESOLVED"

    # Reusability — the point of a product-level mapping
    #
    # The backend stores a MATCHED result and reuses it for every line carrying
    # this product, anywhere, indefinitely. A product whose article depends on
    # *which request* it appears in therefore has no single correct answer, and
    # saving one would be worse than saving nothing.
    CONTEXT_DEPENDENT = "CONTEXT_DEPENDENT"

    # Model behaviour
    MODEL_PROPOSED_UNKNOWN_ARTICLE = "MODEL_PROPOSED_UNKNOWN_ARTICLE"
    MODEL_CONFIDENCE_CLAMPED = "MODEL_CONFIDENCE_CLAMPED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    CONTRADICTORY_EVIDENCE = "CONTRADICTORY_EVIDENCE"
    BATCHED_INFERENCE = "BATCHED_INFERENCE"


class ProductMatchingError(Exception):
    """Run-level failure surfaced as an HTTP error with a stable code."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str, *, code: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class InvalidRequestError(ProductMatchingError):
    status_code = 400
    code = "invalid_request"


class ProductsUnavailableError(ProductMatchingError):
    """Requested products that this organization cannot see.

    Deliberately does not distinguish "no such product" from "belongs to another
    organization": telling them apart would confirm the existence of another
    organization's products, which the brief forbids.
    """

    status_code = 404
    code = "products_unavailable"


class RunNotFoundError(ProductMatchingError):
    status_code = 404
    code = "run_not_found"


class RunNotCompletedError(ProductMatchingError):
    status_code = 409
    code = "run_not_completed"


class IdempotencyConflictError(ProductMatchingError):
    status_code = 409
    code = "idempotency_conflict"


class DependencyUnavailableError(ProductMatchingError):
    status_code = 503
    code = "dependency_unavailable"


__all__ = [
    "DependencyUnavailableError",
    "ErrorCode",
    "IdempotencyConflictError",
    "InvalidRequestError",
    "ProductMatchingError",
    "ProductsUnavailableError",
    "RunNotCompletedError",
    "RunNotFoundError",
    "WarningCode",
]
