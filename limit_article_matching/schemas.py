"""API contract for the product limit-article matching service.

Field names follow the brief exactly (camelCase on the wire). Input models are
`extra="forbid"`: a backend typo must fail loudly at 400 rather than be silently
ignored, because a dropped `productIds` entry means a product quietly never gets
analyzed.

The unit of everything here is the **product**, not the purchase-request line.
That is the whole point of this contract: one product has one matching article
within an organization, so the backend stores the answer and reuses it across
every line that carries that product, indefinitely. Nothing request-shaped —
request id, line id, budget year, the article a line currently sits on — appears
in a result or in its fingerprint, precisely so the stored mapping stays valid
no matter which request it came from.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# The brief's proposed ceiling for one call. Deduplication happens before the
# check, so 600 ids collapsing to 400 distinct ones is accepted.
MAX_PRODUCTS_PER_RUN = 500


class MatchStatus(StrEnum):
    MATCHED = "MATCHED"
    AMBIGUOUS = "AMBIGUOUS"
    NO_MATCH = "NO_MATCH"
    ERROR = "ERROR"


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ── requests ────────────────────────────────────────────────────────────────


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    organization_id: int = Field(
        alias="organizationId",
        description="Owns both the requested products and the candidate articles.",
    )
    product_ids: list[int] = Field(
        alias="productIds",
        description=(
            "organization_product.id values — not line ids, corporate item ids, "
            "product codes or MXIK ids."
        ),
    )

    @field_validator("organization_id")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("must be a positive integer")
        return value

    @field_validator("product_ids")
    @classmethod
    def _normalized_ids(cls, value: list[int]) -> list[int]:
        """Sorted and deduplicated, which is also what idempotency hashes.

        The brief asks for exactly this normalization so that the same intent
        expressed in a different order is recognised as the same run.
        """
        if not value:
            raise ValueError("productIds must be a non-empty array")
        if any(v <= 0 for v in value):
            raise ValueError("productIds must contain positive integers")
        normalized = sorted(set(value))
        if len(normalized) > MAX_PRODUCTS_PER_RUN:
            raise ValueError(
                f"productIds holds {len(normalized)} distinct ids, above the limit of "
                f"{MAX_PRODUCTS_PER_RUN} per call"
            )
        return normalized


# ── evidence echoed back ────────────────────────────────────────────────────


class MxikEvidence(BaseModel):
    """One MXIK link that was actually analyzed.

    A product may carry several. The brief requires all valid links to be
    gathered rather than the first one taken, so this is a list on the item and
    `isPlaceholder` says which of them carry no real classification.
    """

    model_config = ConfigDict(populate_by_name=True)

    mxik_id: int = Field(alias="mxikId")
    classification_id: int | None = Field(default=None, alias="classificationId")
    code: str | None = None
    name: str | None = None
    name_ru: str | None = Field(default=None, alias="nameRu")
    name_uz_latn: str | None = Field(default=None, alias="nameUzLatn")
    group: str | None = None
    class_name: str | None = Field(default=None, alias="class")
    position: str | None = None
    subposition: str | None = None
    brand: str | None = None
    attribute: str | None = None
    type: str | None = None
    is_placeholder: bool = Field(default=False, alias="isPlaceholder")
    source: str = Field(
        default="uz_goods_classification",
        description="Which table the link came from — the brief asks for the real source of every attribute.",
    )


class ProductInput(BaseModel):
    """The analyzed snapshot: every product field the recommendation consumed."""

    model_config = ConfigDict(populate_by_name=True)

    product_id: int = Field(alias="productId")
    organization_id: int | None = Field(default=None, alias="organizationId")
    product_code: str | None = Field(default=None, alias="productCode")
    product_name: str | None = Field(default=None, alias="productName")
    deleted: bool | None = None
    updated_at: datetime | None = Field(default=None, alias="updatedAt")
    mxik_links: list[MxikEvidence] = Field(default_factory=list, alias="mxikLinks")


class EvidenceItem(BaseModel):
    """One cited fact. `field` is the source path, `value` its actual content."""

    model_config = ConfigDict(populate_by_name=True)

    source: str = Field(description="product | mxik | catalog")
    field: str
    value: str


class AlternativeCandidate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    limit_article_id: int = Field(alias="limitArticleId")
    article_code: str | None = Field(default=None, alias="articleCode")
    article_name: str | None = Field(default=None, alias="articleName")
    score: float = Field(ge=0.0, le=1.0)
    reason: str


class ProductMatchItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    product_id: int = Field(alias="productId")
    product_code: str | None = Field(default=None, alias="productCode")
    product_name: str | None = Field(default=None, alias="productName")
    source_fingerprint: str = Field(alias="sourceFingerprint")

    match_status: MatchStatus = Field(alias="matchStatus")
    matched_limit_article_id: int | None = Field(default=None, alias="matchedLimitArticleId")
    matched_article_code: str | None = Field(default=None, alias="matchedArticleCode")
    matched_article_name: str | None = Field(default=None, alias="matchedArticleName")
    suggested_parent_limit_article_id: int | None = Field(
        default=None, alias="suggestedParentLimitArticleId"
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str
    input: ProductInput
    evidence: list[EvidenceItem] = Field(default_factory=list)
    alternatives: list[AlternativeCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error_code: str | None = Field(default=None, alias="errorCode")

    @model_validator(mode="after")
    def _enforce_status_contract(self) -> "ProductMatchItem":
        """The brief's invariants, enforced at the boundary rather than trusted.

        A MATCHED item without a target, or a NO_MATCH carrying one, would let
        the backend persist a mapping the model never actually chose — and a
        stored mapping is reused everywhere, so the damage would not stay local.
        """
        if self.match_status is MatchStatus.MATCHED:
            if self.matched_limit_article_id is None:
                raise ValueError("MATCHED requires matchedLimitArticleId")
            if self.confidence is None:
                raise ValueError("MATCHED requires a confidence")
        elif self.matched_limit_article_id is not None:
            raise ValueError(f"{self.match_status} must not carry matchedLimitArticleId")

        if self.match_status in (MatchStatus.NO_MATCH, MatchStatus.ERROR) and self.confidence is not None:
            raise ValueError(f"{self.match_status} must have a null confidence")
        if self.match_status is MatchStatus.ERROR and not self.error_code:
            raise ValueError("ERROR requires errorCode")
        if self.match_status is not MatchStatus.ERROR and self.error_code:
            raise ValueError("errorCode belongs only to ERROR results")
        return self

    @property
    def storable(self) -> bool:
        """Whether the backend may persist this as a reusable product mapping.

        Only a MATCHED result qualifies. An AMBIGUOUS, NO_MATCH or ERROR outcome
        may be kept as analysis history, but storing it as a mapping would make
        an unresolved judgement look settled.
        """
        return self.match_status is MatchStatus.MATCHED


# ── responses ───────────────────────────────────────────────────────────────


class RunCounts(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_products: int = Field(alias="totalProducts")
    processed_products: int = Field(alias="processedProducts")
    matched: int = 0
    ambiguous: int = 0
    no_match: int = Field(default=0, alias="noMatch")
    error: int = 0


class CreateRunResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    status: RunStatus
    organization_id: int = Field(alias="organizationId")
    product_ids: list[int] = Field(alias="productIds")


class RunStatusResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    status: RunStatus
    organization_id: int = Field(alias="organizationId")
    product_ids: list[int] = Field(alias="productIds")
    source_snapshot_at: datetime | None = Field(default=None, alias="sourceSnapshotAt")
    started_at: datetime | None = Field(default=None, alias="startedAt")
    completed_at: datetime | None = Field(default=None, alias="completedAt")
    expires_at: datetime | None = Field(default=None, alias="expiresAt")
    # Null on a rules-only run, which the brief explicitly allows.
    model_version: str | None = Field(default=None, alias="modelVersion")
    rules_version: str = Field(alias="rulesVersion")
    fingerprint_schema_version: str = Field(alias="fingerprintSchemaVersion")
    catalog_fingerprint: str | None = Field(default=None, alias="catalogFingerprint")
    counts: RunCounts
    error: dict[str, Any] | None = None


class RunResultsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    organization_id: int = Field(alias="organizationId")
    items: list[ProductMatchItem]
    next_cursor: str | None = Field(default=None, alias="nextCursor")


class ApiError(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    code: str
    message: str
    trace_id: str = Field(alias="traceId")


__all__ = [
    "MAX_PRODUCTS_PER_RUN",
    "AlternativeCandidate",
    "ApiError",
    "CreateRunRequest",
    "CreateRunResponse",
    "EvidenceItem",
    "MatchStatus",
    "MxikEvidence",
    "ProductInput",
    "ProductMatchItem",
    "RunCounts",
    "RunResultsResponse",
    "RunStatus",
    "RunStatusResponse",
]
