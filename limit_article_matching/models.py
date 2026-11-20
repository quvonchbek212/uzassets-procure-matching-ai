"""Domain objects built from the snapshot rows.

Two jobs live here. First, turning flat query rows into `Product` and
`CatalogArticle`, including the data-quality judgements the brief demands — a
deleted product, a product owned by another organization, an MXIK link that
carries no real classification.

Second, building the canonical payload that gets fingerprinted. For a product
mapping that payload is deliberately narrow: product identity, organization,
deletion flag, names and codes, and every consumed MXIK link. Nothing
request-shaped goes in. The brief is explicit — "It must not include plan
request IDs, line versions or line article assignments" — and the reason is
structural: the backend stores this mapping and reuses it for every line
carrying the product, so a fingerprint that moved whenever a line moved would
mark every stored mapping stale for no reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

from limit_article_matching.errors import WarningCode
from limit_article_matching.fingerprint import fingerprint
from limit_article_matching.normalization import normalize, token_set


def _as_str(value: Any) -> str | None:
    """Codes stay strings so leading zeroes survive; empty becomes None."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = str(value).strip()
    return text or None


def _as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1"}:
        return True
    if text in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dt_or_none(value: Any) -> datetime | None:
    return value if isinstance(value, datetime) else None


# ── MXIK ────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class MxikLink:
    """One MXIK record reached from a product.

    `source` records which table produced the link. The brief asks for the real
    source of every attribute used, and a product reaches MXIK two ways: through
    `uz_goods_classification` rows, and through `organization_product.mxik_id`
    directly. Both are gathered; `Product.mxik_links` deduplicates by `mxik_id`.
    """

    mxik_id: int
    classification_id: int | None
    code: str | None
    name: str | None
    name_ru: str | None
    name_uz_latn: str | None
    group: str | None
    class_name: str | None
    position: str | None
    subposition: str | None
    brand: str | None
    attribute: str | None
    type: str | None
    source: str

    @property
    def has_taxonomy(self) -> bool:
        return any((self.group, self.class_name, self.position, self.subposition))

    def is_placeholder(self, product_name: str | None) -> bool:
        """True when this link carries no real classification.

        The platform mints an internal MXIK row per product when no official
        MXIK was assigned: `type='INTERNAL_PRODUCT_CODE'`, a synthetic `INTP-…`
        code, every taxonomy level null, and a `name` that simply repeats the
        product name. Verified against the live data, where every MXIK row
        reachable from the 2026 plan cohort looks like this.

        This matters beyond bookkeeping. Presenting such a row as a second
        source would double-count one fact: the model would see `product_name`
        and `mxik_name` saying the same words and read it as two independent
        sources agreeing.
        """
        if (self.type or "").strip().upper() == "INTERNAL_PRODUCT_CODE":
            return True
        if (self.code or "").strip().upper().startswith("INTP-"):
            return True
        if self.has_taxonomy:
            return False
        return bool(self.name) and normalize(self.name) == normalize(product_name)

    def evidence_text(self) -> str:
        parts = (
            self.name, self.name_ru, self.name_uz_latn, self.group, self.class_name,
            self.position, self.subposition, self.brand, self.attribute,
        )
        return " ".join(p for p in parts if p)

    def taxonomy_signature(self) -> tuple[str, str]:
        """What two links must agree on to be considered consistent."""
        return normalize(self.group), normalize(self.class_name)

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "mxikId": self.mxik_id,
            "classificationId": self.classification_id,
            "code": self.code,
            "name": self.name,
            "nameRu": self.name_ru,
            "nameUzLatn": self.name_uz_latn,
            "group": self.group,
            "class": self.class_name,
            "position": self.position,
            "subposition": self.subposition,
            "brand": self.brand,
            "attribute": self.attribute,
            "type": self.type,
            "source": self.source,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any], *, source: str) -> "MxikLink | None":
        mxik_id = _int_or_none(row.get("mxik_id"))
        if mxik_id is None:
            return None
        return cls(
            mxik_id=mxik_id,
            classification_id=_int_or_none(row.get("classification_id")),
            code=_as_str(row.get("mxik_kod")),
            name=_as_str(row.get("mxik_name")),
            name_ru=_as_str(row.get("mxik_name_ru")),
            name_uz_latn=_as_str(row.get("mxik_name_uz_latn")),
            group=_as_str(row.get("mxik_group_name")),
            class_name=_as_str(row.get("mxik_class_name")),
            position=_as_str(row.get("mxik_position_name")),
            subposition=_as_str(row.get("mxik_subposition_name")),
            brand=_as_str(row.get("mxik_brand_name")),
            attribute=_as_str(row.get("mxik_attribute_name")),
            type=_as_str(row.get("mxik_type")),
            source=source,
        )


# ── product ─────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class Product:
    """One product with every MXIK link that was found for it."""

    id: int
    organization_id: int | None
    name: str | None
    code: str | None
    deleted: bool | None
    updated_at: datetime | None
    mxik_links: list[MxikLink] = field(default_factory=list)

    warnings: list[str] = field(default_factory=list)
    source_fingerprint: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_rows(
        cls, product_row: dict[str, Any], mxik_rows: list[tuple[dict[str, Any], str]]
    ) -> "Product":
        """Assemble a product and deduplicate its MXIK links by `mxik_id`.

        A product can reach the same MXIK through more than one classification
        row, and again through `organization_product.mxik_id`. The brief asks
        for all valid links gathered and deduplicated — so the first occurrence
        of each `mxik_id` wins, and links are ordered by id to keep the
        fingerprint stable.
        """
        links: dict[int, MxikLink] = {}
        for row, source in mxik_rows:
            link = MxikLink.from_row(row, source=source)
            if link is not None and link.mxik_id not in links:
                links[link.mxik_id] = link

        product = cls(
            id=int(product_row["id"]),
            organization_id=_int_or_none(product_row.get("organization_id")),
            name=_as_str(product_row.get("product_name")),
            code=_as_str(product_row.get("product_kod")),
            deleted=_as_bool(product_row.get("deleted")),
            updated_at=_dt_or_none(product_row.get("updated_at")),
            mxik_links=[links[key] for key in sorted(links)],
            raw=product_row,
        )
        product.warnings = product._derive_warnings()
        product.source_fingerprint = fingerprint(product.fingerprint_payload())
        return product

    # ── evidence views ──────────────────────────────────────────────────────

    @property
    def usable_mxik_links(self) -> list[MxikLink]:
        """Links that carry a real classification, placeholders excluded."""
        return [link for link in self.mxik_links if not link.is_placeholder(self.name)]

    @property
    def has_real_mxik(self) -> bool:
        return bool(self.usable_mxik_links)

    @property
    def has_any_evidence(self) -> bool:
        """Is there anything *describing* the product to reason from?

        A deleted product is excluded outright: the brief treats deleted records
        as unusable evidence, and a mapping stored from one would outlive the
        product it describes.
        """
        if self.deleted:
            return False
        return bool(self.name) or self.has_real_mxik

    def evidence_text(self) -> str:
        """Free text the candidate scorer reads. Original values, not normalized."""
        parts = [self.name]
        # Placeholder links repeat the product name; including them would let one
        # word count twice in every token overlap score.
        parts += [link.evidence_text() for link in self.usable_mxik_links]
        return " ".join(p for p in parts if p)

    def taxonomy_text(self) -> str:
        parts: list[str] = []
        for link in self.usable_mxik_links:
            parts += [link.group, link.class_name, link.position, link.subposition]
        return " ".join(p for p in parts if p)

    def _derive_warnings(self) -> list[str]:
        out: list[str] = []
        if self.deleted:
            out.append(WarningCode.DELETED_PRODUCT)
        if not self.name:
            out.append(WarningCode.EMPTY_PRODUCT_NAME)
        if not self.mxik_links:
            out.append(WarningCode.MISSING_MXIK)
        elif not self.has_real_mxik:
            out.append(WarningCode.MXIK_NOT_CLASSIFIED)

        usable = self.usable_mxik_links
        if len(usable) > 1:
            signatures = {link.taxonomy_signature() for link in usable if link.has_taxonomy}
            if len(signatures) > 1:
                # Two official classifications that disagree. Neither can be
                # assumed, and the disagreement itself is evidence of doubt.
                out.append(WarningCode.CONFLICTING_MXIK_LINKS)

        if self.name and not usable:
            out.append(WarningCode.PRODUCT_NAME_ONLY_EVIDENCE)
        return list(dict.fromkeys(out))

    def fingerprint_payload(self) -> dict[str, Any]:
        """Product identity and evidence only — nothing request-shaped.

        Ordering is fixed (links sorted by `mxik_id` at construction) so the same
        product hashes identically however the database returned its rows.
        """
        return {
            "productId": self.id,
            "organizationId": self.organization_id,
            "productName": self.name,
            "productCode": self.code,
            "deleted": self.deleted,
            "updatedAt": self.updated_at,
            "mxikLinks": [link.fingerprint_payload() for link in self.mxik_links],
        }


# ── catalog ─────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class CatalogArticle:
    id: int
    code: str | None
    name: str | None
    organization_id: int | None
    department_id: int | None
    parent_article_id: int | None
    external_1c_id: str | None
    is_active: bool | None
    updated_at: datetime | None
    # Extra parent paths from treasury_limit_article_parent_link. That table does
    # not exist in the live schema today; the brief still names it, so the link
    # rows are read when present and simply stay empty when not.
    extra_parent_ids: list[int] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    _tokens: frozenset[str] | None = field(default=None, repr=False)

    @property
    def tokens(self) -> frozenset[str]:
        if self._tokens is None:
            self._tokens = token_set(self.name)
        return self._tokens

    @property
    def normalized_name(self) -> str:
        return normalize(self.name)

    @property
    def all_parent_ids(self) -> list[int]:
        seen: list[int] = []
        for candidate in [self.parent_article_id, *self.extra_parent_ids]:
            if candidate is not None and candidate not in seen:
                seen.append(candidate)
        return seen

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CatalogArticle":
        return cls(
            id=int(row["id"]),
            code=_as_str(row.get("article_code")),
            name=_as_str(row.get("article_name")),
            organization_id=_int_or_none(row.get("organization_id")),
            department_id=_int_or_none(row.get("department_id")),
            parent_article_id=_int_or_none(row.get("parent_article_id")),
            external_1c_id=_as_str(row.get("external_1c_id")),
            is_active=_as_bool(row.get("is_active")),
            updated_at=_dt_or_none(row.get("updated_at")),
            raw=row,
        )

    def fingerprint_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "articleCode": self.code,
            "articleName": self.name,
            "organizationId": self.organization_id,
            "departmentId": self.department_id,
            "parentArticleId": self.parent_article_id,
            "extraParentIds": sorted(self.extra_parent_ids),
            "external1cId": self.external_1c_id,
            "isActive": self.is_active,
            "updatedAt": self.updated_at,
        }


@dataclass(slots=True)
class Catalog:
    organization_id: int
    articles: list[CatalogArticle]
    by_id: dict[int, CatalogArticle]
    duplicate_name_ids: dict[str, list[int]]
    catalog_fingerprint: str

    @classmethod
    def build(
        cls,
        organization_id: int,
        rows: list[dict[str, Any]],
        parent_link_rows: list[dict[str, Any]] | None = None,
    ) -> "Catalog":
        articles = [CatalogArticle.from_row(row) for row in rows]
        by_id = {a.id: a for a in articles}

        for link in parent_link_rows or []:
            child = _int_or_none(link.get("article_id"))
            parent = _int_or_none(link.get("parent_article_id"))
            if child is None or parent is None or child not in by_id:
                continue
            article = by_id[child]
            if parent != article.parent_article_id and parent not in article.extra_parent_ids:
                article.extra_parent_ids.append(parent)

        duplicates: dict[str, list[int]] = {}
        for article in articles:
            key = article.normalized_name
            if key:
                duplicates.setdefault(key, []).append(article.id)
        duplicate_name_ids = {k: sorted(v) for k, v in duplicates.items() if len(v) > 1}

        payload = {
            "organizationId": organization_id,
            "articleIds": sorted(by_id),
            "articles": [a.fingerprint_payload() for a in sorted(articles, key=lambda a: a.id)],
        }
        return cls(
            organization_id=organization_id,
            articles=articles,
            by_id=by_id,
            duplicate_name_ids=duplicate_name_ids,
            catalog_fingerprint=fingerprint(payload),
        )

    def duplicates_for(self, article: CatalogArticle) -> list[int]:
        return self.duplicate_name_ids.get(article.normalized_name, [])

    def resolve_parent(self, article_id: int) -> tuple[int | None, WarningCode | None]:
        """An unambiguous parent, or None with the reason it could not be given.

        The brief allows `suggestedParentLimitArticleId` only when a valid parent
        path is unambiguous — several paths, or a parent that is not itself an
        eligible article, means null plus a hierarchy warning.
        """
        article = self.by_id.get(article_id)
        if article is None:
            return None, WarningCode.PARENT_PATH_UNRESOLVED
        parents = article.all_parent_ids
        if not parents:
            return None, None  # a root article legitimately has no parent
        if len(parents) > 1:
            return None, WarningCode.AMBIGUOUS_PARENT_PATH
        parent_id = parents[0]
        if parent_id not in self.by_id or parent_id == article_id:
            return None, WarningCode.PARENT_PATH_UNRESOLVED
        return parent_id, None


__all__ = ["Catalog", "CatalogArticle", "MxikLink", "Product"]
