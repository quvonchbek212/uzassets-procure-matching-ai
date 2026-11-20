"""Per-product inference and the validation that makes its output safe to store.

The model proposes; this module disposes. Nothing the model returns reaches the
backend unchecked, and the stakes are higher here than for a line-level
recommendation: a MATCHED result becomes a stored mapping that the backend
applies to every line carrying the product, so a bad answer is not reviewed once
and forgotten — it is applied silently, everywhere, until something invalidates
it.

  * An article id that is not in the organization's catalog is discarded, not
    passed through.
  * A confidence below the review threshold turns a MATCHED into a review result
    rather than a silent low-quality mapping.
  * Two candidates within the ambiguity margin become AMBIGUOUS even when the
    model picked one, because "the model was slightly more sure" is not evidence.
  * Same-name articles that nothing distinguishes become AMBIGUOUS in code. Their
    names are identical, so a model that picks one cannot have picked it on the
    name.
  * `contextDependent` forces AMBIGUOUS regardless of what else the model said —
    a mapping that depends on the request has no single right answer to store.
  * Evidence entries are dropped unless they carry a real field and value, so an
    explanation can never be backed by invented citations.

Products with no usable evidence never reach the model at all. There is nothing
to classify, the answer is NO_MATCH by definition, and skipping the call keeps a
large batch's cost proportional to its real content.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

from limit_article_matching.candidates import ScoredCandidate, shortlist
from limit_article_matching.errors import ErrorCode, WarningCode
from limit_article_matching.models import Catalog, Product
from limit_article_matching.prompts import (
    SYSTEM_PROMPT,
    build_batch_user_prompt,
    build_user_prompt,
)
from limit_article_matching.schemas import (
    AlternativeCandidate,
    EvidenceItem,
    MatchStatus,
    MxikEvidence,
    ProductInput,
    ProductMatchItem,
)
from limit_article_matching.settings import MatchingConfig

logger = logging.getLogger(__name__)

_VALID_SOURCES = {"product", "mxik", "catalog"}
_MAX_EXPLANATION_CHARS = 1000


@dataclass(slots=True)
class Decision:
    """The validated outcome for one product, before it is bound to a result item."""

    status: MatchStatus
    matched_article_id: int | None
    confidence: float | None
    explanation: str
    evidence: list[EvidenceItem]
    alternatives: list[AlternativeCandidate]
    warnings: list[str]
    error_code: str | None = None

    def with_warnings(self, extra: list[str]) -> "Decision":
        if not extra:
            return self
        return Decision(
            status=self.status,
            matched_article_id=self.matched_article_id,
            confidence=self.confidence,
            explanation=self.explanation,
            evidence=self.evidence,
            alternatives=self.alternatives,
            warnings=list(dict.fromkeys([*self.warnings, *extra])),
            error_code=self.error_code,
        )


class ProductMatcher:
    """Turns one product into one validated decision."""

    def __init__(self, ai_client: Any, catalog: Catalog, config: MatchingConfig) -> None:
        self.ai = ai_client
        self.catalog = catalog
        self.config = config

    # ── entry points ────────────────────────────────────────────────────────

    def decide(self, product: Product) -> Decision:
        shortcut = self._decide_without_model(product)
        if shortcut is not None:
            return shortcut

        candidates, fits_whole = self._candidates_for(product)
        prompt = build_user_prompt(
            product,
            candidates,
            duplicate_name_ids=self.catalog.duplicate_name_ids,
            ordered_by_relevance=not fits_whole,
        )
        raw = self._generate(prompt)
        return self._validate(
            raw,
            product=product,
            candidates=candidates,
            base_warnings=list(dict.fromkeys(product.warnings)),
        )

    def decide_batch(self, products: Sequence[Product]) -> list[Decision | None]:
        """One prompt, several products. `None` means "ask again individually".

        Only safe when every product in the batch sees the *same* candidate list,
        which is true on the whole-catalog path and false once shortlists are
        per-product. That case declines rather than quietly showing one product
        another product's candidates.

        Nothing here is trusted to come back complete. An item the model skipped,
        duplicated, or answered with an unknown itemId yields `None`, and the
        caller re-asks that product on its own — a batch must never be able to
        drop a product or hand one product's answer to another.
        """
        if not products:
            return []

        outcomes: list[Decision | None] = [None] * len(products)
        pending: list[tuple[int, Product]] = []
        for index, product in enumerate(products):
            shortcut = self._decide_without_model(product)
            if shortcut is not None:
                outcomes[index] = shortcut
            else:
                pending.append((index, product))

        if not pending:
            return outcomes

        candidates, fits_whole = self._candidates_for(pending[0][1])
        if not fits_whole:
            return outcomes

        items = [(position + 1, product) for position, (_, product) in enumerate(pending)]
        prompt = build_batch_user_prompt(
            items,
            candidates,
            duplicate_name_ids=self.catalog.duplicate_name_ids,
            ordered_by_relevance=False,
        )
        # No retry: a batch that timed out will time out again on the same prompt,
        # and the individual fallback is the better second attempt.
        raw = self._generate(
            prompt,
            max_tokens=self._batch_max_tokens(len(items)),
            timeout_sec=self._batch_timeout(len(items)),
            attempts=1,
        )

        by_item = self._read_batch_results(raw, expected=len(items))
        for position, (index, product) in enumerate(pending):
            payload = by_item.get(position + 1)
            if payload is None:
                continue  # left as None: the caller re-asks this one alone
            outcomes[index] = self._validate(
                payload,
                product=product,
                candidates=candidates,
                base_warnings=list(dict.fromkeys(product.warnings))
                + [WarningCode.BATCHED_INFERENCE],
            )
        return outcomes

    # ── shared setup ────────────────────────────────────────────────────────

    def _decide_without_model(self, product: Product) -> Decision | None:
        """The answers that need no inference, so a batch never wastes room."""
        base_warnings = list(dict.fromkeys(product.warnings))

        if not self.catalog.articles:
            return Decision(
                status=MatchStatus.NO_MATCH,
                matched_article_id=None,
                confidence=None,
                explanation=(
                    "Tashkilotning aktiv limit modda katalogi bo'sh, shuning uchun "
                    "hech qanday nomzod modda mavjud emas."
                ),
                evidence=[],
                alternatives=[],
                warnings=base_warnings + [WarningCode.EMPTY_CANDIDATE_CATALOG],
            )

        if not product.has_any_evidence:
            return Decision(
                status=MatchStatus.NO_MATCH,
                matched_article_id=None,
                confidence=None,
                explanation=self._no_evidence_explanation(product),
                evidence=[],
                alternatives=[],
                warnings=base_warnings,
            )
        return None

    def _candidates_for(self, product: Product) -> tuple[list[ScoredCandidate], bool]:
        """The candidate set and whether it is the whole catalog.

        A small catalog goes to the model whole: shortlisting can only lose a
        correct answer, and it buys nothing once every article fits the prompt.
        """
        fits_whole = len(self.catalog.articles) <= self.config.full_catalog_threshold
        candidates = shortlist(
            product,
            self.catalog,
            size=len(self.catalog.articles) if fits_whole else self.config.candidate_shortlist_size,
            keep_zero_score=fits_whole,
        )
        return candidates, fits_whole

    # ── inference ───────────────────────────────────────────────────────────

    def _generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        timeout_sec: int | None = None,
        attempts: int | None = None,
    ) -> dict[str, Any]:
        """One inference, retried on the failures that are worth retrying.

        Two observed modes justify this. A cold model answers 200 with an *empty*
        completion while it loads, which the client surfaces as an error — without
        a retry the first few products of every run would come back ERROR purely
        because the model had not warmed up. And a model occasionally emits
        unparsable JSON, which a second sample usually fixes.

        Deterministic failures are not retried: a 4xx means the request itself is
        wrong, and repeating it only wastes the run's time budget.
        """
        last: Exception | None = None
        budget = attempts or self.config.max_attempts
        for attempt in range(budget):
            try:
                return self.ai.generate_json(
                    prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens,
                    timeout_sec=timeout_sec or self.config.timeout_sec,
                )
            except Exception as exc:  # noqa: BLE001 — re-raised below
                if not _is_retryable(exc) or attempt == budget - 1:
                    raise
                last = exc
                logger.info(
                    "Inference attempt %d/%d failed (%s); retrying",
                    attempt + 1,
                    budget,
                    str(exc)[:120],
                )
                time.sleep(self.config.retry_backoff_sec * (attempt + 1))
        raise last if last else RuntimeError("inference produced no result")

    def _batch_max_tokens(self, item_count: int) -> int:
        """Room for every item's answer, or the batch truncates mid-array."""
        return min(
            self.config.batch_max_tokens_per_item * item_count + 200,
            self.config.batch_max_tokens_cap,
        )

    def _batch_timeout(self, item_count: int) -> int:
        """Generation time grows with the number of answers, so the wait must too.

        Leaving the single-item timeout on a nine-item batch is what made the
        first live batched run fail: ~3,000 tokens of output does not fit a budget
        sized for ~340.
        """
        return self.config.timeout_sec + self.config.batch_timeout_per_item_sec * item_count

    @staticmethod
    def _read_batch_results(raw: dict[str, Any], *, expected: int) -> dict[int, dict[str, Any]]:
        """Map itemId -> payload, dropping anything that cannot be trusted."""
        results = raw.get("results")
        if not isinstance(results, list):
            # Some models answer a single-item batch with a bare object.
            results = [raw] if expected == 1 and "matchStatus" in raw else []

        by_item: dict[int, dict[str, Any]] = {}
        seen_twice: set[int] = set()
        for position, entry in enumerate(results):
            if not isinstance(entry, dict):
                continue
            item_id = _read_int(entry.get("itemId"))
            if item_id is None and len(results) == expected:
                # No itemId but the array is complete and ordered: position is a
                # usable fallback. A short or long array gets no such benefit.
                item_id = position + 1
            if item_id is None or not (1 <= item_id <= expected):
                continue
            if item_id in by_item:
                # A repeated itemId means the model lost track; neither copy can
                # be trusted, so both go back for an individual answer.
                seen_twice.add(item_id)
                continue
            by_item[item_id] = entry

        for item_id in seen_twice:
            by_item.pop(item_id, None)
        return by_item

    # ── validation ──────────────────────────────────────────────────────────

    def _validate(
        self,
        raw: dict[str, Any],
        *,
        product: Product,
        candidates: list[ScoredCandidate],
        base_warnings: list[str],
    ) -> Decision:
        warnings = list(base_warnings)
        offered_ids = {c.id for c in candidates}

        status = self._read_status(raw)
        article_id = _read_int(raw.get("matchedLimitArticleId"))
        confidence = _read_confidence(raw.get("confidence"))
        explanation = _read_text(raw.get("explanation"), _MAX_EXPLANATION_CHARS)
        evidence = self._read_evidence(raw.get("evidence"))
        alternatives = self._read_alternatives(raw.get("alternatives"), exclude=article_id)
        context_dependent = raw.get("contextDependent") is True

        # An id outside the catalog is a hallucination; an id inside the catalog
        # but outside what was offered is legal but unexpected, so it is kept and
        # flagged rather than dropped.
        if article_id is not None and article_id not in self.catalog.by_id:
            warnings.append(WarningCode.MODEL_PROPOSED_UNKNOWN_ARTICLE)
            article_id = None
            status = MatchStatus.AMBIGUOUS if alternatives else MatchStatus.NO_MATCH
        elif article_id is not None and article_id not in offered_ids:
            warnings.append(WarningCode.MODEL_PROPOSED_UNKNOWN_ARTICLE)

        if status is MatchStatus.MATCHED and article_id is None:
            status = MatchStatus.AMBIGUOUS if alternatives else MatchStatus.NO_MATCH
        if status is not MatchStatus.MATCHED:
            article_id = None

        # A mapping that depends on the request has no single right answer to
        # store, whatever else the model concluded.
        if context_dependent:
            warnings.append(WarningCode.CONTEXT_DEPENDENT)
            if status is MatchStatus.MATCHED:
                alternatives = self._promote_to_alternative(article_id, confidence, alternatives)
            status, article_id, confidence = MatchStatus.AMBIGUOUS, None, None

        # Same-name articles: their names are identical, so a choice between them
        # cannot have been made on the name. Nothing else is available at product
        # level, so this is a review result by definition.
        if article_id is not None:
            siblings = [
                i
                for i in self.catalog.duplicates_for(self.catalog.by_id[article_id])
                if i != article_id
            ]
            if siblings:
                warnings.append(WarningCode.DUPLICATE_ARTICLE_NAME)
                alternatives = self._as_alternatives([article_id, *siblings], alternatives)
                status, article_id, confidence = MatchStatus.AMBIGUOUS, None, None

        if status is MatchStatus.MATCHED:
            if confidence is None:
                confidence = 0.5
                warnings.append(WarningCode.MODEL_CONFIDENCE_CLAMPED)

            top_alternative = alternatives[0].score if alternatives else 0.0
            if confidence < self.config.min_confidence_for_match:
                warnings.append(WarningCode.LOW_CONFIDENCE)
                alternatives = self._promote_to_alternative(article_id, confidence, alternatives)
                status, article_id, confidence = MatchStatus.AMBIGUOUS, None, None
            elif top_alternative and (confidence - top_alternative) < self.config.ambiguity_margin:
                warnings.append(WarningCode.CONTRADICTORY_EVIDENCE)
                alternatives = self._promote_to_alternative(article_id, confidence, alternatives)
                status, article_id, confidence = MatchStatus.AMBIGUOUS, None, None

        if status is not MatchStatus.MATCHED:
            confidence = None

        if status is MatchStatus.AMBIGUOUS and not alternatives:
            # AMBIGUOUS means "two or more plausible candidates"; without any to
            # show, the honest answer is that nothing was established.
            status = MatchStatus.NO_MATCH

        if not explanation:
            explanation = self._fallback_explanation(status, article_id)

        return Decision(
            status=status,
            matched_article_id=article_id,
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            alternatives=alternatives[: self.config.max_alternatives],
            warnings=list(dict.fromkeys(warnings)),
        )

    def _read_status(self, raw: dict[str, Any]) -> MatchStatus:
        value = str(raw.get("matchStatus") or "").strip().upper()
        if value in {MatchStatus.MATCHED, MatchStatus.AMBIGUOUS, MatchStatus.NO_MATCH}:
            return MatchStatus(value)
        # ERROR is never the model's to declare — it means the pipeline failed,
        # not that the model decided the product was unclassifiable.
        return MatchStatus.NO_MATCH

    def _read_evidence(self, raw: Any) -> list[EvidenceItem]:
        if not isinstance(raw, list):
            return []
        items: list[EvidenceItem] = []
        for entry in raw[:12]:
            if not isinstance(entry, dict):
                continue
            field_name = _read_text(entry.get("field"), 120)
            value = _read_text(entry.get("value"), 400)
            if not field_name or not value:
                continue
            source = str(entry.get("source") or "").strip().lower()
            items.append(
                EvidenceItem(
                    source=source if source in _VALID_SOURCES else "product",
                    field=field_name,
                    value=value,
                )
            )
        return items

    def _read_alternatives(self, raw: Any, *, exclude: int | None) -> list[AlternativeCandidate]:
        if not isinstance(raw, list):
            return []
        out: list[AlternativeCandidate] = []
        seen: set[int] = set()
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            article_id = _read_int(entry.get("limitArticleId") or entry.get("articleId"))
            if article_id is None or article_id == exclude or article_id in seen:
                continue
            article = self.catalog.by_id.get(article_id)
            if article is None:
                continue  # same rule as the primary: catalog or nothing
            seen.add(article_id)
            out.append(
                AlternativeCandidate(
                    limitArticleId=article_id,
                    articleCode=article.code,
                    articleName=article.name,
                    score=_read_confidence(entry.get("score")) or 0.0,
                    reason=_read_text(entry.get("reason"), 300) or "Nomzod sifatida ko'rib chiqildi.",
                )
            )
        out.sort(key=lambda a: -a.score)
        return out

    def _promote_to_alternative(
        self, article_id: int | None, confidence: float | None, alternatives: list[AlternativeCandidate]
    ) -> list[AlternativeCandidate]:
        """Keep a demoted pick visible — the reviewer still needs to see it."""
        if article_id is None:
            return alternatives
        article = self.catalog.by_id.get(article_id)
        if article is None or any(a.limit_article_id == article_id for a in alternatives):
            return alternatives
        promoted = AlternativeCandidate(
            limitArticleId=article_id,
            articleCode=article.code,
            articleName=article.name,
            score=confidence or 0.0,
            reason=(
                "Model shu moddani tanladi, lekin uni saqlanadigan mapping sifatida "
                "tasdiqlash uchun dalil yetarli emas."
            ),
        )
        return sorted([promoted, *alternatives], key=lambda a: -a.score)

    def _as_alternatives(
        self, article_ids: list[int], existing: list[AlternativeCandidate]
    ) -> list[AlternativeCandidate]:
        """Put a set of tied articles in front of the reviewer, scores intact."""
        by_id = {a.limit_article_id: a for a in existing}
        out: list[AlternativeCandidate] = []
        for article_id in article_ids:
            if article_id in by_id:
                out.append(by_id.pop(article_id))
                continue
            article = self.catalog.by_id.get(article_id)
            if article is None:
                continue
            out.append(
                AlternativeCandidate(
                    limitArticleId=article_id,
                    articleCode=article.code,
                    articleName=article.name,
                    score=0.5,
                    reason=(
                        "Bir xil nomli moddalardan biri; ularni ajratadigan dalil "
                        "topilmadi, shuning uchun tanlovni odam qilishi kerak."
                    ),
                )
            )
        return out + list(by_id.values())

    def _no_evidence_explanation(self, product: Product) -> str:
        if product.deleted:
            return (
                "Mahsulot manba tizimida o'chirilgan deb belgilangan, shuning uchun undan "
                "qayta ishlatiladigan modda mapping'i yaratilmadi."
            )
        if not product.name and not product.mxik_links:
            return (
                "Mahsulotda na nom, na MXIK tasnifi bor; xarajat moddasini aniqlash uchun "
                "dalil yetarli emas."
            )
        return (
            "Mahsulotda tasniflash uchun yetarli dalil yo'q — nom bo'sh yoki MXIK "
            "tasnifi mavjud emas."
        )

    def _fallback_explanation(self, status: MatchStatus, article_id: int | None) -> str:
        if status is MatchStatus.MATCHED and article_id is not None:
            article = self.catalog.by_id.get(article_id)
            name = article.name if article else str(article_id)
            return f"Mahsulot dalili '{name}' moddasiga mos keladi."
        if status is MatchStatus.AMBIGUOUS:
            return "Bir nechta modda bir xil darajada mos keladi; tanlash uchun qo'shimcha dalil kerak."
        return "Mavjud dalil biror moddani ishonchli tanlash uchun yetarli emas."


# ── result assembly ─────────────────────────────────────────────────────────


def build_result_item(product: Product, decision: Decision, catalog: Catalog) -> ProductMatchItem:
    """Bind a decision to its product and resolve the suggested parent."""
    warnings = list(decision.warnings)
    matched_code: str | None = None
    matched_name: str | None = None
    suggested_parent: int | None = None

    if decision.matched_article_id is not None:
        article = catalog.by_id.get(decision.matched_article_id)
        if article is not None:
            matched_code, matched_name = article.code, article.name
        suggested_parent, parent_warning = catalog.resolve_parent(decision.matched_article_id)
        if parent_warning is not None:
            warnings.append(parent_warning)

    return ProductMatchItem(
        productId=product.id,
        productCode=product.code,
        productName=product.name,
        sourceFingerprint=product.source_fingerprint,
        matchStatus=decision.status,
        matchedLimitArticleId=decision.matched_article_id,
        matchedArticleCode=matched_code,
        matchedArticleName=matched_name,
        suggestedParentLimitArticleId=suggested_parent,
        confidence=decision.confidence,
        explanation=decision.explanation,
        input=_product_input(product),
        evidence=decision.evidence,
        alternatives=decision.alternatives,
        warnings=list(dict.fromkeys(warnings)),
        errorCode=decision.error_code,
    )


def build_error_item(product: Product, code: ErrorCode, message: str) -> ProductMatchItem:
    """A product that could not be analyzed still comes back, never omitted."""
    return ProductMatchItem(
        productId=product.id,
        productCode=product.code,
        productName=product.name,
        sourceFingerprint=product.source_fingerprint,
        matchStatus=MatchStatus.ERROR,
        matchedLimitArticleId=None,
        confidence=None,
        explanation=message,
        input=_product_input(product),
        evidence=[],
        alternatives=[],
        warnings=list(dict.fromkeys(product.warnings)),
        errorCode=str(code),
    )


def _product_input(product: Product) -> ProductInput:
    """The analyzed snapshot the brief requires on every item."""
    return ProductInput(
        productId=product.id,
        organizationId=product.organization_id,
        productCode=product.code,
        productName=product.name,
        deleted=product.deleted,
        updatedAt=product.updated_at,
        mxikLinks=[
            MxikEvidence(
                mxikId=link.mxik_id,
                classificationId=link.classification_id,
                code=link.code,
                name=link.name,
                nameRu=link.name_ru,
                nameUzLatn=link.name_uz_latn,
                group=link.group,
                **{"class": link.class_name},
                position=link.position,
                subposition=link.subposition,
                brand=link.brand,
                attribute=link.attribute,
                type=link.type,
                isPlaceholder=link.is_placeholder(product.name),
                source=link.source,
            )
            for link in product.mxik_links
        ],
    )


# ── small readers ───────────────────────────────────────────────────────────


def _is_retryable(exc: Exception) -> bool:
    """Transient failures only.

    A 4xx is the request's own fault and will fail identically next time; an
    empty completion, a timeout, a 5xx or unparsable JSON are all states that a
    second attempt genuinely clears.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500:
        return False
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "empty generation", "timed out", "timeout", "malformed json", "not valid json",
            "no choices", "temporarily", "connection", "500", "502", "503", "504",
        )
    )


def _read_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _read_confidence(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    # Models occasionally answer on a 0-100 scale despite the instruction.
    if number > 1.0:
        number = number / 100.0 if number <= 100.0 else 1.0
    return max(0.0, min(1.0, number))


def _read_text(value: Any, limit: int) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())[:limit]


__all__ = ["Decision", "ProductMatcher", "build_error_item", "build_result_item"]
