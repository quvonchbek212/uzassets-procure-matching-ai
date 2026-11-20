"""Shortlisting the catalog before inference.

An organization's catalog can run to hundreds or thousands of articles. Putting
all of them in a prompt is neither affordable nor accurate — models pick worse
from long lists. So a cheap deterministic scorer narrows the field, and the model
chooses within it.

The scorer never decides the answer. It only decides what the model is allowed to
see, and it is tuned to be *generous*: a candidate wrongly kept costs a few
prompt tokens, a candidate wrongly dropped can never be recovered. Two rules from
the brief are enforced here rather than left to the model:

  * **Numbers are not evidence of an article code.** Model numbers, document
    numbers and UUID fragments must not match article codes, so purely numeric
    and alphanumeric-model tokens are removed from the query before scoring, and
    article codes are never compared against product text at all.
  * **Product-level evidence only.** There is deliberately no department or
    request-context signal here. A product mapping is stored and reused across
    every request, so scoring it against one request's department would bake a
    local accident into a global answer.

On a small catalog the shortlist is skipped entirely — see `LineMatcher` in
matcher.py — and the scorer's output is used only for ordering, if at all.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from limit_article_matching.models import Catalog, CatalogArticle, Product
from limit_article_matching.normalization import contains_phrase, coverage, jaccard, stems

# A token that is only digits, or digits glued to a short letter run (M12, 40W,
# GOST-2590, 5W-30, X20H80) is a specification, not a category word.
_MODEL_NUMBER_RE = re.compile(r"^[0-9]+$|^[0-9]+[a-z]{1,3}$|^[a-z]{1,3}[0-9]+$")


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    article: CatalogArticle
    score: float
    signals: tuple[str, ...]

    @property
    def id(self) -> int:
        return self.article.id


def _query_tokens(product: Product) -> frozenset[str]:
    """Category-bearing tokens from the product's evidence, numbers excluded."""
    return frozenset(t for t in stems(product.evidence_text()) if not _MODEL_NUMBER_RE.match(t))


def _taxonomy_tokens(product: Product) -> frozenset[str]:
    """MXIK taxonomy levels — the strongest categorical signal when present."""
    return frozenset(t for t in stems(product.taxonomy_text()) if not _MODEL_NUMBER_RE.match(t))


def score_article(
    article: CatalogArticle,
    *,
    query: frozenset[str],
    taxonomy: frozenset[str],
    evidence_text: str,
) -> ScoredCandidate:
    article_tokens = article.tokens
    signals: list[str] = []

    if not article_tokens or not (query or taxonomy):
        return ScoredCandidate(article=article, score=0.0, signals=())

    # How much of the *article name* the evidence accounts for. Asymmetric on
    # purpose — a two-word article fully covered by a long product description is
    # a strong hit, and symmetric Jaccard would bury it.
    name_coverage = coverage(article_tokens, query)
    overlap = jaccard(article_tokens, query)
    taxonomy_coverage = coverage(article_tokens, taxonomy)

    score = 0.55 * name_coverage + 0.20 * overlap + 0.25 * taxonomy_coverage
    if name_coverage >= 0.99:
        signals.append("article_name_fully_covered")
    if taxonomy_coverage > 0:
        signals.append("mxik_taxonomy_overlap")

    # A whole-word phrase hit ("yoqilgi", "ehtiyot qism") is worth more than the
    # same tokens scattered across unrelated text.
    if article.name and contains_phrase(evidence_text, article.name):
        score += 0.15
        signals.append("article_name_phrase_in_evidence")

    return ScoredCandidate(article=article, score=min(score, 1.0), signals=tuple(signals))


def shortlist(
    product: Product,
    catalog: Catalog,
    *,
    size: int,
    keep_zero_score: bool = False,
) -> list[ScoredCandidate]:
    """The `size` best candidates for this product.

    `keep_zero_score` is how the caller asks for the whole catalog: when every
    article fits one prompt, an article the lexical scorer failed to connect must
    still be offered, because the scorer is a poor judge here. On the live data
    it ranked "Tabiiy gaz" (natural gas, the commodity) top for a product reading
    "Tabiiy gaz filtra" — a gas *filter*, i.e. a spare part — purely on shared
    words. That is exactly the oil / oil-filter confusion the brief warns about,
    so the scorer decides ordering at most, never eligibility.
    """
    query = _query_tokens(product)
    taxonomy = _taxonomy_tokens(product)
    evidence_text = product.evidence_text()

    scored = [
        score_article(article, query=query, taxonomy=taxonomy, evidence_text=evidence_text)
        for article in catalog.articles
    ]
    scored.sort(key=lambda c: (-c.score, c.article.id))

    picked = [c for c in scored[:size] if keep_zero_score or c.score > 0.0]

    # An evidence-free product scores zero everywhere. Sending an empty shortlist
    # would make it look like an empty catalog; sending the head of the catalog
    # lets the model answer NO_MATCH for the real reason instead.
    if not picked:
        picked = scored[:size]
    return picked


__all__ = ["ScoredCandidate", "score_article", "shortlist"]
