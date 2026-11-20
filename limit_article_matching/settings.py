"""Plan request limit-article matching — module configuration.

Reads the environment directly (same pattern as risk_forecast/settings.py) but
takes every database value from the already-provisioned REAL_DB_* / DB_SSH_*
block in .env. That credential is the read-only `read-user` on `uzassetsdb`;
nothing in this module ever issues a write, and the extraction session asserts
that at the transaction level (see db_reader.SnapshotSession).

New LAM_* variables are all optional with working defaults, so the service runs
against the existing .env untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int, *, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    """Read-only access to uzassetsdb, straight from the existing .env keys."""

    host: str
    port: int
    user: str
    password: str
    name: str
    schema: str
    use_ssh_tunnel: bool
    ssh_host: str
    ssh_port: int
    ssh_username: str
    ssh_private_key_path: str
    ssh_passphrase: str
    local_bind_port: int
    connect_timeout_sec: int
    statement_timeout_ms: int

    @property
    def configured(self) -> bool:
        return bool(self.user and self.password and self.name)

    def missing_keys(self) -> list[str]:
        missing: list[str] = []
        if not self.user:
            missing.append("REAL_DB_USER")
        if not self.password:
            missing.append("REAL_DB_PASSWORD")
        if not self.name:
            missing.append("REAL_DB_NAME")
        if self.use_ssh_tunnel:
            if not self.ssh_host:
                missing.append("DB_SSH_HOST")
            if not self.ssh_username:
                missing.append("DB_SSH_USERNAME")
            if not self.ssh_private_key_path:
                missing.append("DB_SSH_PRIVATE_KEY_PATH")
        return missing


@dataclass(frozen=True, slots=True)
class MatchingConfig:
    """Inference and result-shaping knobs."""

    ai_model: str
    temperature: float
    max_tokens: int
    timeout_sec: int
    max_attempts: int
    retry_backoff_sec: float
    max_concurrency: int
    candidate_shortlist_size: int
    full_catalog_threshold: int
    max_alternatives: int
    min_confidence_for_match: float
    ambiguity_margin: float
    dedupe_identical_evidence: bool
    batch_size: int
    batch_max_tokens_per_item: int
    batch_max_tokens_cap: int
    batch_timeout_per_item_sec: int


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Run lifecycle: storage, paging and retention."""

    storage_path: str
    default_page_size: int
    max_page_size: int
    retention_hours: int
    max_products_per_run: int
    rules_version: str
    fingerprint_schema_version: str


@dataclass(frozen=True, slots=True)
class LimitArticleMatchingConfig:
    database: DatabaseConfig = field(default_factory=lambda: _database_config())
    matching: MatchingConfig = field(default_factory=lambda: _matching_config())
    run: RunConfig = field(default_factory=lambda: _run_config())


def _database_config() -> DatabaseConfig:
    db_host = (os.getenv("DB_HOST") or "127.0.0.1").strip()
    db_port = _env_int("DB_PORT", 5432, min_value=1, max_value=65535)
    return DatabaseConfig(
        host=(os.getenv("REAL_DB_HOST") or db_host).strip() or db_host,
        port=_env_int("REAL_DB_PORT", db_port, min_value=1, max_value=65535),
        user=(os.getenv("REAL_DB_USER") or "").strip(),
        password=os.getenv("REAL_DB_PASSWORD") or "",
        name=(os.getenv("REAL_DB_NAME") or "").strip(),
        schema=(os.getenv("REAL_DB_SCHEMA") or "public").strip() or "public",
        use_ssh_tunnel=_env_bool("REAL_DB_USE_SSH_TUNNEL", False),
        ssh_host=(os.getenv("DB_SSH_HOST") or "").strip(),
        ssh_port=_env_int("DB_SSH_PORT", 22, min_value=1, max_value=65535),
        ssh_username=(os.getenv("DB_SSH_USERNAME") or "").strip(),
        ssh_private_key_path=(os.getenv("DB_SSH_PRIVATE_KEY_PATH") or "").strip(),
        ssh_passphrase=os.getenv("DB_SSH_PASSPHRASE") or "",
        local_bind_port=_env_int("DB_LOCAL_BIND_PORT", 15432, min_value=0, max_value=65535),
        connect_timeout_sec=_env_int("LAM_DB_CONNECT_TIMEOUT_SEC", 10, min_value=1),
        # A runaway extraction on the production database is the one way this
        # read-only service can still hurt the platform, so every statement is
        # bounded server-side as well as by the client timeout.
        statement_timeout_ms=_env_int("LAM_DB_STATEMENT_TIMEOUT_MS", 120_000, min_value=1000),
    )


def _matching_config() -> MatchingConfig:
    return MatchingConfig(
        # GEMINI or GEMMA — resolved through services.ai_client_factory, so this
        # module inherits whichever provider the platform is pointed at.
        ai_model=(os.getenv("LAM_AI_MODEL") or os.getenv("DEFAULT_AI_MODEL") or "GEMINI").strip().upper(),
        temperature=_env_float("LAM_AI_TEMPERATURE", 0.0),
        max_tokens=_env_int("LAM_AI_MAX_TOKENS", 1400, min_value=256),
        timeout_sec=_env_int("LAM_AI_TIMEOUT_SEC", 120, min_value=5),
        # 2 attempts covers a cold model load and one unlucky sample; more
        # would just extend a large run without changing the outcome.
        max_attempts=_env_int("LAM_AI_MAX_ATTEMPTS", 2, min_value=1, max_value=5),
        retry_backoff_sec=_env_float("LAM_AI_RETRY_BACKOFF_SEC", 2.0),
        max_concurrency=_env_int("LAM_AI_MAX_CONCURRENCY", _env_int("AI_MAX_CONCURRENCY", 4), min_value=1),
        # The catalog can hold thousands of articles; only the shortlist reaches
        # the prompt. Validation still runs against the full catalog.
        candidate_shortlist_size=_env_int("LAM_CANDIDATE_SHORTLIST", 25, min_value=5, max_value=100),
        # A catalog this small fits in one prompt whole, so nothing is shortlisted
        # away and shortlisting cannot cost recall. Organization 1 has 43 active
        # articles (verified 2026-09-08), so today every run takes this path.
        full_catalog_threshold=_env_int("LAM_FULL_CATALOG_THRESHOLD", 60, min_value=0, max_value=300),
        max_alternatives=_env_int("LAM_MAX_ALTERNATIVES", 3, min_value=0, max_value=3),
        min_confidence_for_match=_env_float("LAM_MIN_CONFIDENCE", 0.55),
        # Two candidates closer than this in confidence are reported AMBIGUOUS
        # rather than silently resolved in favour of the first.
        ambiguity_margin=_env_float("LAM_AMBIGUITY_MARGIN", 0.08),
        dedupe_identical_evidence=_env_bool("LAM_DEDUPE_IDENTICAL_EVIDENCE", True),
        # Products per inference call. A single-line prompt is ~94% fixed
        # overhead (rules + the whole catalog), so batching is where the
        # order-of-magnitude saving lives. 1 keeps the one-at-a-time path,
        # which is the safe default until a batch size is measured against
        # it on real lines.
        batch_size=_env_int("LAM_BATCH_SIZE", 1, min_value=1, max_value=50),
        # Output room per item. Too little truncates the JSON array mid-item
        # and costs the whole batch, so this is deliberately generous.
        batch_max_tokens_per_item=_env_int("LAM_BATCH_TOKENS_PER_ITEM", 320, min_value=120),
        batch_max_tokens_cap=_env_int("LAM_BATCH_TOKENS_CAP", 8000, min_value=1000),
        # A batch writes roughly one answer per item, so it needs roughly one
        # item's worth of generation time per item. Leaving the single-line
        # timeout in place is what made the first batched run time out at 120s
        # and fall back after wasting four minutes.
        batch_timeout_per_item_sec=_env_int("LAM_BATCH_TIMEOUT_PER_ITEM_SEC", 20, min_value=5),
    )


def _run_config() -> RunConfig:
    default_store = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "var",
        "product_limit_article_matching.sqlite3",
    )
    return RunConfig(
        storage_path=(os.getenv("LAM_STORAGE_PATH") or default_store).strip(),
        default_page_size=_env_int("LAM_DEFAULT_PAGE_SIZE", 200, min_value=1, max_value=1000),
        max_page_size=_env_int("LAM_MAX_PAGE_SIZE", 1000, min_value=1, max_value=1000),
        retention_hours=_env_int("LAM_RESULT_RETENTION_HOURS", 168, min_value=1),
        # The brief proposes 500 distinct products per call; this is the
        # server-side ceiling behind that, so a raised client limit still has
        # something bounding it.
        max_products_per_run=_env_int("LAM_MAX_PRODUCTS_PER_RUN", 500, min_value=1, max_value=5000),
        rules_version=(os.getenv("LAM_RULES_VERSION") or "product-rules-1.0.0").strip(),
        # Bumped only when the canonical fingerprint encoding changes; the
        # backend pins it to reproduce fingerprints.
        fingerprint_schema_version=(os.getenv("LAM_FINGERPRINT_SCHEMA_VERSION") or "1.0.0").strip(),
    )


@lru_cache(maxsize=1)
def get_config() -> LimitArticleMatchingConfig:
    return LimitArticleMatchingConfig()


__all__ = [
    "DatabaseConfig",
    "LimitArticleMatchingConfig",
    "MatchingConfig",
    "RunConfig",
    "get_config",
]
