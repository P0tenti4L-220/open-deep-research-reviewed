"""Frozen price table and deterministic cost estimation.

The product records provider usage; a price table turns that usage into a *cost
estimate*. The two must never be conflated:

* ``provider_billed_cost_usd`` is what the provider charged. We do not have an invoice
  parser, so it stays ``None`` until a bill is reconciled;
* ``estimated_model_cost_usd`` is usage x frozen price, recomputable from the ledger
  and the price snapshot hash;
* ``known_cost_lower_bound_usd`` sums only rows whose price was known. It is a floor,
  not a total.

A missing price, a missing token category or a null price entry never becomes ``0``.
The paid-run preflight refuses to start while the table is unverified.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION = "odr-pricing-v1"
CATEGORIES = ("uncached_input", "cache_read", "cache_write", "output")

METHOD_EXACT = "exact_categories"
METHOD_UPPER_BOUND = "upper_bound_estimate"
METHOD_UNKNOWN = "unknown"


@dataclass(frozen=True)
class PriceEntry:
    """One model's per-million-token prices (``None`` means not published)."""

    model: str
    provider: str
    uncached_input: Optional[float]
    cache_read: Optional[float]
    cache_write: Optional[float]
    output: Optional[float]
    source_url: str = ""
    fetched_at_utc: str = ""
    effective_from: Optional[str] = None

    def has_all_prices(self) -> bool:
        return all(getattr(self, category) is not None for category in CATEGORIES)


@dataclass(frozen=True)
class PriceTable:
    """Loaded price snapshot plus its content hash."""

    path: Optional[Path]
    status: str
    models: Dict[str, PriceEntry]
    raw: Dict[str, Any]
    snapshot_hash: str

    def entry(self, model: str) -> Optional[PriceEntry]:
        return self.models.get(model)

    def is_verified(self) -> bool:
        return self.status == "verified"


def default_price_path() -> Optional[Path]:
    """Return the configured price table path, if one exists.

    ``ODR_PRICING_PATH`` wins. Otherwise use the frozen table bundled with the
    runtime package, so an installation does not need local benchmark files.
    """
    candidates = []
    env = os.environ.get("ODR_PRICING_PATH")
    if env:
        candidates.append(Path(env))
    candidates.append(Path(__file__).with_name("default_pricing.json"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def load_table(path: Optional[str | os.PathLike] = None) -> PriceTable:
    """Load a frozen table; an absent or malformed table is the empty table."""
    resolved = Path(path) if path else default_price_path()
    raw: Dict[str, Any] = {}
    if resolved is not None and resolved.is_file():
        try:
            raw = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - a broken table is "no prices", not zero
            raw = {}
    models: Dict[str, PriceEntry] = {}
    for model_id, row in (raw.get("models") or {}).items():
        if not isinstance(row, dict):
            continue
        models[str(model_id)] = PriceEntry(
            model=str(row.get("model") or model_id),
            provider=str(row.get("provider") or ""),
            uncached_input=row.get("uncached_input"),
            cache_read=row.get("cache_read"),
            cache_write=row.get("cache_write"),
            output=row.get("output"),
            source_url=str(row.get("source_url") or ""),
            fetched_at_utc=str(row.get("fetched_at_utc") or ""),
            effective_from=row.get("effective_from"),
        )
    digest = hashlib.sha256(json.dumps(raw, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest() if raw else ""
    return PriceTable(
        path=resolved,
        status=str(raw.get("status") or "missing"),
        models=models,
        raw=raw,
        snapshot_hash=f"sha256:{digest}" if digest else "",
    )


def estimate_cost(
    table: PriceTable,
    *,
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cache_read_tokens: Optional[int] = None,
    cache_write_tokens: Optional[int] = None,
    reasoning_tokens: Optional[int] = None,
) -> Tuple[Optional[float], str, Dict[str, Any]]:
    """Return ``(cost_usd, method, detail)`` for one attempt.

    ``reasoning_tokens`` is *not* added to output when the provider already includes
    it there (our usage contract takes ``output_tokens`` as the billed total); the
    detail records that decision instead of double counting.
    """
    detail: Dict[str, Any] = {
        "model": model,
        "price_snapshot_hash": table.snapshot_hash,
        "price_status": table.status,
        "categories_used": [],
        "reasoning_tokens_included_in_output": reasoning_tokens is not None,
    }
    entry = table.entry(model)
    if entry is None or not entry.has_all_prices():
        detail["missing_reason"] = "model_price_unknown" if entry is None else "price_category_null"
        return None, METHOD_UNKNOWN, detail
    if input_tokens is None and output_tokens is None:
        detail["missing_reason"] = "provider_usage_missing"
        return None, METHOD_UNKNOWN, detail
    if table.status != "verified":
        detail["missing_reason"] = f"price_table_{table.status}"
        return None, METHOD_UNKNOWN, detail

    cached = int(cache_read_tokens or 0)
    cached_write = int(cache_write_tokens or 0)
    total_input = int(input_tokens or 0)
    uncached = max(0, total_input - cached - cached_write)
    out = int(output_tokens or 0)
    method = METHOD_EXACT if (cache_read_tokens is not None and cache_write_tokens is not None) else METHOD_UPPER_BOUND
    if cache_read_tokens is None:
        # Unknown cache split: charge every input token as uncached. This is the
        # conservative upper bound for budgeting and is labelled as such.
        uncached = total_input
        detail["missing_reason"] = "cache_split_unknown_charged_as_uncached"
    cost = (
        uncached * float(entry.uncached_input)
        + cached * float(entry.cache_read)
        + cached_write * float(entry.cache_write)
        + out * float(entry.output)
    ) / 1_000_000.0
    detail["categories_used"] = ["uncached_input", "cache_read", "cache_write", "output"]
    detail["uncached_input_tokens"] = uncached
    detail["cache_read_tokens"] = cached
    detail["cache_write_tokens"] = cached_write
    detail["output_tokens"] = out
    detail["currency"] = "USD"
    return round(cost, 8), method, detail


__all__ = [
    "CATEGORIES",
    "METHOD_EXACT",
    "METHOD_UNKNOWN",
    "METHOD_UPPER_BOUND",
    "PriceEntry",
    "PriceTable",
    "default_price_path",
    "estimate_cost",
    "load_table",
]
