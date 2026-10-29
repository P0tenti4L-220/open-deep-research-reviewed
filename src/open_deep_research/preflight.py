"""Pre-flight checks for a paid research run.

A paid run must fail *before* the research graph starts when a prerequisite is
missing. The failure modes this guards against are all silent today:

* no explicit authorization (``ODR_ALLOW_PAID_RUNS=1``);
* no amount cap, so "how much may this cost" has no answer;
* an unverified or incomplete price table, so usage could only be reported as zero;
* a missing provider credential (existence does not prove access, but absence
  guarantees failure);
* a model whose structured-output / tool-calling support is not declared, which would
  fail in the middle of a run after spending tokens.

The amount cap is also *applied*: the worst-case published price per token turns it
into a token ceiling, and the smaller of that and the configured token cap is bound to
the run. The token admission check (``ArtifactStore.reserve_model_call``) then enforces
it for every request, including the current request's own estimate.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from open_deep_research import pricing
from open_deep_research.configuration import Configuration

PROVIDER_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

ROLE_MODELS = (
    ("research", "research_model"),
    ("compression", "compression_model"),
    ("final_report", "final_report_model"),
    ("summarization", "summarization_model"),
    ("reviewer", "reviewer_model"),
)

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"


@dataclass
class Check:
    """One named precondition and its verdict."""

    name: str
    status: str
    detail: str = ""


@dataclass
class PreflightReport:
    """The whole pre-flight result, serialisable into the run's record."""

    checks: List[Check] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    models: Dict[str, str] = field(default_factory=dict)
    price_snapshot_hash: str = ""
    price_table_path: str = ""
    max_cost_usd: Optional[float] = None
    worst_price_per_million: Optional[float] = None
    token_ceiling: Optional[int] = None
    applied_max_total_tokens: Optional[int] = None

    @property
    def ok(self) -> bool:
        return not self.errors and all(check.status != STATUS_FAIL for check in self.checks)

    def add(self, name: str, status: str, detail: str = "", *, error: Optional[str] = None) -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))
        if error:
            self.errors.append(error)

    def as_dict(self) -> Dict[str, Any]:
        payload = {key: value for key, value in asdict(self).items()}
        payload["ok"] = self.ok
        payload["checks"] = [asdict(check) for check in self.checks]
        return payload


def planned_models(config: Configuration) -> Dict[str, str]:
    """Return ``role -> model id`` for every model the run can call."""
    models: Dict[str, str] = {}
    for role, field_name in ROLE_MODELS:
        value = getattr(config, field_name, None)
        if role == "reviewer" and not value:
            value = getattr(config, "research_model", None)
        if value:
            models[role] = str(getattr(value, "value", value))
    return models


def required_api_key(model_id: str) -> str:
    """Return the credential variable for a ``provider:model`` id ("" if unknown)."""
    provider = str(model_id or "").split(":", 1)[0].strip().lower()
    return PROVIDER_KEYS.get(provider, "")


def _worst_price_per_million(table: pricing.PriceTable, models: List[str]) -> Optional[float]:
    """Return the highest published per-1M price any planned model can incur.

    Output is usually the most expensive category; taking the maximum over every
    category and model gives a ceiling that does not depend on the input/output split.
    """
    worst: Optional[float] = None
    for model in models:
        entry = table.entry(model)
        if entry is None or not entry.has_all_prices():
            return None
        for category in pricing.CATEGORIES:
            value = getattr(entry, category)
            if value is None:
                return None
            worst = float(value) if worst is None else max(worst, float(value))
    return worst


def run_preflight(
    config: Configuration,
    *,
    table: Optional[pricing.PriceTable] = None,
    env: Optional[Dict[str, str]] = None,
    require_authorization: bool = True,
) -> PreflightReport:
    """Check one planned run; never raises for a missing prerequisite."""
    environment = os.environ if env is None else env
    report = PreflightReport(models=planned_models(config))

    # 1. Explicit authorization: a paid run is never implicit.
    if require_authorization:
        authorized = str(environment.get("ODR_ALLOW_PAID_RUNS") or "").strip() == "1"
        report.add(
            "authorization",
            STATUS_PASS if authorized else STATUS_FAIL,
            "ODR_ALLOW_PAID_RUNS=1" if authorized else "paid runs are not authorized",
            error="" if authorized else "authorization_missing: set ODR_ALLOW_PAID_RUNS=1 after a recorded amount cap",
        )
    else:
        report.add("authorization", STATUS_SKIP, "not required for this run profile")

    # 2. Amount cap: a positive finite number, or there is nothing to enforce.
    cap = getattr(config, "max_cost_usd", None)
    try:
        cap_value = float(cap) if cap is not None else None
    except (TypeError, ValueError):
        cap_value = None
    if cap_value is None or not math.isfinite(cap_value) or cap_value <= 0:
        report.add(
            "amount_cap",
            STATUS_FAIL,
            f"max_cost_usd={cap!r}",
            error="amount_cap_missing: max_cost_usd must be a positive number before a paid run",
        )
    else:
        report.max_cost_usd = cap_value
        report.add("amount_cap", STATUS_PASS, f"max_cost_usd={cap_value:g} USD")

    # 3. Price table: verified, complete, and covering every planned model.
    resolved_table = table or pricing.load_table(getattr(config, "pricing_path", None) or None)
    report.price_snapshot_hash = resolved_table.snapshot_hash
    report.price_table_path = str(resolved_table.path or "")
    if not resolved_table.is_verified():
        report.add(
            "pricing",
            STATUS_FAIL,
            f"status={resolved_table.status!r} path={report.price_table_path or 'none'}",
            error=f"pricing_unverified: price table status is {resolved_table.status!r}",
        )
    else:
        missing = [model for model in report.models.values() if (entry := resolved_table.entry(model)) is None or not entry.has_all_prices()]
        if missing:
            report.add("pricing", STATUS_FAIL, f"no complete price for {sorted(set(missing))}", error=f"pricing_incomplete: {sorted(set(missing))}")
        else:
            report.add("pricing", STATUS_PASS, f"verified snapshot {report.price_snapshot_hash[:19]} for {len(set(report.models.values()))} model(s)")

    # 4. Declared capabilities: structured output and tool calling are hard requirements.
    unsupported: List[str] = []
    for model in sorted(set(report.models.values())):
        declared = ((resolved_table.raw.get("models") or {}).get(model) or {}).get("capabilities") or {}
        if declared.get("structured_output") is not True or declared.get("tool_calling") is not True:
            unsupported.append(model)
    if unsupported:
        report.add(
            "capabilities",
            STATUS_FAIL,
            f"not declared for {unsupported}",
            error="capability_undeclared: structured output / tool calling must be declared for every planned model",
        )
    else:
        report.add("capabilities", STATUS_PASS, "structured output and tool calling declared for every planned model")

    # 5. Credentials: existence only. It never proves authentication or balance.
    missing_keys = sorted({key for model in report.models.values() if (key := required_api_key(model)) and not str(environment.get(key) or "").strip()})
    if missing_keys:
        report.add("credentials", STATUS_FAIL, f"missing {missing_keys}", error=f"credentials_missing: {missing_keys}")
    else:
        report.add("credentials", STATUS_PASS, "provider key(s) present (existence only; not an authentication or balance check)")

    # 6. Retrieval: the configured search API needs its own credential.
    search_api = str(getattr(getattr(config, "search_api", ""), "value", getattr(config, "search_api", "")) or "").lower()
    if search_api == "tavily":
        present = bool(str(environment.get("TAVILY_API_KEY") or "").strip())
        report.add("search", STATUS_PASS if present else STATUS_FAIL, "tavily" if present else "TAVILY_API_KEY missing", error="" if present else "search_credentials_missing: TAVILY_API_KEY")
    else:
        report.add("search", STATUS_SKIP, f"search_api={search_api or 'unset'}")

    # 7. Apply the amount cap as a token ceiling (worst published price per token).
    if report.max_cost_usd and report.ok:
        worst = _worst_price_per_million(resolved_table, sorted(set(report.models.values())))
        if not worst:
            report.add("token_ceiling", STATUS_FAIL, "no usable price to derive a ceiling", error="token_ceiling_underivable: worst-case price is unknown")
        else:
            report.worst_price_per_million = worst
            ceiling = int(report.max_cost_usd * 1_000_000 / worst)
            report.token_ceiling = ceiling
            configured = getattr(config, "max_total_tokens", None)
            applied = min(int(configured), ceiling) if configured is not None else ceiling
            report.applied_max_total_tokens = applied
            report.add(
                "token_ceiling",
                STATUS_PASS,
                f"cap {report.max_cost_usd:g} USD / worst {worst:g} USD per 1M = {ceiling} tokens; applied {applied}",
            )
    return report


__all__ = [
    "Check",
    "PROVIDER_KEYS",
    "PreflightReport",
    "planned_models",
    "required_api_key",
    "run_preflight",
]