"""Run-scoped observability: local canonical records plus one optional backend.

Design contract: ``docs/context-review-design/observability-plan-2026-10-26``.
This module is the *only* observability boundary of the product. It keeps three
things apart, because conflating them is how trace trees end up lying:

* **the local record** (:class:`LocalRecorder`) - append-only events and one
  materialised row per span, written under the run's own artifact directory. It
  is the persistent fact used for reconciliation, not a copy of a platform trace.
* **the framework callbacks** (:class:`TelemetryCallbackHandler`) - the actual
  graph/node/model/tool runs LangGraph emits. They carry the native run ids and
  the provider's own timings.
* **the business spans** (``session.span``) - application semantics the framework
  cannot know: a dispatch batch, one model *request* versus its attempts, one
  review batch, an auxiliary call that bypasses the ledger.

Everything shares one identifier table (``trace-map.json``). Two invariants:

1. A container span never owns tokens or cost. Only the lowest unique native
   generation or the single usage row of an attempt is aggregate-able.
2. Observation never changes the research path. Every record is written inside a
   best-effort guard; a broken exporter never re-runs a model call.

The default backend is ``none``: no session, no files, no callbacks, no network.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from langchain_core.callbacks import BaseCallbackHandler

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "odr-observability-v1"
TELEMETRY_KEY = "__odr_telemetry__"

# Span kinds. ``run`` and ``stage`` are business spans; ``native_*`` come from the
# framework callback handler and carry provider timings/usage for cross-checking.
SPAN_RUN = "run"
SPAN_STAGE = "stage"
SPAN_AGENT = "agent"
SPAN_MODEL_REQUEST = "llm_request"
SPAN_MODEL_ATTEMPT = "llm_attempt"
SPAN_TOOL = "tool"
SPAN_REVIEW = "review"
SPAN_FRAMEWORK_CHAIN = "framework_chain"
SPAN_NATIVE_LLM = "native_llm"
SPAN_NATIVE_TOOL = "native_tool"

TERMINAL_OK = "ok"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
STATUS_TIMEOUT = "timeout"
STATUS_BUDGET_DENIED = "budget_denied"
STATUS_INCOMPLETE = "incomplete"
STATUS_NOT_STARTED = "not_started"

METRIC_MEASURED = "measured"
METRIC_UNKNOWN = "unknown"
METRIC_NOT_AVAILABLE = "not_available"
METRIC_NOT_APPLICABLE = "not_applicable"

EXPORT_DISABLED = "disabled"
EXPORT_PENDING = "pending"
EXPORT_CONFIRMED = "confirmed"
EXPORT_FAILED = "failed"
EXPORT_PARTIAL = "partial"
EXPORT_NOT_RUN = "not_run"

BACKEND_NONE = "none"
BACKEND_LOCAL = "local"
BACKEND_LANGSMITH = "langsmith"
BACKEND_LANGFUSE = "langfuse"
_BACKENDS = {BACKEND_NONE, BACKEND_LOCAL, BACKEND_LANGSMITH, BACKEND_LANGFUSE}

CONTENT_METADATA_ONLY = "metadata_only"
CONTENT_REDACTED = "redacted"
CONTENT_SYNTHETIC_FULL = "synthetic_full"
_CONTENT_MODES = {CONTENT_METADATA_ONLY, CONTENT_REDACTED, CONTENT_SYNTHETIC_FULL}


class TelemetryError(RuntimeError):
    """Raised only when ``require_complete_trace`` makes missing telemetry fatal."""


# --------------------------------------------------------------------------- #
# Sanitisation: an allowlist summary, never "filter the obvious secrets"
# --------------------------------------------------------------------------- #
# Only fields the analysis actually needs may leave the process. Unknown keys are
# dropped instead of guessed at: a task's private canary is usually a *value* whose
# key looks harmless, so "copy everything except /token|key/" leaks by default.
_ALLOWED_ATTRIBUTES = frozenset(
    {
        "schema_version",
        "span_kind",
        "operation",
        "status",
        "status_detail",
        "metric_status",
        "missing_reason",
        "error_class",
        "error_hash",
        "role",
        "agent_instance_id",
        "parent_agent_instance_id",
        "task_id",
        "research_task_id",
        "task_status",
        "request_id",
        "attempt_id",
        "tool_call_id",
        "tool_name",
        "dispatch_batch_id",
        "iteration",
        "attempt_number",
        "review_id",
        "review_stage",
        "batch_index",
        "batch_count",
        "agent_count",
        "overflow_count",
        "failed_count",
        "succeeded_count",
        # Query units are billing-relevant: the acceptance test reads this count from
        # the search.raw span, and dropping it silently hid the only in-span proof.
        "query_count",
        "verdict",
        "complete",
        "checked_count",
        "unchecked_count",
        "report_version",
        "report_hash",
        "recovery_event_id",
        "evidence_count",
        "evidence_ids_count",
        "model",
        "actual_model",
        "provider",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "reasoning_tokens",
        "estimated_input_tokens",
        "budget_tokens",
        "budget_denied",
        "cost_value",
        "cost_currency",
        "cost_basis",
        "cost_usd",
        "cost_method",
        "price_snapshot_hash",
        "price_version",
        "counts_for_billing",
        "native_tokens_present",
        "native_callback_expected",
        "execution_id",
        "channel",
        "streamed",
        "ttft_ms",
        "duration_ms",
        "framework_run_id",
        "parent_framework_run_id",
        "framework_name",
        "tag_count",
        "http_attempts",
        "transport_attempts",
        "queue_wait_ms",
        "dropped_attributes",
        "sample_rate",
        "backend",
        "content_mode",
        "gate",
        "reason_code",
        "objective",
    }
)

# Text-bearing fields are hashed (metadata_only) or truncated + redacted.
_TEXT_ATTRIBUTES = frozenset({"objective", "message", "summary", "note", "reason"})

_SCRUB_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(sk-[a-z0-9_-]{8,})"),
    re.compile(r"(?i)\b(bearer)\s+[a-z0-9._\-]+"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)([?&](?:api[_-]?key|token|access_token|key)=)[^&\s]+"),
)

_MAX_ATTRIBUTE_CHARS = 8_000
_MAX_STRING_CHARS = 200


def _scrub_string(value: str, *, limit: int) -> str:
    """Return ``value`` with credential-like substrings removed and length bounded."""
    text = value
    for pattern in _SCRUB_PATTERNS:
        text = pattern.sub("<redacted>", text)
    if len(text) > limit:
        text = text[:limit] + "…<truncated>"
    return text


def sanitize_attributes(attributes: Optional[Mapping[str, Any]], *, content_mode: str = CONTENT_METADATA_ONLY) -> Dict[str, Any]:
    """Return the safe, allowlisted summary of ``attributes``.

    ``metadata_only`` keeps identifiers, counts, timings and hashes; any free text
    is replaced by its digest. ``redacted`` keeps a scrubbed, truncated excerpt of
    the allowlisted text fields. ``synthetic_full`` is for synthetic pilot data and
    keeps a larger excerpt, still scrubbed.
    """
    if not attributes:
        return {}
    out: Dict[str, Any] = {}
    dropped = 0
    for raw_key, raw_value in attributes.items():
        key = str(raw_key)
        if key not in _ALLOWED_ATTRIBUTES:
            dropped += 1
            continue
        value = raw_value
        if isinstance(value, bool) or isinstance(value, int) or isinstance(value, float):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            out[key] = value
        elif isinstance(value, str):
            if key in _TEXT_ATTRIBUTES and content_mode == CONTENT_METADATA_ONLY:
                out[key] = f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"
            else:
                limit = _MAX_STRING_CHARS if content_mode != CONTENT_SYNTHETIC_FULL else 4_000
                out[key] = _scrub_string(value, limit=limit)
        elif value is None:
            out[key] = None
        else:
            # Nested objects, stores, runnables and paths never leave as-is.
            dropped += 1
    if dropped:
        out["dropped_attributes"] = dropped
    if len(json.dumps(out, ensure_ascii=False, default=str)) > _MAX_ATTRIBUTE_CHARS:
        return {"truncated_payload": True, "dropped_attributes": dropped + len(out)}
    return out


def classify_exception(exc: BaseException) -> str:
    """Map an exception to a stable span status."""
    if isinstance(exc, asyncio.CancelledError):
        return STATUS_CANCELLED
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return STATUS_TIMEOUT
    if type(exc).__name__ == "BudgetExhausted":
        return STATUS_BUDGET_DENIED
    return STATUS_ERROR


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _new_execution_id() -> str:
    return f"exe-{uuid.uuid4().hex[:12]}"


def _new_span_id() -> str:
    return f"sp-{uuid.uuid4().hex[:12]}"


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _error_fingerprint(exc: BaseException) -> Tuple[str, str]:
    """Return ``(class, hash)`` for an observation error, never its message."""
    return type(exc).__name__, f"sha256:{hashlib.sha256(str(exc).encode('utf-8')).hexdigest()[:16]}"


def execution_dir(run_dir: str | os.PathLike, execution_id: str) -> Path:
    """Return the per-execution record directory for one run."""
    return Path(run_dir) / "observability" / "executions" / execution_id


def execution_index(run_dir: str | os.PathLike) -> List[Dict[str, Any]]:
    """Return the appended ``executions.jsonl`` rows (newest last)."""
    path = Path(run_dir) / "observability" / "executions.jsonl"
    rows: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            payload = json.loads(line)
        except Exception:  # noqa: BLE001 - a torn line must not hide earlier rows
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def latest_execution_dir(run_dir: str | os.PathLike) -> Optional[Path]:
    """Return the newest execution directory of a run, if any.

    Legacy single-directory records (``observability/manifest.json``) are not
    execution directories: a caller that reads them must label them ``legacy``
    because their spans may mix more than one execution.
    """
    base = Path(run_dir) / "observability" / "executions"
    if not base.is_dir():
        return None
    candidates = [path for path in base.iterdir() if path.is_dir() and (path / "manifest.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def rebuild_execution_index(run_dir: str | os.PathLike) -> List[Dict[str, Any]]:
    """Rebuild ``executions.jsonl`` by scanning execution manifests.

    The index is an append-only convenience, not the fact ledger: if it is lost or
    corrupted the manifests are still the truth and can restore it.
    """
    root = Path(run_dir) / "observability"
    base = root / "executions"
    rows: List[Dict[str, Any]] = []
    if base.is_dir():
        for path in sorted(base.iterdir(), key=lambda item: item.stat().st_mtime):
            try:
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event": "recovered",
                    "execution_id": manifest.get("execution_id") or path.name,
                    "odr_run_id": manifest.get("odr_run_id"),
                    "previous_execution_id": manifest.get("previous_execution_id"),
                    "backend": manifest.get("backend"),
                    "at_utc": manifest.get("created_at_utc") or _iso(_utc_now()),
                    "path": str(path),
                }
            )
    if rows:
        root.mkdir(parents=True, exist_ok=True)
        _atomic_json(root / "executions.jsonl.rebuilt", rows)
    return rows


def record_layout(run_dir: str | os.PathLike) -> Dict[str, Any]:
    """Describe where a run's canonical record lives without reading spans.

    ``schema`` is ``execution`` when execution directories exist, ``legacy`` when
    only the pre-execution single directory does, and ``none`` when neither exists.
    """
    root = Path(run_dir) / "observability"
    executions = [path for path in (root / "executions").glob("*") if (path / "manifest.json").exists()] if (root / "executions").is_dir() else []
    if executions:
        return {"schema": "execution", "root": str(root), "executions": sorted(path.name for path in executions), "latest": max(executions, key=lambda path: path.stat().st_mtime).name}
    if (root / "manifest.json").exists():
        return {"schema": "legacy", "root": str(root), "executions": [], "latest": None}
    return {"schema": "none", "root": str(root), "executions": [], "latest": None}


def sdk_versions() -> Dict[str, str]:
    """Return the versions that decide whether a trace behaves as tested."""
    versions: Dict[str, str] = {"python": ".".join(str(part) for part in os.sys.version_info[:3])}
    try:
        from importlib.metadata import version

        for name in ("langgraph", "langchain", "langchain-core", "langsmith"):
            try:
                versions[name] = version(name)
            except Exception:  # noqa: BLE001 - absent package is information, not an error
                versions[name] = "missing"
    except Exception:  # noqa: BLE001
        pass
    return versions


# --------------------------------------------------------------------------- #
# Run specification
# --------------------------------------------------------------------------- #
@dataclass
class RunSpec:
    """Host-owned facts about one execution. The model never supplies any of them."""

    odr_run_id: str
    execution_id: str = field(default_factory=_new_execution_id)
    previous_execution_id: Optional[str] = None
    previous_execution_reason: Optional[str] = None
    thread_id: Optional[str] = None
    experiment_id: Optional[str] = None
    arm: Optional[str] = None
    repeat: Optional[int] = None
    dataset_task_id: Optional[str] = None
    model_mode: str = "real"
    retrieval_mode: str = "live"
    telemetry_backend: str = BACKEND_NONE
    content_mode: str = CONTENT_METADATA_ONLY
    sample_rate: float = 1.0
    require_complete_trace: bool = False
    export_deadline_seconds: float = 5.0
    source_snapshot_hash: Optional[str] = None
    config_hash: Optional[str] = None
    corpus_hash: Optional[str] = None
    actual_models: Dict[str, str] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        run_id: str,
        thread_id: Optional[str] = None,
        dataset_task_id: Optional[str] = None,
        arm: Optional[str] = None,
        repeat: Optional[int] = None,
        experiment_id: Optional[str] = None,
        model_mode: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        previous_execution_id: Optional[str] = None,
        previous_execution_reason: Optional[str] = None,
        source_snapshot_hash: Optional[str] = None,
        corpus_hash: Optional[str] = None,
    ) -> "RunSpec":
        """Build the spec from the resolved :class:`Configuration`."""
        backend = str(getattr(config, "observability_backend", BACKEND_NONE) or BACKEND_NONE).lower()
        content_mode = str(getattr(config, "observability_content_mode", CONTENT_METADATA_ONLY) or CONTENT_METADATA_ONLY)
        models = {
            "research": str(getattr(config, "research_model", "") or ""),
            "compression": str(getattr(config, "compression_model", "") or ""),
            "final_report": str(getattr(config, "final_report_model", "") or ""),
            "summarization": str(getattr(config, "summarization_model", "") or ""),
            "reviewer": str(getattr(config, "reviewer_model", "") or getattr(config, "research_model", "") or ""),
        }
        resolved_arm = arm or (str(getattr(config, "experiment_arm", "") or "") or os.environ.get("EXPERIMENT_ARM") or None)
        if getattr(config, "context_policy", None) is not None:
            policy = str(getattr(config.context_policy, "value", config.context_policy))
            reviewer = bool(getattr(config, "reviewer_enabled", False))
            if not resolved_arm:
                resolved_arm = {("baseline", False): "A", ("scoped", False): "B", ("baseline", True): "C", ("scoped", True): "D"}.get((policy, reviewer))
        raw_rate = getattr(config, "observability_sample_rate", None)
        raw_deadline = getattr(config, "observability_export_deadline_seconds", None)
        return cls(
            odr_run_id=run_id,
            previous_execution_id=previous_execution_id,
            previous_execution_reason=previous_execution_reason,
            thread_id=thread_id,
            experiment_id=experiment_id or os.environ.get("ODR_EXPERIMENT_ID") or None,
            arm=resolved_arm,
            repeat=int(repeat if repeat is not None else (os.environ.get("ODR_REPEAT") or 0) or 0) or None,
            dataset_task_id=dataset_task_id or None,
            model_mode=model_mode or os.environ.get("ODR_MODEL_MODE") or "real",
            retrieval_mode=retrieval_mode or os.environ.get("ODR_RETRIEVAL_MODE") or "live",
            telemetry_backend=backend,
            content_mode=content_mode if content_mode in _CONTENT_MODES else CONTENT_METADATA_ONLY,
            sample_rate=float(raw_rate) if raw_rate is not None else 1.0,
            require_complete_trace=bool(getattr(config, "observability_require_complete_trace", False)),
            export_deadline_seconds=float(raw_deadline) if raw_deadline is not None else 5.0,
            source_snapshot_hash=source_snapshot_hash or os.environ.get("ODR_SOURCE_SNAPSHOT") or None,
            # Machine paths and observation settings are not research semantics:
            # hashing them would make two equivalent runs look different.
            config_hash=_digest(
                config.model_dump(
                    mode="json",
                    exclude={
                        "record_requests",
                        "artifact_root",
                        "observability_backend",
                        "observability_content_mode",
                        "observability_sample_rate",
                        "observability_export_deadline_seconds",
                        "observability_require_complete_trace",
                    },
                )
            )
            if hasattr(config, "model_dump")
            else None,
            corpus_hash=corpus_hash or os.environ.get("ODR_CORPUS_HASH") or None,
            actual_models=models,
        )

    def root_metadata(self) -> Dict[str, Any]:
        """Return the metadata inherited by every framework run of this execution."""
        return {
            "schema_version": SCHEMA_VERSION,
            "odr_run_id": self.odr_run_id,
            "odr_execution_id": self.execution_id,
            "odr_previous_execution_id": self.previous_execution_id,
            "odr_experiment_id": self.experiment_id,
            "odr_arm": self.arm,
            "odr_repeat": self.repeat,
            "odr_dataset_task_id": self.dataset_task_id,
            "odr_thread_id": self.thread_id,
            "odr_model_mode": self.model_mode,
            "odr_retrieval_mode": self.retrieval_mode,
            "odr_telemetry_backend": self.telemetry_backend,
            "odr_content_mode": self.content_mode,
            "odr_source_snapshot_hash": self.source_snapshot_hash,
            "odr_config_hash": self.config_hash,
            "odr_corpus_hash": self.corpus_hash,
        }

    def manifest(self) -> Dict[str, Any]:
        """Return the immutable header written to ``observability/manifest.json``."""
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _iso(_utc_now()),
            "clock": {"duration": "time.monotonic", "timestamp": "datetime.now(timezone.utc)"},
            "execution_id": self.execution_id,
            "previous_execution_id": self.previous_execution_id,
            "previous_execution_reason": self.previous_execution_reason,
            "odr_run_id": self.odr_run_id,
            "thread_id": self.thread_id,
            "experiment": {"experiment_id": self.experiment_id, "arm": self.arm, "repeat": self.repeat, "dataset_task_id": self.dataset_task_id},
            "modes": {"model": self.model_mode, "retrieval": self.retrieval_mode},
            "backend": self.telemetry_backend,
            "content_mode": self.content_mode,
            "sample_rate": self.sample_rate,
            "export_deadline_seconds": self.export_deadline_seconds,
            "require_complete_trace": self.require_complete_trace,
            "source_snapshot_hash": self.source_snapshot_hash,
            "config_hash": self.config_hash,
            "corpus_hash": self.corpus_hash,
            "actual_models": dict(self.actual_models),
            "sdk_versions": sdk_versions(),
        }


# --------------------------------------------------------------------------- #
# Local recorder
# --------------------------------------------------------------------------- #
class LocalRecorder:
    """Append-only local record: ``events.jsonl`` plus materialised ``spans.jsonl``.

    One execution writes exactly one directory (``executions/<execution_id>/``). The
    directory is created exclusively by the session; this class only appends inside
    it, so a second execution of the same business run can never share a file with
    the first one.
    """

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "events.jsonl"
        self.spans_path = self.directory / "spans.jsonl"
        self.outbox_dir = self.directory / "outbox"
        self._lock = threading.RLock()
        self.event_count = 0

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        _atomic_json(self.directory / "manifest.json", dict(manifest))

    def event(self, payload: Mapping[str, Any]) -> None:
        record = dict(payload)
        with self._lock:
            self.event_count += 1
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def span(self, record: Mapping[str, Any]) -> None:
        with self._lock:
            with self.spans_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(record), ensure_ascii=False, default=str) + "\n")

    def write_json(self, name: str, payload: Any) -> None:
        _atomic_json(self.directory / name, payload)

    def read_spans(self) -> List[Dict[str, Any]]:
        """Return the materialised spans (tolerant of a truncated last line)."""
        rows, _ = self.read_spans_checked()
        return rows

    def read_spans_checked(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return spans plus an integrity report for torn/foreign rows.

        A truncated tail (process killed mid-write) is not silently "complete": the
        caller gets the count and the byte offset so a collector can say so.
        """
        rows: List[Dict[str, Any]] = []
        torn = 0
        try:
            lines = self.spans_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return rows, {"torn_lines": 0, "rows": 0, "readable": False}
        for line in lines:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:  # noqa: BLE001 - a torn write must not hide earlier rows
                torn += 1
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows, {"torn_lines": torn, "rows": len(rows), "readable": True}


def _atomic_json(path: Path, payload: Any) -> None:
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
@dataclass
class ExportReceipt:
    """Result of one bounded export attempt."""

    backend: str
    status: str
    reason: str = ""
    remote_confirmed: bool = False
    trace_id: Optional[str] = None
    url: Optional[str] = None
    duration_ms: Optional[float] = None


def is_local_scope(backend: Any) -> bool:
    """Return True when the backend must emit nothing (``none``/``local``/unset)."""
    name = str(getattr(backend, "value", backend) or "none").strip().lower()
    return name in {BACKEND_NONE, BACKEND_LOCAL, ""}


@contextlib.contextmanager
def suppress_inherited_tracing() -> Any:
    """Disable inherited LangSmith tracing for the scope.

    ``local`` and ``none`` promise that this execution emits nothing. An inherited
    exporter (env ``LANGSMITH_TRACING`` or an outer ``tracing_context``) would upload
    the run anyway, so the scope is entered with tracing off. An absent SDK means
    nothing can be exported and the scope is a no-op.
    """
    try:
        from langsmith.run_helpers import tracing_context
    except Exception:  # noqa: BLE001 - absent SDK means nothing can be exported
        yield
        return
    with tracing_context(enabled=False):
        yield


class TelemetryBackend:
    """Minimal backend contract: start a tracing scope, then stop with a receipt."""

    name = BACKEND_NONE

    def __init__(self, spec: RunSpec):
        self.spec = spec

    def start(self) -> Tuple[str, Any]:
        """Return ``(status, context_manager)`` for the invocation scope."""
        return EXPORT_DISABLED, contextlib.nullcontext()

    def stop(self, *, receipt: ExportReceipt, deadline_seconds: float) -> ExportReceipt:
        return receipt


class LocalBackend(TelemetryBackend):
    """Local records only: the default for benchmarks, zero network.

    A parent process may have enabled LangSmith tracing (environment or an explicit
    tracer). "local" means this execution emits no telemetry, so the scope is entered
    with tracing *disabled*; without that, an inherited exporter would upload the run
    even though the manifest says local.
    """

    name = BACKEND_LOCAL

    def start(self) -> Tuple[str, Any]:
        return EXPORT_DISABLED, suppress_inherited_tracing()


# --------------------------------------------------------------------------- #
# Outbound sanitisation (platform payloads, not the local record)
# --------------------------------------------------------------------------- #
# The local record is already allowlisted. An exporter has its *own* payload shape
# (run name, tags, serialized runnable, error text, events, attachments), and the
# installed SDK's ``hide_inputs`` only clears some of them. A tag or an exception
# message is a perfectly good place to leak a task canary, so the whole payload is
# rebuilt from an allowlist before it leaves the process.
_EXPORT_TAG_RE = re.compile(r"^(?:odr|odr_[a-z_]+(?::[A-Za-z0-9._:/-]{1,120})?|backend:[a-z]+|langsmith:nostream)$")
# Outbound run names are opaque labels. Anything that does not look like one of our
# own operation names is replaced by a digest: a "node-<canary>" name is not safe just
# because it contains no credential-looking substring.
_EXPORT_NAME_RE = re.compile(r"^(?:odr|stage|agent|review|research|writer|supervisor|llm|tool|context|compress|extract|search|report|run)[._a-z0-9:-]{0,80}$")
_EXPORT_NAME_ALLOWLIST = frozenset({"LangGraph", "RunnableSequence", "RunnableLambda", "RunnableBinding", "ChannelWrite", "ChannelRead", "should_continue", "odr.run"})


def safe_export_name(value: Any) -> str:
    """Return an allowlisted name or a stable digest, never arbitrary text."""
    text = str(value or "")
    if text in _EXPORT_NAME_ALLOWLIST or _EXPORT_NAME_RE.match(text):
        return text[:120]
    return f"name#{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


_EXPORT_RUN_KEYS = frozenset(
    {
        "id",
        "trace_id",
        "parent_run_id",
        "dotted_order",
        "name",
        "run_name",
        "run_type",
        "start_time",
        "end_time",
        "status",
        "error",
        "tags",
        "metadata",
        "inputs",
        "outputs",
        "extra",
        "events",
        "serialized",
        "attachments",
        "session_name",
        "project_name",
        "revision_id",
        "reference_example_id",
    }
)


def _is_remote_tracer(handler: Any) -> bool:
    """Return True for a callback handler that would export off-process."""
    cls = type(handler)
    name = cls.__name__
    module = (getattr(cls, "__module__", "") or "").lower()
    if name in {"LangChainTracer", "LangSmithTracer", "LangfuseCallbackHandler", "LangfuseTracer"}:
        return True
    if name.lower().endswith("tracer") and ("langsmith" in module or "langfuse" in module):
        return True
    return False


def sanitize_export_tags(tags: Any) -> List[str]:
    """Return only opaque, allowlisted tags (never free text)."""
    cleaned: List[str] = []
    for tag in tags or []:
        text = str(tag)
        if _EXPORT_TAG_RE.match(text) and len(text) <= 160:
            cleaned.append(text)
    return cleaned


def sanitize_export_error(error: Any) -> Optional[str]:
    """Return ``Class#hash`` for an error; never the message or traceback."""
    if error is None:
        return None
    if isinstance(error, BaseException):
        error_class, error_hash = _error_fingerprint(error)
    else:
        error_class, error_hash = "RemoteError", f"sha256:{hashlib.sha256(str(error).encode('utf-8')).hexdigest()[:16]}"
    return f"{error_class}#{error_hash}"


def sanitize_export_events(events: Any) -> List[Dict[str, Any]]:
    """Rebuild LangSmith run events from an allowlist."""
    cleaned: List[Dict[str, Any]] = []
    for event in events or []:
        if not isinstance(event, Mapping):
            continue
        row: Dict[str, Any] = {}
        if isinstance(event.get("name"), str):
            row["name"] = safe_export_name(event["name"])
        for key in ("start_time", "end_time"):
            if event.get(key) is not None:
                row[key] = event[key]
        if isinstance(event.get("status"), str):
            row["status"] = _scrub_string(event["status"], limit=40)
        row["inputs"] = {}
        row["outputs"] = {}
        row["metadata"] = sanitize_attributes(event.get("metadata"), content_mode=CONTENT_METADATA_ONLY) if isinstance(event.get("metadata"), Mapping) else {}
        row["tags"] = sanitize_export_tags(event.get("tags"))
        error = sanitize_export_error(event.get("error"))
        if error:
            row["error"] = error
        cleaned.append(row)
    return cleaned


def sanitize_export_payload(payload: Mapping[str, Any], *, content_mode: str = CONTENT_METADATA_ONLY) -> Dict[str, Any]:
    """Return the outbound-safe version of one platform run payload.

    Unknown keys are dropped. Inputs, outputs, serialized runnables and attachments
    are replaced by a digest summary; metadata is allowlisted; tags and part names are
    bounded opaque identifiers; errors keep only class + hash.
    """
    out: Dict[str, Any] = {}
    dropped = 0
    for raw_key, value in payload.items():
        key = str(raw_key)
        if key not in _EXPORT_RUN_KEYS:
            dropped += 1
            continue
        if key in {"inputs", "outputs"}:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str) if value not in (None, {}) else ""
            out[key] = {} if content_mode == CONTENT_METADATA_ONLY or not text else {"sha256": _digest(value)[7:], "chars": len(text)}
        elif key == "serialized":
            out[key] = None
        elif key == "attachments":
            dropped += 1
        elif key == "extra":
            out[key] = sanitize_attributes(value, content_mode=content_mode) if isinstance(value, Mapping) else {}
        elif key == "metadata":
            out[key] = sanitize_attributes(value, content_mode=content_mode) if isinstance(value, Mapping) else {}
        elif key == "tags":
            out[key] = sanitize_export_tags(value)
        elif key == "events":
            out[key] = sanitize_export_events(value)
        elif key == "error":
            out[key] = sanitize_export_error(value)
        elif key in {"name", "run_name", "session_name", "project_name"}:
            out[key] = safe_export_name(value)
        elif key == "status":
            out[key] = _scrub_string(str(value), limit=40)
        else:
            out[key] = value
    if dropped:
        out["_odr_dropped_keys"] = dropped
    return out


class SanitizedLangSmithClient:
    """Proxy that sanitises every write payload before delegating to the SDK client.

    Read paths (``flush``, ``read_run`` and anything else) are forwarded unchanged, so
    the adapter's bounded flush and future read-back keep working.
    """

    def __init__(self, inner: Any, *, content_mode: str = CONTENT_METADATA_ONLY):
        self._inner = inner
        self._content_mode = content_mode

    def _clean(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        return sanitize_export_payload(payload, content_mode=self._content_mode)

    def create_run(self, **kwargs: Any) -> Any:
        return self._inner.create_run(**self._clean(kwargs))

    def update_run(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.update_run(*args, **self._clean(kwargs))

    def batch_ingest_runs(self, **kwargs: Any) -> Any:
        cleaned = dict(kwargs)
        for key in ("create", "update"):
            items = kwargs.get(key)
            if isinstance(items, (list, tuple)):
                cleaned[key] = [self._clean(item) if isinstance(item, Mapping) else item for item in items]
        return self._inner.batch_ingest_runs(**cleaned)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class LangSmithBackend(TelemetryBackend):
    """LangSmith adapter using the SDK tracing context and one shared client.

    Only one enabling mechanism is used. The client applies the content policy
    itself (``hide_inputs`` / ``hide_outputs``), so opaque framework inputs and
    outputs are not uploaded by the automatic tracing path we cannot enumerate.
    """

    name = BACKEND_LANGSMITH

    def __init__(self, spec: RunSpec, *, client_factory: Any = None):
        super().__init__(spec)
        self._client_factory = client_factory
        self.client: Any = None
        self.reason = ""

    def _credentials(self) -> Tuple[str, str, str]:
        api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY") or ""
        project = os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or "odr-observability"
        endpoint = os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get("LANGCHAIN_ENDPOINT") or "https://api.smith.langchain.com"
        return api_key, project, endpoint

    def start(self) -> Tuple[str, Any]:
        api_key, project, endpoint = self._credentials()
        tracing = str(os.environ.get("LANGSMITH_TRACING") or os.environ.get("LANGCHAIN_TRACING_V2") or "").strip().lower()
        if tracing in {"0", "false", "no", "off"}:
            self.reason = "LANGSMITH_TRACING is disabled"
            return EXPORT_NOT_RUN, contextlib.nullcontext()
        if not api_key and self._client_factory is None:
            self.reason = "missing LANGSMITH_API_KEY; no upload attempted"
            return EXPORT_NOT_RUN, contextlib.nullcontext()
        try:
            from langsmith.run_helpers import tracing_context

            if self._client_factory is not None:
                self.client = self._client_factory(api_url=endpoint, api_key=api_key)
            else:
                from langsmith import Client

                self.client = Client(
                    api_url=endpoint,
                    api_key=api_key,
                    hide_inputs=self.spec.content_mode == CONTENT_METADATA_ONLY,
                    hide_outputs=self.spec.content_mode == CONTENT_METADATA_ONLY,
                    hide_metadata=self.spec.content_mode == CONTENT_METADATA_ONLY,
                )
            # Every outbound write is rebuilt from an allowlist: the SDK's
            # hide_* switches do not cover tags, error text or serialized runnables.
            self.client = SanitizedLangSmithClient(self.client, content_mode=self.spec.content_mode)
            scope = tracing_context(
                enabled=True,
                project_name=project,
                client=self.client,
                tags=_export_tags(self.spec),
                metadata=self.spec.root_metadata(),
            )
            return EXPORT_PENDING, scope
        except Exception as exc:  # noqa: BLE001 - telemetry must not break the run
            self.reason = f"{type(exc).__name__}: {str(exc)[:120]}"
            LOGGER.warning("LangSmith backend could not start: %s", self.reason)
            return EXPORT_FAILED, contextlib.nullcontext()

    def stop(self, *, receipt: ExportReceipt, deadline_seconds: float) -> ExportReceipt:
        started = time.monotonic()
        if self.client is None:
            # A failed start is not the same as "never used": keep the failure.
            if receipt.status not in {EXPORT_FAILED, EXPORT_PARTIAL}:
                receipt.status = EXPORT_NOT_RUN
                receipt.reason = receipt.reason or self.reason or "backend never started"
            receipt.duration_ms = round((time.monotonic() - started) * 1000, 3)
            return receipt
        outcome: Dict[str, Any] = {}

        def _flush() -> None:
            try:
                flush = getattr(self.client, "flush", None)
                if callable(flush):
                    flush()
                outcome["ok"] = True
            except Exception as exc:  # noqa: BLE001
                outcome["error"] = exc

        # The deadline is enforced, not merely measured: a slow/hung exporter must
        # not hold the research process open (the daemon thread is abandoned).
        thread = threading.Thread(target=_flush, name="odr-langsmith-flush", daemon=True)
        thread.start()
        thread.join(max(0.0, float(deadline_seconds)))
        if thread.is_alive():
            receipt.status = EXPORT_PARTIAL
            receipt.reason = f"flush did not finish within the {deadline_seconds:g}s deadline"
        elif "error" in outcome:
            error = outcome["error"]
            receipt.status = EXPORT_FAILED
            receipt.reason = f"{type(error).__name__}: {str(error)[:120]}"
        else:
            # A local flush only proves "handed to the SDK"; remote completeness is
            # not confirmed without a read-back. Never report confirmed from flush.
            receipt.status = EXPORT_PENDING
            receipt.reason = "flush submitted; remote completeness not confirmed"
        receipt.duration_ms = round((time.monotonic() - started) * 1000, 3)
        return receipt


class LangfuseBackend(TelemetryBackend):
    """Optional Langfuse route; not part of the first delivered version.

    Selecting it without the locked SDK/version verification returns ``not_run``
    instead of pretending a trace was exported. The installation and end-to-end
    acceptance remain explicit TODO work in the handover report.
    """

    name = BACKEND_LANGFUSE

    def start(self) -> Tuple[str, Any]:
        try:
            import langfuse  # noqa: F401
        except Exception:  # noqa: BLE001
            return EXPORT_NOT_RUN, contextlib.nullcontext()
        return EXPORT_NOT_RUN, contextlib.nullcontext()

    def stop(self, *, receipt: ExportReceipt, deadline_seconds: float) -> ExportReceipt:
        receipt.status = EXPORT_NOT_RUN
        receipt.reason = "Langfuse adapter is not implemented in this revision"
        return receipt


def _export_tags(spec: RunSpec) -> List[str]:
    """Safe, filterable identifiers: metadata objects stay hidden by policy."""
    tags = [tag for tag in spec.tags if tag]
    tags.extend(["odr", f"backend:{BACKEND_LANGSMITH}"])
    for prefix, value in (
        ("odr_run", spec.odr_run_id),
        ("odr_execution", spec.execution_id),
        ("odr_experiment", spec.experiment_id),
        ("odr_arm", spec.arm),
        ("odr_task", spec.dataset_task_id),
    ):
        if value not in (None, ""):
            tags.append(f"{prefix}:{str(value)[:120]}")
    return tags


def make_backend(spec: RunSpec) -> TelemetryBackend:
    """Return the one backend this run may use (never two at once)."""
    if spec.telemetry_backend == BACKEND_LANGSMITH:
        return LangSmithBackend(spec)
    if spec.telemetry_backend == BACKEND_LANGFUSE:
        return LangfuseBackend(spec)
    if spec.telemetry_backend == BACKEND_LOCAL:
        return LocalBackend(spec)
    return TelemetryBackend(spec)


# --------------------------------------------------------------------------- #
# Session
# --------------------------------------------------------------------------- #
_CURRENT_SESSION: ContextVar[Optional["RunTelemetry"]] = ContextVar("odr_telemetry", default=None)
_CURRENT_SPAN: ContextVar[Optional[str]] = ContextVar("odr_span", default=None)
_CURRENT_ATTEMPT: ContextVar[Optional[str]] = ContextVar("odr_attempt", default=None)
_CURRENT_REQUEST: ContextVar[Optional[str]] = ContextVar("odr_request", default=None)


def current_session() -> Optional["RunTelemetry"]:
    """Return the session active in this task/context, if any."""
    return _CURRENT_SESSION.get()


def session_from_config(config: Optional[Mapping[str, Any]]) -> Optional["RunTelemetry"]:
    """Return the run-scoped session bound by the host (config first, then context)."""
    if config:
        value = (config.get("configurable") or {}).get(TELEMETRY_KEY) if isinstance(config, Mapping) else None
        if isinstance(value, RunTelemetry):
            return value
    return current_session()


def current_span_id() -> Optional[str]:
    """Return the business span that new stages should nest under."""
    return _CURRENT_SPAN.get()


def current_attempt_id() -> Optional[str]:
    """Return the model attempt currently in flight, if any."""
    return _CURRENT_ATTEMPT.get()


@contextlib.contextmanager
def attempt_context(attempt_id: str, request_id: str = "") -> Iterator[None]:
    """Make the in-flight attempt visible to the framework callback handler."""
    attempt_token = _CURRENT_ATTEMPT.set(attempt_id)
    request_token = _CURRENT_REQUEST.set(request_id) if request_id else None
    try:
        yield
    finally:
        try:
            _CURRENT_ATTEMPT.reset(attempt_token)
            if request_token is not None:
                _CURRENT_REQUEST.reset(request_token)
        except (ValueError, RuntimeError):  # pragma: no cover - different task
            _CURRENT_ATTEMPT.set(None)
            if request_token is not None:
                _CURRENT_REQUEST.set(None)


class _NullSpan:
    """A span-shaped no-op used when no session is active."""

    def __enter__(self) -> "_NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    async def __aenter__(self) -> "_NullSpan":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def set(self, **attributes: Any) -> "_NullSpan":
        return self

    def fail(self, *, error_class: str = "", status: str = STATUS_ERROR) -> "_NullSpan":
        return self


def span(operation: str, *, kind: str = SPAN_STAGE, attributes: Optional[Mapping[str, Any]] = None, links: Optional[Mapping[str, Any]] = None) -> Any:
    """Open a business span on the active session (no-op without one)."""
    session = current_session()
    if session is None or not session.enabled:
        return _NullSpan()
    return session.span(operation, kind=kind, attributes=attributes, links=links)


def record_event(event_type: str, **payload: Any) -> None:
    """Record a domain event on the active session (no-op without one)."""
    session = current_session()
    if session is not None:
        session.record_event(event_type, **payload)


class _SpanScope:
    """A business span usable as both a sync and an async context manager."""

    def __init__(self, session: "RunTelemetry", operation: str, *, kind: str, attributes: Optional[Mapping[str, Any]] = None, links: Optional[Mapping[str, Any]] = None):
        self.session = session
        self.operation = operation
        self.kind = kind
        self.extra = dict(attributes or {})
        self.links = dict(links or {})
        self.span_id = _new_span_id()
        self.parent_span_id = session.parent_span_for_new_span()
        self._start_mono = session.clock()
        self._started_at = _utc_now()
        self._token: Any = None
        self._closed = False
        self.status: Optional[str] = None
        self._forced_status: Optional[str] = None
        self.attributes: Dict[str, Any] = {}

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "_SpanScope":
        self._begin()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self._finish(classify_exception(exc), exc)
        else:
            self._finish(TERMINAL_OK, None)
        return False

    async def __aenter__(self) -> "_SpanScope":
        self._begin()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self._finish(classify_exception(exc), exc)
        else:
            self._finish(TERMINAL_OK, None)
        return False

    def _begin(self) -> None:
        if self._closed:
            return
        self._token = _CURRENT_SPAN.set(self.span_id)
        self.session._record_start(self)

    def set(self, **attributes: Any) -> "_SpanScope":
        """Attach business attributes (sanitised on write)."""
        self.attributes.update({k: v for k, v in attributes.items() if v is not None})
        return self

    def fail(self, *, error_class: str = "", status: str = STATUS_ERROR) -> "_SpanScope":
        """Mark a span that returned normally but whose business result is an error.

        A tool that converts its exception into an error *result* must not look like
        a successful tool call in the trace.
        """
        self._forced_status = status
        if error_class:
            self.attributes.setdefault("error_class", error_class)
        if self.attributes.get("task_status") is None:
            self.attributes["task_status"] = "error"
        return self

    def _finish(self, status: str, exc: Optional[BaseException]) -> None:
        if self._closed:
            return
        self._closed = True
        status = self._forced_status or status
        self.status = status
        if self._token is not None:
            try:
                _CURRENT_SPAN.reset(self._token)
            except (ValueError, RuntimeError):  # pragma: no cover - different task
                _CURRENT_SPAN.set(self.parent_span_id)
        duration_ms = round((self.session.clock() - self._start_mono) * 1000, 3)
        self.session._record_end(self, status=status, exc=exc, duration_ms=duration_ms)


class RunTelemetry:
    """One execution's observation session (see module docstring)."""

    def __init__(
        self,
        spec: RunSpec,
        *,
        run_dir: Path,
        store: Any = None,
        backend: Optional[TelemetryBackend] = None,
        clock: Any = time.monotonic,
        enabled: bool = True,
    ):
        self.spec = spec
        self.store = store
        self.clock = clock
        self.enabled = enabled and spec.telemetry_backend in _BACKENDS and spec.telemetry_backend != BACKEND_NONE
        self.backend = backend or make_backend(spec)
        self.directory = Path(run_dir) / "observability"
        self.executions_dir = self.directory / "executions"
        self.executions_index_path = self.directory / "executions.jsonl"
        self.execution_dir = self.executions_dir / spec.execution_id
        self.recorder: Optional[LocalRecorder] = None
        self.degraded = False
        self.telemetry_errors: List[Dict[str, str]] = []
        self._io_lock = threading.RLock()
        if self.enabled:
            try:
                self._create_execution_directory()
                self.recorder = LocalRecorder(self.execution_dir)
                self.recorder.write_manifest(spec.manifest())
                self._append_execution_index("started")
            except Exception as exc:  # noqa: BLE001
                # An unwritable record is a measurement failure, not a research
                # failure: keep the session disabled so nothing pretends to be recorded.
                self.degraded = True
                self.enabled = False
                self._note_error("init", exc)
                if spec.require_complete_trace:
                    raise TelemetryError(f"observability could not open an execution record: {_error_fingerprint(exc)[0]}") from exc
        self.root_span_id: Optional[str] = None
        self._spans: Dict[str, Dict[str, Any]] = {}
        self._open: Dict[str, Dict[str, Any]] = {}
        self._native: List[Dict[str, Any]] = []
        self._framework_map: Dict[str, str] = {}
        self._attempts: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._started = False
        self._closed = False
        self._finished = False
        self._product_status = ""
        self._export_status = EXPORT_DISABLED if spec.telemetry_backend in {BACKEND_NONE, BACKEND_LOCAL} else EXPORT_NOT_RUN
        self._export_reason = ""
        self._backend_scope: Any = None
        self._root_token: Any = None
        self._backend_trace_id: Optional[str] = None
        self._handlers: List["TelemetryCallbackHandler"] = []
        # A local/none execution emits no platform telemetry for its whole lifetime, not
        # only inside ``invocation()``: a parent process may have inherited tracing from
        # the environment or an outer scope, and the manifest must not say "local" while
        # an inherited exporter uploads the run.
        self._tracing_stack: Optional[contextlib.ExitStack] = None
        if spec.telemetry_backend in {BACKEND_NONE, BACKEND_LOCAL}:
            self._suppress_inherited_tracing()

    # -- tracing ownership -------------------------------------------------
    def _suppress_inherited_tracing(self) -> None:
        """Push a tracing-disabled scope for the execution's lifetime (never raises)."""
        try:
            stack = contextlib.ExitStack()
            stack.enter_context(suppress_inherited_tracing())
        except Exception as exc:  # noqa: BLE001 - an absent SDK can export nothing anyway
            LOGGER.warning("could not suppress inherited tracing: %s", type(exc).__name__)
            return
        self._tracing_stack = stack

    def _restore_tracing(self) -> None:
        """Pop the execution's tracing scope (idempotent, never raises)."""
        stack, self._tracing_stack = self._tracing_stack, None
        if stack is None:
            return
        try:
            stack.close()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("could not restore inherited tracing: %s", type(exc).__name__)

    # -- record layout -----------------------------------------------------
    def _create_execution_directory(self) -> None:
        """Create this execution's directory exclusively; never reuse an old id."""
        self.executions_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                os.mkdir(self.execution_dir)
                return
            except FileExistsError:
                self.spec.execution_id = _new_execution_id()
                self.execution_dir = self.executions_dir / self.spec.execution_id
        raise RuntimeError("could not allocate a unique execution directory")

    def _append_execution_index(self, event: str, **extra: Any) -> None:
        """Append one ``executions.jsonl`` row (the index, not the fact ledger)."""
        row = {
            "schema_version": SCHEMA_VERSION,
            "event": event,
            "execution_id": self.spec.execution_id,
            "odr_run_id": self.spec.odr_run_id,
            "previous_execution_id": self.spec.previous_execution_id,
            "previous_execution_reason": self.spec.previous_execution_reason,
            "backend": self.spec.telemetry_backend,
            "at_utc": _iso(_utc_now()),
            "path": str(self.execution_dir),
            **extra,
        }
        with self._io_lock:
            self.executions_index_path.parent.mkdir(parents=True, exist_ok=True)
            with self.executions_index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _note_error(self, operation: str, exc: BaseException) -> None:
        """Remember an observation failure without writing through the failed writer."""
        error_class, error_hash = _error_fingerprint(exc)
        self.degraded = True
        if len(self.telemetry_errors) < 50:
            self.telemetry_errors.append({"operation": operation, "error_class": error_class, "error_hash": error_hash})
        LOGGER.warning("observability %s failed: %s (%s)", operation, error_class, error_hash)
        recorder = self.recorder
        if recorder is None:
            return
        try:  # never recurse through the guarded writer that just failed
            recorder.event(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": f"{self.spec.execution_id}:telemetry_error:{uuid.uuid4().hex[:8]}",
                    "event_type": "telemetry_error",
                    "execution_id": self.spec.execution_id,
                    "odr_run_id": self.spec.odr_run_id,
                    "at_utc": _iso(_utc_now()),
                    "operation": operation,
                    "error_class": error_class,
                    "error_hash": error_hash,
                }
            )
        except Exception:  # noqa: BLE001 - the writer is broken; memory keeps the fact
            pass

    def _guard(self, operation: str, func: Any, default: Any = None) -> Any:
        """Run one observation step; an I/O/SDK error never reaches the product.

        ``Exception`` deliberately excludes ``CancelledError``, ``KeyboardInterrupt``
        and ``SystemExit``: observation may not swallow a cancellation or a signal.
        """
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            self._note_error(operation, exc)
            return default

    def record_layout(self) -> Dict[str, Any]:
        """Return this session's actual record location (execution-scoped)."""
        return {
            "schema": "execution",
            "root": str(self.directory),
            "execution_id": self.spec.execution_id,
            "path": str(self.execution_dir) if self.enabled else None,
            "degraded": self.degraded,
            "errors": list(self.telemetry_errors),
        }

    # -- construction ------------------------------------------------------
    @classmethod
    def open(cls, spec: RunSpec, *, run_dir: Path, store: Any = None, backend: Optional[TelemetryBackend] = None, clock: Any = time.monotonic) -> "RunTelemetry":
        """Open a session, honouring sampling and never raising for observation."""
        enabled = spec.telemetry_backend not in {BACKEND_NONE, ""}
        if enabled and spec.sample_rate < 1.0:
            enabled = random.random() <= max(0.0, spec.sample_rate)
        if spec.telemetry_backend not in _BACKENDS:
            LOGGER.warning("Unknown observability backend %r; recording locally only", spec.telemetry_backend)
            spec.telemetry_backend = BACKEND_LOCAL
        try:
            return cls(spec, run_dir=run_dir, store=store, backend=backend, clock=clock, enabled=enabled)
        except Exception as exc:  # noqa: BLE001
            if spec.require_complete_trace or spec.telemetry_backend not in {BACKEND_NONE, BACKEND_LOCAL}:
                if spec.require_complete_trace:
                    raise TelemetryError(f"observability could not start: {exc}") from exc
            LOGGER.warning("Observability disabled after an init error: %s", exc)
            return cls(spec, run_dir=run_dir, store=store, backend=TelemetryBackend(spec), clock=clock, enabled=False)

    # -- root wiring -------------------------------------------------------
    def attach_root(self, config: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None, tags: Optional[Sequence[str]] = None, run_name: Optional[str] = None) -> Dict[str, Any]:
        """Merge callbacks/metadata/tags into a copy of the root config.

        ``callbacks`` stays a top-level RunnableConfig key: putting it into
        ``configurable`` would let it be treated as a model setting (and would not
        reach the framework's run tree at all).
        """
        bound: Dict[str, Any] = dict(config)
        if not self.enabled:
            return bound
        bound["callbacks"] = self._merge_callbacks(bound.get("callbacks"))
        existing_meta = dict(bound.get("metadata") or {})
        # Host-owned keys win over anything already in the config.
        existing_meta.update(self.spec.root_metadata())
        if metadata:
            for key, value in metadata.items():
                existing_meta.setdefault(str(key), value)
        bound["metadata"] = existing_meta
        existing_tags = list(bound.get("tags") or [])
        for tag in list(self.spec.tags) + [str(t) for t in (tags or [])] + ["odr"]:
            if tag and tag not in existing_tags:
                existing_tags.append(tag)
        bound["tags"] = existing_tags
        bound["run_name"] = run_name or bound.get("run_name") or "odr.run"
        bound["configurable"] = {**(bound.get("configurable") or {}), TELEMETRY_KEY: self}
        return bound

    def _merge_callbacks(self, existing: Any) -> List[Any]:
        """Return existing handlers plus ours, de-duplicated by identity.

        In local mode an explicit remote tracer (``LangChainTracer`` or any
        ``langsmith`` tracer) would upload the run behind the local backend's back.
        Ordinary host callbacks/sentinels are preserved; the remote tracer is dropped
        and, under ``require_complete_trace`` (strict local), the contradiction is a
        stable configuration error instead.
        """
        handlers: List[Any] = []
        manager = getattr(existing, "handlers", None)
        if manager is not None and not isinstance(existing, (list, tuple)):
            handlers = list(manager)
        elif existing is None:
            handlers = []
        elif isinstance(existing, (list, tuple)):
            handlers = list(existing)
        else:  # a single handler object
            handlers = [existing]
        if self.spec.telemetry_backend == BACKEND_LOCAL:
            remote = [handler for handler in handlers if _is_remote_tracer(handler)]
            if remote:
                if self.spec.require_complete_trace:
                    raise TelemetryError(
                        "local telemetry conflicts with an explicit remote tracer "
                        f"({', '.join(type(handler).__name__ for handler in remote)}); "
                        "configure 'none'/'langsmith' or remove the tracer"
                    )
                handlers = [handler for handler in handlers if not _is_remote_tracer(handler)]
                self._note_error("local_remote_tracer_dropped", RuntimeError("explicit remote tracer removed in local mode"))
        for handler in handlers:
            if isinstance(handler, TelemetryCallbackHandler) and handler.session is self:
                # Already attached to this run: never create a second billing run.
                return handlers
        handler = TelemetryCallbackHandler(self)
        self._handlers.append(handler)
        handlers.append(handler)
        return handlers

    def child_config(self, config: Mapping[str, Any], *, metadata: Optional[Mapping[str, Any]] = None, tags: Optional[Sequence[str]] = None, run_name: Optional[str] = None) -> Dict[str, Any]:
        """Return an isolated config for one branch (no shared nested dicts)."""
        child: Dict[str, Any] = dict(config)
        base_meta = dict(config.get("metadata") or {})
        base_meta.update(self.spec.root_metadata())
        if metadata:
            base_meta.update({k: v for k, v in metadata.items() if v is not None})
        child["metadata"] = base_meta
        child_tags = list(config.get("tags") or [])
        for tag in tags or []:
            if tag and tag not in child_tags:
                child_tags.append(tag)
        child["tags"] = child_tags
        if run_name:
            child["run_name"] = run_name
        child["configurable"] = {**(config.get("configurable") or {})}
        return child

    # -- invocation --------------------------------------------------------
    @contextlib.contextmanager
    def invocation(self) -> Iterator["RunTelemetry"]:
        """Enter the run scope: tracing context, contextvars and the root span."""
        if not self.enabled:
            yield self
            return
        self._started = True
        try:
            self._export_status, self._backend_scope = self.backend.start()
        except Exception as exc:  # noqa: BLE001
            self._export_status = EXPORT_FAILED
            self._export_reason = f"{type(exc).__name__}: {exc}"
            self._note_error("backend_start", exc)
            self._backend_scope = contextlib.nullcontext()
        if self.spec.require_complete_trace and self._export_status in {EXPORT_FAILED, EXPORT_NOT_RUN}:
            raise TelemetryError(f"complete trace required but backend is {self._export_status}: {self._export_reason}")
        entered = False
        session_token = _CURRENT_SESSION.set(self)
        try:
            # The backend scope is a third-party context manager: entering it must
            # not be able to abort the research, so it gets the same guarded path.
            if self._backend_scope is not None:
                try:
                    self._backend_scope.__enter__()
                    entered = True
                except Exception as exc:  # noqa: BLE001
                    self._note_error("backend_enter", exc)
                    self._export_status = EXPORT_FAILED
            root = self.span("odr.run", kind=SPAN_RUN, attributes={"backend": self.spec.telemetry_backend})
            with root:
                self.root_span_id = root.span_id
                self._record_event_guarded("execution_started", {"execution_id": self.spec.execution_id, "odr_run_id": self.spec.odr_run_id})
                yield self
                self._record_event_guarded("execution_stopped", {"execution_id": self.spec.execution_id, "product_status": self._product_status})
        finally:
            try:
                _CURRENT_SESSION.reset(session_token)
            except (ValueError, RuntimeError):  # pragma: no cover
                _CURRENT_SESSION.set(None)
            if entered:
                try:
                    self._backend_scope.__exit__(None, None, None)
                except Exception as exc:  # noqa: BLE001
                    self._note_error("backend_exit", exc)
            self._backend_scope = None

    def parent_span_for_new_span(self) -> Optional[str]:
        """Return this session's current span, never a leak from another session."""
        candidate = current_span_id()
        if candidate and (candidate in self._open or candidate in self._spans):
            return candidate
        return self.root_span_id

    # -- spans -------------------------------------------------------------
    def span(self, operation: str, *, kind: str = SPAN_STAGE, attributes: Optional[Mapping[str, Any]] = None, links: Optional[Mapping[str, Any]] = None) -> _SpanScope:
        """Create a business span (a no-op scope when observation is disabled)."""
        if not self.enabled:
            scope = _SpanScope(self, operation, kind=kind, attributes=attributes, links=links)
            scope._begin = lambda: None  # type: ignore[method-assign]
            scope._finish = lambda status, exc: None  # type: ignore[method-assign]
            return scope
        return _SpanScope(self, operation, kind=kind, attributes=attributes, links=links)

    # Aliases used by the design document.
    stage = span

    def _record_start(self, scope: _SpanScope) -> None:
        record = {
            "span_id": scope.span_id,
            "parent_span_id": scope.parent_span_id,
            "kind": scope.kind,
            "operation": scope.operation,
            "started_at_utc": _iso(scope._started_at),
            "status": "running",
        }
        with self._lock:
            self._open[scope.span_id] = {
                "record": record,
                "start_mono": scope._start_mono,
                "started_at": scope._started_at,
            }
        self._record_event_guarded("span_start", {"span_id": scope.span_id, "parent_span_id": scope.parent_span_id, "kind": scope.kind, "operation": scope.operation})

    def _record_end(self, scope: _SpanScope, *, status: str, exc: Optional[BaseException], duration_ms: float) -> None:
        with self._lock:
            entry = self._open.pop(scope.span_id, None)
        try:
            attributes = sanitize_attributes({**scope.extra, **scope.attributes, **scope.links}, content_mode=self.spec.content_mode)
        except Exception as error:  # noqa: BLE001 - a hostile attribute must not abort a span
            self._note_error("sanitize", error)
            attributes = {}
        record: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "execution_id": self.spec.execution_id,
            "odr_run_id": self.spec.odr_run_id,
            "span_id": scope.span_id,
            "parent_span_id": scope.parent_span_id,
            "kind": scope.kind,
            "operation": scope.operation,
            "started_at_utc": _iso(scope._started_at),
            "ended_at_utc": _iso(_utc_now()),
            "duration_ms": max(0.0, duration_ms),
            "status": status,
            "attributes": attributes,
        }
        error_class = type(exc).__name__ if exc is not None else str(scope.attributes.get("error_class") or "")
        if error_class:
            record["error_class"] = error_class
        if exc is not None:
            record["error_hash"] = f"sha256:{hashlib.sha256(str(exc).encode('utf-8')).hexdigest()[:16]}"
        with self._lock:
            self._spans[scope.span_id] = record
            if scope.kind == SPAN_MODEL_ATTEMPT:
                self._attempts[attributes.get("attempt_id") or scope.span_id] = record
        # The span is already materialised in memory; a full disk loses the file row,
        # never the research result and never the in-process reconciliation.
        if self.recorder is not None:
            self._guard("span_write", lambda: self.recorder.span(record))
        if scope.kind == SPAN_MODEL_ATTEMPT:
            self._guard("native_bind", lambda: self._bind_native_by_window(scope.span_id, scope._start_mono, self.clock()))
        self._record_event_guarded(
            "span_end",
            {"span_id": scope.span_id, "kind": scope.kind, "operation": scope.operation, "status": status, "duration_ms": round(duration_ms, 3)},
        )

    # -- events / correlation ---------------------------------------------
    def record_event(self, event_type: str, **payload: Any) -> None:
        """Record one domain event, sanitising every value."""
        self._record_event_guarded(event_type, payload)

    def _record_event_guarded(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self.recorder is None:
            return
        try:
            safe = sanitize_attributes({k: v for k, v in payload.items() if k in _ALLOWED_ATTRIBUTES}, content_mode=self.spec.content_mode)
            self.recorder.event(
                {
                    "schema_version": SCHEMA_VERSION,
                    "event_id": f"{self.spec.execution_id}:{event_type}:{uuid.uuid4().hex[:8]}",
                    "event_type": event_type,
                    "execution_id": self.spec.execution_id,
                    "odr_run_id": self.spec.odr_run_id,
                    "span_id": current_span_id(),
                    "at_utc": _iso(_utc_now()),
                    **safe,
                }
            )
        except Exception as exc:  # noqa: BLE001 - observation must never break a run
            self._note_error(f"event:{event_type}", exc)

    def bind_native_run(self, framework_run_id: str, *, parent_run_id: Optional[str] = None, span_id: Optional[str] = None) -> None:
        """Add one framework run to the local crosswalk."""
        if not self.enabled:
            return
        with self._lock:
            if span_id:
                self._framework_map[framework_run_id] = span_id
            if parent_run_id is None and self._backend_trace_id is None and self.spec.telemetry_backend == BACKEND_LANGSMITH:
                # In LangSmith the root run id is the trace id.
                self._backend_trace_id = framework_run_id

    def register_framework_span(self, framework_run_id: str, span_id: str, *, parent_framework_run_id: Optional[str], kind: str, operation: str, started_at: datetime, started_mono: float) -> None:
        """Register a native run's span so its children can find their parent."""
        if not self.enabled:
            return
        with self._lock:
            self._framework_map[framework_run_id] = span_id
            if parent_framework_run_id is None and self.root_span_id:
                self._framework_map.setdefault("__root__", self.root_span_id)

    def _bind_native_by_window(self, attempt_span_id: str, start_mono: float, end_mono: float) -> None:
        """Fallback binding when the callback ran outside the attempt's context.

        The contextvar normally carries the attempt id; this matches the native
        generation by its overlap with the attempt's own window instead of trusting
        wall-clock order across tasks.
        """
        with self._lock:
            candidates = [item for item in self._native if not item.get("attempt_id") and item.get("span_id") and item.get("start_mono") is not None]
            matches = [item for item in candidates if start_mono - 0.001 <= item["start_mono"] <= end_mono + 0.001]
            # Only an unambiguous single generation may be joined by window. If two
            # overlap the same attempt (e.g. a nested model call), leave them unbound
            # and let reconciliation expose the gap instead of guessing.
            if len(matches) == 1:
                item = matches[0]
                item["attempt_id"] = attempt_span_id
                record = self._spans.get(item["span_id"])
                if record is not None:
                    record.setdefault("attributes", {})["attempt_id"] = attempt_span_id

    def note_native_span(self, span: Dict[str, Any]) -> None:
        """Remember one native generation for reconciliation."""
        with self._lock:
            self._native.append(span)

    # -- finish / close ----------------------------------------------------
    def finish(self, product_status: str, *, error: Optional[BaseException] = None) -> None:
        """Record the product verdict (not the telemetry verdict)."""
        if not self.enabled or self._finished:
            return
        self._finished = True
        self._product_status = str(product_status or "")
        self._record_event_guarded("run_finished", {"status_detail": self._product_status[:120], "error_class": type(error).__name__ if error else None})

    def close(self, *, export_deadline_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Close open spans, export within a bound, write reconciliation."""
        if not self.enabled or self._closed or self.recorder is None:
            self._closed = True
            self._restore_tracing()
            return {"enabled": False, "degraded": self.degraded, "telemetry_errors": list(self.telemetry_errors)}
        self._closed = True
        deadline = float(export_deadline_seconds if export_deadline_seconds is not None else self.spec.export_deadline_seconds)
        if not self._finished:
            self.finish(STATUS_INCOMPLETE if self._started else STATUS_NOT_STARTED)
        # Spans still open mean the process exited (or a stage leaked): record the
        # observed duration as incomplete instead of dropping them.
        for span_id, entry in list(self._open.items()):
            record = entry["record"]
            record.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "execution_id": self.spec.execution_id,
                    "odr_run_id": self.spec.odr_run_id,
                    "ended_at_utc": _iso(_utc_now()),
                    "duration_ms": round(max(0.0, (self.clock() - entry["start_mono"]) * 1000), 3),
                    "status": STATUS_INCOMPLETE,
                    "missing_reason": "process_exited_before_span_end",
                    "attributes": {},
                }
            )
            self._guard("span_write", lambda row=record: self.recorder.span(row))
            with self._lock:
                self._spans[span_id] = record
        self._open.clear()
        for handler in self._handlers:
            try:
                handler.finalize_incomplete()
            except Exception as exc:  # noqa: BLE001
                self._note_error("finalize_native", exc)
        receipt = self._export(deadline)
        reconciliation = self.reconcile()
        try:
            _, integrity = self.recorder.read_spans_checked()
        except Exception:  # noqa: BLE001
            integrity = {"torn_lines": None, "rows": None, "readable": False}
        reconciliation["record_integrity"] = integrity
        self._guard(
            "trace_map_write",
            lambda: self.recorder.write_json(
                "trace-map.json",
                {
                    "execution_id": self.spec.execution_id,
                    "odr_run_id": self.spec.odr_run_id,
                    "framework_runs": dict(self._framework_map),
                    "attempts": dict(self._attempts),
                    "backend": self.spec.telemetry_backend,
                    "backend_trace_id": self._backend_trace_id,
                    "backend_url": receipt.url,
                },
            ),
        )
        self._guard(
            "export_state_write",
            lambda: self.recorder.write_json(
                "export-state.json",
                {
                    "backend": receipt.backend,
                    "status": receipt.status,
                    "reason": receipt.reason,
                    "remote_confirmed": receipt.remote_confirmed,
                    "duration_ms": receipt.duration_ms,
                    "deadline_seconds": deadline,
                    "attempts": 1,
                },
            ),
        )
        self._guard("reconciliation_write", lambda: self.recorder.write_json("reconciliation.json", reconciliation))
        self._guard("execution_index_stop", lambda: self._append_execution_index("stopped", status=receipt.status, degraded=self.degraded, product_status=self._product_status))
        self._restore_tracing()
        return {
            "enabled": True,
            "export": receipt.status,
            "reconciliation_ok": reconciliation.get("ok"),
            "local_accounting_status": reconciliation.get("local_accounting_status"),
            "remote_trace_status": reconciliation.get("remote_trace_status"),
            "cost_completeness": reconciliation.get("cost_completeness"),
            "degraded": self.degraded,
            "execution_id": self.spec.execution_id,
        }

    def _export(self, deadline: float) -> ExportReceipt:
        receipt = ExportReceipt(
            backend=self.spec.telemetry_backend,
            status=self._export_status if self._export_status else EXPORT_DISABLED,
            reason=self._export_reason,
            trace_id=self._backend_trace_id,
        )
        try:
            receipt = self.backend.stop(receipt=receipt, deadline_seconds=deadline)
        except Exception as exc:  # noqa: BLE001
            receipt.status = EXPORT_FAILED
            receipt.reason = f"{type(exc).__name__}: {str(exc)[:120]}"
        self._export_status = receipt.status
        return receipt

    # -- reconciliation ----------------------------------------------------
    def reconcile(self) -> Dict[str, Any]:
        """Compare spans, native generations, reservations and the usage ledger.

        The old version answered one boolean and could call an attempt with no
        native generation and no usage row "ok". Three independent conclusions are
        reported instead, each with its own denominator:

        * ``local_accounting_status`` - were all *sent* attempts recorded exactly once,
          with one native generation and one usage row each;
        * ``remote_trace_status`` - did the platform trace get confirmed (never guessed
          from a local flush);
        * ``cost_completeness`` - is every usage row priced (unknown is not free).

        A provider that cannot emit native callbacks is reported as ``unsupported``
        for the native column; it does not silently become a pass.
        """
        spans = list(self._spans.values())
        span_ids = {span["span_id"] for span in spans}
        parents = {span.get("parent_span_id") for span in spans if span.get("parent_span_id")}
        orphan_spans = sorted(parent for parent in parents if parent not in span_ids)
        attempts = [span for span in spans if span.get("kind") == SPAN_MODEL_ATTEMPT]
        attempt_ids = [span.get("attributes", {}).get("attempt_id") or span["span_id"] for span in attempts]
        native = list(self._native)
        native_per_attempt: Dict[str, int] = {}
        unbound_native: List[str] = []
        for item in native:
            attempt_id = item.get("attempt_id") or ""
            if attempt_id:
                native_per_attempt[attempt_id] = native_per_attempt.get(attempt_id, 0) + 1
            elif item.get("span_id"):
                unbound_native.append(item["span_id"])
        duplicate_native = sorted(attempt for attempt, count in native_per_attempt.items() if count > 1)
        # Native callbacks are expected for LangChain chat models. A plain test double
        # (or a provider without the callback) is marked unsupported at the attempt,
        # so its absence is a capability statement, not a fabricated pass.
        native_expected = [span for span in attempts if span.get("attributes", {}).get("native_callback_expected", True) is not False]
        native_unsupported = [span.get("attributes", {}).get("attempt_id") or span["span_id"] for span in attempts if span.get("attributes", {}).get("native_callback_expected") is False]
        missing_native = sorted(attempt for attempt in (span.get("attributes", {}).get("attempt_id") or span["span_id"] for span in native_expected) if attempt not in native_per_attempt)
        usage_rows, unattributed_usage = self._usage_rows_for_execution()
        usage_attempts = {row.get("attempt_id") for row in usage_rows if row.get("attempt_id")}
        missing_usage = sorted(attempt for attempt in attempt_ids if attempt not in usage_attempts)
        orphan_usage = sorted(attempt for attempt in usage_attempts if attempt and attempt not in set(attempt_ids))
        unknown_usage = sorted(row.get("attempt_id") or row.get("request_id") or "?" for row in usage_rows if row.get("input_tokens") is None and row.get("output_tokens") is None)
        reservations = self._reservations()
        unfinished = sorted(entry.get("id") or entry.get("attempt_id") or "?" for entry in reservations if not entry.get("settled"))
        failed = [span["span_id"] for span in spans if span.get("status") in {STATUS_ERROR, STATUS_TIMEOUT, STATUS_BUDGET_DENIED, STATUS_CANCELLED, STATUS_INCOMPLETE}]
        known_usage = sum(1 for row in usage_rows if row.get("input_tokens") is not None or row.get("output_tokens") is not None)
        coverage = (len(attempt_ids) - len(missing_usage)) / len(attempt_ids) if attempt_ids else None
        native_coverage = (len(native_expected) - len(missing_native)) / len(native_expected) if native_expected else None

        local_fail = bool(missing_usage or orphan_usage or missing_native or duplicate_native or orphan_spans or unbound_native)
        if local_fail:
            local_status = "fail"
        elif not attempts and not usage_rows:
            local_status = "unknown"
        else:
            local_status = "pass"
        if self.degraded and local_status == "pass":
            # A record writer failed: "pass" would overclaim what was actually stored.
            local_status = "unknown"

        known_costs = [row.get("cost_usd") for row in usage_rows if row.get("cost_usd") is not None]
        if not usage_rows:
            cost_status = "complete" if not attempts else "unknown"
        elif len(known_costs) == len(usage_rows) and not unfinished:
            cost_status = "complete"
        elif known_costs:
            cost_status = "partial"
        else:
            cost_status = "unknown"
        if unfinished and cost_status == "complete":
            cost_status = "partial"

        remote_status = self._remote_trace_status()
        expected_remote = sorted(
            span["span_id"]
            for span in spans
            if self.spec.telemetry_backend not in {BACKEND_NONE, BACKEND_LOCAL} and span.get("kind") in {SPAN_RUN, SPAN_STAGE, SPAN_AGENT, SPAN_REVIEW}
        ) if self.spec.telemetry_backend not in {BACKEND_NONE, BACKEND_LOCAL} else []
        returned_remote = getattr(self, "_remote_span_ids", None)
        missing_remote = sorted(set(expected_remote) - set(returned_remote)) if returned_remote is not None else []
        unexpected_remote = sorted(set(returned_remote) - set(expected_remote)) if returned_remote is not None else []

        ok = local_status != "fail" and remote_status not in {"fail", "error"}
        return {
            "schema_version": SCHEMA_VERSION,
            "execution_id": self.spec.execution_id,
            "backend": self.spec.telemetry_backend,
            "record_schema": "execution",
            "export_status": self._export_status,
            "degraded": self.degraded,
            "telemetry_errors": list(self.telemetry_errors),
            # Three independent conclusions, never collapsed into one boolean.
            "local_accounting_status": local_status,
            "remote_trace_status": remote_status,
            "cost_completeness": cost_status,
            "ok": ok,
            "counts": {
                "spans": len(spans),
                "attempt_spans": len(attempts),
                "native_llm_spans": len([span for span in spans if span.get("kind") == SPAN_NATIVE_LLM]),
                "usage_rows": len(usage_rows),
                "unattributed_usage_rows": len(unattributed_usage),
                "unknown_usage_rows": len(unknown_usage),
                "failed_spans": len(failed),
                "expected_native_attempts": len(native_expected),
                "unsupported_native_attempts": len(native_unsupported),
                "unfinished_reservations": len(unfinished),
            },
            "span_coverage": coverage,
            "attempt_usage_coverage": coverage,
            "native_coverage": native_coverage,
            "token_known_coverage": (known_usage / len(usage_rows)) if usage_rows else None,
            "missing_usage_attempts": missing_usage,
            "orphan_usage_attempts": orphan_usage,
            "missing_native_attempts": missing_native,
            "unsupported_native_attempts": sorted(native_unsupported),
            "unbound_native_spans": unbound_native,
            "duplicate_native_attempts": duplicate_native,
            "unknown_usage_attempts": unknown_usage,
            "unfinished_reservations": unfinished,
            "orphan_spans": orphan_spans,
            "failed_span_ids": failed,
            "expected_remote_span_ids": expected_remote,
            "missing_remote_span_ids": missing_remote,
            "unexpected_remote_span_ids": unexpected_remote,
            "remote_readback_available": returned_remote is not None,
            "cost_known_rows": len(known_costs),
            "cost_unknown_rows": len(usage_rows) - len(known_costs),
            "known_cost_lower_bound_usd": round(sum(known_costs), 6) if known_costs else None,
            "provider_billed_cost_usd": None,
            "cost_method": "frozen_price_estimation" if known_costs else "unknown",
        }

    def _remote_trace_status(self) -> str:
        if self.spec.telemetry_backend in {BACKEND_NONE, BACKEND_LOCAL}:
            return "not_applicable"
        if self._export_status == EXPORT_CONFIRMED:
            return "pass"
        if self._export_status in {EXPORT_PENDING, EXPORT_PARTIAL}:
            return "pending"
        if self._export_status == EXPORT_FAILED:
            return "fail"
        return "error"

    def _reservations(self) -> List[Dict[str, Any]]:
        getter = getattr(self.store, "reservations", None)
        if not callable(getter):
            return []
        try:
            rows = list(getter())
        except Exception:  # noqa: BLE001
            return []
        scoped: List[Dict[str, Any]] = []
        for row in rows:
            execution_id = row.get("execution_id") or ""
            if execution_id in ("", self.spec.execution_id):
                scoped.append(row)
        return scoped

    def _usage_rows(self) -> List[Dict[str, Any]]:
        """Read the run's own usage ledger (never the platform's copy)."""
        rows, _ = self._usage_rows_for_execution()
        return rows

    def _usage_rows_for_execution(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return ``(rows of this execution, rows without an execution id)``.

        A row with no ``execution_id`` is legacy/manual: it is kept visible and
        counted separately, because silently attaching it to every execution would
        double-count a re-run.
        """
        rows: List[Dict[str, Any]] = []
        if self.store is not None:
            try:
                rows = [row.model_dump(mode="json") if hasattr(row, "model_dump") else dict(row) for row in self.store.metadata.usage.values()]
            except Exception:  # noqa: BLE001
                rows = []
        else:
            path = Path(self.spec_run_dir()) / "metrics" / "usage.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows = list(payload.get("usage") or [])
            except Exception:  # noqa: BLE001
                rows = []
        scoped = [row for row in rows if (row.get("execution_id") or "") == self.spec.execution_id]
        unattributed = [row for row in rows if not (row.get("execution_id") or "")]
        # A ledger row with no execution id is legacy/manual. With one execution in the
        # run it can only belong to that execution; with several it is reported as
        # unattributed instead of being counted in every re-run's totals.
        if unattributed and self._execution_count() <= 1:
            scoped = scoped + unattributed
        return scoped, unattributed

    def _execution_count(self) -> int:
        try:
            return len([path for path in self.executions_dir.iterdir() if path.is_dir()])
        except OSError:
            return 1

    def spec_run_dir(self) -> Path:
        return self.directory.parent

    # -- introspection -----------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Return an in-memory summary (used by tests and the benchmark result)."""
        return {
            "enabled": self.enabled,
            "backend": self.spec.telemetry_backend,
            "execution_id": self.spec.execution_id,
            "spans": len(self._spans),
            "open_spans": len(self._open),
            "native_spans": len(self._native),
            "export_status": self._export_status,
            "directory": str(self.directory) if self.enabled else None,
            "execution_dir": str(self.execution_dir) if self.enabled else None,
            "degraded": self.degraded,
            "telemetry_errors": list(self.telemetry_errors),
        }

    def spans(self) -> List[Dict[str, Any]]:
        """Return materialised spans in memory (test/debug view)."""
        return list(self._spans.values())


# --------------------------------------------------------------------------- #
# Framework callback handler
# --------------------------------------------------------------------------- #
class TelemetryCallbackHandler(BaseCallbackHandler):
    """Record native graph/node/model/tool runs into the local record.

    This is deliberately not a second tracer: it creates no billing generation.
    The usage ledger stays the only billable source; native token counts are
    stored with ``counts_for_billing=False`` for reconciliation.

    Callbacks stay inline so the attempt contextvar is visible when a run starts
    and the ordering of nested runs is preserved.
    """

    run_inline = True

    def __init__(self, session: RunTelemetry):
        self.session = session
        self._open: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # -- helpers -----------------------------------------------------------
    def _parent(self, parent_run_id: Optional[str]) -> Optional[str]:
        if parent_run_id:
            mapped = self.session._framework_map.get(str(parent_run_id))
            if mapped:
                return mapped
        return current_span_id() or self.session.root_span_id

    def _start(self, kind: str, operation: str, run_id: Any, parent_run_id: Any, *, attributes: Optional[Mapping[str, Any]] = None) -> None:
        if not self.session.enabled:
            return
        span_id = _new_span_id()
        framework_id = str(run_id)
        parent = self._parent(str(parent_run_id) if parent_run_id else None)
        started = _utc_now()
        started_mono = self.session.clock()
        record = {
            "schema_version": SCHEMA_VERSION,
            "execution_id": self.session.spec.execution_id,
            "odr_run_id": self.session.spec.odr_run_id,
            "span_id": span_id,
            "parent_span_id": parent,
            "kind": kind,
            "operation": operation,
            "started_at_utc": _iso(started),
            "status": "running",
            "framework_run_id": framework_id,
            "attributes": {},
        }
        with self._lock:
            self._open[framework_id] = {"record": record, "start_mono": started_mono, "started_at": started, "attributes": dict(attributes or {})}
        self.session.register_framework_span(framework_id, span_id, parent_framework_run_id=str(parent_run_id) if parent_run_id else None, kind=kind, operation=operation, started_at=started, started_mono=started_mono)
        self.session.bind_native_run(framework_id, parent_run_id=str(parent_run_id) if parent_run_id else None, span_id=span_id)
        self.session._record_event_guarded("native_span_start", {"span_id": span_id, "kind": kind, "operation": operation, "framework_run_id": framework_id})

    def _end(self, run_id: Any, *, status: str = TERMINAL_OK, usage: Optional[Mapping[str, Any]] = None, error: Optional[BaseException] = None) -> None:
        if not self.session.enabled:
            return
        framework_id = str(run_id)
        with self._lock:
            entry = self._open.pop(framework_id, None)
        if entry is None:
            return
        record = entry["record"]
        duration_ms = round(max(0.0, (self.session.clock() - entry["start_mono"]) * 1000), 3)
        record.update({"ended_at_utc": _iso(_utc_now()), "duration_ms": duration_ms, "status": status})
        attributes = dict(entry.get("attributes") or {})
        attempt_id = current_attempt_id() or ""
        request_id = _CURRENT_REQUEST.get() or ""
        if attempt_id:
            attributes["attempt_id"] = attempt_id
        if request_id:
            attributes["request_id"] = request_id
        if usage:
            attributes.update({k: v for k, v in usage.items() if k in _ALLOWED_ATTRIBUTES})
        if error is not None:
            record["error_class"] = type(error).__name__
            record["error_hash"] = f"sha256:{hashlib.sha256(str(error).encode('utf-8')).hexdigest()[:16]}"
        record["attributes"] = sanitize_attributes(attributes, content_mode=self.session.spec.content_mode)
        with self.session._lock:
            self.session._spans[record["span_id"]] = record
        if self.session.recorder is not None:
            self.session.recorder.span(record)
        if record["kind"] == SPAN_NATIVE_LLM:
            self.session.note_native_span(
                {
                    "span_id": record["span_id"],
                    "attempt_id": record.get("attributes", {}).get("attempt_id") or "",
                    "model": record.get("attributes", {}).get("model"),
                    "start_mono": entry["start_mono"],
                    "input_tokens": record.get("attributes", {}).get("input_tokens"),
                    "output_tokens": record.get("attributes", {}).get("output_tokens"),
                }
            )
        self.session._record_event_guarded(
            "native_span_end",
            {"span_id": record["span_id"], "kind": record["kind"], "operation": record["operation"], "status": status, "duration_ms": duration_ms},
        )

    def finalize_incomplete(self) -> None:
        """Materialise native runs that never ended before the process stopped."""
        with self._lock:
            entries = list(self._open.items())
            self._open.clear()
        for _framework_id, entry in entries:
            record = entry["record"]
            record.update(
                {
                    "ended_at_utc": _iso(_utc_now()),
                    "duration_ms": round(max(0.0, (self.session.clock() - entry["start_mono"]) * 1000), 3),
                    "status": STATUS_INCOMPLETE,
                    "missing_reason": "process_exited_before_native_run_end",
                    "attributes": sanitize_attributes(entry.get("attributes") or {}, content_mode=self.session.spec.content_mode),
                }
            )
            with self.session._lock:
                self.session._spans[record["span_id"]] = record
            if self.session.recorder is not None:
                self.session.recorder.span(record)

    # -- chain -------------------------------------------------------------
    def on_chain_start(self, serialized: Any, inputs: Any, *, run_id: Any, parent_run_id: Any = None, tags: Any = None, metadata: Any = None, **kwargs: Any) -> None:
        try:
            name = ""
            if isinstance(serialized, Mapping):
                name = str(serialized.get("name") or serialized.get("id") or "")
            self._start(SPAN_FRAMEWORK_CHAIN, name or "chain", run_id, parent_run_id, attributes={"framework_name": name[:120] or "chain", "tag_count": len(tags or [])})
        except Exception:  # noqa: BLE001
            LOGGER.debug("on_chain_start observation failed", exc_info=True)

    def on_chain_end(self, outputs: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        try:
            self._end(run_id)
        except Exception:  # noqa: BLE001
            LOGGER.debug("on_chain_end observation failed", exc_info=True)

    def on_chain_error(self, error: BaseException, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        try:
            self._end(run_id, status=STATUS_ERROR, error=error)
        except Exception:  # noqa: BLE001
            LOGGER.debug("on_chain_error observation failed", exc_info=True)

    # -- model -------------------------------------------------------------
    def _model_name(self, serialized: Any, metadata: Any) -> str:
        for source in (metadata or {}, serialized or {}):
            if isinstance(source, Mapping):
                for key in ("ls_model_name", "model_name", "model"):
                    value = source.get(key)
                    if value:
                        return str(value)[:120]
        return "unknown"

    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: Any, parent_run_id: Any = None, tags: Any = None, metadata: Any = None, **kwargs: Any) -> None:
        self._llm_start(serialized, run_id, parent_run_id, tags, metadata)

    def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: Any, parent_run_id: Any = None, tags: Any = None, metadata: Any = None, **kwargs: Any) -> None:
        self._llm_start(serialized, run_id, parent_run_id, tags, metadata)

    def _llm_start(self, serialized: Any, run_id: Any, parent_run_id: Any, tags: Any, metadata: Any) -> None:
        try:
            self._start(
                SPAN_NATIVE_LLM,
                "llm.generate",
                run_id,
                parent_run_id,
                attributes={
                    "model": self._model_name(serialized, metadata),
                    "channel": "text",
                    "streamed": False,
                    "counts_for_billing": False,
                    "tag_count": len(tags or []),
                },
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug("llm start observation failed", exc_info=True)

    def on_llm_end(self, response: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        try:
            self._end(run_id, usage=_native_usage(response))
        except Exception:  # noqa: BLE001
            LOGGER.debug("llm end observation failed", exc_info=True)

    def on_llm_error(self, error: BaseException, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        try:
            self._end(run_id, status=STATUS_ERROR, error=error)
        except Exception:  # noqa: BLE001
            LOGGER.debug("llm error observation failed", exc_info=True)

    # -- tools -------------------------------------------------------------
    def on_tool_start(self, serialized: Any, input_str: Any, *, run_id: Any, parent_run_id: Any = None, tags: Any = None, metadata: Any = None, inputs: Any = None, **kwargs: Any) -> None:
        try:
            name = ""
            if isinstance(serialized, Mapping):
                name = str(serialized.get("name") or "")
            self._start(SPAN_NATIVE_TOOL, name or "tool", run_id, parent_run_id, attributes={"tool_name": name[:120] or "tool"})
        except Exception:  # noqa: BLE001
            LOGGER.debug("tool start observation failed", exc_info=True)

    def on_tool_end(self, output: Any, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        try:
            self._end(run_id)
        except Exception:  # noqa: BLE001
            LOGGER.debug("tool end observation failed", exc_info=True)

    def on_tool_error(self, error: BaseException, *, run_id: Any, parent_run_id: Any = None, **kwargs: Any) -> None:
        try:
            self._end(run_id, status=STATUS_ERROR, error=error)
        except Exception:  # noqa: BLE001
            LOGGER.debug("tool error observation failed", exc_info=True)


def _native_usage(response: Any) -> Dict[str, Any]:
    """Extract provider-reported usage without inventing missing values."""
    usage: Dict[str, Any] = {}
    llm_output = getattr(response, "llm_output", None) or {}
    token_usage = {}
    if isinstance(llm_output, Mapping):
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if not token_usage:
        for generation_list in getattr(response, "generations", []) or []:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                meta = getattr(message, "usage_metadata", None)
                if meta:
                    token_usage = meta
                    break
            if token_usage:
                break
    if isinstance(token_usage, Mapping) and token_usage:
        usage["input_tokens"] = token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
        usage["output_tokens"] = token_usage.get("completion_tokens", token_usage.get("output_tokens"))
        details = token_usage.get("input_token_details") or {}
        if isinstance(details, Mapping) and details.get("cache_read") is not None:
            usage["cache_read_tokens"] = details.get("cache_read")
        usage["native_tokens_present"] = usage.get("input_tokens") is not None or usage.get("output_tokens") is not None
        usage["metric_status"] = METRIC_MEASURED if usage["native_tokens_present"] else METRIC_UNKNOWN
    else:
        usage["native_tokens_present"] = False
        usage["metric_status"] = METRIC_UNKNOWN
        usage["missing_reason"] = "provider_returned_no_usage"
    return usage


# --------------------------------------------------------------------------- #
# Timing helpers (used by tests and by the metrics collector)
# --------------------------------------------------------------------------- #
def union_ms(intervals: Iterable[Tuple[float, float]]) -> float:
    """Return the length of the union of ``(start, end)`` intervals, in ms.

    Overlapping parallel children must be merged before subtracting from a parent:
    subtracting the *sum* of child durations produces negative self time.
    """
    ordered = sorted((max(0.0, float(s)), max(0.0, float(e))) for s, e in intervals if e is not None)
    total = 0.0
    current_start: Optional[float] = None
    current_end: Optional[float] = None
    for start, end in ordered:
        if end < start:
            continue
        if current_start is None:
            current_start, current_end = start, end
            continue
        if start <= (current_end or 0.0):
            current_end = max(current_end or 0.0, end)
        else:
            total += (current_end or 0.0) - current_start
            current_start, current_end = start, end
    if current_start is not None:
        total += (current_end or 0.0) - current_start
    return round(total, 3)


def batch_timing(children: Sequence[Tuple[float, float]], *, batch_start: float = 0.0, batch_end: Optional[float] = None) -> Dict[str, Any]:
    """Return batch wall / agent work / union / uncovered time for one batch.

    ``children`` are relative ``(start, end)`` pairs. ``agent_work_ms`` is the sum
    of the children's own durations (what they spent working, which double counts
    parallel time); ``uncovered_ms`` is the parent interval not covered by the
    union of children, i.e. the part that is not explained by a child at all.
    """
    normalized = [(float(s), float(e)) for s, e in children]
    end = float(batch_end) if batch_end is not None else max((e for _, e in normalized), default=float(batch_start))
    wall = max(0.0, end - float(batch_start))
    work = round(sum(max(0.0, e - s) for s, e in normalized), 3)
    covered = union_ms(normalized)
    return {
        "batch_wall_ms": round(wall, 3),
        "agent_work_ms": work,
        "union_ms": covered,
        "uncovered_ms": round(max(0.0, wall - covered), 3),
        "slowest_branch_ms": round(max((max(0.0, e - s) for s, e in normalized), default=0.0), 3),
    }