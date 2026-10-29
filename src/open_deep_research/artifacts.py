"""受控记录与读取层（ArtifactStore）。

设计依据：``docs/context-review-design/CONTEXT_AND_STORAGE.md``。

本模块把三件事分开：

1. **记录**：完整请求、上下文清单、证据原文和指标写入磁盘，供审计与回放。
2. **授权**：身份由宿主绑定（``Identity``），读取证据必须持有与当前任务、运行
   匹配的授权；文件名、ID、目录都不是凭证。
3. **进入本次请求**：由 :mod:`open_deep_research.context` 决定哪些片段进入模型
   的请求。存储完整不等于模型可见。

威胁范围与文档一致：同一 Python 进程内的隔离，防止错误组装、工具误用、外部
材料诱导越权和并发串线。没有任意文件读取工具时（默认部署），模型永远接触不到
真实路径；即使给了 ``read_evidence`` 工具，路径也只由程序按 ``evidence_id``
解析，并检查目录穿越与符号链接。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from open_deep_research.state import (
    ClaimRecord,
    ContextManifest,
    EvidenceRecord,
    ResearchBrief,
    RevalidationEvent,
    TaskResult,
    UsageRecord,
)

LOGGER = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ASCII_TERM_RE = re.compile(r"[A-Za-z0-9_./:-]{2,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
# Credential-named fields only. A substring match on "token" also redacted token
# *counts* (``estimated_input_tokens``), which destroyed the audit trail for exactly
# the budget events an operator needs to read.
_SECRET_KEY_RE = re.compile(
    r"^(?:.*[_-])?(?:api[_-]?key|key|keys|secret|secrets|token|tokens_secret|password|passwd|authorization|credential|credentials|bearer)$",
    re.IGNORECASE,
)

# Roles that may read their own store folder.
_PRIVATE_ROLES = {"researcher", "supervisor", "writer"}
# Roles whose events are operator-only and never readable by sibling agents.
_OPERATOR_ONLY_ROLES = {"reviewer"}


def _canonical_source_url(url: str) -> str:
    """Normalize a source url for duplicate detection (scheme/host case, fragment)."""
    value = str(url or "").strip()
    if not value:
        return ""
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(value)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


class ArtifactError(Exception):
    """Base class for structured artifact errors."""


class PermissionDenied(ArtifactError):
    """The identity is not allowed to read or publish this object."""

    status = "permission_denied"


class MissingEvidence(ArtifactError):
    """The referenced evidence does not exist or was not archived."""

    status = "missing"


class InvalidArtifactId(ArtifactError):
    """The identifier or selector is malformed or attempts path traversal."""

    status = "invalid_id"


class PublishRejected(ArtifactError):
    """The publication violates a contract (version, ownership, status)."""

    status = "rejected"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp with a stable format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def search_terms(text: str) -> List[str]:
    """Return retrieval terms, including CJK bigrams when the text has no spaces.

    A Chinese request such as ``企业版的数据保留期`` is one unspaced run. Treating the
    whole run as a single term makes exact-substring matching fail even when the
    archived sentence obviously contains the condition. Bigrams keep the lookup
    lexical (no model, no translation) while making unspaced Chinese searchable.
    ASCII words are unchanged.
    """
    terms: List[str] = []
    seen: set = set()

    def add(term: str) -> None:
        lowered = term.lower()
        if lowered and lowered not in seen:
            seen.add(lowered)
            terms.append(lowered)

    for match in _ASCII_TERM_RE.findall(text or ""):
        add(match)
    for run in _CJK_RUN_RE.findall(text or ""):
        if len(run) == 1:
            add(run)
        else:
            for index in range(len(run) - 1):
                add(run[index : index + 2])
    return terms


def hash_obj(obj: Any) -> str:
    """Return a stable hash of a JSON-serialisable object."""
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def claim_digest(claim: Any) -> str:
    """Return a stable digest of the parts of a claim a re-check vouches for.

    A receipt binds a claim's *content*, not just its id: editing the statement (or
    swapping its evidence) after the check must make the receipt stale, so the digest
    covers the statement, kind, evidence refs, requirement ids and version.
    """
    data = claim.model_dump(mode="json") if hasattr(claim, "model_dump") else dict(claim or {})
    payload = {
        "claim_id": data.get("claim_id"),
        "statement": data.get("statement"),
        "kind": data.get("kind"),
        "evidence_refs": list(data.get("evidence_refs") or []),
        "requirement_ids": list(data.get("requirement_ids") or []),
        "version": int(data.get("version", 1) or 1),
    }
    return hash_obj(payload)


def safe_id(value: Optional[str], fallback: str = "") -> str:
    """Sanitise an identifier for use as a path component.

    Only ``[A-Za-z0-9._-]`` is accepted; everything else is replaced. The host never
    builds a path from user-provided or model-provided text without this function.
    """
    raw = str(value or fallback or "")
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", raw).strip(".-")[:120]
    if not cleaned or cleaned in {".", ".."}:
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", fallback or "unknown")[:120] or "unknown"
    return cleaned


def validate_id(value: str, what: str = "id") -> str:
    """Validate an opaque identifier (evidence id, request id, ...).

    Raises:
        InvalidArtifactId: if the value contains path separators, traversal or
            unexpected characters.
    """
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise InvalidArtifactId(f"Invalid {what}")
    return value


def redact_secrets(value: Any, key: str = "") -> Any:
    """Recursively remove credential-like values from audit payloads."""
    if isinstance(value, dict):
        return {
            k: ("<redacted>" if _SECRET_KEY_RE.search(str(k)) and v not in (None, "") else redact_secrets(v, str(k)))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item, key) for item in value]
    return value


@dataclass(frozen=True)
class Identity:
    """Host-bound identity of one agent invocation.

    Constructed by the runtime (``identity_for``); never parsed from model output or
    from tool arguments.
    """

    run_id: str
    agent_instance_id: str
    role: str
    task_id: Optional[str] = None
    parent_id: Optional[str] = None

    @property
    def restricted(self) -> bool:
        """Return True when this identity may only read its own task-scoped data."""
        return self.role == "researcher" and bool(self.task_id)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation for logs."""
        return asdict(self)

    def visible_private_agents(self) -> List[str]:
        """Return which ``agents/<id>/`` folders this identity may read."""
        if self.role in _OPERATOR_ONLY_ROLES:
            return []
        if self.role in _PRIVATE_ROLES:
            return [self.agent_instance_id]
        return []


@dataclass
class ToolResult:
    """Structured result of one tool execution."""

    status: str = "ok"
    content: str = ""
    evidence_ids: List[str] = field(default_factory=list)
    artifact_refs: List[str] = field(default_factory=list)
    error_class: str = ""
    error: str = ""
    usage: Optional[UsageRecord] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """Return True when the tool produced usable output."""
        return self.status == "ok"


# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #
def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# In-memory metadata (index, grants, usage) mirrored to disk
# --------------------------------------------------------------------------- #
@dataclass
class _Grant:
    evidence_id: str
    grantee_agent_id: str
    grantee_task_id: str
    granted_by: str
    granted_at: str = field(default_factory=utc_now)


class _RunMeta:
    """Thread-safe in-memory metadata for one run, mirrored to JSON files."""

    def __init__(self, run_dir: Path):
        self.lock = threading.RLock()
        self.run_dir = run_dir
        self.run_id = run_dir.name
        self.evidence: Dict[str, Dict[str, Any]] = {}
        self.grants: Dict[str, List[_Grant]] = {}
        self.usage: Dict[str, UsageRecord] = {}
        # Open budget reservations by reservation id (never persisted as "settled").
        self.reservations: Dict[str, Dict[str, Any]] = {}
        self.usage_unknown_calls = 0
        self.tool_calls = 0
        self.model_calls = 0
        self.published: Dict[str, Dict[str, Any]] = {}
        # Non-evidence tool results (repo/MCP output) that the model was promised it
        # could recover. Kept separate from the evidence index so an offloaded tool
        # log never masquerades as a citable source.
        self.tool_artifacts: Dict[str, Dict[str, Any]] = {}
        # The graph a run was started with; a resumed thread must stay on it.
        self.graph_name = ""
        self._load()

    # -- persistence -------------------------------------------------------
    def _index_path(self) -> Path:
        return self.run_dir / "evidence" / "index.json"

    def _grants_path(self) -> Path:
        return self.run_dir / "control" / "grant_ledger.json"

    def _usage_path(self) -> Path:
        return self.run_dir / "metrics" / "usage.json"

    def _artifacts_path(self) -> Path:
        return self.run_dir / "artifacts" / "index.json"

    def _load(self) -> None:
        try:
            index = json.loads(self._index_path().read_text(encoding="utf-8"))
            for entry in index.get("evidence", []):
                eid = entry.get("evidence_id")
                if eid:
                    self.evidence[eid] = entry
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - corrupted index must not crash the run
            LOGGER.warning("Could not read evidence index for run %s", self.run_id)
        try:
            index = json.loads(self._artifacts_path().read_text(encoding="utf-8"))
            for entry in index.get("artifacts", []):
                artifact_id = entry.get("artifact_id")
                if artifact_id:
                    self.tool_artifacts[artifact_id] = entry
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001 - corrupted index must not crash the run
            LOGGER.warning("Could not read tool artifact index for run %s", self.run_id)
        try:
            ledger = json.loads(self._grants_path().read_text(encoding="utf-8"))
            for entry in ledger.get("grants", []):
                grant = _Grant(**{k: entry[k] for k in ("evidence_id", "grantee_agent_id", "grantee_task_id", "granted_by", "granted_at")})
                self.grants.setdefault(grant.evidence_id, []).append(grant)
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            LOGGER.warning("Could not read grant ledger for run %s", self.run_id)
        try:
            usage = json.loads(self._usage_path().read_text(encoding="utf-8"))
            self.usage_unknown_calls = int(usage.get("unknown_usage_calls", 0))
            self.tool_calls = int(usage.get("tool_calls", 0))
            self.model_calls = int(usage.get("model_calls", 0))
            # Restore the per-attempt rows so role counts survive a restart. They are
            # keyed by attempt id; a run that cancelled an in-flight call also keeps
            # its open reservation, which counts as a started attempt.
            for row in usage.get("usage", []) or []:
                try:
                    record = UsageRecord(**row)
                except Exception:  # noqa: BLE001 - one bad row must not drop the ledger
                    continue
                self.usage[record.attempt_id or record.request_id or f"anon-{len(self.usage) + 1}"] = record
            for row in usage.get("reservations", []) or []:
                if isinstance(row, dict) and row.get("id"):
                    self.reservations[row["id"]] = dict(row)
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            LOGGER.warning("Could not read usage file for run %s", self.run_id)
        # Rebuild published refs from disk so restarts never re-publish blindly.
        published_dir = self.run_dir / "published" / "results"
        if published_dir.exists():
            for path in published_dir.glob("*/*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    key = f"{data.get('task_id')}@{data.get('version', 1)}"
                    self.published[key] = {
                        "ref": f"published://results/{data.get('task_id')}/{data.get('version', 1)}",
                        "published_by": data.get("_published_by", ""),
                    }
                except Exception:  # noqa: BLE001
                    continue

    def _persist_index(self) -> None:
        payload = {"run_id": self.run_id, "updated_at": utc_now(), "evidence": list(self.evidence.values())}
        _atomic_write(self._index_path(), json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    def _persist_artifacts(self) -> None:
        payload = {"run_id": self.run_id, "updated_at": utc_now(), "artifacts": list(self.tool_artifacts.values())}
        _atomic_write(self._artifacts_path(), json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    def _persist_grants(self) -> None:
        grants = [asdict(g) for items in self.grants.values() for g in items]
        _atomic_write(self._grants_path(), json.dumps({"run_id": self.run_id, "grants": grants}, indent=2, ensure_ascii=False))

    def persist_usage(self) -> None:
        with self.lock:
            payload = {
                "run_id": self.run_id,
                "updated_at": utc_now(),
                "model_calls": self.model_calls,
                "tool_calls": self.tool_calls,
                "unknown_usage_calls": self.usage_unknown_calls,
                "usage": [_jsonable(u) for u in self.usage.values()],
                "reservations": list(self.reservations.values()),
            }
            _atomic_write(self._usage_path(), json.dumps(payload, indent=2, ensure_ascii=False))


# Global registry so that all identities in one process share metadata even when a
# second store instance is created for the same root.
_META: Dict[str, _RunMeta] = {}
_META_LOCK = threading.Lock()


def reset_registry() -> None:
    """Forget in-memory run metadata (for tests and process restarts)."""
    with _META_LOCK:
        _META.clear()


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #
class ArtifactStore:
    """Filesystem-backed artifact store with capability-style access checks."""

    def __init__(self, root: Optional[str | os.PathLike] = None, run_id: Optional[str] = None):
        base = Path(root or os.environ.get("ARTIFACT_ROOT") or ".odr_runs").expanduser().resolve()
        self.root = base
        self.run_id = safe_id(run_id or f"run-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}", "run")
        self.run_dir = self.root / self.run_id
        with _META_LOCK:
            meta = _META.get(str(self.run_dir))
            if meta is None:
                meta = _RunMeta(self.run_dir)
                _META[str(self.run_dir)] = meta
            self._meta = meta
        # Where the verified price table came from (audit trail; None = built-in table).
        self.pricing_path: Optional[str] = None
        (self.run_dir / "evidence" / "objects").mkdir(parents=True, exist_ok=True)

    # -- convenience -------------------------------------------------------
    @property
    def metadata(self) -> _RunMeta:
        """Return the in-memory metadata (index/grants/usage) for this run."""
        return self._meta

    def _role_dir(self, identity: Identity) -> Optional[Path]:
        """Return the private folder an identity owns, if any."""
        if identity.role in _OPERATOR_ONLY_ROLES:
            return self.run_dir / "agents" / safe_id(identity.agent_instance_id)
        if identity.role in _PRIVATE_ROLES:
            return self.run_dir / "agents" / safe_id(identity.agent_instance_id)
        return None

    def write_run_manifest(self, manifest: Dict[str, Any]) -> Path:
        """Write ``manifest.json`` (credentials removed)."""
        payload = redact_secrets(_jsonable(manifest))
        path = self.run_dir / "manifest.json"
        _atomic_write(path, json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return path

    def write_inputs(self, messages_text: str, brief: Optional[ResearchBrief], request_meta: Dict[str, Any]) -> None:
        """Write operator-auditable inputs (never injected verbatim into requests)."""
        safe_meta = redact_secrets(_jsonable(request_meta))
        _atomic_write(self.run_dir / "inputs" / "user_messages.json", json.dumps({"content": messages_text, "metadata": safe_meta}, indent=2, ensure_ascii=False))
        if brief is not None:
            _atomic_write(self.run_dir / "inputs" / "research_brief.json", json.dumps(_jsonable(brief), indent=2, ensure_ascii=False))

    # -- evidence ----------------------------------------------------------
    def _safe_run_dir(self) -> Path:
        return self.run_dir.resolve()

    def _resolve_inside(self, candidate: Path) -> Path:
        """Resolve ``candidate`` and refuse to leave the run directory or follow links."""
        run_root = self._safe_run_dir()
        resolved = candidate.resolve()
        if not str(resolved).startswith(str(run_root) + os.sep):
            raise PermissionDenied("Artifact path escapes the run directory")
        # Refuse symlinks anywhere under the run directory.
        current = run_root
        for part in candidate.relative_to(self.run_dir).parts:
            current = current / part
            if current.is_symlink():
                raise PermissionDenied("Artifact path traverses a symbolic link")
        return resolved

    def _object_path(self, content_hash: str) -> Path:
        if not re.match(r"^[0-9a-f]{64}$", content_hash or ""):
            raise InvalidArtifactId("Invalid content hash")
        return self.run_dir / "evidence" / "objects" / f"{content_hash}.txt"

    def archive_document(self, identity: Identity, text: str, *, url: str = "", title: str = "") -> Dict[str, Any]:
        """Archive the original text of a source and return its descriptor.

        The object is addressed by content hash, so a changed page is a new object
        and old evidence keeps pointing at the old snapshot.

        Returns:
            A dict with ``content_hash``, ``chars`` and ``url``.
        """
        content_hash = sha256_text(text)
        path = self._object_path(content_hash)
        if not path.exists():
            _atomic_write(path, text)
        return {"content_hash": content_hash, "chars": len(text), "url": url, "title": title}

    def archive_tool_result(
        self,
        identity: Identity,
        text: str,
        *,
        title: str = "",
        call_id: str = "",
    ) -> Dict[str, Any]:
        """Archive a non-evidence tool result and make it recoverable by id.

        Repository/MCP output is not a citable source, but it is still material the
        model may need after a trim. It gets its own owner-scoped index entry so
        ``request_context`` can read it back without polluting the evidence manifest.
        """
        doc = self.archive_document(identity, text, title=title)
        artifact_id = f"artifact://tool-results/{doc['content_hash']}"
        owner = {"task_id": identity.task_id or "", "agent_instance_id": identity.agent_instance_id}
        with self._meta.lock:
            entry = self._meta.tool_artifacts.get(artifact_id)
            if entry is None:
                entry = {
                    "artifact_id": artifact_id,
                    "content_hash": doc["content_hash"],
                    "chars": doc["chars"],
                    "title": title,
                    "url": "",
                    "call_id": call_id,
                    "created_at": utc_now(),
                    "owners": [owner],
                }
                self._meta.tool_artifacts[artifact_id] = entry
            elif owner not in entry.setdefault("owners", []):
                entry["owners"].append(owner)
            self._meta._persist_artifacts()
        self._append_agent_event(
            identity,
            {"type": "artifact_created", "artifact_id": artifact_id, "chars": doc["chars"], "call_id": call_id},
        )
        return dict(entry)

    def _artifact_entry(self, artifact_id: str) -> Dict[str, Any]:
        if not isinstance(artifact_id, str) or not re.match(r"^artifact://tool-results/[0-9a-f]{16,64}$", artifact_id):
            raise InvalidArtifactId("Invalid tool artifact id")
        with self._meta.lock:
            entry = self._meta.tool_artifacts.get(artifact_id)
        if entry is None:
            raise MissingEvidence(f"Tool artifact {artifact_id} does not exist")
        return entry

    @staticmethod
    def _check_artifact_access(identity: Identity, entry: Dict[str, Any]) -> None:
        owners = entry.get("owners") or []
        if any(
            owner.get("agent_instance_id") == identity.agent_instance_id
            or (identity.task_id and owner.get("task_id") == identity.task_id)
            for owner in owners
        ):
            return
        raise PermissionDenied("Tool artifact is not readable by this identity")

    def can_read_artifact(self, identity: Identity, artifact_id: str) -> bool:
        """Return whether ``identity`` owns this archived tool result (no event)."""
        entry = self._meta.tool_artifacts.get(artifact_id)
        if entry is None:
            return False
        try:
            self._check_artifact_access(identity, entry)
        except PermissionDenied:
            return False
        return True

    def read_artifact(
        self,
        identity: Identity,
        artifact_id: str,
        *,
        query: str = "",
        max_chars: int = 900,
    ) -> Dict[str, Any]:
        """Read a bounded, owner-checked window of one archived tool result."""
        entry = self._artifact_entry(artifact_id)
        try:
            self._check_artifact_access(identity, entry)
        except PermissionDenied:
            self._append_agent_event(identity, {"type": "artifact_read_denied", "artifact_id": artifact_id})
            raise
        path = self._resolve_inside(self._object_path(entry["content_hash"]))
        body = path.read_text(encoding="utf-8")
        cap = max(200, int(max_chars))
        text = self._best_query_window(body, query, cap) if query else ""
        if not text:
            text = body[:cap]
        self._append_agent_event(identity, {"type": "read_artifact", "artifact_id": artifact_id})
        return {
            "status": "ok",
            "artifact_id": artifact_id,
            "title": entry.get("title", ""),
            "url": entry.get("url", ""),
            "retrieved_at": entry.get("created_at", ""),
            "content_hash": entry.get("content_hash", ""),
            "excerpt": text,
            "truncated": len(text) < len(body),
        }

    def search_artifacts(
        self,
        identity: Identity,
        terms: Iterable[str],
        *,
        limit: int = 8,
        max_scan: int = 200,
    ) -> List[Dict[str, Any]]:
        """Rank this identity's own archived tool results by keyword overlap."""
        wanted = [str(term).strip().lower() for term in terms if str(term).strip()]
        if not wanted:
            return []
        ranked: List[Dict[str, Any]] = []
        scanned = 0
        with self._meta.lock:
            entries = list(self._meta.tool_artifacts.items())
        for artifact_id, entry in entries:
            if not self.can_read_artifact(identity, artifact_id):
                continue
            if scanned >= max_scan:
                break
            scanned += 1
            metadata = " ".join(str(entry.get(key) or "") for key in ("title", "url", "call_id")).lower()
            score = sum(1 for term in wanted if term in metadata)
            matched_in = "metadata" if score else ""
            if not score:
                try:
                    path = self._resolve_inside(self._object_path(entry["content_hash"]))
                    body = path.read_text(encoding="utf-8").lower()
                except Exception:  # noqa: BLE001 - a lost object is a missing source, not a crash
                    body = ""
                score = sum(1 for term in wanted if term in body)
                matched_in = "full_text" if score else ""
            if score:
                ranked.append(
                    {
                        "artifact_id": artifact_id,
                        "score": score,
                        "title": entry.get("title", ""),
                        "url": entry.get("url", ""),
                        "matched_in": matched_in,
                    }
                )
        ranked.sort(key=lambda item: (-item["score"], item["artifact_id"]))
        return ranked[: max(1, int(limit))]

    def add_evidence(
        self,
        identity: Identity,
        *,
        content: str,
        source_url: str,
        title: str = "",
        excerpt: str = "",
        locator: Optional[Dict[str, Any]] = None,
        origin: str = "tool_result",
        trust_label: str = "unverified",
        is_counter_evidence: bool = False,
        public: bool = False,
        attempt: int = 0,
    ) -> EvidenceRecord:
        """Store one evidence record and index it for the current identity.

        The same document delivered again (same url and content hash, typically the
        same page returned by two queries) reuses the existing evidence id instead of
        becoming a second "source" with a different excerpt. A second task gets a read
        grant, so reuse does not silently hide the document from it.
        """
        doc = self.archive_document(identity, content, url=source_url, title=title)
        canonical = _canonical_source_url(source_url)
        reused_id = ""
        with self._meta.lock:
            for existing_id, existing in self._meta.evidence.items():
                if existing.get("content_hash") != doc["content_hash"]:
                    continue
                if _canonical_source_url(existing.get("original_url", "")) != canonical:
                    continue
                reused_id = existing_id
                reused = dict(existing)
                task_attempts = existing.setdefault("_task_attempts", {})
                attempts = task_attempts.setdefault(identity.task_id or "", [])
                if attempt not in attempts:
                    attempts.append(attempt)
                self._meta._persist_index()
                break
        if reused_id:
            if not reused.get("public") and (reused.get("_owner_task") or "") != (identity.task_id or ""):
                self.grant_evidence(
                    reused_id,
                    grantee_agent_id=identity.agent_instance_id,
                    grantee_task_id=identity.task_id or "",
                    granted_by="archive_reuse",
                )
            self._append_agent_event(identity, {"type": "evidence_reused", "evidence_id": reused_id, "url": source_url})
            return EvidenceRecord(**{key: value for key, value in reused.items() if not key.startswith("_")})
        exact = excerpt or content[:600]
        if exact and exact not in content:
            # The excerpt must be locatable in the archived object; otherwise keep the
            # head of the document as the exact excerpt and record nothing as a quote.
            exact = content[:600]
        with self._meta.lock:
            seq = len(self._meta.evidence) + 1
            evidence_id = f"E{seq}"
            record = EvidenceRecord(
                evidence_id=evidence_id,
                source_id=hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:16] if source_url else doc["content_hash"][:16],
                task_id=identity.task_id or "",
                agent_instance_id=identity.agent_instance_id,
                run_id=self.run_id,
                original_url=source_url,
                canonical_url=source_url,
                title=title,
                retrieved_at=utc_now(),
                content_hash=doc["content_hash"],
                locator={"kind": "document", "note": "excerpt locatable in archived object", **(locator or {})},
                exact_excerpt=exact,
                origin=origin,
                trust_label=trust_label,
                public=public,
                is_counter_evidence=is_counter_evidence,
            )
            entry = record.model_dump(mode="json")
            entry["_owner_task"] = identity.task_id or ""
            entry["_owner_attempt"] = attempt
            entry["_task_attempts"] = {identity.task_id or "": [attempt]}
            entry["_owner_agent"] = identity.agent_instance_id
            entry["_registered_by"] = identity.agent_instance_id
            self._meta.evidence[evidence_id] = entry
            self._meta._persist_index()
        _atomic_write(
            self.run_dir / "evidence" / "records" / f"{evidence_id}.json",
            json.dumps({k: v for k, v in entry.items() if not k.startswith("_")}, indent=2, ensure_ascii=False, default=str),
        )
        self._append_agent_event(identity, {"type": "evidence_created", "evidence_id": evidence_id, "url": source_url})
        return record

    def bulk_archive(
        self,
        identity: Identity,
        mapping: Iterable[Dict[str, Any]],
        *,
        attempt: int = 0,
    ) -> List[EvidenceRecord]:
        """Archive search results ``[{"url","title","content"}]`` in order."""
        records: List[EvidenceRecord] = []
        for item in mapping:
            content = item.get("content") or ""
            if not content.strip():
                continue
            records.append(
                self.add_evidence(
                    identity,
                    content=content,
                    source_url=item.get("url", ""),
                    title=item.get("title", ""),
                    excerpt=item.get("excerpt", ""),
                    origin=item.get("origin", "tool_result"),
                    trust_label=item.get("trust_label", "unverified"),
                    is_counter_evidence=bool(item.get("is_counter_evidence", False)),
                    attempt=attempt,
                )
            )
        return records

    def _evidence_entry(self, evidence_id: str) -> Dict[str, Any]:
        validate_id(evidence_id, "evidence id")
        with self._meta.lock:
            entry = self._meta.evidence.get(evidence_id)
        if entry is None:
            raise MissingEvidence(f"Evidence {evidence_id} does not exist")
        return entry

    def _check_evidence_access(self, identity: Identity, entry: Dict[str, Any]) -> None:
        # Unknown id and forbidden id raise the same error: no enumeration oracle.
        if entry.get("public"):
            return
        owner_task = entry.get("_owner_task") or entry.get("task_id") or ""
        if identity.restricted:
            if owner_task and owner_task == identity.task_id:
                return
            with self._meta.lock:
                grants = list(self._meta.grants.get(entry["evidence_id"], []))
            if any(g.grantee_agent_id == identity.agent_instance_id and g.grantee_task_id == identity.task_id for g in grants):
                return
            raise PermissionDenied("Evidence is not readable by this identity")
        if identity.role in {"supervisor", "writer"}:
            # The supervisor and the writer only see published/approved evidence.
            if entry.get("_published"):
                return
            raise PermissionDenied("Evidence is not published to this identity")
        if identity.role == "reviewer":
            return
        raise PermissionDenied("Evidence is not readable by this identity")

    def get_evidence(self, identity: Identity, evidence_id: str) -> EvidenceRecord:
        """Return one evidence record if the identity may read it.

        Raises:
            PermissionDenied: for forbidden and unknown ids alike.
            MissingEvidence: never for ids the identity cannot see.
        """
        entry = self._evidence_entry(evidence_id)
        try:
            self._check_evidence_access(identity, entry)
        except PermissionDenied:
            self._append_agent_event(identity, {"type": "read_denied", "evidence_id": evidence_id})
            raise
        self._append_agent_event(identity, {"type": "read_evidence", "evidence_id": evidence_id})
        return EvidenceRecord(**{k: v for k, v in entry.items() if not k.startswith("_")})

    def read_evidence_text(self, identity: Identity, evidence_id: str) -> str:
        """Return the archived original document behind an evidence id."""
        record = self.get_evidence(identity, evidence_id)
        path = self._resolve_inside(self._object_path(record.content_hash))
        body = path.read_text(encoding="utf-8")
        if sha256_text(body) != record.content_hash:
            raise ArtifactError(f"archived object for {evidence_id} no longer matches its content_hash")
        return body

    def read_excerpt(
        self,
        identity: Identity,
        evidence_id: str,
        *,
        selector: str = "excerpt",
        max_tokens: int = 400,
        query: str = "",
    ) -> Dict[str, Any]:
        """Read a bounded, permission-checked excerpt of one evidence record.

        Args:
            selector: ``excerpt`` (stored excerpt), ``full`` (whole document up to
                ``max_tokens``) or ``query`` (text window around a query string).
            max_tokens: hard cap on returned characters (``max_tokens * 4``).
        """
        record = self.get_evidence(identity, evidence_id)
        char_cap = max(200, int(max_tokens) * 4)
        if selector == "full":
            body = self.read_evidence_text(identity, evidence_id)
            text = body[:char_cap]
        elif selector == "query":
            body = self.read_evidence_text(identity, evidence_id)
            text = self._best_query_window(body, query, char_cap) or record.exact_excerpt[:char_cap]
        else:
            text = record.exact_excerpt[:char_cap]
        return {
            "status": "ok",
            "evidence_id": record.evidence_id,
            "url": record.original_url,
            "title": record.title,
            "retrieved_at": record.retrieved_at,
            "content_hash": record.content_hash,
            "excerpt": text,
            "truncated": len(text) >= char_cap,
        }

    @staticmethod
    def _best_query_window(body: str, query: str, char_cap: int) -> str:
        """Return the passage of ``body`` that best matches the query terms.

        A natural-language need ("the v2 retention exception") rarely appears verbatim,
        so an exact substring search used to fall back to the document head - exactly
        the place where the exception is *not*. The body is scored in overlapping
        windows instead; the best window is returned verbatim.
        """
        terms = search_terms(query or "")
        if not terms or not body:
            return ""
        lowered = body.lower()
        window = max(200, int(char_cap))
        stride = max(100, window // 2)
        best_score = 0
        best_start = 0
        for start in range(0, max(1, len(body) - stride), stride):
            chunk = lowered[start : start + window]
            score = sum(1 for term in terms if term in chunk)
            if score > best_score:
                best_score = score
                best_start = start
        if not best_score:
            return ""
        return body[best_start : best_start + char_cap]

    def grant_evidence(self, evidence_id: str, *, grantee_agent_id: str, grantee_task_id: str, granted_by: str) -> None:
        """Grant read access of one evidence item to one task (host policy only)."""
        entry = self._evidence_entry(evidence_id)
        validate_id(grantee_agent_id, "grantee id")
        if entry.get("public"):
            return
        with self._meta.lock:
            grants = self._meta.grants.setdefault(evidence_id, [])
            if any(g.grantee_agent_id == grantee_agent_id and g.grantee_task_id == grantee_task_id for g in grants):
                return
            grants.append(_Grant(evidence_id=evidence_id, grantee_agent_id=grantee_agent_id, grantee_task_id=grantee_task_id, granted_by=granted_by))
            self._meta._persist_grants()
        self._append_event("control", {"type": "grant", "evidence_id": evidence_id, "to_agent": grantee_agent_id, "to_task": grantee_task_id, "by": granted_by})

    def mark_public(self, identity: Identity, evidence_ids: Iterable[str], *, reason: str = "") -> None:
        """Mark evidence as published so the supervisor/writer may read it."""
        with self._meta.lock:
            for eid in evidence_ids:
                entry = self._meta.evidence.get(eid)
                if entry is None:
                    continue
                entry["public"] = True
                entry["_published"] = True
                entry["_published_reason"] = reason
            self._meta._persist_index()
        self._append_agent_event(identity, {"type": "evidence_published", "evidence_ids": list(evidence_ids), "reason": reason})

    def evidence_for_task(self, task_id: str, attempt: Optional[int] = None) -> List[str]:
        """Return sources delivered to this task, including reused snapshots."""
        with self._meta.lock:
            return [
                eid
                for eid, entry in self._meta.evidence.items()
                if (
                    entry.get("_owner_task") == task_id
                    and (attempt is None or entry.get("_owner_attempt") == attempt)
                ) or (
                    task_id in entry.get("_task_attempts", {})
                    and (attempt is None or attempt in entry["_task_attempts"][task_id])
                )
            ]

    def counter_evidence_for(self, identities: Iterable[Identity]) -> List[str]:
        """Return ids of counter-evidence archived by the given invocations."""
        agents = {i.agent_instance_id for i in identities}
        with self._meta.lock:
            return [eid for eid, entry in self._meta.evidence.items() if entry.get("is_counter_evidence") and entry.get("_owner_agent") in agents]

    def evidence_index(self) -> List[Dict[str, Any]]:
        """Return the evidence index entries of this run (host-internal, ids only)."""
        with self._meta.lock:
            return [{"evidence_id": eid, "title": entry.get("title", ""), "url": entry.get("original_url", "")} for eid, entry in self._meta.evidence.items()]

    def read_evidence_for_review(
        self,
        evidence_ids: Iterable[str],
        limit: Optional[int] = None,
        focus: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Reviewer-only bulk excerpt reader (scope is the packet's evidence list).

        ``limit=None`` reads *every* id in the packet. The old default of 24 silently
        turned archived evidence into "missing" once a run had more than 24 sources,
        which made the reviewer report a false gap and could block a correct report.
        A caller that really wants a bounded read passes an explicit limit and must
        treat the remainder as "not shown" rather than "not archived".

        ``focus`` maps an evidence id to the claim statement/quote the review must
        check. When the stored opening excerpt does not contain that text, a window
        around it is added as ``relevant_excerpt`` with its offsets, so a fact in the
        middle of a long page is visible instead of being called unsupported.
        """
        from open_deep_research.utils import select_relevant_window

        ids = list(dict.fromkeys(evidence_ids))
        if limit is not None:
            ids = ids[: int(limit)]
        out: List[Dict[str, Any]] = []
        for eid in ids:
            entry = self._meta.evidence.get(eid)
            if not entry:
                continue
            excerpt = entry.get("exact_excerpt", "") or ""
            shown = excerpt[:1200]
            row: Dict[str, Any] = {
                "evidence_id": eid,
                "url": entry.get("original_url", ""),
                "title": entry.get("title", ""),
                "retrieved_at": entry.get("retrieved_at", ""),
                "content_hash": entry.get("content_hash", ""),
                "exact_excerpt": shown,
                "excerpt_truncated": len(excerpt) > len(shown),
                "is_counter_evidence": bool(entry.get("is_counter_evidence")),
            }
            needle = str((focus or {}).get(eid) or "").strip()
            if needle and needle not in shown:
                try:
                    path = self._resolve_inside(self._object_path(entry["content_hash"]))
                    body = path.read_text(encoding="utf-8")
                except Exception:  # noqa: BLE001 - an unreadable object is not a support claim
                    body = ""
                if body:
                    window, start, end = select_relevant_window(body, query=needle, limit=1200)
                    if window and window not in shown:
                        row["relevant_excerpt"] = window
                        row["relevant_span"] = {"content_hash": entry["content_hash"], "start": start, "end": end}
                        row["relevant_match"] = bool(window.strip())
                        row["verbatim_match"] = needle in body
            out.append(row)
        return out

    def can_read_evidence(self, identity: Identity, evidence_id: str) -> bool:
        """Return whether ``identity`` may read this evidence id (no event written).

        Used by the recovery search to filter candidates *before* reading them, so a
        denied id is not disclosed and the index is not enumerated to the model.
        """
        entry = self._meta.evidence.get(evidence_id)
        if entry is None:
            return False
        try:
            self._check_evidence_access(identity, entry)
        except PermissionDenied:
            return False
        return True

    def search_evidence(
        self,
        identity: Identity,
        terms: Iterable[str],
        *,
        limit: int = 8,
        scan_full_text: bool = True,
        max_scan: int = 200,
    ) -> List[Dict[str, Any]]:
        """Rank this identity's readable evidence by keyword overlap.

        The metadata (title, excerpt, url) is searched first. When that finds nothing,
        the archived objects are scanned (bounded by ``max_scan``) so a term that only
        appears in the omitted middle of a long document can still be recovered; the
        excerpt alone must not become the whole searchable world.

        Returns:
            ``[{"evidence_id", "score", "title", "url", "matched_in"}]``, best first.
        """
        wanted = [str(term).strip().lower() for term in terms if str(term).strip()]
        if not wanted:
            return []
        ranked: List[Dict[str, Any]] = []
        with self._meta.lock:
            entries = list(self._meta.evidence.items())
        scanned = 0
        for eid, entry in entries:
            if not self.can_read_evidence(identity, eid):
                continue
            if scanned >= max_scan:
                break
            scanned += 1
            haystack = " ".join(
                str(entry.get(key) or "") for key in ("title", "original_url", "exact_excerpt")
            ).lower()
            score = sum(1 for term in wanted if term in haystack)
            matched_in = "metadata" if score else ""
            if not score and scan_full_text:
                try:
                    record = EvidenceRecord(**{k: v for k, v in entry.items() if not k.startswith("_")})
                    path = self._resolve_inside(self._object_path(record.content_hash))
                    body = path.read_text(encoding="utf-8").lower()
                except Exception:  # noqa: BLE001 - a lost object is a missing source, not a crash
                    body = ""
                score = sum(1 for term in wanted if term in body)
                matched_in = "full_text" if score else ""
            if score:
                ranked.append(
                    {
                        "evidence_id": eid,
                        "score": score,
                        "title": entry.get("title", ""),
                        "url": entry.get("original_url", ""),
                        "matched_in": matched_in,
                    }
                )
        ranked.sort(key=lambda item: (-item["score"], item["evidence_id"]))
        return ranked[: max(1, int(limit))]

    # -- published results -------------------------------------------------
    def publish_result(self, identity: Identity, task_result: TaskResult, expected_version: Optional[int] = None) -> Dict[str, Any]:
        """Publish one task result idempotently.

        Publication means "may be distributed", not "verified".

        Raises:
            PublishRejected: on version conflicts or when the publishing identity does
                not own the task.
        """
        if identity.role != "researcher":
            raise PermissionDenied("Only a researcher may publish its own task result")
        if identity.task_id and task_result.task_id != identity.task_id:
            raise PermissionDenied("A researcher may only publish its own task")
        if task_result.status.value == "failed" and not task_result.summary:
            raise PublishRejected("A failed task without any result cannot be published")
        key = f"{task_result.task_id}@{task_result.version}"
        with self._meta.lock:
            existing = self._meta.published.get(key)
            if existing:
                return dict(existing)
            previous = [
                v for k, v in self._meta.published.items() if k.startswith(f"{task_result.task_id}@")
            ]
            if expected_version is not None and previous and expected_version not in {
                int(k.split("@", 1)[1]) for k in self._meta.published if k.startswith(f"{task_result.task_id}@")
            }:
                raise PublishRejected(f"Stale expected_version {expected_version}")
            path = self.run_dir / "published" / "results" / safe_id(task_result.task_id) / f"{task_result.version}.json"
            _atomic_write(path, json.dumps({**_jsonable(task_result), "_published_by": identity.agent_instance_id, "_published_at": utc_now()}, indent=2, ensure_ascii=False))
            ref = {
                "task_id": task_result.task_id,
                "version": task_result.version,
                "ref": f"published://results/{task_result.task_id}/{task_result.version}",
                "published_by": identity.agent_instance_id,
            }
            self._meta.published[key] = ref
        self._append_event("control", {"type": "publish", "task_id": task_result.task_id, "version": task_result.version, "by": identity.agent_instance_id})
        return dict(ref)

    def get_published(self, task_id: str, version: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Read one published result (host-internal; callers must re-check scope)."""
        safe_task = safe_id(task_id)
        folder = self.run_dir / "published" / "results" / safe_task
        if not folder.exists():
            return None
        candidates = sorted(folder.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        if not candidates:
            return None
        chosen = None
        for path in candidates:
            if version is not None and path.stem == str(version):
                chosen = path
        if chosen is None:
            chosen = candidates[-1]
        try:
            return json.loads(chosen.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def published_summaries(self) -> List[Dict[str, Any]]:
        """Return short summaries of all published results (supervisor projection)."""
        out: List[Dict[str, Any]] = []
        folder = self.run_dir / "published" / "results"
        if not folder.exists():
            return out
        for task_dir in sorted(folder.iterdir()):
            if not task_dir.is_dir():
                continue
            files = sorted(task_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
            if not files:
                continue
            try:
                data = json.loads(files[-1].read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            out.append(
                {
                    "task_id": data.get("task_id"),
                    "version": data.get("version", 1),
                    "status": data.get("status", "complete"),
                    "summary": (data.get("summary") or "")[:600],
                    "evidence_refs": data.get("evidence_refs", [])[:20],
                    "gaps": data.get("gaps", [])[:5],
                    "conflicts": data.get("conflicts", [])[:5],
                }
            )
        return out

    def write_claims(self, identity: Identity, claims: Iterable[ClaimRecord]) -> None:
        """Persist claims produced by a finding extractor to ``published/claims``."""
        for claim in claims:
            self.write_claim_version(claim, owner_task=identity.task_id or "", owner_agent=identity.agent_instance_id)

    def write_claim_version(self, claim: ClaimRecord, *, owner_task: str, owner_agent: str) -> None:
        """Write one claim version, keeping the task/agent that first produced it.

        A host-applied revision is written by the supervisor identity, but the claim
        still belongs to the researcher task whose evidence it cites: the owner fields
        are carried over so readiness review can still see it.
        """
        path = self.run_dir / "published" / "claims" / safe_id(claim.claim_id) / f"{claim.version}.json"
        _atomic_write(
            path,
            json.dumps(
                {
                    **_jsonable(claim),
                    "_owner_task": owner_task,
                    "_owner_agent": owner_agent,
                    "_revision_written_by": "host",
                },
                indent=2,
                ensure_ascii=False,
            ),
        )
        self._append_event(
            "control",
            {
                "type": "claim_version",
                "claim_id": claim.claim_id,
                "version": claim.version,
                "status": getattr(claim.status, "value", claim.status),
                "supersedes": claim.supersedes_version,
            },
        )

    def bind_quote(self, evidence_id: str, quote: str) -> Dict[str, Any]:
        """Locate a verbatim quote in an archived object (host validation only).

        This is not a read interface for a model: it returns offsets and the content
        hash, never the document. Callers must first establish that the evidence id is
        in the run's scope (a claim's existing reference or an id published to it).
        """
        if not quote or len(quote) > 800:
            raise ArtifactError("quote must be 1-800 characters")
        entry = self._evidence_entry(evidence_id)
        path = self._resolve_inside(self._object_path(entry["content_hash"]))
        try:
            body = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ArtifactError(f"archived object for {evidence_id} is unreadable: {exc}") from exc
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != entry["content_hash"]:
            raise ArtifactError(f"archived object for {evidence_id} no longer matches its content_hash")
        start = body.find(quote)
        if start < 0:
            raise ArtifactError(f"quote is not verbatim in the archived object of {evidence_id}")
        return {
            "evidence_id": evidence_id,
            "content_hash": entry["content_hash"],
            "start": start,
            "end": start + len(quote),
            "quote": quote,
        }

    def all_claims(self) -> List[Dict[str, Any]]:
        """Return every claim written by finding extractors (host-internal)."""
        out: List[Dict[str, Any]] = []
        folder = self.run_dir / "published" / "claims"
        if not folder.exists():
            return out
        for claim_dir in sorted(folder.iterdir()):
            if not claim_dir.is_dir():
                continue
            files = sorted(claim_dir.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
            if not files:
                continue
            try:
                out.append(json.loads(files[-1].read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                continue
        return out

    def active_claims(self) -> List[Dict[str, Any]]:
        """Return only the claims that are still active (latest version, not withdrawn).

        Writer, coverage and report review must read this set; ``all_claims`` stays the
        audit view so a withdrawn claim and its counter-evidence remain readable.
        """
        return [
            claim
            for claim in self.all_claims()
            if str(claim.get("status") or "active") == "active"
        ]

    def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Return the latest stored version of one claim id, or None."""
        safe = safe_id(claim_id)
        folder = self.run_dir / "published" / "claims" / safe
        if not folder.exists():
            return None
        files = sorted(folder.glob("*.json"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
        if not files:
            return None
        try:
            return json.loads(files[-1].read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    # -- books / ledgers ---------------------------------------------------
    def append_ledger(self, name: str, entry: Dict[str, Any]) -> None:
        """Append one entry to a host-written control ledger file."""
        path = self.run_dir / "control" / f"{safe_id(name)}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"at": utc_now(), **redact_secrets(entry)}, ensure_ascii=False, default=str)
        with self._meta.lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def append_event(self, entry: Dict[str, Any]) -> None:
        """Append one entry to the global ``control/events.jsonl``."""
        self.append_ledger("events", entry)

    def reserve_search_queries(self, count: int, maximum: int, *, execution_id: str = "") -> str:
        """Reserve query units before transport; failed calls still consume allowance."""
        path = self.run_dir / "control" / "search_usage.jsonl"
        with self._meta.lock:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
            used = sum(int(row.get("query_count", 0)) for row in rows if row.get("phase") == "reserved")
            if count < 0 or used + count > maximum:
                raise ArtifactError(f"search query budget exhausted: {used}+{count} exceeds {maximum}")
            request_id = "search-" + uuid.uuid4().hex[:12]
            self.append_ledger("search_usage", {"request_id": request_id, "phase": "reserved", "query_count": count,
                                               "maximum": maximum, "execution_id": execution_id, "endpoint": "tavily.search"})
            return request_id

    def cached_extraction(self, identity: Identity, evidence_id: str, key: str) -> Optional[str]:
        """Read a successful extraction only after checking source access."""
        self.get_evidence(identity, evidence_id)
        path = self.run_dir / "evidence" / "extractions" / f"{safe_id(key)}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str(payload["content"]) if payload.get("evidence_id") == evidence_id else None

    def reserve_page_fetch(self, url: str, maximum: int) -> Optional[str]:
        """Bound optional direct page reads, including failures, across the run."""
        with self._meta.lock:
            path = self.run_dir / "control" / "page_fetch_usage.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            if sum(row.get("phase") == "reserved" for row in rows) >= maximum:
                return None
            request_id = "page-" + uuid.uuid4().hex[:12]
            self.append_ledger("page_fetch_usage", {"request_id": request_id, "url": url, "phase": "reserved", "maximum": maximum, "transport": "direct_https"})
            return request_id

    def cache_extraction(self, evidence_id: str, key: str, content: str) -> None:
        """Keep successful extraction output for an identical source/model/request."""
        path = self.run_dir / "evidence" / "extractions" / f"{safe_id(key)}.json"
        _atomic_write(path, json.dumps({"evidence_id": evidence_id, "content": content}, ensure_ascii=False))

    def claim_review_findings(self) -> List[Dict[str, Any]]:
        """Fold the host's persistent claim findings without deleting their history."""
        path = self.run_dir / "control" / "claim_review_findings.jsonl"
        if not path.exists():
            return []
        entries: Dict[str, Dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = str(row["finding_id"])
            entries[key] = {**entries.get(key, {}), **row}
        return list(entries.values())

    # -- correction (revalidation) ledger ----------------------------------
    def _revalidation_path(self) -> Path:
        return self.run_dir / "control" / "revalidation.jsonl"

    def revalidation_events(self) -> List[Dict[str, Any]]:
        """Return the folded correction events, newest state per event id."""
        path = self._revalidation_path()
        events: Dict[str, Dict[str, Any]] = {}
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        for line in lines:
            try:
                entry = json.loads(line)
            except Exception:  # noqa: BLE001 - one corrupt line must not hide the rest
                continue
            event_id = entry.get("event_id")
            if event_id:
                events[event_id] = {**events.get(event_id, {}), **entry}
        return sorted(events.values(), key=lambda item: str(item.get("created_at", "")))

    def open_revalidations(self) -> List[Dict[str, Any]]:
        """Return correction events that no later result has resolved yet."""
        return [event for event in self.revalidation_events() if event.get("status") == "open"]

    def record_revalidation(self, event: RevalidationEvent | Dict[str, Any]) -> Dict[str, Any]:
        """Append one correction event (host-owned; models cannot call this)."""
        payload = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event)
        payload.setdefault("created_at", utc_now())
        self.append_ledger("revalidation", payload)
        self.append_event({"type": "revalidation_opened", "event_id": payload.get("event_id"), "task_id": payload.get("task_id"), "requirements": payload.get("affected_requirement_ids", [])})
        return payload

    def resolve_revalidation(self, event_id: str, *, resolved_by: str, reason: str = "") -> None:
        """Mark one correction event resolved (keeps the full history on disk)."""
        existing = next((event for event in self.revalidation_events() if event.get("event_id") == event_id), None)
        if existing is None:
            return
        self.append_ledger(
            "revalidation",
            {
                "event_id": event_id,
                "status": "resolved",
                "resolved_at": utc_now(),
                "resolved_by": resolved_by,
                "resolution": reason,
                "task_id": existing.get("task_id", ""),
                "affected_requirement_ids": existing.get("affected_requirement_ids", []),
            },
        )
        self.append_event({"type": "revalidation_resolved", "event_id": event_id, "by": resolved_by, "reason": reason})

    # -- revalidation receipts ---------------------------------------------
    def evidence_snapshot(self, evidence_ids: Iterable[str]) -> List[Dict[str, Any]]:
        """Return the current identity of each evidence record a receipt vouches for."""
        snapshot: List[Dict[str, Any]] = []
        with self._meta.lock:
            for evidence_id in evidence_ids or []:
                entry = self._meta.evidence.get(str(evidence_id))
                if not entry:
                    continue
                snapshot.append(
                    {
                        "evidence_id": str(evidence_id),
                        "content_hash": entry.get("content_hash", ""),
                        "document_version": int(entry.get("document_version", 1) or 1),
                    }
                )
        return snapshot

    def revalidation_receipts(self, event_id: str = "") -> List[Dict[str, Any]]:
        """Return the receipts issued for one correction event (or for the whole run)."""
        path = self.run_dir / "control" / "revalidation_receipts.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [row for row in rows if not event_id or str(row.get("event_id") or "") == str(event_id)]

    def record_revalidation_receipt(self, receipt: Any) -> Dict[str, Any]:
        """Append one independent receipt to the append-only receipt ledger."""
        payload = receipt.model_dump(mode="json") if hasattr(receipt, "model_dump") else dict(receipt or {})
        payload.setdefault("created_at", utc_now())
        payload["checker_kind"] = payload.get("checker_kind") or "independent_reviewer"
        self.append_ledger("revalidation_receipts", payload)
        self.append_event(
            {
                "type": "revalidation_receipt_issued",
                "event_id": payload.get("event_id"),
                "receipt_id": payload.get("receipt_id"),
                "review_id": payload.get("review_id"),
                "claims": payload.get("claim_ids", []),
            }
        )
        return payload

    def verify_revalidation_receipt(self, receipt: Any, event: Any, claims: Any) -> Tuple[bool, str]:
        """Re-check one receipt against the *current* event, claims and evidence.

        Returns ``(ok, reason)``. A stale hash (the claim was edited, the archived
        object replaced) or an unchecked claim keeps the correction event open; the
        caller records the reason so a rejected receipt is auditable.
        """
        row = receipt.model_dump(mode="json") if hasattr(receipt, "model_dump") else dict(receipt or {})
        event_row = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event or {})
        if str(row.get("event_id") or "") != str(event_row.get("event_id") or ""):
            return False, "event_mismatch"
        if row.get("checker_kind") != "independent_reviewer" or not str(row.get("checker_identity") or "").strip():
            return False, "not_an_independent_reviewer"
        current: Dict[str, Any] = {}
        if isinstance(claims, dict):
            current = {str(key): value for key, value in claims.items()}
        else:
            for claim in claims or []:
                data = claim.model_dump(mode="json") if hasattr(claim, "model_dump") else dict(claim or {})
                current[str(data.get("claim_id"))] = data
        claim_ids = [str(cid) for cid in (row.get("claim_ids") or [])]
        if not claim_ids:
            return False, "no_claim_checked"
        for claim_id in claim_ids:
            if claim_id not in current:
                return False, f"claim_missing:{claim_id}"
            expected = (row.get("claim_hashes") or {}).get(claim_id)
            if not expected or claim_digest(current[claim_id]) != expected:
                return False, f"stale_claim_hash:{claim_id}"
        for evidence_id in row.get("evidence_ids") or []:
            entry = self._meta.evidence.get(str(evidence_id))
            if not entry:
                return False, f"evidence_missing:{evidence_id}"
            expected = (row.get("evidence_hashes") or {}).get(str(evidence_id))
            if not expected or entry.get("content_hash") != expected:
                return False, f"stale_evidence_hash:{evidence_id}"
        verdicts = row.get("verdicts") or {}
        for claim_id in claim_ids:
            verdict = str(verdicts.get(claim_id) or "")
            if verdict not in {"verified_supported", "verified_contradicted"}:
                return False, f"claim_not_verified:{claim_id}"
        affected = {str(rid) for rid in (event_row.get("affected_requirement_ids") or [])}
        covered = {str(rid) for claim_id in claim_ids for rid in (current.get(claim_id, {}).get("requirement_ids") or [])}
        if affected and not affected.intersection(covered):
            return False, "requirements_not_covered"
        return True, ""

    def valid_revalidation_receipt(self, event: Any, claims: Any) -> Optional[Dict[str, Any]]:
        """Return the newest receipt that still verifies for this event, if any."""
        row = event.model_dump(mode="json") if hasattr(event, "model_dump") else dict(event or {})
        for receipt in reversed(self.revalidation_receipts(str(row.get("event_id") or ""))):
            ok, _reason = self.verify_revalidation_receipt(receipt, row, claims)
            if ok:
                return receipt
        return None

    def _append_event(self, _scope: str, entry: Dict[str, Any]) -> None:
        self.append_event(entry)

    def _append_agent_event(self, identity: Identity, entry: Dict[str, Any]) -> None:
        folder = self._role_dir(identity)
        if folder is None:
            return
        if identity.role in {"supervisor", "writer"} and not folder.parent.exists():
            return
        path = folder / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"at": utc_now(), "agent_instance_id": identity.agent_instance_id, "task_id": identity.task_id, **entry}, ensure_ascii=False, default=str)
        with self._meta.lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def append_agent_event(self, identity: Identity, entry: Dict[str, Any]) -> None:
        """Public wrapper to record an event for one agent invocation."""
        if identity.role in _OPERATOR_ONLY_ROLES or identity.role in _PRIVATE_ROLES:
            self._append_agent_event(identity, entry)

    def agent_events(self, identity: Identity, *, limit: int = 200) -> List[Dict[str, Any]]:
        """Return this invocation's own event rows (newest last, bounded).

        Used by the runtime to rebuild the working state from what actually happened
        (tool results, failures, recoveries) instead of trusting a model summary.
        """
        folder = self._role_dir(identity)
        if folder is None:
            return []
        path = folder / "events.jsonl"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events: List[Dict[str, Any]] = []
        for line in lines[-max(1, int(limit)) :]:
            try:
                events.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
        return events

    def write_working_memory(self, identity: Identity, memory: Dict[str, Any]) -> None:
        """Persist the compact working memory of one invocation."""
        folder = self._role_dir(identity)
        if folder is None:
            return
        _atomic_write(folder / "working_memory.json", json.dumps(_jsonable(memory), indent=2, ensure_ascii=False, default=str))

    def write_task_card(self, identity: Identity, task_card: Dict[str, Any]) -> None:
        """Persist the task card of a researcher invocation."""
        folder = self.run_dir / "agents" / safe_id(identity.agent_instance_id)
        _atomic_write(folder / "task_card.json", json.dumps(_jsonable(task_card), indent=2, ensure_ascii=False, default=str))

    # -- request / context records ----------------------------------------
    def write_request(self, identity: Identity, request_id: str, record_requests: bool, payload: Dict[str, Any], manifest: ContextManifest) -> None:
        """Write ``contexts/<id>.manifest.json`` and, when enabled, ``requests/<id>.json``.

        The records are observations: they never feed back into model input.
        """
        folder = self.run_dir / "agents" / safe_id(identity.agent_instance_id)
        if record_requests:
            safe_payload = redact_secrets(payload)
            _atomic_write(folder / "requests" / f"{safe_id(request_id)}.json", json.dumps(safe_payload, indent=2, ensure_ascii=False, default=str))
        _atomic_write(folder / "contexts" / f"{safe_id(request_id)}.manifest.json", json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False, default=str))

    # -- metrics -----------------------------------------------------------
    def record_usage(self, usage: UsageRecord) -> None:
        """Record one model-call attempt (``metrics/calls.jsonl`` + aggregate).

        Every attempt has its own row: a request that is retried three times is three
        model calls in the global total **and** in its role's count. The row key is the
        attempt id, so retries never overwrite each other, and a settled budget
        reservation is consumed exactly once here.

        Note: the store is a *file* lock; atomicity between concurrent reservations is
        provided by :meth:`reserve_model_call` under the same process-wide metadata
        lock, which is the only writer of this run.
        """
        with self._meta.lock:
            self._meta.model_calls += 1
            if usage.input_tokens is None and usage.output_tokens is None:
                self._meta.usage_unknown_calls += 1
            key = usage.attempt_id or usage.request_id or f"anon-{self._meta.model_calls}"
            self._meta.usage[key] = usage
            reservation_id = usage.reservation_id
            if reservation_id and reservation_id in self._meta.reservations:
                reservation = self._meta.reservations[reservation_id]
                if reservation.get("settled") is not True:
                    reservation["settled"] = True
                    reservation["settled_at"] = utc_now()
                    reservation["settled_attempt_id"] = key
                    reservation["settled_tokens"] = (usage.input_tokens or 0) + (usage.output_tokens or 0)
                    reservation["usage_known"] = usage.input_tokens is not None or usage.output_tokens is not None
            self._meta.persist_usage()
            # One append-only row per attempt, written under the same lock so the
            # ledger order matches the order of the aggregate.
            path = self.run_dir / "control" / "calls.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"at": utc_now(), "kind": "model", **_jsonable(usage)}, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _committed_tokens(entry: Dict[str, Any]) -> int:
        """Tokens one reservation still holds against the run-wide cap.

        A settled request whose provider reported usage commits what it actually spent:
        the estimate was only a hold. Keeping ``max(settled, estimate)`` made every
        settled call keep its conservative output charge forever, so a run with 40k real
        tokens could be refused at 190k committed (observed in the F-D acceptance run).
        A call with unknown usage keeps its estimate, since the charge is unknown.
        """
        estimate = int(entry.get("estimated_input_tokens", 0)) + int(entry.get("estimated_output_tokens", 0))
        if not entry.get("settled"):
            return estimate
        if entry.get("usage_known"):
            return int(entry.get("settled_tokens", 0) or 0)
        return estimate

    def reserve_model_call(
        self,
        *,
        request_id: str,
        attempt_id: str,
        role: str,
        agent_instance_id: str,
        max_total_model_calls: int,
        max_total_tokens: int,
        estimated_input_tokens: int = 0,
        # Conservative per-attempt output charge when the provider reports no usage.
        unknown_output_tokens: int = 4000,
        role_cap: Optional[int] = None,
        execution_id: str = "",
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Atomically reserve one model call against the run-wide budget.

        The check and the write happen under one lock, so concurrent reservations
        (parallel researchers) cannot both pass on the last remaining call. A granted
        reservation counts as a call from that moment on, which is what makes the cap
        hold even when the request later times out or is cancelled; the settlement only
        refines the token side. An unknown token total is charged as the conservative
        estimate, never as free usage.
        """
        with self._meta.lock:
            settled = [u for u in self._meta.usage.values() if not u.reservation_id or self._meta.reservations.get(u.reservation_id, {}).get("settled")]
            open_reservations = [entry for entry in self._meta.reservations.values() if not entry.get("settled")]
            # Calls already settled + calls already granted but still in flight. A grant
            # therefore consumes the allowance immediately, which is what makes the cap
            # hold under concurrency, retries and timeouts.
            used_calls = len(settled) + len(open_reservations)
            # Usage rows that did not come from a reservation (legacy/manual rows).
            unreserved = sum((u.input_tokens or 0) + (u.output_tokens or 0) for u in self._meta.usage.values() if not u.attempt_id)
            reserved_tokens = sum(self._committed_tokens(entry) for entry in self._meta.reservations.values())
            used_tokens = unreserved + reserved_tokens
            # An explicit 0 is a hard cap ("no call may start"), not "unlimited".
            if max_total_model_calls is not None and used_calls >= int(max_total_model_calls):
                return False, f"model call budget exhausted ({used_calls}/{int(max_total_model_calls)})", None
            # The request's own hold must fit too: comparing only what is already
            # committed let a 400 + 4000 request through at 99500/100000 and overshoot.
            requested_tokens = max(0, int(estimated_input_tokens or 0)) + max(0, int(unknown_output_tokens or 0))
            if max_total_tokens is not None:
                cap_tokens = int(max_total_tokens)
                if requested_tokens and used_tokens + requested_tokens > cap_tokens:
                    return False, f"token budget exhausted ({used_tokens} committed + {requested_tokens} requested > {cap_tokens})", None
                if not requested_tokens and used_tokens >= cap_tokens:
                    return False, f"token budget exhausted ({used_tokens}/{cap_tokens})", None
            if role_cap is not None:
                # Role allowance = settled attempts for the role + this role's in-flight
                # reservations, so a retry cannot slip past a role cap either.
                role_used = sum(1 for u in settled if (u.role or "unknown") == role) + sum(
                    1 for entry in open_reservations if (entry.get("role") or "unknown") == role
                )
                if role_used >= int(role_cap):
                    return False, f"role budget exhausted for {role} ({role_used}/{int(role_cap)})", None
            estimated_output = max(0, int(unknown_output_tokens or 0))
            reservation = {
                "id": f"{attempt_id}",
                "request_id": request_id,
                "attempt_id": attempt_id,
                "role": role,
                "agent_instance_id": agent_instance_id,
                "reserved_at": utc_now(),
                "estimated_input_tokens": int(estimated_input_tokens or 0),
                "estimated_output_tokens": estimated_output,
                "execution_id": execution_id,
                "reserved_model_calls": 1,
                "settled": False,
            }
            self._meta.reservations[attempt_id] = reservation
            self._meta.persist_usage()
        return True, "", dict(reservation)

    def reservation(self, reservation_id: str) -> Optional[Dict[str, Any]]:
        """Return one budget reservation (traceability)."""
        with self._meta.lock:
            entry = self._meta.reservations.get(reservation_id)
            return dict(entry) if entry else None

    def reservations(self) -> List[Dict[str, Any]]:
        """Return every budget reservation of the run, in reservation order."""
        with self._meta.lock:
            return [dict(entry) for entry in self._meta.reservations.values()]

    def record_tool_call(self, usage: Dict[str, Any]) -> None:
        """Record one tool call row in ``metrics/calls.jsonl``."""
        with self._meta.lock:
            self._meta.tool_calls += 1
            self._meta.persist_usage()
        self.append_ledger("calls", {"kind": "tool", **redact_secrets(usage)})

    def totals(self) -> Dict[str, Any]:
        """Return aggregate usage numbers (used by the global budget)."""
        with self._meta.lock:
            usage = list(self._meta.usage.values())
            costs = [u.cost_usd for u in usage]
            known_costs = [cost for cost in costs if cost is not None]
            # A missing price is not a free call. Report the total only when every
            # attempt has a known cost; otherwise null plus an explicit unknown count.
            cost_usd = sum(known_costs) if known_costs and len(known_costs) == len(costs) else (0.0 if not costs else None)
            # Split totals: a recomputable estimate, a lower bound of what is known, and
            # what the provider actually billed are kept apart, never mixed into one
            # number that would look authoritative while hiding an unpriced attempt.
            estimated = sum(known_costs) if known_costs and len(known_costs) == len(costs) else (0.0 if not costs else None)
            bounded = sum(known_costs) if known_costs else (0.0 if not costs else None)
            billed_rows = [u.provider_billed_cost_usd for u in usage if u.provider_billed_cost_usd is not None]
            billed = sum(billed_rows) if billed_rows and len(billed_rows) == len(usage) else None
            methods: Dict[str, int] = {}
            for row in usage:
                methods[row.cost_method or "unknown"] = methods.get(row.cost_method or "unknown", 0) + 1
            if not usage:
                completeness = "unknown"
            elif len(known_costs) == len(costs):
                completeness = "complete" if all((row.cost_method or "unknown") == "exact_categories" for row in usage) else "partial"
            else:
                completeness = "partial"
            return {
                "model_calls": self._meta.model_calls,
                "tool_calls": self._meta.tool_calls,
                "unknown_usage_calls": self._meta.usage_unknown_calls,
                "cost_unknown_calls": len(costs) - len(known_costs),
                "input_tokens": sum((u.input_tokens or 0) for u in usage),
                "output_tokens": sum((u.output_tokens or 0) for u in usage),
                "cost_usd": cost_usd,
                "estimated_model_cost_usd": estimated,
                "known_cost_lower_bound_usd": bounded,
                "provider_billed_cost_usd": billed,
                "cost_completeness": completeness,
                "cost_methods": methods,
            }

    def calls_by_role(self) -> Dict[str, int]:
        """Return the number of recorded model attempts per role.

        Rows are keyed per attempt, so a request retried three times counts as three.
        A reservation that was granted but never settled (the call is still in flight,
        or the process died mid-request) is counted as an unknown-usage attempt: the
        provider may already have charged it.
        """
        with self._meta.lock:
            counts: Dict[str, int] = {}
            for usage in self._meta.usage.values():
                role = usage.role or "unknown"
                counts[role] = counts.get(role, 0) + 1
            # Granted-but-unsettled reservations are started attempts with unknown usage.
            for entry in self._meta.reservations.values():
                if entry.get("settled"):
                    continue
                role = entry.get("role") or "unknown"
                counts[role] = counts.get(role, 0) + 1
            return counts

    def write_metrics_summary(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Write ``metrics/summary.json`` and return its content.

        Rows are per attempt; an attempt whose provider reported no usage keeps
        ``unknown_usage_calls`` so its cost is never silently reported as zero.
        """
        by_role: Dict[str, Dict[str, Any]] = {}
        with self._meta.lock:
            for usage in self._meta.usage.values():
                role = usage.role or "unknown"
                row = by_role.setdefault(role, {"calls": 0, "attempts_settled": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cost_known_calls": 0, "cost_unknown_calls": 0, "unknown_usage_calls": 0})
                row["calls"] += 1
                row["attempts_settled"] += 1
                if usage.input_tokens is None and usage.output_tokens is None:
                    row["unknown_usage_calls"] += 1
                row["input_tokens"] += usage.input_tokens or 0
                row["output_tokens"] += usage.output_tokens or 0
                if usage.cost_usd is None:
                    row["cost_unknown_calls"] += 1
                else:
                    row["cost_known_calls"] += 1
                    row["cost_usd"] += usage.cost_usd
            for row in by_role.values():
                if row["cost_unknown_calls"]:
                    row["cost_usd"] = None
            unreserved = sum(1 for entry in self._meta.reservations.values() if not entry.get("settled"))
        summary = {
            "run_id": self.run_id,
            "generated_at": utc_now(),
            "totals": self.totals(),
            "by_role": by_role,
            "unsettled_reservations": unreserved,
            **(extra or {}),
        }
        _atomic_write(self.run_dir / "metrics" / "summary.json", json.dumps(summary, indent=2, ensure_ascii=False, default=str))
        return summary

    def _persist_index(self) -> None:
        payload = {"run_id": self.run_id, "updated_at": utc_now(), "evidence": list(self.evidence.values())}
        _atomic_write(self._index_path(), json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    def bind_graph(self, name: str) -> None:
        """Remember which compiled graph this run was started with (host-only)."""
        if not name:
            return
        with self._meta.lock:
            self._meta.graph_name = str(name)
        _atomic_write(self._meta.run_dir / "control" / "graph_binding.json", json.dumps({"run_id": self.run_id, "graph": str(name), "at": utc_now()}, indent=2))

    def bound_graph(self) -> str:
        """Return the graph bound to this run ("" when unknown / never bound)."""
        with self._meta.lock:
            if self._meta.graph_name:
                return self._meta.graph_name
        try:
            data = json.loads((self._meta.run_dir / "control" / "graph_binding.json").read_text(encoding="utf-8"))
            return str(data.get("graph") or "")
        except Exception:  # noqa: BLE001 - an unreadable binding is simply unknown
            return ""

    # -- reports -----------------------------------------------------------
    def write_report(self, identity: Identity, version: int, text: str, stage: str = "draft") -> str:
        """Write a report version and return its path (relative to the run)."""
        folder = self.run_dir / "reports"
        name = "final.md" if stage == "final" else f"draft-{version}.md"
        path = folder / name
        _atomic_write(path, text)
        self.append_event({"type": "report_written", "stage": stage, "version": version, "hash": sha256_text(text), "by": identity.agent_instance_id})
        return str(path.relative_to(self.run_dir))

    def write_review_coverage(self, payload: Dict[str, Any]) -> None:
        """Write ``reports/review-coverage.json``."""
        _atomic_write(self.run_dir / "reports" / "review-coverage.json", json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, default=str))

    def write_review_artifact(self, review_id: str, name: str, payload: Any) -> None:
        """Write ``agents/reviewer-<id>/<name>.json``."""
        folder = self.run_dir / "agents" / f"reviewer-{safe_id(review_id)}"
        _atomic_write(folder / f"{safe_id(name)}.json", json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, default=str))


_STORES: Dict[tuple, ArtifactStore] = {}


def resolve_artifact_root() -> str:
    """Resolve the configured artifact root (``None`` disables on-disk storage)."""
    return os.environ.get("ARTIFACT_ROOT") or ".odr_runs"


def get_store(run_id: str, root: Optional[str] = None) -> ArtifactStore:
    """Return the process-wide store for ``run_id`` (created on first use)."""
    resolved_root = str(Path(root or resolve_artifact_root()).expanduser().resolve())
    key = (resolved_root, safe_id(run_id))
    store = _STORES.get(key)
    if store is None:
        store = ArtifactStore(root=resolved_root, run_id=safe_id(run_id))
        _STORES[key] = store
    return store


def identity_for(
    *,
    run_id: str,
    role: str,
    agent_instance_id: str,
    task_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> Identity:
    """Build a host-bound identity; the only sanctioned constructor in graph code."""
    return Identity(
        run_id=safe_id(run_id, "run"),
        agent_instance_id=safe_id(agent_instance_id),
        role=role,
        task_id=task_id,
        parent_id=parent_id,
    )
