"""受控上下文回读（``request_context``）。

设计依据：``docs/context-review-design/DYNAMIC_CONTEXT.md`` §7。

裁剪之后，原文仍在 ``ArtifactStore`` 里；但“文件存在”不等于模型拿得回来。本模块
提供唯一的模型可见回读入口：

```text
request_context(need, purpose, known_refs?, expected_detail?)
→ 按当前身份检索获准证据
→ 返回 ContextPatch（found / partial / unavailable / access_denied）
```

边界：

- 身份来自宿主绑定的 ``IDENTITY_CONTEXT``，不是模型参数；模型只能给“需要什么”，
  不能给文件路径、agent id 或权限。
- 先查精确 ref，再做关键词/元数据检索，最后才扫描获准原文；摘要没提到的东西也
  可能被找回。
- 找不到返回 ``unavailable``；无权限与不存在返回同一类拒绝信息，不构成枚举预言机；
  返回大小有硬上限，超出部分明确标 ``truncated``。
- 每次调用都写 agent 事件，供工作记忆、指标和遗漏纠正使用。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_core.tools import tool

from open_deep_research.artifacts import (
    ArtifactError,
    ArtifactStore,
    Identity,
    PermissionDenied,
    search_terms,
    sha256_text,
    utc_now,
)
from open_deep_research.configuration import IDENTITY_CONTEXT
from open_deep_research.state import ContextPatch, Requirement

CONTEXT_RECOVERY_DESCRIPTION = (
    "Read a specific passage back from material that was trimmed out of your context. "
    "Name what you need (a version, an exception, a number, a definition) and why; the "
    "runtime searches the archived originals you are allowed to read and returns bounded "
    "verbatim excerpts with source ids. Use it before asserting something you could not "
    "see, or when a conclusion depends on an early condition or a late exception."
)

_EVIDENCE_RE = re.compile(r"^(?:evidence://)?(E\d+)$")
_ARTIFACT_RE = re.compile(r"^artifact://tool-results/[0-9a-f]{16,64}$")
REF_CHARS = 3200
PER_ITEM_CHARS = 900
MAX_ITEMS = 4


def _terms(text: str) -> List[str]:
    """Return ASCII words plus CJK bigrams (shared with the archive search)."""
    return search_terms(text)


def _split_refs(known_refs: Optional[Iterable[str]]) -> List[str]:
    refs: List[str] = []
    for raw in known_refs or []:
        if not raw:
            continue
        for part in re.split(r"[,\s]+", str(raw)):
            candidate = part.strip()
            if candidate and candidate not in refs:
                refs.append(candidate)
    return refs


def recover_context(
    identity: Identity,
    store: Optional[ArtifactStore],
    *,
    need: str,
    purpose: str = "",
    known_refs: Optional[Sequence[str]] = None,
    expected_detail: str = "",
    budget_chars: int = REF_CHARS,
) -> ContextPatch:
    """Search the identity's readable archive and return bounded verbatim excerpts."""
    need = (need or "").strip()
    refs = _split_refs(known_refs)
    if not need and not refs:
        return ContextPatch(need="", purpose=purpose, status="unavailable", still_missing=["need or known_refs is required"])
    if store is None:
        return ContextPatch(need=need, purpose=purpose, status="unavailable", still_missing=["no artifact store in this run"])
    patch = ContextPatch(need=need, purpose=purpose)
    query_terms = _terms(need + " " + expected_detail)
    candidates: List[Dict[str, Any]] = []
    # 1) Exact refs the model already knows about. Both citable evidence ids and
    #    offloaded tool-result pointers are accepted: a pointer printed by the runtime
    #    must be a promise it can keep.
    for ref in refs:
        if _ARTIFACT_RE.match(ref):
            patch.queried_refs += 1
            if store.can_read_artifact(identity, ref):
                candidates.append({"artifact_id": ref, "score": 10_000, "title": "", "url": "", "matched_in": "known_ref"})
            else:
                patch.denied_refs.append(ref)
            continue
        evidence_match = _EVIDENCE_RE.match(ref)
        if not evidence_match:
            patch.still_missing.append(ref)
            continue
        evidence_id = evidence_match.group(1)
        patch.queried_refs += 1
        if store.can_read_evidence(identity, evidence_id):
            candidates.append({"evidence_id": evidence_id, "score": 10_000, "title": "", "url": "", "matched_in": "known_ref"})
        else:
            patch.denied_refs.append(evidence_id)
    # 2) Keyword/metadata search over evidence and archived tool results, then a
    #    bounded full-text scan.
    if query_terms:
        found = store.search_evidence(identity, query_terms, limit=MAX_ITEMS)
        patch.queried_refs += len(found)
        known = {item["evidence_id"] for item in candidates if item.get("evidence_id")}
        candidates.extend(item for item in found if item["evidence_id"] not in known)
        found_artifacts = store.search_artifacts(identity, query_terms, limit=MAX_ITEMS)
        patch.queried_refs += len(found_artifacts)
        known_artifacts = {item["artifact_id"] for item in candidates if item.get("artifact_id")}
        candidates.extend(item for item in found_artifacts if item["artifact_id"] not in known_artifacts)
    if not candidates:
        patch.status = "access_denied" if patch.denied_refs else "unavailable"
        patch.still_missing = [need] if need else list(patch.denied_refs)
        return patch

    used = 0
    seen_text: set = set()
    for item in candidates:
        if len(patch.exact_excerpts) >= MAX_ITEMS:
            patch.truncated = True
            break
        remaining = budget_chars - used
        if remaining <= 120:
            patch.truncated = True
            break
        ref = item.get("artifact_id") or item["evidence_id"]
        try:
            if item.get("artifact_id"):
                excerpt = store.read_artifact(
                    identity,
                    item["artifact_id"],
                    query=need,
                    max_chars=min(PER_ITEM_CHARS, remaining),
                )
            else:
                excerpt = store.read_excerpt(
                    identity,
                    item["evidence_id"],
                    selector="query" if need else "excerpt",
                    max_tokens=max(80, min(PER_ITEM_CHARS, remaining) // 4),
                    query=need,
                )
        except PermissionDenied:
            patch.denied_refs.append(ref)
            continue
        except ArtifactError:
            patch.still_missing.append(ref)
            continue
        text = str(excerpt.get("excerpt") or "")
        digest = sha256_text(text)
        if not text.strip() or digest in seen_text:
            continue
        seen_text.add(digest)
        used += len(text)
        item_truncated = bool(excerpt.get("truncated"))
        if item_truncated:
            # A single cut excerpt is not a complete patch: consumers use this flag
            # to decide whether the answer may still be missing its decisive part.
            patch.truncated = True
        patch.restored_refs.append(ref)
        patch.source_versions.append(
            {
                "ref": ref,
                "evidence_id": item.get("evidence_id", ""),
                "content_hash": excerpt.get("content_hash", ""),
                "retrieved_at": excerpt.get("retrieved_at", ""),
                "title": excerpt.get("title", ""),
                "url": excerpt.get("url", ""),
            }
        )
        patch.exact_excerpts.append(
            {
                "ref": ref,
                "evidence_id": item.get("evidence_id", ""),
                "artifact_id": item.get("artifact_id", ""),
                "title": excerpt.get("title", "") or item.get("title", ""),
                "url": excerpt.get("url", "") or item.get("url", ""),
                "excerpt": text,
                "truncated": item_truncated,
                "matched_in": item.get("matched_in", ""),
            }
        )
    patch.status = "found" if patch.restored_refs else ("access_denied" if patch.denied_refs else "unavailable")
    if patch.status == "found" and (patch.truncated or patch.still_missing):
        patch.status = "partial"
    if not patch.still_missing and query_terms and patch.status == "unavailable":
        patch.still_missing = [need]
    return patch


def render_patch(patch: ContextPatch) -> str:
    """Render one patch for the model: status first, then ids, excerpts and limits."""
    header = f"[request_context] status={patch.status} restored={len(patch.restored_refs)}"
    if patch.affected_requirement_ids:
        header += " affected=" + ",".join(patch.affected_requirement_ids)
    lines: List[str] = [header]
    if patch.status == "access_denied":
        lines.append("This material exists but is not readable by the current task. Ask the supervisor for an authorised hand-off; do not infer its content.")
        return "\n".join(lines)
    if patch.status == "unavailable":
        lines.append("No readable archived passage matched this need. This does not prove the information does not exist: try a narrower need, another known id, or return to search.")
        return "\n".join(lines)
    for item in patch.exact_excerpts:
        ref = item.get("ref") or item.get("artifact_id") or item.get("evidence_id", "")
        title = item.get("title") or item.get("url") or ""
        lines.append(f"\n--- [{ref}] {title} ---")
        if item.get("url"):
            lines.append(f"URL: {item['url']}")
        lines.append(str(item.get("excerpt") or ""))
        if item.get("truncated"):
            lines.append(f"[excerpt truncated; read more with known_refs=[\"{ref}\"] and a narrower need]")
    if patch.still_missing:
        lines.append("\nstill missing: " + ", ".join(patch.still_missing[:6]))
    if patch.denied_refs:
        lines.append("not readable by this task: " + ", ".join(patch.denied_refs[:6]))
    if patch.truncated:
        lines.append("The patch hit its size limit; the remaining material stays archived and can be requested separately.")
    return "\n".join(lines)


def match_requirements(
    patch: ContextPatch,
    requirements: Sequence[Requirement],
    *,
    need: str = "",
) -> List[str]:
    """Return requirement ids whose subject appears in the recovered material.

    This is a deterministic lexical test, not a semantic proof: it decides which
    conclusions must be re-checked, and the reviewer still has to resolve them. An
    empty result means "no requirement name matched", not "nothing was affected".
    """
    haystack = (need + " " + " ".join(str(item.get("excerpt", "")) for item in patch.exact_excerpts)).lower()
    if not haystack.strip():
        return []
    matched: List[str] = []
    for requirement in requirements or []:
        statement = (getattr(requirement, "statement", "") or "").lower()
        terms = [term for term in _terms(statement) if len(term) >= 3 or re.search(r"[\u4e00-\u9fff]", term)][:8]
        if terms and sum(1 for term in terms if term in haystack) >= max(1, min(2, len(terms) // 2)):
            matched.append(requirement.requirement_id)
    return matched


def record_patch(store: Optional[ArtifactStore], identity: Identity, patch: ContextPatch) -> None:
    """Write the patch event used by working memory and metrics (never raises)."""
    if store is None:
        return
    try:
        store.append_agent_event(
            identity,
            {
                "type": "context_patch",
                "created_at": utc_now(),
                "need": patch.need[:400],
                "purpose": patch.purpose[:200],
                "status": patch.status,
                "restored_refs": patch.restored_refs,
                "affected_requirement_ids": patch.affected_requirement_ids,
                "truncated": patch.truncated,
                "denied_refs": patch.denied_refs,
            },
        )
        store.append_event(
            {
                "type": "context_recovery",
                "task_id": identity.task_id,
                "agent_instance_id": identity.agent_instance_id,
                "status": patch.status,
                "restored": len(patch.restored_refs),
                "truncated": patch.truncated,
            }
        )
    except Exception:  # noqa: BLE001 - observation must not break a tool call
        return


def request_context_text(
    *,
    need: str,
    purpose: str = "",
    known_refs: Optional[Sequence[str]] = None,
    expected_detail: str = "",
) -> str:
    """Host-entry for the tool: resolve identity, recover, record, render."""
    context = IDENTITY_CONTEXT.get() or {}
    identity = context.get("identity")
    store = context.get("store")
    if not isinstance(identity, Identity):
        return "[request_context] status=unavailable This tool is not available in the current runtime context."
    patch = recover_context(
        identity,
        store,
        need=need,
        purpose=purpose,
        known_refs=known_refs,
        expected_detail=expected_detail,
    )
    record_patch(store, identity, patch)
    return render_patch(patch)


@tool("request_context", description=CONTEXT_RECOVERY_DESCRIPTION)
def request_context(
    need: str,
    purpose: str = "",
    known_refs: Optional[List[str]] = None,
    expected_detail: str = "",
) -> str:
    """Read a specific passage back from archived material you are allowed to see.

    Args:
        need: What exactly you need to see (version, exception, number, definition).
        purpose: Why this turn needs it (used for the recovery log).
        known_refs: Optional evidence ids such as ``E12`` that you already saw.
        expected_detail: Optional extra keywords to narrow the search.
    """
    return request_context_text(need=need, purpose=purpose, known_refs=known_refs, expected_detail=expected_detail)


__all__ = [
    "CONTEXT_RECOVERY_DESCRIPTION",
    "match_requirements",
    "recover_context",
    "record_patch",
    "render_patch",
    "request_context",
    "request_context_text",
]