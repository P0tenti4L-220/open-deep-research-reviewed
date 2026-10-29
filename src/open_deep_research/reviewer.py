"""独立审查者（Reviewer）。

设计依据：``docs/context-review-design/README.md`` §4 与
``DYNAMIC_CONTEXT.md``。

边界：

- 审查输入 ``ReviewPacket`` 完全由程序从账本、证据库和已发布结果构造；主管不能
  删改、不能注入“已核验”布尔值，也不能用一个提示词跳过门禁。
- 审查者处于新上下文，只收到目标、待审决定、必要证据和未解决问题，不继承主管
  对话。它不能派发任务、修改目标或扩大读取范围。
- 模型输出只是**未验证的意见**。:func:`validate_decision` 用程序检查 verdict
  枚举、证据 id 是否存在、必填字段和返工上限；任何异常（超时、解析失败、预算
  耗尽、伪造 evidence id）都不得自动变成 ``pass``。
- 报告正文本身就是引用点：正文出现而账本未归档的 id 进入 ``missing_evidence_ids``，
  已归档的 id 必须随审查包送到审查者（S1）。
- 审查是否完成是程序事实：只要有一个必需批次超时、解析失败或用尽预算，即使已有
  输出也不得完整通过；``review_complete`` 为假时路由不会 ``finalize``（S3）。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from open_deep_research import observability as obs
from open_deep_research.artifacts import (
    ArtifactStore,
    Identity,
    claim_digest,
    hash_obj,
    utc_now,
)
from open_deep_research.configuration import Configuration
from open_deep_research.state import (
    RawReviewOutput,
    ReviewDecision,
    ReviewFinding,
    ReviewPacket,
    ReviewStage,
    RevalidationReceipt,
    Verdict,
)

LOGGER = logging.getLogger(__name__)

ISSUE_TYPES = {
    "unsupported_claim",
    "contradicted_claim",
    "overreach",
    "missing_requirement",
    "unresolved_conflict",
    "scope_drift",
    "no_progress",
    "premature_completion",
    "tool_failure_ignored",
    "other",
}

STAGE_QUESTIONS = {
    ReviewStage.PLAN: "Does the proposed research plan cover every mandatory requirement, avoid an obviously wrong premise, and stay in scope?",
    ReviewStage.PROGRESS: "Is the research still making progress toward the mandatory requirements, or is it looping without new evidence? What concrete next step (or stop) is justified?",
    ReviewStage.READINESS: "Are all mandatory requirements either answered with supporting evidence or explicitly recorded as unknown? Is it justified to stop researching now?",
    ReviewStage.REPORT: "Does every factual claim in the draft that matters have locatable supporting evidence with its qualifiers preserved? Is each mandatory requirement covered?",
    ReviewStage.TERMINATION: "The budget is exhausted. Which parts are supported enough to deliver, and which limits must the partial report state?",
}


def _brief_constraints(brief: Any) -> List[str]:
    constraints = list(getattr(brief, "constraints", []) or [])
    for requirement in getattr(brief, "requirements", []) or []:
        if getattr(requirement, "mandatory", False):
            constraints.append(f"{requirement.requirement_id}: {requirement.statement}")
    return constraints


# Reviewer budgets: one call must see the whole expected scope. A draft that does not
# fit is split into segments and reviewed across several calls, never silently cut.
REVIEWER_BUDGET_CHARS = 24000  # ~ reviewer_context_budget (8000 tokens) * 3
REVIEW_MAX_SEGMENTS = 12
_REVIEW_MIN_SEGMENT_CHARS = 400
_CITATION_RE = re.compile(r"\[(E[0-9]+)\]")


def draft_evidence_ids(draft: str) -> List[str]:
    """Return the evidence ids the report body itself cites, in first-seen order.

    Both permitted citation forms are read: ``[E1]`` and ``[E1](https://...)``. The
    link target alone is not a citation, and an id that no claim mentions is still
    part of the review scope once the writer put it in the draft (S1).
    """
    ids: List[str] = []
    for match in _CITATION_RE.finditer(draft or ""):
        if match.group(1) not in ids:
            ids.append(match.group(1))
    return ids

# Requirement state words the reviewer may use about a requirement in the draft.
def _segments(text: str, *, budget_chars: int = REVIEWER_BUDGET_CHARS, max_segments: int = REVIEW_MAX_SEGMENTS) -> List[Dict[str, Any]]:
    """Split a draft into contiguous segments that together contain the whole text.

    Segments never drop characters: concatenating the ``text`` of every segment
    reproduces the input exactly, so nothing can fall outside the review scope. When a
    draft needs more segments than ``max_segments``, the extra segments are still
    produced and flagged ``within_budget=False``: they are reviewable, and a request
    that cannot carry them is split again by ``run_review``.
    """
    text = text or ""
    if not text:
        return [{"segment_id": "S1", "char_start": 0, "char_end": 0, "text": "", "within_budget": True}]
    if len(text) <= budget_chars:
        return [{"segment_id": "S1", "char_start": 0, "char_end": len(text), "text": text, "within_budget": True}]
    count = max(2, math.ceil(len(text) / budget_chars))
    target = math.ceil(len(text) / count)
    segments: List[Dict[str, Any]] = []
    pos = 0
    while pos < len(text):
        if len(segments) == count - 1:
            end = len(text)
        else:
            budget_end = min(len(text), pos + target)
            # Prefer a paragraph boundary near the target, else the hard cut.
            window_start = max(pos + int(target * 0.85), pos + 1)
            newline = text.rfind("\n", window_start, budget_end)
            end = newline + 1 if newline > 0 else budget_end
            if end <= pos:
                end = budget_end
        index = len(segments) + 1
        segments.append({"segment_id": f"S{index}", "char_start": pos, "char_end": end, "text": text[pos:end], "within_budget": index <= max_segments})
        pos = end
        if len(segments) >= 4 * max_segments:  # pragma: no cover - pathological guard
            break
    if pos < len(text):  # pragma: no cover - only if the guard above triggered
        segments[-1]["text"] += text[pos:]
        segments[-1]["char_end"] = len(text)
    return segments


def _draft_blocks(draft: str) -> List[Dict[str, Any]]:
    """Split a draft into paragraph blocks that keep their heading context.

    Blocks are the unit for the "unbound assertion" check: a short factual paragraph
    written after a cited section must not borrow that section's citations. A block
    that directly follows a Sources/References heading is flagged so that only genuine
    citation entries inside it can be exempted (see :func:`_is_reference_entry`); the
    heading itself never exempts a factual sentence.
    """
    blocks: List[Dict[str, Any]] = []
    in_reference_section = False
    for raw in re.split(r"\n\s*\n", draft or ""):
        block = raw.strip()
        if not block:
            continue
        heading = next((line.strip() for line in block.splitlines() if line.lstrip().startswith("#")), "")
        if heading:
            in_reference_section = bool(_REFERENCE_HEADING.match(heading))
        blocks.append({"index": len(blocks) + 1, "text": block, "reference_section": in_reference_section})
    return blocks


_REFERENCE_HEADING = re.compile(r"^#{1,6}\s*(sources?|references?|bibliography|citations?|footnotes?)\b", re.IGNORECASE)
_URL_ONLY = re.compile(r"^\s*<?https?://\S+>?\s*$")
_URL_IN_TEXT = re.compile(r"https?://\S+|www\.\S+")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)]|\[\d+\])\s+")
_BRACKET_ENTRY = re.compile(r"^\s*\[\d+\]")
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
_SENTENCE_PUNCT = re.compile(r"[.!?。！？](?:\s|$)")
_YEAR_CITATION = re.compile(r"\b(?:19|20)\d{2}\b")


def _is_reference_entry(line: str, *, in_reference_section: bool = False) -> bool:
    """Return True only for a bibliography *pointer* line, not a factual sentence.

    A line is exempt from the assertion check when it is structurally a source
    pointer: a bare URL, a ``[1]``-prefixed bibliography entry, a list item whose body
    is a markdown link, or a short label followed by a URL. Inside an explicit
    Sources/References section a year-bearing citation entry is also accepted (APA
    style entries carry no URL).

    A bullet or numbered item that states a fact - no link, no URL, no citation
    marker - is a claim and stays in scope. Merely sitting under a heading (or being
    the one entry of a list) does not exempt anything (S2).
    """
    text = (line or "").strip()
    if not text or text.startswith("#"):
        return True
    if _URL_ONLY.match(text):
        return True
    if _BRACKET_ENTRY.match(text):
        return True
    marker = _LIST_MARKER.match(text)
    body = text[marker.end():].strip() if marker else text
    if not body:
        return True
    if _MARKDOWN_LINK.search(body):
        remainder = _MARKDOWN_LINK.sub("", body).strip(" \t.,;:)]}>-")
        if not remainder:
            return True
    if not _URL_IN_TEXT.search(body):
        if marker and in_reference_section and _YEAR_CITATION.search(body):
            return True
        return False
    if not marker:
        # A prose paragraph that happens to contain a URL is still prose.
        return False
    prose = _URL_IN_TEXT.sub("", _MARKDOWN_LINK.sub("", body))
    prose = prose.strip(" \t.,;:)]}>-")
    if len(prose) > 160:
        return False
    return not bool(_SENTENCE_PUNCT.search(prose + " "))


def _is_reference_block(text: str, *, in_reference_section: bool = False) -> bool:
    """Return True when every non-heading line of the block is a source pointer."""
    lines = [line for line in (text or "").splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if not lines:
        return True
    return all(_is_reference_entry(line, in_reference_section=in_reference_section) for line in lines)


def _assertions(text: str, *, in_reference_section: bool = False) -> int:
    """Count sentence-like claims in a draft section (structural, not semantic).

    Reference entries are removed line by line, so a mixed block (one source pointer
    plus one uncited factual bullet) is still detected. Headings carry no assertion of
    their own.
    """
    kept: List[str] = []
    for line in (text or "").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _is_reference_entry(line, in_reference_section=in_reference_section):
            continue
        kept.append(line)
    if not kept:
        return 0
    sentences = [part for part in re.split(r"(?<=[.!?])\s+|\n+", "\n".join(kept)) if len(part.strip()) >= 25]
    return len(sentences)


def _section_requirement_hits(section: Dict[str, Any], requirement_ids: Sequence[str]) -> List[str]:
    """Return the requirement statements this section restates (lexical pre-check)."""
    lowered = (section.get("text") or "").lower()
    hits: List[str] = []
    for rid in requirement_ids:
        statement = str(rid).lower()
        if len(statement) >= 12 and statement in lowered:
            hits.append(rid)
    return hits


def _unbound_assertions(draft: str, claims: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """List draft paragraphs whose factual sentences cite no evidence at all.

    A paragraph that makes assertions but mentions no claim id and no archived evidence
    id was written from the model's own knowledge or from unverified notes; the
    reviewer cannot check it, so the program records it as unbound.
    """
    known_claim_ids = {str(c.get("claim_id")) for c in claims if c.get("claim_id")}
    unbound: List[Dict[str, Any]] = []
    for block in _draft_blocks(draft):
        text = block["text"]
        assertions = _assertions(text, in_reference_section=bool(block.get("reference_section")))
        if not assertions:
            continue
        # Headings and bullet labels carry the citations of their body: judge the text.
        if any(cid and cid in text for cid in known_claim_ids) or _CITATION_RE.search(text):
            continue
        unbound.append({"segment_id": f"P{block['index']}", "assertions": assertions, "excerpt": text[:200]})
    return unbound


def _claim_review_material(claim: Dict[str, Any], excerpts: Sequence[Dict[str, Any]]) -> str:
    """Bind a finding to content; a version-only bump is not a repair."""
    refs = list(claim.get("evidence_refs") or [])
    return hash_obj({
        "statement": claim.get("statement"),
        "kind": claim.get("kind", "fact"),
        "evidence_refs": refs,
        "support_span": claim.get("support_span") or {},
        "qualifiers": claim.get("qualifiers") or [],
        "evidence_hashes": {row["evidence_id"]: row.get("content_hash", "") for row in excerpts if row["evidence_id"] in refs},
    })


def build_review_packet(
    stage: ReviewStage,
    *,
    review_id: str,
    run_id: str,
    brief: Any,
    proposal: str,
    store: Optional[ArtifactStore],
    task_ledger: Optional[Dict[str, Any]] = None,
    claims: Optional[Sequence[Dict[str, Any]]] = None,
    evidence_ids: Optional[Sequence[str]] = None,
    failures: Optional[Sequence[str]] = None,
    remaining_budget: Optional[Dict[str, Any]] = None,
    draft: str = "",
    draft_hash: str = "",
    extra_counter_evidence: Optional[Sequence[str]] = None,
) -> ReviewPacket:
    """Construct a review packet from program-owned ledgers only.

    The supervisor's own narrative is passed as ``proposal`` and is explicitly
    labelled as an unverified proposal; it is never treated as a source.

    Delivery expectations are derived here, not by the reviewer:

    * ``missing_evidence_ids``: claim references with no archived evidence (so no
      reviewer could have read them);
    * ``delivery_required_claim_ids``: at the report stage every claim;
    * ``report_segments``: the whole draft, in contiguous segments (R2);
    * ``unbound_assertions``: draft sections that cite neither a claim nor evidence;
    * ``scope_issues``: stage preconditions (e.g. a report with no claims at all).
    """
    claims = list(claims or [])
    evidence_ids = list(dict.fromkeys(list(evidence_ids or [])))
    active_ids = {str(claim.get("claim_id")) for claim in claims}
    prior_findings = [entry for entry in (store.claim_review_findings() if store is not None else []) if entry.get("status") == "open" and entry.get("claim_id") in active_ids]
    for entry in prior_findings:
        evidence_ids.extend(eid for eid in entry.get("finding", {}).get("evidence_refs", []) if eid not in evidence_ids)
    counter = list(dict.fromkeys(list(extra_counter_evidence or [])))
    # Every evidence id a claim relies on is in scope, so the reviewer can check the
    # cited source rather than only the supervisor's summary of it.
    for claim in claims:
        for ref in claim.get("evidence_refs", []) or []:
            if ref not in evidence_ids:
                evidence_ids.append(ref)
    for cid in counter:
        if cid not in evidence_ids:
            evidence_ids.append(cid)
    # S1: the report body is itself a citation site. An id the writer put in the draft
    # is inside the review scope even when no claim references it, so its source must be
    # shown to the reviewer; an id that was never archived must surface as missing
    # instead of slipping through as "already cited".
    for eid in draft_evidence_ids(draft):
        if eid not in evidence_ids:
            evidence_ids.append(eid)
    # Read every id in scope. A fixed default limit turned archived sources into
    # "missing" once a run had more than 24 of them, which produced false gaps and
    # blocked correct reports. Truncated excerpts are reported separately: they are
    # readable, just cut, and must not be counted as absent.
    focus: Dict[str, str] = {}
    for claim in claims:
        span = claim.get("support_span") or {}
        needle = str(span.get("quote") or claim.get("statement") or "").strip()
        for ref in claim.get("evidence_refs", []) or []:
            if needle and ref not in focus:
                focus[ref] = needle
    excerpts = store.read_evidence_for_review(evidence_ids, focus=focus) if store is not None else []
    span_checks: Dict[str, Dict[str, Any]] = {}
    for claim in claims:
        span = claim.get("support_span") or {}
        if not span or store is None:
            continue
        try:
            eid = str(span.get("evidence_id") or "")
            if eid not in claim.get("evidence_refs", []):
                raise ValueError("support span is not one of the claim's references")
            actual = store.bind_quote(eid, str(span.get("quote") or ""))
            if any(actual.get(key) != span.get(key) for key in ("content_hash", "start", "end", "quote")):
                raise ValueError("support span does not match the archived offsets/hash")
            span_checks[str(claim["claim_id"])] = {"support_span_verified": True}
        except Exception as exc:  # noqa: BLE001 - invalid support cannot become verified
            span_checks[str(claim["claim_id"])] = {"support_span_verified": False, "support_span_error": str(exc)}
    found = {entry["evidence_id"] for entry in excerpts}
    missing = [eid for eid in evidence_ids if eid not in found]
    truncated = [entry["evidence_id"] for entry in excerpts if entry.get("excerpt_truncated")]
    open_issues: List[str] = []
    if missing:
        open_issues.append(f"Referenced evidence not archived: {', '.join(missing)}")
    if truncated:
        open_issues.append(
            "These sources are archived but only their opening excerpt is shown here (the full original exists): "
            + ", ".join(truncated)
            + ". Do not treat a cut excerpt as missing evidence; qualify any conclusion that needs the omitted part."
        )
    revalidation_issues: List[str] = []
    if store is not None:
        for event in store.open_revalidations():
            requirement_ids = ", ".join(event.get("affected_requirement_ids") or [])
            revalidation_issues.append(
                f"{event.get('event_id')}: material {', '.join(event.get('restored_refs') or [])} was restored after "
                f"conclusions for {requirement_ids or 'this task'} were formed; those conclusions need re-checking."
            )
        if revalidation_issues:
            open_issues.extend(revalidation_issues)
    ledger = task_ledger or {}
    coverage_matrix = []
    for requirement in getattr(brief, "requirements", []) or []:
        rid = requirement.requirement_id
        covering_claims = [c for c in claims if rid in (c.get("requirement_ids") or [])]
        covering = [c.get("claim_id") for c in covering_claims]
        covering_evidence = sorted({r for c in covering_claims for r in (c.get("evidence_refs") or [])})
        available = [eid for eid in covering_evidence if eid in found]
        # A requirement with archived material is only a *candidate*: no review has
        # semantically verified the claim yet. ``verified`` is reserved for a claim a
        # previous review accepted, so the first review is not circular.
        verified = any(str(c.get("support_status") or "") == "supported" for c in covering_claims)
        coverage_matrix.append(
            {
                "requirement_id": rid,
                "statement": requirement.statement,
                "mandatory": bool(requirement.mandatory),
                "claim_ids": covering,
                "evidence_ids": covering_evidence,
                "available_evidence_ids": available,
                "state": "verified" if available and verified else ("candidate" if available else "missing"),
            }
        )
    for task_id, entry in ledger.items():
        if entry.get("status") in {"failed", "partial"}:
            open_issues.append(f"Task {task_id} ended {entry.get('status')}: {(entry.get('error') or 'no result')[:200]}")
    claim_ids = [c.get("claim_id") for c in claims if c.get("claim_id")]
    open_issues.extend(
        f"Unclosed {entry['finding_id']} for {entry['claim_id']}: {entry['finding']['required_action']}"
        for entry in prior_findings
    )
    segments = _segments(draft) if draft else []
    unbound = _unbound_assertions(draft, claims) if draft else []
    scope_issues: List[str] = [f"{cid}: invalid archived support span: {entry['support_span_error']}" for cid, entry in span_checks.items() if entry.get("support_span_error")]
    if stage == ReviewStage.REPORT:
        if not claims:
            scope_issues.append("report submitted with no claim list: no factual statement in it can be verified")
        if not segments or not any((s.get("text") or "").strip() for s in segments):
            scope_issues.append("report draft is empty")
        if unbound:
            scope_issues.append(f"{len(unbound)} draft section(s) assert facts without citing any claim or evidence id")
    packet = ReviewPacket(
        review_id=review_id,
        stage=stage,
        run_id=run_id,
        objective=getattr(brief, "objective", "") if brief is not None else "",
        mandatory_constraints=_brief_constraints(brief),
        requirement_ids=[r.requirement_id for r in (getattr(brief, "requirements", []) or [])],
        coverage_matrix=coverage_matrix,
        open_issues=open_issues + list(failures or []),
        failures=list(failures or []),
        remaining_budget=dict(remaining_budget or {}),
        claim_ids=claim_ids,
        evidence_ids=evidence_ids,
        contradictory_evidence_ids=counter,
        evidence_excerpts=excerpts,
        evidence_truncated_ids=truncated,
        revalidation_issues=revalidation_issues,
        prior_claim_findings=prior_findings,
        claims=[
            {
                "claim_id": c.get("claim_id"),
                "statement": c.get("statement"),
                "kind": c.get("kind", "fact"),
                "evidence_refs": c.get("evidence_refs", []),
                "requirement_ids": c.get("requirement_ids", []),
                "qualifiers": c.get("qualifiers", []),
                "version": c.get("version", 1),
                "status": c.get("status", "active"),
                "support_status": c.get("support_status", "unchecked"),
                "support_span": c.get("support_span", {}),
                "unbound_reason": c.get("unbound_reason", ""),
                "unknown_refs": c.get("unknown_refs", []),
                **span_checks.get(str(c.get("claim_id")), {}),
            }
            for c in claims
        ],
        proposed_action=proposal,
        report_sections=[{"title": seg["segment_id"], "segment_id": seg["segment_id"], "text": seg["text"]} for seg in segments],
        draft_hash=draft_hash,
        allowed_read_scope=evidence_ids,
        review_coverage_target="all mandatory requirements and all claims listed in this packet",
        delivery_required_claim_ids=list(claim_ids) if stage == ReviewStage.REPORT else [],
        available_evidence_ids=[eid for eid in evidence_ids if eid in found],
        missing_evidence_ids=missing,
        report_segments=segments,
        draft_assertion_count=sum(_assertions(block["text"], in_reference_section=bool(block.get("reference_section"))) for block in _draft_blocks(draft)) if draft else 0,
        unbound_assertions=unbound,
        scope_issues=scope_issues,
        blocking_issue_count=len(missing) + len([i for i in open_issues if i.startswith("Task ") and " failed" in i]),
    )
    packet.packet_hash = packet_hash(packet)
    return packet


def missing_delivery_requirements(packet: ReviewPacket) -> List[str]:
    """Return which report deliverables a packet would demand (deterministic)."""
    return list(packet.delivery_required_claim_ids)


def packet_hash(packet: ReviewPacket) -> str:
    """Hash the review-relevant content; identical packets replay for free."""
    payload = packet.model_dump(mode="json")
    payload.pop("packet_hash", None)
    payload.pop("remaining_budget", None)
    payload.pop("review_id", None)
    return hash_obj(payload)


def render_packet(packet: ReviewPacket) -> str:
    """Render a packet as the reviewer's user message (no supervisor history)."""
    lines: List[str] = [f"<ReviewPacket stage=\"{packet.stage.value}\" id=\"{packet.review_id}\">"]
    lines.append(f"<Question>{STAGE_QUESTIONS[packet.stage]}</Question>")
    lines.append(f"<Objective>{packet.objective}</Objective>")
    lines.append("<MandatoryRequirements>")
    for item in packet.coverage_matrix:
        lines.append(
            f"- {item['requirement_id']} [{'mandatory' if item['mandatory'] else 'optional'}] {item['statement']} "
            f"(state={item['state']}; claim_ids={item['claim_ids']}; evidence_ids={item['evidence_ids']})"
        )
    lines.append("</MandatoryRequirements>")
    if packet.missing_evidence_ids:
        lines.append("<MissingEvidence>")
        lines.append("Claims cite these ids, but no archived evidence with that id exists, so they cannot be verified:")
        lines.extend(f"- {eid}" for eid in packet.missing_evidence_ids)
        lines.append("</MissingEvidence>")
    if packet.unbound_assertions:
        lines.append("<UnboundAssertions>")
        lines.append("These draft sections assert facts but cite no claim id and no [E<n>] evidence id:")
        for item in packet.unbound_assertions:
            lines.append(f"- {item['segment_id']} ({item['assertions']} assertion(s)): {item['excerpt']}")
        lines.append("</UnboundAssertions>")
    if packet.mandatory_constraints:
        lines.append("<Constraints>")
        lines.extend(f"- {c}" for c in packet.mandatory_constraints)
        lines.append("</Constraints>")
    if packet.claims:
        lines.append("<Claims>")
        lines.append("Host checks each claim's own SUPPORT SPAN: location, not semantic entailment; it can fall outside the shared excerpt.")
        for claim in packet.claims:
            lines.append(
                f"- {claim.get('claim_id')} ({claim.get('kind')}) v{claim.get('version', 1)} support={claim.get('support_status')} "
                f"evidence={claim.get('evidence_refs')} requirements={claim.get('requirement_ids')} qualifiers={claim.get('qualifiers')}: {claim.get('statement')}"
            )
            span = claim.get("support_span") or {}
            if span.get("quote"):
                lines.append(
                    f"  SUPPORT SPAN evidence_id={span.get('evidence_id')} host_verbatim_verified={bool(claim.get('support_span_verified'))} content_hash={span.get('content_hash')} chars {span.get('start')}-{span.get('end')}: {span.get('quote')}"
                )
            if claim.get("unbound_reason") or claim.get("unknown_refs"):
                lines.append(f"  UNBOUND: {claim.get('unbound_reason') or ('unknown ids: ' + ', '.join(claim.get('unknown_refs') or []))}")
        lines.append("</Claims>")
    lines.append("<Evidence>")
    for entry in packet.evidence_excerpts:
        counter = " COUNTER-EVIDENCE" if entry.get("is_counter_evidence") else ""
        cut = " [excerpt cut to the opening 1200 chars; the full original is archived]" if entry.get("excerpt_truncated") else ""
        lines.append(
            f"- {entry['evidence_id']}{counter} {entry.get('title', '')} <{entry.get('url', '')}> retrieved {entry.get('retrieved_at', '')}{cut}\n"
            f"  EXCERPT: {entry.get('exact_excerpt', '')}"
        )
        if entry.get("relevant_excerpt"):
            span = entry.get("relevant_span") or {}
            lines.append(
                f"  TASK-RELEVANT WINDOW content_hash={span.get('content_hash')} chars {span.get('start')}-{span.get('end')}"
                f" (quote present in the archive: {bool(entry.get('relevant_match'))}): {entry['relevant_excerpt']}"
            )
    lines.append("</Evidence>")
    lines.append("<PriorClaimFindings>")
    lines.append(json.dumps([
        {key: entry[key] for key in ("finding_id", "claim_id", "finding")}
        for entry in packet.prior_claim_findings
    ], ensure_ascii=False, indent=2))
    lines.append("</PriorClaimFindings>")
    lines.append("finding_resolutions may use only finding_id values in PriorClaimFindings. If the list is empty, return finding_resolutions=[]. Revalidation and OpenIssues ids are not claim finding ids.")
    if packet.revalidation_issues:
        lines.append("<Revalidation>")
        lines.append("Newly recovered material may invalidate conclusions formed before it was available:")
        lines.extend(f"- {issue}" for issue in packet.revalidation_issues)
        lines.append("</Revalidation>")
    if packet.open_issues:
        lines.append("<OpenIssues>")
        lines.extend(f"- {issue}" for issue in packet.open_issues)
        lines.append("</OpenIssues>")
    if packet.proposed_action:
        lines.append("<ProposedAction status=\"unverified proposal, not evidence\">")
        lines.append(packet.proposed_action[:4000])
        lines.append("</ProposedAction>")
    if packet.report_segments:
        selected = list(packet.review_segments or [])
        shown = [seg for seg in packet.report_segments if not selected or str(seg.get("segment_id")) in selected]
        lines.append(f"<DraftSections segments=\"{len(shown)} of {len(packet.report_segments)}; contiguous, no elision\">")
        if selected and len(shown) < len(packet.report_segments):
            lines.append("This request shows part of the draft. Check every segment shown here.")
        else:
            lines.append("Check every segment. A pass requires reading all of them.")
        for segment in shown:
            lines.append(f"<Segment id=\"{segment['segment_id']}\" chars=\"{segment.get('char_start', 0)}-{segment.get('char_end', 0)}\">")
            lines.append(segment.get("text") or "")
            lines.append("</Segment>")
        lines.append("</DraftSections>")
    lines.append(f"<Budget>{json.dumps(packet.remaining_budget, ensure_ascii=False)}</Budget>")
    lines.append("</ReviewPacket>")
    return "\n".join(lines)


def _split_union(value: Any) -> List[str]:
    """Split a field that a model may fill with several ids (``"C1, C2"``)."""
    return [part for part in re.split(r"[,;，；\s]+", str(value or "").strip()) if part]


def validate_decision(
    raw: Optional[RawReviewOutput],
    packet: ReviewPacket,
    *,
    error: str = "",
    review_cost: float = 0.0,
    review_model_calls: int = 0,
) -> ReviewDecision:
    """Apply program checks to a raw reviewer output.

    Rules (design: README §4.2, IMPLEMENTATION_HANDOFF §6):

    * A finding that cites a non-existent ``evidence_ref`` invalidates the review.
    * At the report stage a ``pass`` becomes ``passed`` (deliverable) only when the
      program's own coverage says the whole deliverable was reviewed:

      - every claim in ``delivery_required_claim_ids`` was checked;
      - every mandatory requirement is covered by archived evidence, or was checked
        and marked unknown by the reviewer;
      - no claim cites evidence that was never archived (``missing_evidence_ids``);
      - no draft segment was left unread (``unreviewed_segment_ids`` empty);
      - the draft is not empty and has no unbound assertions (facts citing neither a
        claim nor evidence), and the scope preconditions hold (``scope_issues``).
    * No output / timeout / parse error -> ``insufficient_evidence`` + not passed.
    * Non-report stages stay about research progress: a pass must cover the claims
      that were put in front of the reviewer.
    """
    required_claims = list(packet.delivery_required_claim_ids or [])
    all_segments = [str(seg.get("segment_id")) for seg in (packet.report_segments or [])]
    requirement_states = {str(item.get("requirement_id")): str(item.get("state") or "missing") for item in packet.coverage_matrix or []}
    mandatory_requirement_ids = [str(item.get("requirement_id")) for item in packet.coverage_matrix or [] if item.get("mandatory")]

    def base(verdict: Verdict, status: str, err: str = "", note: str = "", protocol_errors: Optional[List[str]] = None) -> ReviewDecision:
        findings = [
            ReviewFinding(
                issue_type="other",
                severity="blocking",
                explanation=(note or err or "reviewer did not return a usable decision"),
                required_action="Re-run the review, or gather the missing evidence; do not deliver the affected content as verified.",
            )
        ]
        return ReviewDecision(
            review_id=packet.review_id,
            stage=packet.stage,
            verdict=verdict,
            findings=findings,
            checked_claim_ids=[],
            unchecked_claim_ids=list(packet.claim_ids),
            unchecked_requirement_ids=list(mandatory_requirement_ids),
            unreviewed_segment_ids=list(all_segments),
            unbound_assertion_count=len(packet.unbound_assertions or []),
            draft_hash=packet.draft_hash,
            packet_hash=packet.packet_hash,
            review_status=status,
            review_cost=review_cost,
            review_model_calls=review_model_calls,
            review_complete=False,
            protocol_errors=list(protocol_errors or []),
            error="; ".join(part for part in ("reviewer protocol error: " + "; ".join(protocol_errors) if protocol_errors else "", err, note) if part),
        )

    if raw is None:
        return base(Verdict.INSUFFICIENT_EVIDENCE, "unreviewed", error or "no reviewer output")

    available_evidence = set(packet.available_evidence_ids or packet.evidence_ids)
    claimed_evidence = set(packet.evidence_ids)
    allowed_claims = set(packet.claim_ids)
    allowed_sections = set(all_segments) | {str(row["segment_id"]) for row in packet.unbound_assertions}
    findings: List[ReviewFinding] = []
    violations: List[str] = []
    protocol_errors: List[str] = []
    field_notes: List[str] = []
    for finding in raw.findings or []:
        # P/S ids are supplied by the host. A known paragraph in the claim field is
        # a segment finding, never an invented claim or evidence gap.
        if finding.claim_id in allowed_sections and not finding.segment_id:
            finding = finding.model_copy(update={"claim_id": "", "segment_id": finding.claim_id})
        section_tokens = _split_union(finding.segment_id)
        if any(token not in allowed_sections for token in section_tokens):
            violations.append(f"finding cites unknown segment id {finding.segment_id!r}")
            protocol_errors.append("finding.segment_id is outside the packet's shown sections")
        claim_tokens = _split_union(finding.claim_id)
        valid_union = len(claim_tokens) > 1 and all(token in allowed_claims for token in claim_tokens)
        if len(claim_tokens) > 1 and not valid_union:
            protocol_errors.append(f"finding.claim_id mixes ids {claim_tokens!r}; not all are valid claims")
        refs = []
        for ref in finding.evidence_refs or []:
            tokens = _split_union(ref)
            if len(tokens) > 1 and any(token not in claimed_evidence for token in tokens):
                protocol_errors.append(f"finding.evidence_refs mixes ids {tokens!r}; not all are in scope")
            refs.extend(tokens)
        refs = list(dict.fromkeys(refs))
        bogus = [ref for ref in refs if ref not in claimed_evidence]
        # A claim id in evidence_refs names no source: it is stripped, never accepted
        # as evidence. It is reported as a field error rather than a fabricated source,
        # because the finding's target is still named in claim_id and a reviewer format
        # slip must not by itself veto an otherwise verified delivery. Unknown tokens
        # (invented E/C ids) stay violations.
        misplaced = [ref for ref in bogus if ref in allowed_claims or ref.startswith("C-")]
        fabricated = [ref for ref in bogus if ref not in misplaced]
        refs = [ref for ref in refs if ref in claimed_evidence]
        if misplaced:
            field_notes.append(
                f"finding.evidence_refs holds claim ids {misplaced}: claim ids belong in claim_id, evidence_refs take archived E ids"
            )
        if fabricated:
            violations.append(f"finding cites non-existent evidence ids: {fabricated}")
        if finding.claim_id and finding.claim_id not in allowed_claims and not valid_union:
            violations.append(f"finding cites unknown claim id {finding.claim_id!r}")
        issue_type = finding.issue_type if finding.issue_type in ISSUE_TYPES else "other"
        for cid in dict.fromkeys(claim_tokens if valid_union else [finding.claim_id]):
            for sid in dict.fromkeys(section_tokens or [""]):
                findings.append(finding.model_copy(update={"claim_id": cid, "segment_id": sid, "evidence_refs": refs, "issue_type": issue_type}))

    raw_checked: List[str] = []
    for entry in raw.checked_claim_ids or []:
        tokens = _split_union(entry)
        if len(tokens) > 1 and not all(token in allowed_claims for token in tokens):
            protocol_errors.append(f"checked_claim_ids mixes ids {tokens!r}; not all are valid claims")
        raw_checked.extend(tokens)
    checked = [cid for cid in dict.fromkeys(raw_checked) if cid in allowed_claims]
    unknown_checked = [cid for cid in raw_checked if cid not in allowed_claims]
    if unknown_checked:
        violations.append(f"reviewer claims to have checked unknown ids: {unknown_checked}")
    unchecked = [cid for cid in packet.claim_ids if cid not in checked]

    def split_checked(values: Sequence[str], allowed: Sequence[str], field: str) -> List[str]:
        tokens: List[str] = []
        for value in values:
            parts = _split_union(value)
            if len(parts) > 1 and any(part not in allowed for part in parts):
                protocol_errors.append(f"{field} mixes ids {parts!r}; not all are in scope")
            tokens.extend(parts)
        return list(dict.fromkeys(tokens))

    requirement_tokens = split_checked(raw.checked_requirement_ids or [], list(requirement_states), "checked_requirement_ids")
    checked_requirements = [rid for rid in requirement_tokens if rid in requirement_states]
    unknown_requirements = [rid for rid in requirement_tokens if rid not in requirement_states]
    if unknown_requirements:
        violations.append(f"reviewer claims to have verified unknown requirement ids: {unknown_requirements}")
    acknowledged_tokens = split_checked(raw.acknowledged_unknown_requirement_ids or [], list(requirement_states), "acknowledged_unknown_requirement_ids")
    acknowledged = [
        rid
        for rid in acknowledged_tokens
        if str(rid) in requirement_states and requirement_states.get(str(rid)) not in {"verified", "covered"}
    ]
    unknown_acknowledged = [rid for rid in acknowledged_tokens if str(rid) not in requirement_states]
    if unknown_acknowledged:
        violations.append(f"reviewer acknowledges unknown requirement ids: {unknown_acknowledged}")
    # ``candidate`` coverage means material exists but nobody has checked it in this
    # review: it counts only when the reviewer explicitly checked that requirement.
    uncovered_requirements = []
    for rid in mandatory_requirement_ids:
        if rid in acknowledged:
            continue
        state = requirement_states.get(rid)
        if state in {"verified", "covered"}:
            continue
        if state == "candidate" and rid in checked_requirements:
            continue
        uncovered_requirements.append(rid)

    segment_tokens = split_checked(raw.checked_segment_ids or [], all_segments, "checked_segment_ids")
    checked_segments = [sid for sid in segment_tokens if sid in all_segments]
    unknown_segments = [sid for sid in segment_tokens if sid not in all_segments]
    if unknown_segments:
        violations.append(f"reviewer claims to have checked unknown segment ids: {unknown_segments}")
    unreviewed_segments = [sid for sid in all_segments if sid not in checked_segments]

    # A later omission cannot erase a previously rejected, unchanged claim.
    by_id = {str(claim.get("claim_id")): claim for claim in packet.claims}
    prior_by_id = {entry["finding_id"]: entry for entry in packet.prior_claim_findings}
    resolutions: Dict[str, str] = {}
    for resolution in raw.finding_resolutions:
        finding_id, reason = resolution.finding_id, resolution.reason
        prior = prior_by_id.get(finding_id)
        if prior is None or len(reason.strip()) < 12 or prior.get("claim_id") not in checked or not set(prior.get("finding", {}).get("evidence_refs", [])).issubset(available_evidence) or (all_segments and unreviewed_segments):
            protocol_errors.append(f"invalid or unreviewed finding resolution {finding_id!r}")
            violations.append(f"finding resolution {finding_id!r} lacks scoped recheck evidence")
        else:
            resolutions[finding_id] = "; ".join(dict.fromkeys([resolutions[finding_id], reason.strip()])) if finding_id in resolutions else reason.strip()
    for prior in packet.prior_claim_findings:
        cid = str(prior["claim_id"])
        claim = by_id.get(cid)
        if claim is not None and prior["finding_id"] not in resolutions and _claim_review_material(claim, packet.evidence_excerpts) == prior.get("material_hash"):
            if not any(f.claim_id == cid and f.severity in {"major", "blocking"} for f in findings):
                findings.append(ReviewFinding(**prior["finding"]))
    blocking_findings = [f for f in findings if f.severity in {"blocking", "major"}]
    notes: List[str] = []
    if packet.missing_evidence_ids:
        notes.append(f"claims cite evidence that was never archived: {packet.missing_evidence_ids}")
    if packet.scope_issues:
        notes.extend(packet.scope_issues)
    if packet.unbound_assertions:
        notes.append(f"{len(packet.unbound_assertions)} draft section(s) assert facts without citing claim or evidence ids")

    verdict = Verdict(raw.verdict)
    status = "passed"
    error_parts = [f"reviewer protocol error: {item}" for item in protocol_errors] + list(violations) + list(field_notes) + notes
    report_stage = packet.stage == ReviewStage.REPORT
    if verdict == Verdict.PASS:
        if violations:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            status = "inconclusive"
        elif packet.claim_ids and not checked:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            status = "inconclusive"
            error_parts.append("pass without any checked claim")
        elif blocking_findings:
            verdict = Verdict.REVISE
            status = "changes_requested"
            error_parts.append("pass with unresolved major/blocking findings is inconsistent; treated as revise")
        elif report_stage and (packet.missing_evidence_ids or packet.scope_issues or packet.unbound_assertions or blocking_findings):
            # R1: a report whose sources do not exist, whose claims were never built,
            # or whose sections cite nothing cannot be delivered as fully verified.
            citation_only = bool(packet.claim_ids and packet.unbound_assertions) and not packet.missing_evidence_ids and all("draft section(s) assert facts without citing" in issue for issue in packet.scope_issues)
            verdict = Verdict.REVISE if citation_only else Verdict.INSUFFICIENT_EVIDENCE
            status = "changes_requested" if citation_only else "inconclusive"
            error_parts.append("pass requires verifiable delivery: missing evidence, unbound assertions or unmet scope preconditions")
            for issue in packet.scope_issues:
                findings.append(
                    ReviewFinding(
                        issue_type="unsupported_claim",
                        severity="blocking",
                        explanation=issue,
                        required_action=(
                            "Write the report only from claims that carry an archived evidence id ([E<n>]); cite every factual sentence, "
                            "or state the missing part as an explicit unknown/limitation."
                        )
                    )
                )
            for item in packet.unbound_assertions:
                findings.append(
                    ReviewFinding(
                        segment_id=str(item["segment_id"]),
                        issue_type="unsupported_claim",
                        severity="major",
                        explanation=f"{item['segment_id']} asserts facts without citing a claim or evidence id: {item.get('excerpt', '')[:120]}",
                        required_action="Attach the archived evidence id that supports this section, or remove the assertion.",
                    )
                )
        elif report_stage and any(cid not in checked for cid in required_claims):
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            status = "inconclusive"
            error_parts.append(f"pass but {len([c for c in required_claims if c not in checked])} claim(s) were not checked")
        elif report_stage and unreviewed_segments:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            status = "inconclusive"
            error_parts.append(f"pass but {len(unreviewed_segments)} draft segment(s) were not read")
        elif uncovered_requirements:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            status = "inconclusive"
            error_parts.append(f"pass but mandatory requirements lack archived evidence: {uncovered_requirements}")
        elif not report_stage and packet.claim_ids and any(cid not in checked for cid in packet.claim_ids):
            violations_note = f"pass but {len(unchecked)} claim(s) were not checked"
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            status = "inconclusive"
            error_parts.append(violations_note)
    elif verdict in {Verdict.REVISE, Verdict.INSUFFICIENT_EVIDENCE, Verdict.STOP}:
        status = "changes_requested"
        if not findings:
            error_parts.append("non-pass verdict without actionable findings")
            status = "inconclusive"
        for finding in findings:
            if not finding.required_action.strip():
                finding.required_action = "Provide a specific, checkable required action."

    # S3: "the review finished" is a program fact, independent of the model's verdict.
    # A required batch that timed out, failed to parse or ran out of budget leaves part
    # of the draft unread; the records of the batches that did run are kept, but they
    # can never authorize a full delivery. This also covers the case where earlier
    # batches already produced a complete-looking ``pass``.
    review_complete = not error
    if error:
        if verdict == Verdict.PASS:
            verdict = Verdict.INSUFFICIENT_EVIDENCE
        if verdict != Verdict.REVISE:
            status = "unreviewed"
        findings.append(
            ReviewFinding(
                issue_type="other",
                severity="blocking",
                explanation=f"the review did not finish: {error}",
                required_action="Re-run the review once the failure clears, or deliver only the reviewed part and state the unreviewed scope.",
            )
        )
        error_parts.insert(0, f"review incomplete: {error}")
    return ReviewDecision(
        review_id=packet.review_id,
        stage=packet.stage,
        verdict=verdict,
        findings=findings,
        checked_claim_ids=checked,
        unchecked_claim_ids=unchecked,
        unchecked_requirement_ids=uncovered_requirements if report_stage else [],
        unreviewed_segment_ids=unreviewed_segments if report_stage else [],
        unbound_assertion_count=len(packet.unbound_assertions),
        unresolved_conflicts=list(raw.unresolved_conflicts or []),
        draft_hash=packet.draft_hash,
        packet_hash=packet.packet_hash,
        review_status=status,
        review_cost=review_cost,
        review_model_calls=review_model_calls,
        review_complete=review_complete,
        protocol_errors=list(protocol_errors),
        finding_resolutions=resolutions,
        error="; ".join(error_parts),
    )


def _record_claim_review_findings(store: ArtifactStore, packet: ReviewPacket, decision: ReviewDecision) -> None:
    """Close a finding only after changed material was independently rechecked."""
    claims = {str(claim.get("claim_id")): claim for claim in packet.claims}
    existing = store.claim_review_findings()
    current_findings = [f for f in decision.findings if f.severity in {"major", "blocking"}]
    can_close = decision.review_complete and not decision.protocol_errors
    for prior in existing:
        if prior.get("status") != "open":
            continue
        cid = str(prior["claim_id"])
        latest = store.get_claim(cid)
        withdrawn = latest is not None and latest.get("status") == "withdrawn"
        claim = claims.get(cid)
        rechecked = (
            can_close and claim is not None and cid in decision.checked_claim_ids
            and not any(f.claim_id == cid for f in current_findings)
            and (_claim_review_material(claim, packet.evidence_excerpts) != prior.get("material_hash") or prior["finding_id"] in decision.finding_resolutions)
        )
        if withdrawn or rechecked:
            store.append_ledger("claim_review_findings", {
                "finding_id": prior["finding_id"], "status": "resolved", "resolved_by": decision.review_id,
                "resolution": "claim withdrawn" if withdrawn else decision.finding_resolutions.get(prior["finding_id"], "changed claim/source material independently rechecked"),
                "packet_hash": decision.packet_hash,
            })
    for finding in current_findings:
        claim = claims.get(finding.claim_id)
        if claim is None:
            continue
        material_hash = _claim_review_material(claim, packet.evidence_excerpts)
        payload = finding.model_dump(mode="json")
        finding_id = "finding-" + hash_obj([finding.claim_id, material_hash, payload])[:16]
        if any(entry["finding_id"] == finding_id and entry.get("status") == "open" for entry in existing):
            continue
        store.append_ledger("claim_review_findings", {
            "finding_id": finding_id, "status": "open", "review_id": decision.review_id,
            "claim_id": finding.claim_id, "material_hash": material_hash, "finding": payload,
            "draft_hash": packet.draft_hash,
        })


def decision_summary(decision: ReviewDecision) -> str:
    """Render a decision as compact text for the supervisor's next request."""
    lines = [
        f"Reviewer decision ({decision.stage.value}, {decision.verdict.value}, status={decision.review_status}).",
        "This is an unverified opinion; fix what is actionable and keep contradicting evidence.",
    ]
    if decision.unchecked_claim_ids:
        lines.append(f"Claims that were NOT checked: {', '.join(decision.unchecked_claim_ids)}")
    if decision.findings:
        lines.append("Required actions:")
        for finding in decision.findings:
            target = finding.claim_id or finding.segment_id or finding.requirement_id or "general"
            lines.append(f"- [{finding.severity}] {target} ({finding.issue_type}): {finding.explanation} -> {finding.required_action}")
    if decision.unresolved_conflicts:
        lines.append(f"Unresolved conflicts: {', '.join(decision.unresolved_conflicts)}")
    if decision.error:
        lines.append(f"Reviewer validation note: {decision.error}")
    return "\n".join(lines)


def _split_text_half(text: str) -> tuple:
    """Cut text into two contiguous halves, preferring a newline near the middle."""
    if len(text) < 2:
        return text, ""
    middle = len(text) // 2
    low = max(1, middle - 200)
    high = min(len(text) - 1, middle + 200)
    newline = text.rfind("\n", low, high) if high > low else -1
    cut = newline + 1 if newline > 0 else middle
    cut = min(max(1, cut), len(text) - 1)
    return text[:cut], text[cut:]


def _restrict_checked_segments(out: RawReviewOutput, requested: Sequence[str]) -> RawReviewOutput:
    """Keep only the segment ids the request actually showed (S3, rule 4).

    A batch can only have read what it was sent. Claiming a segment id that this
    request did not contain is dropped, so a self-reported "checked everything" cannot
    cover a batch that never ran.
    """
    allowed = set(requested)
    claimed = list(dict.fromkeys(out.checked_segment_ids or []))
    kept = [sid for sid in claimed if sid in allowed]
    if len(kept) == len(claimed):
        return out
    return out.model_copy(update={"checked_segment_ids": kept})


async def run_review(
    *,
    packet: ReviewPacket,
    reviewer_identity: Identity,
    store: Optional[ArtifactStore],
    model: Any,
    config: Configuration,
    model_name: str,
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ReviewDecision:
    """Run one review in fresh request(s) and validate the result.

    Reviews are replayed from ``cache`` when the packet hash is unchanged: the same
    packet cannot be charged twice.

    Under ``context_policy=scoped`` the draft is packed into contiguous units that fit
    one request. Packing measures *the request that will be sent* (same messages, tools
    and response schema), never an average per-batch guess:

    * a segment larger than one request is split into contiguous character ranges
      (``S1.1``, ``S1.2`` ...); no character is dropped, and a parent segment counts as
      reviewed only when every sub-range was read;
    * if the fixed part of the request (role rules, claims, evidence, requirement
      matrix) already exceeds the budget, or the smallest reviewable piece cannot fit,
      no request is sent and the decision records the unreviewed scope (S6);
    * any batch failure (timeout, parse failure, budget stop) vetoes a full pass even
      when earlier batches produced output, and a batch may only claim the segment ids
      that request actually showed (S3);
    * once a batch returns ``stop`` the remaining batches are not started (S4).
    """
    from open_deep_research.context import BudgetExhausted, ContextBuilder, ModelCallBudget, assemble, role_rules
    from open_deep_research.utils import get_today_str

    if store is not None:
        store.write_review_artifact(packet.review_id, "packet", packet)
    if cache is not None and packet.packet_hash in cache:
        decision = ReviewDecision(**cache[packet.packet_hash]).model_copy(update={"review_id": packet.review_id, "review_model_calls": 0, "review_cost": 0.0})
        if store is not None:
            store.write_review_artifact(packet.review_id, "decision", decision)
            store.append_event({"type": "review_reused", "review_id": packet.review_id, "packet_hash": packet.packet_hash})
        return decision
    rules = role_rules("reviewer", date=get_today_str())
    builder = ContextBuilder(config, store)
    # The reviewer has its own call allowance (per attempt, including retries) on top
    # of the run-wide budget: both are reserved atomically before every attempt.
    reviewer_budget = ModelCallBudget(store, config)
    role_cap = max(0, int(getattr(config, "max_review_model_calls", 0) or 0)) or None
    scoped = str(getattr(config.context_policy, "value", config.context_policy)) == "scoped"

    original_segments = [dict(segment) for segment in (packet.report_segments or [])]
    parent_order = [str(segment.get("segment_id")) for segment in original_segments]
    units: List[Dict[str, Any]] = [
        {
            "segment_id": sid,
            "text": segment.get("text") or "",
            "char_start": int(segment.get("char_start") or 0),
            "char_end": int(segment.get("char_end") or 0),
            "parent": sid,
        }
        for segment, sid in zip(original_segments, parent_order)
    ]

    def build_messages(candidate: ReviewPacket) -> List[BaseMessage]:
        return [SystemMessage(content=rules), HumanMessage(content=render_packet(candidate))]

    def items_for(candidate: ReviewPacket) -> List[Dict[str, Any]]:
        return [
            {"kind": "role", "ref": "roles/reviewer.md", "reason": "role_policy"},
            {"kind": "review_packet", "ref": candidate.review_id, "reason": "review_scope"},
        ]

    def measure(candidate: ReviewPacket) -> Any:
        return builder.build(
            reviewer_identity,
            "reviewer",
            build_messages(candidate),
            response_schema=RawReviewOutput,
            items=items_for(candidate),
            model=model_name,
            declared_max_tokens=config.reviewer_model_max_tokens,
        )

    def candidate_for(segment_pool: Sequence[Dict[str, Any]], selected: Sequence[str]) -> ReviewPacket:
        """A packet copy that shows exactly ``selected`` out of ``segment_pool``."""
        candidate = packet.model_copy(
            update={
                "report_segments": [dict(segment) for segment in segment_pool],
                "report_sections": [
                    {"title": segment["segment_id"], "segment_id": segment["segment_id"], "text": segment.get("text") or ""}
                    for segment in segment_pool
                ],
                "review_segments": list(selected),
            }
        )
        candidate.packet_hash = packet.packet_hash
        return candidate

    def finish(decision: ReviewDecision, *, execution_error: str, batches: int) -> ReviewDecision:
        if store is not None:
            _record_claim_review_findings(store, packet, decision)
            store.write_review_artifact(packet.review_id, "decision", decision)
            store.append_event(
                {
                    "type": "review",
                    "review_id": packet.review_id,
                    "stage": packet.stage.value,
                    "verdict": decision.verdict.value,
                    "status": decision.review_status,
                    "error": decision.error,
                    "batches": batches,
                    "calls": decision.review_model_calls,
                    "complete": decision.review_complete,
                }
            )
        if cache is not None and not execution_error:
            cache[packet.packet_hash] = decision.model_dump(mode="json")
        obs.record_event(
            "review_decision",
            review_id=packet.review_id,
            review_stage=packet.stage.value,
            verdict=str(getattr(decision.verdict, "value", decision.verdict)),
            complete=bool(decision.review_complete),
            checked_count=len(getattr(decision, "checked_claim_ids", []) or []),
            unchecked_count=len(getattr(decision, "unchecked_requirement_ids", []) or []),
            batch_count=batches,
        )
        return decision

    def refuse(reason: str) -> ReviewDecision:
        """Return "not reviewed" without sending anything (the budget cannot fit it)."""
        if store is not None:
            store.append_event({"type": "review_not_sent", "review_id": packet.review_id, "reason": reason})
        return finish(validate_decision(None, packet, error=reason, review_model_calls=0), execution_error=reason, batches=0)

    unchanged = [
        entry for entry in packet.prior_claim_findings
        if not (packet.draft_hash and entry.get("draft_hash") and packet.draft_hash != entry["draft_hash"])
        if any(str(claim.get("claim_id")) == entry["claim_id"] and _claim_review_material(claim, packet.evidence_excerpts) == entry["material_hash"] for claim in packet.claims)
    ]
    if unchanged:
        # No new paid opinion can close an unchanged rejected claim. Return the
        # existing required actions with no claimed checks and no provider call.
        raw = RawReviewOutput(verdict="revise", summary="Previously rejected claim/source material is unchanged.", findings=[ReviewFinding(**entry["finding"]) for entry in unchanged])
        return finish(validate_decision(raw, packet, review_model_calls=0), execution_error="", batches=0)

    batches: List[List[str]] = []
    budget = 0
    if not units:
        # Non-report stages and empty drafts carry no draft segments: one request.
        batches = [[]]
    elif not scoped:
        batches = [[unit["segment_id"] for unit in units]]
    else:
        fixed = measure(candidate_for([], []))
        budget = fixed.budget_tokens
        if fixed.estimated_input_tokens > budget:
            return refuse(
                f"reviewer context budget {budget} cannot fit the fixed part of the review request "
                f"({fixed.estimated_input_tokens} estimated input tokens); no request was sent"
            )
        # Split any segment that cannot fit one request on its own, keeping contiguous
        # character ranges so the text can be reassembled exactly.
        while True:
            oversized = next(
                (
                    unit
                    for unit in units
                    if measure(candidate_for(units, [unit["segment_id"]])).estimated_input_tokens > budget
                ),
                None,
            )
            if oversized is None:
                break
            if len(oversized["text"]) <= _REVIEW_MIN_SEGMENT_CHARS:
                return refuse(
                    f"review segment {oversized['segment_id']} cannot fit the reviewer context budget {budget} "
                    f"({len(oversized['text'])} characters); no request was sent"
                )
            left, right = _split_text_half(oversized["text"])
            offset = oversized["char_start"]
            replacements: List[Dict[str, Any]] = []
            for part_index, part in enumerate((left, right), start=1):
                replacements.append(
                    {
                        "segment_id": f"{oversized['segment_id']}.{part_index}",
                        "text": part,
                        "char_start": offset,
                        "char_end": offset + len(part),
                        "parent": oversized["parent"],
                    }
                )
                offset += len(part)
            index = units.index(oversized)
            units[index : index + 1] = replacements
        # Greedy packing, in draft order, over measured candidates.
        current: List[Dict[str, Any]] = []
        for unit in units:
            trial = current + [unit]
            if current and measure(candidate_for(units, [item["segment_id"] for item in trial])).estimated_input_tokens > budget:
                batches.append([item["segment_id"] for item in current])
                current = [unit]
            else:
                current = trial
        if current:
            batches.append([item["segment_id"] for item in current])

    segments_by_parent: Dict[str, List[str]] = {}
    for unit in units:
        segments_by_parent.setdefault(unit["parent"], []).append(unit["segment_id"])

    raw_outputs: List[RawReviewOutput] = []
    execution_error = ""
    calls = 0
    for index, batch in enumerate(batches, start=1):
        candidate = candidate_for(units, batch)
        ctx = measure(candidate)
        if ctx.estimated_input_tokens > ctx.budget_tokens:
            if store is not None:
                store.append_event(
                    {
                        "type": "review_batch_over_budget",
                        "review_id": packet.review_id,
                        "batch": index,
                        "batches": len(batches),
                        "segment_ids": batch,
                        "estimated_input_tokens": ctx.estimated_input_tokens,
                        "budget_tokens": ctx.budget_tokens,
                        "counting_method": ctx.manifest.counting_method if ctx.manifest is not None else "",
                    }
                )
            if scoped:
                execution_error = (
                    f"review batch {index} of {len(batches)} was not sent: {ctx.estimated_input_tokens} estimated "
                    f"input tokens exceed the reviewer context budget {ctx.budget_tokens}"
                )
                break
        batch_span = obs.span(
            "review.batch",
            kind=obs.SPAN_REVIEW,
            attributes={
                "review_id": packet.review_id,
                "review_stage": packet.stage.value,
                "batch_index": index,
                "batch_count": len(batches),
                "role": "reviewer",
                "agent_instance_id": reviewer_identity.agent_instance_id,
            },
        )
        try:
            # A reviewer answer that does not satisfy the decision schema is a bad
            # response, not a negative verdict: retry it once (each attempt is ledgered)
            # instead of turning an empty object into "insufficient evidence" and
            # finalizing a partial run on a parsing accident (seen in the F-D run).
            for attempt_index in range(2):
                calls += 1
                with batch_span:
                    out = await asyncio.wait_for(
                        assemble(
                            model,
                            ctx,
                            store=store,
                            record_requests=config.record_requests,
                            max_tokens=config.reviewer_model_max_tokens,
                            model_name=model_name,
                            retries=config.max_structured_output_retries,
                            budget=reviewer_budget,
                            role_cap=role_cap,
                        ),
                        timeout=config.review_timeout_seconds,
                    )
                    batch_span.set(verdict=str(getattr(getattr(out, "verdict", None), "value", "") or ""), checked_count=len(getattr(out, "checked_segment_ids", []) or []))
                try:
                    if isinstance(out, dict):
                        out = RawReviewOutput(**out)
                    elif not isinstance(out, RawReviewOutput):
                        out = RawReviewOutput.model_validate(json.loads(str(getattr(out, "content", "")) or "{}"))
                    break
                except (ValidationError, json.JSONDecodeError) as exc:
                    if store is not None:
                        store.append_event(
                            {
                                "type": "review_output_invalid",
                                "review_id": packet.review_id,
                                "batch": index,
                                "attempt": attempt_index + 1,
                                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                            }
                        )
                    if attempt_index:
                        raise
        except asyncio.TimeoutError:
            batch_span.fail(error_class="TimeoutError", status=obs.STATUS_TIMEOUT)
            execution_error = f"review timed out (batch {index} of {len(batches)})"
            break
        except BudgetExhausted as exc:
            # A refused allowance is a budget stop, not a provider failure: record it
            # as such (the attempts that did run are already in the ledger).
            batch_span.fail(error_class=type(exc).__name__, status=obs.STATUS_BUDGET_DENIED)
            execution_error = f"review stopped by the budget: {exc.reason} (batch {index} of {len(batches)})"
            break
        except Exception as exc:  # noqa: BLE001 - any failure must not become a pass
            batch_span.fail(error_class=type(exc).__name__)
            execution_error = f"review failed: {type(exc).__name__}: {str(exc)[:200]}"
            break
        except asyncio.CancelledError:
            # The caller stopped us: the attempt is counted and the cancellation is
            # propagated untouched.
            raise
        claimed = list(dict.fromkeys(out.checked_segment_ids or []))
        out = _restrict_checked_segments(out, batch)
        if len(out.checked_segment_ids or []) != len(claimed) and store is not None:
            store.append_event(
                {
                    "type": "review_ignored_unrequested_segments",
                    "review_id": packet.review_id,
                    "batch": index,
                    "claimed_not_shown": [sid for sid in claimed if sid not in set(batch)],
                }
            )
        raw_outputs.append(out)
        if out.verdict == Verdict.STOP and index < len(batches):
            # A stop is a decision to end the run: do not spend further review calls on
            # batches that can no longer change the outcome (S4).
            if store is not None:
                store.append_event(
                    {
                        "type": "review_stopped_early",
                        "review_id": packet.review_id,
                        "batch": index,
                        "skipped_batches": len(batches) - index,
                        "skipped_segment_ids": [sid for rest in batches[index:] for sid in rest],
                    }
                )
            break

    # Map reviewed units back to the packet's own segment ids. Coverage is judged over
    # the whole review (a split segment counts only when every contiguous sub-range was
    # read in some batch), while an id no request ever showed stays unknown and is
    # reported by ``validate_decision``.
    unit_ids = {unit["segment_id"] for unit in units}
    claimed_units = {sid for out in raw_outputs for sid in (out.checked_segment_ids or [])}
    covered = [
        parent
        for parent in parent_order
        if segments_by_parent.get(parent) and all(sid in claimed_units for sid in segments_by_parent[parent])
    ]
    extras = [sid for sid in claimed_units if sid not in unit_ids and sid not in covered]
    translated = [out.model_copy(update={"checked_segment_ids": []}) for out in raw_outputs]
    merged = merge_review_outputs(translated) if translated else None
    if merged is not None:
        merged = merged.model_copy(update={"checked_segment_ids": covered + extras})
    decision = validate_decision(merged, packet, error=execution_error, review_model_calls=calls)
    return finish(decision, execution_error=execution_error, batches=len(batches))


_VERDICT_PRIORITY = {"stop": 4, "insufficient_evidence": 3, "revise": 2, "pass": 1}


def merge_review_outputs(outputs: Sequence[RawReviewOutput]) -> RawReviewOutput:
    """Merge per-batch opinions into one pessimistic, order-independent opinion.

    The merged verdict keeps the strongest signal any batch produced, using the
    explicit priority ``stop > insufficient_evidence > revise > pass`` (S4). A ``stop``
    is a request to end the run, so it is never downgraded into "keep revising"; the
    batches' summaries, findings and scope records are all kept, which keeps the stop
    reason and the unreviewed range visible.

    Checked claims / segments / requirements are the union of what the batches say they
    checked (the program still filters them), and explicit-unknown requirement ids are
    merged the same way instead of being dropped (S5).
    """
    if len(outputs) == 1:
        return outputs[0]

    def union(name: str) -> List[str]:
        seen: List[str] = []
        for out in outputs:
            for value in getattr(out, name) or []:
                if value not in seen:
                    seen.append(value)
        return seen

    verdict = max((out.verdict for out in outputs), key=lambda value: _VERDICT_PRIORITY.get(value, 0))
    return RawReviewOutput(
        verdict=verdict,
        summary=" | ".join(out.summary for out in outputs if out.summary),
        findings=[finding for out in outputs for finding in (out.findings or [])],
        checked_claim_ids=union("checked_claim_ids"),
        checked_requirement_ids=union("checked_requirement_ids"),
        checked_segment_ids=union("checked_segment_ids"),
        acknowledged_unknown_requirement_ids=union("acknowledged_unknown_requirement_ids"),
        unresolved_conflicts=union("unresolved_conflicts"),
        finding_resolutions=[entry for out in outputs for entry in out.finding_resolutions],
    )


def issue_revalidation_receipts(
    *,
    store: Optional[ArtifactStore],
    decision: ReviewDecision,
    packet: ReviewPacket,
    checker_identity: str,
    report_version: int = 0,
) -> List[Dict[str, Any]]:
    """Issue receipts for correction events this review actually re-checked.

    Recovery opens an event; only an independent check can close it. The reviewer's
    ``checked_claim_ids`` is the evidence that a claim was read, and this function turns
    that into a receipt bound to the packet, the claim hashes and the evidence hashes.
    An unchecked claim, an incomplete review or a contradicted claim is recorded as
    such; none of them is silently upgraded to "verified".
    """
    if store is None or not decision.review_complete or not decision.packet_hash:
        return []
    claims = {str(claim.get("claim_id")): claim for claim in store.active_claims()}
    findings: Dict[str, List[ReviewFinding]] = {}
    for finding in decision.findings:
        if finding.claim_id:
            findings.setdefault(finding.claim_id, []).append(finding)
    issued: List[Dict[str, Any]] = []
    for event in store.open_revalidations():
        affected = {str(rid) for rid in (event.get("affected_requirement_ids") or [])}
        if not affected:
            continue
        task_id = str(event.get("task_id") or "")
        relevant = [
            claim
            for claim in claims.values()
            if affected.intersection(str(rid) for rid in (claim.get("requirement_ids") or []))
            and (not task_id or str(claim.get("_owner_task") or "") == task_id)
        ]
        if not relevant:
            continue
        verdicts: Dict[str, str] = {}
        for claim in relevant:
            cid = str(claim.get("claim_id"))
            if cid not in set(decision.checked_claim_ids):
                verdicts[cid] = "insufficient_evidence"
                continue
            blocking = [f for f in findings.get(cid, []) if f.severity in {"blocking", "major"}]
            verdicts[cid] = "verified_contradicted" if blocking else "verified_supported"
        evidence_ids = sorted({ref for claim in relevant for ref in (claim.get("evidence_refs") or [])})
        snapshot = store.evidence_snapshot(evidence_ids)
        receipt = RevalidationReceipt(
            receipt_id=f"receipt-{decision.review_id}-{event.get('event_id')}",
            event_id=str(event.get("event_id")),
            review_id=decision.review_id,
            packet_hash=decision.packet_hash,
            task_id=task_id,
            claim_ids=sorted(verdicts),
            claim_hashes={cid: claim_digest(claim) for cid, claim in claims.items() if cid in verdicts},
            claim_versions={cid: int(claims[cid].get("version", 1) or 1) for cid in verdicts},
            requirement_ids=sorted(affected),
            evidence_ids=[item["evidence_id"] for item in snapshot],
            evidence_hashes={item["evidence_id"]: item["content_hash"] for item in snapshot},
            evidence_versions={item["evidence_id"]: item["document_version"] for item in snapshot},
            verdicts=verdicts,
            report_version=report_version,
            checker_identity=checker_identity,
            checker_kind="independent_reviewer",
            created_at=utc_now(),
        )
        issued.append(store.record_revalidation_receipt(receipt))
    return issued


def apply_review(
    decision: ReviewDecision,
    *,
    current_hash: str,
    remaining_budget: Dict[str, Any],
    revision_rounds: int,
    max_revision_rounds: int,
    stage: ReviewStage,
) -> str:
    """Turn a validated decision into a routing action (program-owned).

    Returns one of ``continue_research | rewrite_report | finalize | finalize_partial``.

    A stale review (draft changed after review) or an unfinished review (a batch that
    timed out, failed to parse or ran out of budget) can never authorize ``finalize``.
    At the report stage the verdict distinguishes the remedies:

    * ``revise`` -> rewrite the report;
    * ``insufficient_evidence`` -> return to research while iterations/budget remain;
    * ``stop`` or an exhausted budget/revision cap -> limited delivery.

    Non-report stages are about research progress, so ``revise`` and
    ``insufficient_evidence`` return to the supervisor; ``pass`` hands off to the
    report phase and ``stop`` delivers what exists as a limited report.
    """
    budget_open = bool(remaining_budget.get("open", True))
    iterations_left = int(remaining_budget.get("iterations_left", 0) or 0)
    revisions_left = revision_rounds < max_revision_rounds

    if decision.protocol_errors:
        # A malformed reviewer answer is not evidence that research is missing. Do
        # not spend a research/repair turn on it, and never let it authorize delivery.
        return "finalize_partial"

    if not decision.review_complete:
        # S3: an unfinished review cannot authorize a delivery, even when the partial
        # verdict said ``pass``.
        if stage == ReviewStage.REPORT:
            return "rewrite_report" if budget_open and revisions_left else "finalize_partial"
        return "continue_research" if budget_open and revisions_left else "finalize_partial"

    if decision.draft_hash and current_hash and decision.draft_hash != current_hash:
        # The reviewed text is no longer the text that would be delivered.
        if stage == ReviewStage.REPORT:
            return "rewrite_report" if budget_open and revisions_left else "finalize_partial"
        return "continue_research" if budget_open and revisions_left else "finalize_partial"

    if decision.verdict == Verdict.PASS and decision.review_status == "passed":
        if not budget_open:
            return "finalize_partial"
        # Defense in depth: a decision cannot authorize a full delivery while it
        # still records unreviewed scope (checked claim / segment / requirement gaps).
        if stage == ReviewStage.REPORT and (
            decision.unchecked_claim_ids
            or decision.unreviewed_segment_ids
            or decision.unchecked_requirement_ids
        ):
            return "finalize_partial"
        return "finalize"

    if decision.verdict == Verdict.STOP:
        return "finalize_partial"
    if not budget_open:
        return "finalize_partial"
    if not revisions_left:
        return "finalize_partial"

    if stage == ReviewStage.REPORT:
        if decision.verdict == Verdict.INSUFFICIENT_EVIDENCE:
            # Missing evidence that the research phase can still gather: go back to
            # research rather than asking the writer to rephrase the same gap.
            return "continue_research" if iterations_left > 0 else "finalize_partial"
        return "rewrite_report"
    return "continue_research"
