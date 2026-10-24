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
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from open_deep_research.artifacts import (
    ArtifactStore,
    Identity,
    hash_obj,
)
from open_deep_research.configuration import Configuration
from open_deep_research.state import (
    RawReviewOutput,
    ReviewDecision,
    ReviewFinding,
    ReviewPacket,
    ReviewStage,
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
    """Split a draft into paragraph blocks that keep their heading.

    Blocks are the unit for the "unbound assertion" check: a short factual paragraph
    written after a cited section must not borrow that section's citations.
    """
    blocks: List[Dict[str, Any]] = []
    for raw in re.split(r"\n\s*\n", draft or ""):
        block = raw.strip()
        if block:
            blocks.append({"index": len(blocks) + 1, "text": block})
    return blocks


_REFERENCE_LINE = re.compile(r"^\s*(\[\d+\]|\d+\.|-|\*)\s+\S")
_REFERENCE_HEADING = re.compile(r"^#{1,6}\s*(sources?|references?|bibliography|citations?|footnotes?)\b", re.IGNORECASE)
_URL_ONLY = re.compile(r"^\s*<?https?://\S+>?\s*$")


def _is_reference_block(text: str) -> bool:
    """Return True for a sources/references list: pointers, not factual assertions."""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return True
    if len(lines) == 1 and _REFERENCE_HEADING.match(lines[0].strip()):
        return True
    body = [line for line in lines if not line.lstrip().startswith("#")]
    if not body:
        return True
    return all(_REFERENCE_LINE.match(line) or _URL_ONLY.match(line) for line in body)


def _assertions(text: str) -> int:
    """Count sentence-like claims in a draft section (structural, not semantic)."""
    if _is_reference_block(text):
        return 0
    sentences = [part for part in re.split(r"(?<=[.!?])\s+|\n+", text or "") if len(part.strip()) >= 25]
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
        assertions = _assertions(text)
        if not assertions:
            continue
        # Headings and bullet labels carry the citations of their body: judge the text.
        if any(cid and cid in text for cid in known_claim_ids) or _CITATION_RE.search(text):
            continue
        unbound.append({"segment_id": f"P{block['index']}", "assertions": assertions, "excerpt": text[:200]})
    return unbound


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
    excerpts = store.read_evidence_for_review(evidence_ids) if store is not None else []
    found = {entry["evidence_id"] for entry in excerpts}
    missing = [eid for eid in evidence_ids if eid not in found]
    open_issues: List[str] = []
    if missing:
        open_issues.append(f"Referenced evidence not archived: {', '.join(missing)}")
    ledger = task_ledger or {}
    coverage_matrix = []
    for requirement in getattr(brief, "requirements", []) or []:
        rid = requirement.requirement_id
        covering = [c.get("claim_id") for c in claims if rid in (c.get("requirement_ids") or [])]
        covering_evidence = sorted({r for c in claims if rid in (c.get("requirement_ids") or []) for r in (c.get("evidence_refs") or [])})
        available = [eid for eid in covering_evidence if eid in found]
        coverage_matrix.append(
            {
                "requirement_id": rid,
                "statement": requirement.statement,
                "mandatory": bool(requirement.mandatory),
                "claim_ids": covering,
                "evidence_ids": covering_evidence,
                "available_evidence_ids": available,
                "state": "covered" if available else "missing",
            }
        )
    for task_id, entry in ledger.items():
        if entry.get("status") in {"failed", "partial"}:
            open_issues.append(f"Task {task_id} ended {entry.get('status')}: {(entry.get('error') or 'no result')[:200]}")
    claim_ids = [c.get("claim_id") for c in claims if c.get("claim_id")]
    segments = _segments(draft) if draft else []
    unbound = _unbound_assertions(draft, claims) if draft else []
    scope_issues: List[str] = []
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
        claims=[
            {
                "claim_id": c.get("claim_id"),
                "statement": c.get("statement"),
                "kind": c.get("kind", "fact"),
                "evidence_refs": c.get("evidence_refs", []),
                "requirement_ids": c.get("requirement_ids", []),
                "qualifiers": c.get("qualifiers", []),
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
        draft_assertion_count=sum(_assertions(s.get("text") or "") for s in segments) if draft else 0,
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
        for claim in packet.claims:
            lines.append(
                f"- {claim.get('claim_id')} ({claim.get('kind')}) evidence={claim.get('evidence_refs')} "
                f"requirements={claim.get('requirement_ids')} qualifiers={claim.get('qualifiers')}: {claim.get('statement')}"
            )
        lines.append("</Claims>")
    lines.append("<Evidence>")
    for entry in packet.evidence_excerpts:
        counter = " COUNTER-EVIDENCE" if entry.get("is_counter_evidence") else ""
        lines.append(
            f"- {entry['evidence_id']}{counter} {entry.get('title', '')} <{entry.get('url', '')}> retrieved {entry.get('retrieved_at', '')}\n"
            f"  EXCERPT: {entry.get('exact_excerpt', '')}"
        )
    lines.append("</Evidence>")
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

    def base(verdict: Verdict, status: str, err: str = "", note: str = "") -> ReviewDecision:
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
            error="; ".join(part for part in (err, note) if part),
        )

    if raw is None:
        return base(Verdict.INSUFFICIENT_EVIDENCE, "unreviewed", error or "no reviewer output")

    available_evidence = set(packet.available_evidence_ids or packet.evidence_ids)
    claimed_evidence = set(packet.evidence_ids)
    allowed_claims = set(packet.claim_ids)
    findings: List[ReviewFinding] = []
    violations: List[str] = []
    for finding in raw.findings or []:
        refs = list(finding.evidence_refs or [])
        bogus = [ref for ref in refs if ref not in claimed_evidence]
        if bogus:
            violations.append(f"finding cites non-existent evidence ids: {bogus}")
            refs = [ref for ref in refs if ref in claimed_evidence]
        if finding.claim_id and finding.claim_id not in allowed_claims:
            violations.append(f"finding cites unknown claim id {finding.claim_id!r}")
        issue_type = finding.issue_type if finding.issue_type in ISSUE_TYPES else "other"
        findings.append(finding.model_copy(update={"evidence_refs": refs, "issue_type": issue_type}))

    checked = [cid for cid in dict.fromkeys(raw.checked_claim_ids or []) if cid in allowed_claims]
    unknown_checked = [cid for cid in (raw.checked_claim_ids or []) if cid not in allowed_claims]
    if unknown_checked:
        violations.append(f"reviewer claims to have checked unknown ids: {unknown_checked}")
    unchecked = [cid for cid in packet.claim_ids if cid not in checked]

    checked_requirements = [rid for rid in dict.fromkeys(raw.checked_requirement_ids or []) if str(rid) in requirement_states]
    unknown_requirements = [rid for rid in (raw.checked_requirement_ids or []) if str(rid) not in requirement_states]
    if unknown_requirements:
        violations.append(f"reviewer claims to have verified unknown requirement ids: {unknown_requirements}")
    acknowledged = [
        rid
        for rid in dict.fromkeys(raw.acknowledged_unknown_requirement_ids or [])
        if str(rid) in requirement_states and requirement_states.get(str(rid)) != "covered"
    ]
    unknown_acknowledged = [rid for rid in (raw.acknowledged_unknown_requirement_ids or []) if str(rid) not in requirement_states]
    if unknown_acknowledged:
        violations.append(f"reviewer acknowledges unknown requirement ids: {unknown_acknowledged}")
    uncovered_requirements = [
        rid for rid in mandatory_requirement_ids if requirement_states.get(rid) != "covered" and rid not in acknowledged
    ]

    checked_segments = [sid for sid in dict.fromkeys(raw.checked_segment_ids or []) if sid in all_segments]
    unknown_segments = [sid for sid in (raw.checked_segment_ids or []) if sid not in all_segments]
    if unknown_segments:
        violations.append(f"reviewer claims to have checked unknown segment ids: {unknown_segments}")
    unreviewed_segments = [sid for sid in all_segments if sid not in checked_segments]

    blocking_findings = [f for f in findings if f.severity == "blocking"]
    notes: List[str] = []
    if packet.missing_evidence_ids:
        notes.append(f"claims cite evidence that was never archived: {packet.missing_evidence_ids}")
    if packet.scope_issues:
        notes.extend(packet.scope_issues)
    if packet.unbound_assertions:
        notes.append(f"{len(packet.unbound_assertions)} draft section(s) assert facts without citing claim or evidence ids")

    verdict = Verdict(raw.verdict)
    status = "passed"
    error_parts = list(violations) + notes
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
            error_parts.append("pass with blocking findings is inconsistent; treated as revise")
        elif report_stage and (packet.missing_evidence_ids or packet.scope_issues or packet.unbound_assertions or blocking_findings):
            # R1: a report whose sources do not exist, whose claims were never built,
            # or whose sections cite nothing cannot be delivered as fully verified.
            verdict = Verdict.INSUFFICIENT_EVIDENCE
            status = "inconclusive"
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
        error="; ".join(error_parts),
    )


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
            target = finding.claim_id or finding.requirement_id or "general"
            lines.append(f"- [{finding.severity}] {target} ({finding.issue_type}): {finding.explanation} -> {finding.required_action}")
    if decision.unresolved_conflicts:
        lines.append(f"Unresolved conflicts: {', '.join(decision.unresolved_conflicts)}")
    if decision.error:
        lines.append(f"Reviewer validation note: {decision.error}")
    return "\n".join(lines)


def trim_packet_for_budget(packet: ReviewPacket, excess_tokens: int) -> ReviewPacket:
    """Shrink excerpts to fit the reviewer budget, recording what was cut.

    Evidence excerpts may be shortened (the reviewer still sees the id, URL and the
    head of the source). The draft is different: a pass needs the *whole* draft, so a
    draft segment that no longer fits is removed **and listed by id** as unreviewed.
    A decision built from a trimmed packet can therefore never report full coverage.
    """
    payload = packet.model_dump(mode="json")
    budget_chars = max(1200, max(800, 4000 - max(0, excess_tokens)) * 4)
    per_item = max(200, budget_chars // max(1, len(payload.get("evidence_excerpts") or [])))
    payload["evidence_excerpts"] = [
        {**entry, "exact_excerpt": (entry.get("exact_excerpt") or "")[:per_item]} for entry in payload.get("evidence_excerpts") or []
    ]
    remaining = budget_chars
    kept: List[Dict[str, Any]] = []
    dropped: List[str] = []
    for segment in payload.get("report_segments") or []:
        text = segment.get("text") or ""
        if kept and len(text) > remaining:
            # Dropping is recorded, never silent: the id goes to ``unreviewed`` below.
            dropped.append(str(segment.get("segment_id")))
            continue
        kept.append(segment)
        remaining = max(0, remaining - len(text))
    payload["report_segments"] = kept
    payload["report_sections"] = [{"title": seg.get("segment_id"), "segment_id": seg.get("segment_id"), "text": seg.get("text")} for seg in kept]
    payload["proposed_action"] = (payload.get("proposed_action") or "")[:2000]
    payload["open_issues"] = list(payload.get("open_issues") or []) + [
        "Some evidence excerpts were shortened to fit the reviewer budget.",
        *([f"Draft segment(s) {', '.join(dropped)} exceed the reviewer budget and are NOT reviewed."] if dropped else []),
    ]
    return ReviewPacket(**payload)


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
    """Run one review in a fresh request and validate the result.

    Reviews are replayed from ``cache`` when the packet hash is unchanged: the same
    packet cannot be charged twice. When the whole packet does not fit the reviewer's
    request budget it is reviewed in contiguous batches of draft segments; the merged
    decision reports exactly the segments, claims and requirements that were covered,
    so a partial review can never authorize a full delivery.
    """
    from open_deep_research.context import BudgetExhausted, ContextBuilder, ModelCallBudget, assemble, role_rules
    from open_deep_research.utils import get_today_str

    if cache is not None and packet.packet_hash in cache:
        cached = cache[packet.packet_hash]
        return ReviewDecision(**cached)
    if store is not None:
        store.write_review_artifact(packet.review_id, "packet", packet)
    rules = role_rules("reviewer", date=get_today_str())
    builder = ContextBuilder(config, store)
    # The reviewer has its own call allowance (per attempt, including retries) on top
    # of the run-wide budget: both are reserved atomically before every attempt.
    reviewer_budget = ModelCallBudget(store, config)
    role_cap = max(0, int(getattr(config, "max_review_model_calls", 0) or 0)) or None
    scoped = str(getattr(config.context_policy, "value", config.context_policy)) == "scoped"

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

    # Split the packet into contiguous draft batches that each fit one request.
    batches: List[List[str]] = [[str(seg.get("segment_id")) for seg in (packet.report_segments or [])]]
    segments_by_id = {str(seg.get("segment_id")): seg for seg in (packet.report_segments or [])}
    packet_ctx = measure(packet)
    if scoped and packet_ctx.estimated_input_tokens > packet_ctx.budget_tokens and len(batches[0]) > 1:
        probes = [measure(packet.model_copy(update={"review_segments": [sid]})) for sid in (segments_by_id[s] for s in batches[0])]
        segment_cost = int(sum(max(1, p.estimated_input_tokens) for p in probes) / len(probes))
        base_cost = int(max(0, packet_ctx.estimated_input_tokens - segment_cost * len(batches[0])))
        per_batch = max(1, (packet_ctx.budget_tokens - base_cost) // max(1, segment_cost))
        batches = [batches[0][i : i + per_batch] for i in range(0, len(batches[0]), per_batch)]
        packet_ctx = measure(packet)
    elif scoped and packet_ctx.estimated_input_tokens > packet_ctx.budget_tokens:
        packet_ctx = measure(trim_packet_for_budget(packet, packet_ctx.estimated_input_tokens - packet_ctx.budget_tokens))

    raw_outputs: List[RawReviewOutput] = []
    error = ""
    calls = 0
    for index, batch in enumerate(batches, start=1):
        candidate = packet.model_copy(update={"review_segments": batch}) if batch else packet
        candidate.packet_hash = packet.packet_hash
        ctx = measure(candidate)
        if scoped and ctx.estimated_input_tokens > ctx.budget_tokens and store is not None:
            store.append_event({"type": "review_batch_over_budget", "review_id": packet.review_id, "batch": index, "batches": len(batches), "segment_ids": batch})
        try:
            calls += 1
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
            if isinstance(out, dict):
                out = RawReviewOutput(**out)
            elif not isinstance(out, RawReviewOutput):
                out = RawReviewOutput.model_validate(json.loads(str(getattr(out, "content", "")) or "{}"))
            raw_outputs.append(out)
        except asyncio.TimeoutError:
            error = f"review timed out (batch {index} of {len(batches)})"
            break
        except BudgetExhausted as exc:
            # A refused allowance is a budget stop, not a provider failure: record it
            # as such (the attempts that did run are already in the ledger).
            error = f"review stopped by the budget: {exc.reason} (batch {index} of {len(batches)})"
            break
        except Exception as exc:  # noqa: BLE001 - any failure must not become a pass
            error = f"review failed: {type(exc).__name__}: {str(exc)[:200]}"
            break
        except asyncio.CancelledError:
            # The caller stopped us (e.g. a timeout around the whole review): the
            # attempt is counted and the cancellation is propagated untouched.
            raise
    merged = merge_review_outputs(raw_outputs) if raw_outputs else None
    decision = validate_decision(merged, packet, error=error, review_model_calls=calls)
    if store is not None:
        store.write_review_artifact(packet.review_id, "decision", decision)
        store.append_event({"type": "review", "review_id": packet.review_id, "stage": packet.stage.value, "verdict": decision.verdict.value, "status": decision.review_status, "error": decision.error, "batches": len(batches), "calls": calls})
    if cache is not None and not error:
        cache[packet.packet_hash] = decision.model_dump(mode="json")
    return decision


def merge_review_outputs(outputs: Sequence[RawReviewOutput]) -> RawReviewOutput:
    """Merge the per-batch reviewer opinions into one (pessimistic) combined opinion.

    The merged verdict is a candidate the program still validates: claims / segments /
    requirements are the union of what the reviewer says it checked, findings are
    concatenated. The verdict is ``pass`` only when every batch passed.
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

    verdicts = [out.verdict for out in outputs]
    verdict = verdicts[0] if all(v == verdicts[0] for v in verdicts) else "revise"
    if "insufficient_evidence" in verdicts and verdict != "pass":
        verdict = "insufficient_evidence"
    return RawReviewOutput(
        verdict=verdict,
        summary=" | ".join(out.summary for out in outputs if out.summary),
        findings=[finding for out in outputs for finding in (out.findings or [])],
        checked_claim_ids=union("checked_claim_ids"),
        checked_requirement_ids=union("checked_requirement_ids"),
        checked_segment_ids=union("checked_segment_ids"),
        unresolved_conflicts=union("unresolved_conflicts"),
    )


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

    A stale review (draft changed after review) can never authorize ``finalize``.
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
