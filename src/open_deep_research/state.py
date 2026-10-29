"""Graph state definitions and data structures for the Deep Research agent.

Besides the LangGraph message states, this module defines the typed contracts from
``docs/context-review-design/CONTEXT_AND_STORAGE.md`` (TaskCard, EvidenceRecord,
ClaimRecord, TaskResult, ReviewPacket, ReviewDecision, ContextManifest).

Every contract model sets ``extra="forbid"``: a model-facing tool argument cannot
smuggle fields (identity, permissions, budgets) that the host did not provide.
"""

import operator
import re
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain_core.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict


###################
# Structured Outputs
###################
class ConductResearch(BaseModel):
    """Call this tool to conduct research on a specific topic."""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""

class Summary(BaseModel):
    """Research summary with key findings."""
    
    summary: str
    key_excerpts: str


class Fact(BaseModel):
    """One atomic fact returned by the compressor, bound to one archived source.

    The host does not trust these fields: ``evidence_id`` must name a source this
    task may read, and ``quote`` must occur verbatim in the archived object. Only
    then does the fact become a claim (still ``unchecked``: locatable is not the
    same as semantically supported).
    """

    statement: str = Field(description="One complete, self-contained factual statement; no headings or citation labels.")
    evidence_id: str = Field(description="The archived evidence id (E<n>) this statement comes from.")
    quote: str = Field(description="A short verbatim excerpt of the archived source that states the fact.")
    requirement_ids: List[str] = Field(default_factory=list, description="Brief requirement ids this fact addresses, if any.")


class CompressedResearch(BaseModel):
    """Structural output of the compression step: one summary plus atomic facts."""

    summary: str = Field(description="Clean, complete findings; keep every relevant detail and do not invent facts.")
    facts: List[Fact] = Field(default_factory=list, description="Atomic facts, each tied to one evidence id and a verbatim quote.")


class ClaimRevision(BaseModel):
    """One proposed change to an existing claim (validated and written by the host)."""

    claim_id: str = Field(description="Existing active claim id to revise or withdraw.")
    action: Literal["revise", "withdraw"] = Field(default="revise")
    statement: str = Field(default="", description="Replacement statement; required for 'revise'.")
    evidence_id: str = Field(default="", description="One archived evidence id that supports the replacement statement.")
    quote: str = Field(default="", description="Verbatim excerpt of that archived source (<= 800 chars).")
    requirement_ids: List[str] = Field(default_factory=list, description="Requirement ids the revised claim addresses; empty keeps the existing mapping.")
    reason: str = Field(default="", description="Why the claim changes (reviewer finding, wrong mapping, withdrawn).")


class ReviseClaims(BaseModel):
    """Apply validated claim revisions after a review asked for corrections.

    The host checks the claim id, evidence permission and verbatim quote; a rejected
    revision is reported back and never written. Withdrawing keeps the original
    record and its counter-evidence in the audit trail.
    """

    revisions: List[ClaimRevision] = Field(description="Claim revisions to apply.")

class ClarifyWithUser(BaseModel):
    """Model for user clarification requests."""
    
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )

class ResearchQuestion(BaseModel):
    """Research question and brief for guiding research."""
    
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class BriefExtraction(BaseModel):
    """Structured user requirements extracted by the trusted runtime (requestor role)."""

    requirements: List[str] = Field(
        default_factory=list,
        description="Each distinct question or deliverable the user asked for, one per item, faithful to the user's wording.",
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Non-negotiable constraints stated by the user: language, dates, region, versions, format, exclusions.",
    )


###################
# Status Enums
###################
class RunStatus(str, Enum):
    """Terminal status of the whole run.

    ``complete`` is only set when the report passed delivery checks (or review was
    explicitly disabled for an ablation arm). Budget exhaustion yields ``partial``.
    """

    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ReviewStatus(str, Enum):
    """Aggregate review status of the run."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    CHANGES_REQUESTED = "changes_requested"
    INCONCLUSIVE = "inconclusive"

class TaskStatus(str, Enum):
    """Status of one task card / task result."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProposalKind(str, Enum):
    """Kinds of supervisor proposals handled by the review gate."""

    DISPATCH = "dispatch"
    PROGRESS = "progress"
    READINESS = "readiness"
    BUDGET_STOP = "budget_stop"
    CONFLICT = "conflict"

class ReviewStage(str, Enum):
    """Review points that the runtime can request."""

    PLAN = "plan"
    PROGRESS = "progress"
    READINESS = "readiness"
    REPORT = "report"
    TERMINATION = "termination"

class Verdict(str, Enum):
    """Verdict produced by the Reviewer (validated by the program)."""

    PASS = "pass"
    REVISE = "revise"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    STOP = "stop"

class ClaimKind(str, Enum):
    """What kind of statement a claim is."""

    FACT = "fact"
    INFERENCE = "inference"
    ASSUMPTION = "assumption"

class SupportStatus(str, Enum):
    """Evidence support status of a claim."""

    UNCHECKED = "unchecked"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT = "insufficient"
    NEEDS_REVALIDATION = "needs_revalidation"


class ClaimStatus(str, Enum):
    """Lifecycle of one claim record.

    A withdrawn claim stays in the version history (and keeps its counter-evidence)
    but no longer counts towards coverage, writing or a delivery check. A revision
    writes a new version of the same claim id instead of editing the old file.
    """

    ACTIVE = "active"
    WITHDRAWN = "withdrawn"


###################
# Contracts
###################
class _Contract(BaseModel):
    """Base class for runtime contracts: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class Requirement(_Contract):
    """One user requirement derived from the request."""

    requirement_id: str
    statement: str
    mandatory: bool = True


class ResearchBrief(_Contract):
    """Structured projection of the user request.

    The brief is produced by the trusted runtime from the user messages; it keeps
    the raw user text so constraints can be re-verified after any compression.
    """

    objective: str
    requirements: List[Requirement] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    scope: str = ""
    as_of_date: str = ""
    output_contract: str = ""
    user_messages: str = ""


class TaskBudget(_Contract):
    """Budget attached to one task card."""

    max_iterations: int = 10
    max_tool_calls: int = 30


class TaskCard(_Contract):
    """Explicit, projected input of one researcher invocation.

    ``task_id`` / ``agent_instance_id`` are assigned by the host, never by the model.
    ``allowed_capability_tags`` are intersected with the role and deployment policy.
    """

    task_id: str
    parent_id: str = "supervisor"
    objective: str
    requirement_ids: List[str] = Field(default_factory=list)
    scope_in: List[str] = Field(default_factory=list)
    scope_out: List[str] = Field(default_factory=list)
    necessary_background: str = ""
    acceptance: List[str] = Field(default_factory=list)
    allowed_capability_tags: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    depends_on_task_ids: List[str] = Field(default_factory=list)
    budget: TaskBudget = Field(default_factory=TaskBudget)
    agent_instance_id: str = ""


class EvidenceRecord(_Contract):
    """A locatable excerpt of a source document.

    Summaries never masquerade as ``exact_excerpt``: the excerpt must be locatable
    in the archived ``objects/<content_hash>.txt``.
    """

    evidence_id: str
    source_id: str
    task_id: str = ""
    agent_instance_id: str = ""
    run_id: str = ""
    original_url: str = ""
    canonical_url: str = ""
    title: str = ""
    retrieved_at: str = ""
    published_at_if_known: Optional[str] = None
    document_version: int = 1
    content_hash: str = ""
    locator: Dict[str, Any] = Field(default_factory=dict)
    exact_excerpt: str = ""
    origin: str = "tool_result"
    trust_label: str = "unverified"
    public: bool = False
    is_counter_evidence: bool = False


class ClaimRecord(_Contract):
    """One atomic claim attached to a task result or the report."""

    claim_id: str
    statement: str
    kind: ClaimKind = ClaimKind.FACT
    evidence_refs: List[str] = Field(default_factory=list)
    support_status: SupportStatus = SupportStatus.UNCHECKED
    requirement_ids: List[str] = Field(default_factory=list)
    qualifiers: List[str] = Field(default_factory=list)
    contradiction_refs: List[str] = Field(default_factory=list)
    version: int = 1
    # --- claim lifecycle -------------------------------------------------
    # ``active`` claims are the only ones writer / coverage / review may use;
    # ``withdrawn`` claims remain readable in the audit history.
    status: ClaimStatus = ClaimStatus.ACTIVE
    supersedes_version: int = 0
    revision_reason: str = ""
    # The exact archived span a claim is bound to (content hash + offsets + quote).
    # It is metadata for the Reviewer, not proof: support stays ``unchecked`` until
    # a review says the statement is entailed.
    support_span: Dict[str, Any] = Field(default_factory=dict)
    # Set when the extraction named a reference the host could not resolve. Such a
    # claim is kept as a visible gap; it is never silently re-pointed at another id.
    unbound_reason: str = ""
    unknown_refs: List[str] = Field(default_factory=list)
    source_urls: List[str] = Field(default_factory=list)


class ToolCallRecord(_Contract):
    """Record of one executed tool call (status is structured, never free text)."""

    tool_call_id: str
    name: str
    status: str = "ok"
    error_class: str = ""
    artifact_refs: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    chars: int = 0


class UsageRecord(_Contract):
    """Token / cost accounting for one model **attempt**.

    ``request_id`` is the logical request; ``attempt_id`` identifies one provider
    invocation of it (retries share the request id and differ here). The ledger is
    keyed by attempt, so ``calls_by_role()`` counts every attempt, including the ones
    that failed, timed out or were cancelled after the request was sent.
    """

    request_id: str = ""
    attempt_id: str = ""
    reservation_id: str = ""
    # Execution boundary: a re-run of the same business run has its own ledger rows,
    # so first-delivery and cumulative spend can be told apart.
    execution_id: str = ""
    agent_instance_id: str = ""
    role: str = ""
    model: str = ""
    # What the provider reports as the served model, when it says so. The requested
    # ``model`` and the served ``actual_model`` are both kept: a silent provider-side
    # substitution must be visible instead of being assumed equal.
    actual_model: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    # ``cost_usd`` is the recomputable estimate (usage x frozen price). Provider-billed
    # and lower-bound totals are separate fields and are never mixed into it.
    cost_usd: Optional[float] = None
    estimated_model_cost_usd: Optional[float] = None
    provider_billed_cost_usd: Optional[float] = None
    cost_method: str = "unknown"
    price_snapshot_hash: str = ""
    latency_s: float = 0.0
    counting_method: str = "unknown"


class TaskResult(_Contract):
    """Result of one task card. Having a result does not make it true."""

    task_id: str
    agent_instance_id: str = ""
    status: TaskStatus = TaskStatus.COMPLETE
    objective: str = ""
    summary: str = ""
    compressed_research: str = ""
    claims: List[ClaimRecord] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    search_coverage: List[str] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    usage: UsageRecord = Field(default_factory=UsageRecord)
    error: str = ""
    error_class: str = ""
    version: int = 1
    raw_notes: Optional[str] = None
    # Requirements whose conclusions this task produced before newly restored
    # material arrived. Downstream must re-check them before delivery.
    needs_revalidation: List[str] = Field(default_factory=list)


class WorkingMemory(_Contract):
    """Compact working state of one agent invocation (not the full history).

    The runtime fills this *deterministically* from the event log (tool results,
    errors, evidence ids and the task card). It is the only part of the earlier
    conversation that is re-projected into the next researcher request, so a trimmed
    history cannot silently drop what the task still needs.
    """

    task_id: str = ""
    objective_ref: str = ""
    active_requirement_ids: List[str] = Field(default_factory=list)
    covered_requirement_ids: List[str] = Field(default_factory=list)
    confirmed_claim_refs: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    failed_approaches: List[str] = Field(default_factory=list)
    next_action: str = ""
    source_event_ids: List[str] = Field(default_factory=list)
    recovery_refs: List[str] = Field(default_factory=list)
    needs_revalidation: List[str] = Field(default_factory=list)
    updated_at: str = ""


class ContextPatch(_Contract):
    """Result of one ``request_context`` recovery call.

    ``status`` distinguishes "not found" from "not allowed" and "only a partial
    match"; recovery never reports an empty answer as proof that the material does
    not exist.
    """

    need: str = ""
    purpose: str = ""
    status: Literal["found", "partial", "unavailable", "access_denied"] = "unavailable"
    restored_refs: List[str] = Field(default_factory=list)
    exact_excerpts: List[Dict[str, Any]] = Field(default_factory=list)
    source_versions: List[Dict[str, Any]] = Field(default_factory=list)
    affected_requirement_ids: List[str] = Field(default_factory=list)
    still_missing: List[str] = Field(default_factory=list)
    denied_refs: List[str] = Field(default_factory=list)
    queried_refs: int = 0
    truncated: bool = False


class RevalidationEvent(_Contract):
    """One bounded correction event: recovered material that may invalidate a result.

    The event is host-owned. It is opened by the runtime when a recovery patch names
    requirements whose conclusions were produced before the new material existed, and
    it is resolved only by a later research result that covers those requirements (or
    by an explicit reviewer decision). Old review passes keep their own hash, so a
    report approved earlier cannot silently stay approved after the correction.
    """

    event_id: str
    trigger: str = "request_context"
    reason: str = ""
    task_id: str = ""
    restored_refs: List[str] = Field(default_factory=list)
    affected_claim_ids: List[str] = Field(default_factory=list)
    affected_requirement_ids: List[str] = Field(default_factory=list)
    status: Literal["open", "resolved", "dismissed"] = "open"
    created_at: str = ""
    resolved_at: str = ""
    resolved_by: str = ""


class RevalidationReceipt(_Contract):
    """Independent proof that one correction event's claims were re-checked.

    A recovery patch only proves that material was *read*. It does not prove that the
    conclusion built before it arrived is still true. This receipt is issued by the
    independent review boundary, binds the checked claims/evidence to their content
    hashes and versions, and is the only artifact that can close an event.

    A contradicted verdict still completes the check (the downstream claim must then be
    revised or qualified); it is not a delivery success.
    """

    receipt_id: str
    event_id: str
    review_id: str
    packet_hash: str
    task_id: str
    claim_ids: List[str] = Field(default_factory=list)
    claim_hashes: Dict[str, str] = Field(default_factory=dict)
    claim_versions: Dict[str, int] = Field(default_factory=dict)
    requirement_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    evidence_hashes: Dict[str, str] = Field(default_factory=dict)
    evidence_versions: Dict[str, int] = Field(default_factory=dict)
    verdicts: Dict[str, str] = Field(default_factory=dict)
    report_version: int = 0
    checker_identity: str = ""
    checker_kind: str = "independent_reviewer"
    created_at: str = ""


class EvictionRecord(_Contract):
    """Record of one item moved out of the working set."""

    artifact_id: str
    reason: str
    recoverable: bool = True
    requirement_ids: List[str] = Field(default_factory=list)


class ContextManifest(_Contract):
    """What one model request actually contained.

    Written next to the exact request payload, keyed by ``request_id``.
    """

    request_id: str
    agent_instance_id: str
    role: str
    policy_version: str = "context-v1"
    ordered_items: List[Dict[str, Any]] = Field(default_factory=list)
    tool_names: List[str] = Field(default_factory=list)
    tool_schema_hashes: Dict[str, str] = Field(default_factory=dict)
    dropped_items: List[EvictionRecord] = Field(default_factory=list)
    estimated_input_tokens: int = 0
    budget_tokens: int = 0
    counting_method: str = "heuristic"
    request_hash: str = ""
    model: str = ""
    observability: str = "full"


class ReviewFinding(_Contract):
    """One actionable finding of the Reviewer."""

    requirement_id: str = ""
    claim_id: str = Field(default="", description="Claim id (C-...) the finding is about; never an evidence id.")
    segment_id: str = Field(default="", description="Shown S/P segment id, not a claim id.")
    issue_type: str = ""
    severity: Literal["blocking", "major", "minor"] = "major"
    evidence_refs: List[str] = Field(
        default_factory=list,
        description="Archived evidence ids only (E1, E2 ...). Put claim ids in claim_id, never here.",
    )
    explanation: str = ""
    required_action: str = ""


class ReviewPacket(_Contract):
    """Program-built review input. The supervisor cannot edit it."""

    review_id: str
    stage: ReviewStage
    run_id: str
    objective: str
    mandatory_constraints: List[str] = Field(default_factory=list)
    requirement_ids: List[str] = Field(default_factory=list)
    coverage_matrix: List[Dict[str, Any]] = Field(default_factory=list)
    open_issues: List[str] = Field(default_factory=list)
    failures: List[str] = Field(default_factory=list)
    remaining_budget: Dict[str, Any] = Field(default_factory=dict)
    claim_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    contradictory_evidence_ids: List[str] = Field(default_factory=list)
    evidence_excerpts: List[Dict[str, Any]] = Field(default_factory=list)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    proposed_action: str = ""
    report_sections: List[Dict[str, Any]] = Field(default_factory=list)
    draft_hash: str = ""
    allowed_read_scope: List[str] = Field(default_factory=list)
    review_coverage_target: str = ""
    packet_hash: str = ""
    # --- R1/R2: what *this* review actually covers, decided by the program ---
    # At the report stage, delivery needs every claim and every mandatory requirement
    # checked, and the whole draft shown to the reviewer. These fields make that
    # expectation machine-checkable instead of relying on the reviewer's claim.
    delivery_required_claim_ids: List[str] = Field(default_factory=list)
    available_evidence_ids: List[str] = Field(default_factory=list)
    missing_evidence_ids: List[str] = Field(default_factory=list)
    report_segments: List[Dict[str, Any]] = Field(default_factory=list)
    draft_assertion_count: int = 0
    unbound_assertions: List[Dict[str, Any]] = Field(default_factory=list)
    scope_issues: List[str] = Field(default_factory=list)
    blocking_issue_count: int = 0
    # Which segments this request shows (empty = all). A long draft is reviewed in
    # contiguous batches, so every segment is eventually read.
    review_segments: List[str] = Field(default_factory=list)
    # Evidence whose archived excerpt was cut to the per-item cap. The reviewer may
    # read the full object through its allowed scope; a cut excerpt is *not* a
    # missing source and must not be reported as one.
    evidence_truncated_ids: List[str] = Field(default_factory=list)
    # Open correction events (recovered material that may invalidate a result).
    revalidation_issues: List[str] = Field(default_factory=list)
    # Unclosed major/blocking findings, bound to the claim and source material.
    prior_claim_findings: List[Dict[str, Any]] = Field(default_factory=list)


class ReviewDecision(_Contract):
    """Reviewer output after program validation of the raw model output.

    ``unreviewed_segment_ids`` / ``unbound_assertion_count`` are the program's own
    record of what was *not* verified; an empty set is required for ``finalize``.
    """

    review_id: str
    stage: ReviewStage
    verdict: Verdict = Verdict.INSUFFICIENT_EVIDENCE
    findings: List[ReviewFinding] = Field(default_factory=list)
    checked_claim_ids: List[str] = Field(default_factory=list)
    unchecked_claim_ids: List[str] = Field(default_factory=list)
    unchecked_requirement_ids: List[str] = Field(default_factory=list)
    unreviewed_segment_ids: List[str] = Field(default_factory=list)
    unbound_assertion_count: int = 0
    unresolved_conflicts: List[str] = Field(default_factory=list)
    draft_hash: str = ""
    packet_hash: str = ""
    review_status: str = "inconclusive"
    review_cost: float = 0.0
    review_model_calls: int = 0
    # S3: False when at least one required batch did not finish (timeout, parse
    # failure, budget stop) or when no usable output exists. A decision that is not
    # complete can never authorize a full delivery, whatever its verdict says.
    review_complete: bool = True
    # Format violations in the reviewer's own output (e.g. several ids in one field
    # that are not all valid). They are protocol errors, not a research gap, and must
    # not be turned into a new research round.
    protocol_errors: List[str] = Field(default_factory=list)
    finding_resolutions: Dict[str, str] = Field(default_factory=dict)
    error: str = ""


class FindingResolution(_Contract):
    """An explicit recheck of an existing finding, validated against this packet."""

    finding_id: str
    reason: str


class SimpleReportReview(BaseModel):
    """A single practical review, without the strict evidence-audit protocol."""

    approved: bool
    feedback: str


class RawReviewOutput(BaseModel):
    """Structured output requested from the Reviewer model (untrusted until checked)."""

    verdict: Literal["pass", "revise", "insufficient_evidence", "stop"] = Field(
        description="Overall verdict for the reviewed proposal, draft or completion request."
    )
    summary: str = Field(description="One or two sentences describing what was checked.")
    finding_resolutions: List[FindingResolution] = Field(default_factory=list, description="Explicit rechecks using only finding_id values from PriorClaimFindings; return [] when that list is empty. Revalidation ids are not finding ids. Omit unresolved findings.")
    findings: List[ReviewFinding] = Field(
        default_factory=list,
        description="Actionable problems. Leave empty only if the verdict is 'pass'.",
    )
    checked_claim_ids: List[str] = Field(
        default_factory=list, description="Claim ids that were actually checked against evidence."
    )
    checked_requirement_ids: List[str] = Field(
        default_factory=list, description="Requirement ids whose coverage was actually verified against the cited evidence."
    )
    checked_segment_ids: List[str] = Field(
        default_factory=list, description="Draft segment ids (S1, S2, ...) that were read in full and checked; a pass requires all of them."
    )
    acknowledged_unknown_requirement_ids: List[str] = Field(
        default_factory=list,
        description="Mandatory requirement ids with no archived evidence that you verified are explicitly reported as an unknown/limitation in the draft.",
    )
    unresolved_conflicts: List[str] = Field(
        default_factory=list, description="Evidence ids that contradict each other and remain unresolved."
    )


###################
# State Definitions
###################

def override_reducer(current_value, new_value):
    """Reducer function that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    elif current_value is None:
        return new_value
    elif new_value is None:
        return current_value
    else:
        return operator.add(current_value, new_value)
    

def task_ledger_reducer(current_value, new_value):
    """Merge the host-written task ledger (keyed upsert, never a blind append)."""
    merged = dict(current_value or {})
    for task_id, entry in (new_value or {}).items():
        merged[task_id] = {**(merged.get(task_id) or {}), **entry}
    return merged


def repair_grant_reducer(current_value, new_value):
    """Merge granted repair turns by review id (idempotent across a subgraph merge)."""
    merged = dict(current_value or {})
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        new_value = new_value.get("value") or {}
    for review_id, stage in (new_value or {}).items():
        merged[str(review_id)] = stage
    return merged


def latest_reducer(current_value, new_value):
    """Keep the newest value (used for one-shot routing signals).

    LangGraph rejects two writes to a plain key in one step; a one-shot hint written by
    both the inner and the outer graph must resolve deterministically instead.
    """
    return new_value if new_value is not None else current_value


_REVIEW_EVENT_KEYS = {
    "reviews": ("review_id",),
    "decisions": ("review_id",),
    "pending": ("review_id",),
    "revisions": ("stage", "review_id"),
    "proposals": ("proposal_id",),
}


def _review_event_identity(key: str, entry: Any) -> Optional[tuple]:
    """Return the stable id of one review event, or None when it has no id."""
    fields = _REVIEW_EVENT_KEYS.get(key)
    if not fields or not isinstance(entry, dict):
        return None
    values = tuple(str(entry.get(field) or "") for field in fields)
    return values if all(values) else None


def review_history_reducer(current_value, new_value):
    """Merge review events by stable id and replace the current open-issue list.

    A compiled subgraph returns its whole ``review_history`` when it finishes, so a
    plain append would count the same review, revision or proposal twice when the
    parent merges the child state. ``open_issues`` is not historical: the latest
    event's open issues are the current ones, while resolved issues disappear.
    """
    history = dict(current_value or {})
    for key, items in (new_value or {}).items():
        if key == "open_issues":
            history[key] = list(items or [])
            continue
        if not isinstance(items, list):
            history[key] = items
            continue
        merged = list(history.get(key) or [])
        seen = {_review_event_identity(key, entry) for entry in merged}
        for entry in items:
            identity = _review_event_identity(key, entry)
            if identity is not None and identity in seen:
                continue
            merged.append(entry)
            if identity is not None:
                seen.add(identity)
        history[key] = merged
    return history


class AgentInputState(MessagesState):
    """InputState is only 'messages'."""

class AgentState(MessagesState):
    """Main agent state containing messages and research data.

    The run-level channels below are owned by the parent graph and shared with the
    supervisor subgraph by key. Keeping them on the parent is what makes the caps
    (research iterations, review events, revision rounds) run-wide instead of
    per-subgraph-invocation. ``review_history``/``task_ledger`` have merge reducers
    because a subgraph returns its full value when it ends.
    """
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]
    raw_notes: Annotated[list[str], override_reducer] = []
    notes: Annotated[list[str], override_reducer] = []
    final_report: str
    # --- Typed run state (design docs: IMPLEMENTATION_HANDOFF §5/§6) ---
    run_id: str
    run_status: str
    review_status: str
    brief: Optional[ResearchBrief]
    coverage_matrix: Annotated[list[dict], override_reducer] = []
    evidence_manifest_ref: str
    metrics_ref: str
    final_report_draft: str
    final_report_draft_hash: str
    report_review_summary: str
    todo_write_notice: str
    # --- Run-level budget / review channels shared with the supervisor subgraph ---
    research_iterations: int
    max_iterations: int
    # Review ids that were granted one extra, consumable supervisor turn, mapped to
    # the review stage that granted it. The allowance is
    # ``max_iterations + len(repair_grants)`` everywhere (entry guard, tool guard and
    # the remaining-budget projection), so a revise cannot be granted by one node and
    # blocked by another. A dict keyed by review id makes a re-merge idempotent.
    repair_grants: Annotated[dict, repair_grant_reducer]
    task_ledger: Annotated[dict, task_ledger_reducer]
    review_history: Annotated[dict, review_history_reducer]
    report_version: int
    report_ready: str
    writing_constraints: str

class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks.

    The supervisor receives only budget / ledger / proposal fields; results are
    fetched from the ArtifactStore through host-bound identities.
    """
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []
    run_id: str
    brief: Optional[ResearchBrief]
    task_ledger: Annotated[dict, task_ledger_reducer] = {}
    review_history: Annotated[dict, review_history_reducer] = {}
    max_iterations: int = 0
    repair_grants: Annotated[dict, repair_grant_reducer] = {}
    review_stage_hint: Annotated[str, latest_reducer] = ""
    # Written by ``apply_review`` / ``validate_proposal`` when the research phase ends;
    # the parent reads these to decide whether the report may still be reviewed.
    report_ready: str = ""
    run_status: str = ""
    review_status: str = ""

class ResearcherState(TypedDict):
    """State for individual researchers conducting research."""
    
    researcher_messages: Annotated[list[MessageLikeRepresentation], operator.add]
    tool_call_iterations: int = 0
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    task_card: Optional[TaskCard]
    run_id: str
    brief: Optional[ResearchBrief]
    working_memory: Optional[WorkingMemory]

class ResearcherOutputState(BaseModel):
    """Output state from individual researchers."""
    
    compressed_research: str
    raw_notes: Annotated[list[str], override_reducer] = []
    task_result: Optional[TaskResult] = None


###################
# Deterministic helpers for requirement / claim records
###################
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n+")
_CONSTRAINT_MARKERS = (
    "must",
    "only",
    "exclude",
    "excluding",
    "no later than",
    "before ",
    "after ",
    "in english",
    "in chinese",
    "in ",
    "format",
    "cite",
    "citations",
    "avoid",
    "limit",
    "at least",
    "at most",
    "please",
    "不要",
    "必须",
    "只需",
    "仅限于",
    "不包括",
    "格式",
    "语言",
    "截至",
)


def requirements_from_text(brief_text: str, user_text: str = "") -> tuple[list[Requirement], list[str]]:
    """Derive stable requirement ids and constraints without an extra model call.

    The brief is the research question; the user text keeps the raw constraints. Both
    are deterministic so that a requirement id means the same thing across turns.
    """
    requirements: list[Requirement] = []
    seen: set[str] = set()
    source = brief_text or user_text
    for sentence in _SENTENCE_SPLIT.split(source or ""):
        text = sentence.strip(" -*\t")
        if len(text) < 8:
            continue
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        requirements.append(Requirement(requirement_id=f"req-{len(requirements) + 1}", statement=text[:300], mandatory=True))
        if len(requirements) >= 8:
            break
    if not requirements:
        requirements.append(Requirement(requirement_id="req-1", statement=(brief_text or user_text or "answer the request")[:300], mandatory=True))
    constraints: list[str] = []
    for sentence in _SENTENCE_SPLIT.split(user_text or ""):
        text = sentence.strip()
        lowered = text.lower()
        if len(text) < 4:
            continue
        if any(marker in lowered for marker in _CONSTRAINT_MARKERS) and text not in constraints:
            constraints.append(text[:300])
        if len(constraints) >= 8:
            break
    return requirements, constraints


_PLAN_PREFIXES = (
    "next step",
    "next steps",
    "todo",
    "to do",
    "follow up",
    "follow-up",
    "we need",
    "need to",
    "should fetch",
    "下一步",
    "接下来",
    "待办",
    "后续",
    "计划",
    "需要",
    "建议",
    "还要",
    "sources",
    "references",
    "来源",
    "参考",
)


def _asserts_something(sentence: str) -> bool:
    """Return False for text that looks like a plan, a question or a bare citation.

    The extractor is structural, so anything containing a URL or ``[E1]`` used to be
    published as a ``fact`` — including ``Next step: fetch https://...`` and bare
    ``[E1] [E4] — Title: https://...`` citation lines. Neither can support a
    requirement, and a Reviewer rightly counts them as fabricated coverage.
    """
    text = sentence.strip(" -*\t#>")
    # A trailing marker ("(evidence E2)", "（evidence E3）") is provenance, not prose:
    # drop it before judging whether anything was actually asserted.
    text = re.sub(r"[\s（(]*(?:evidence|摘录|来源)\s*[:：]?\s*E\d+[\s）)]*$", "", text, flags=re.IGNORECASE).strip(" -*\t#>")
    if not text:
        return False
    if text.endswith(("?", "？")):
        return False
    # A dangling label ("... is summarized as stating:") announces content that is not
    # there. Publishing it as a fact gave the Reviewer claims it could neither support
    # nor refute, and blocked three readiness rounds in the F-D acceptance run.
    if text.endswith((":", "：")):
        return False
    if text.lower().startswith(_PLAN_PREFIXES):
        return False
    # "Nothing was found" is an observation about the search, not a fact about the
    # world, and it goes stale the moment a later search returns the page. Such claims
    # contradicted their own evidence packet in every live acceptance attempt.
    if re.search(
        r"(not (?:found|returned|in the consulted)|no search|cannot confirm|could not (?:find|confirm|verify)|"
        r"未(?:找到|检索到|发现|能确认|能核实)|没有(?:找到|检索到|发现)|无法(?:确认|核实|验证)|未能(?:找到|确认))",
        text,
        re.IGNORECASE,
    ):
        return False
    # A tool log ("tavily_search returned 3 results") describes the run, not the world.
    if re.search(r"\b(tavily_search|think_tool|request_context|ConductResearch)\b", text):
        return False
    # First-person process talk is the agent describing what it will do, not a fact
    # about the world ("I must not accept it as established without checking ...").
    if re.match(r"^(i|we|let me|my|our|我|我们|本人)\b", text, re.IGNORECASE) or text.startswith(("我", "我们")):
        return False
    # A citation line without a statement: remove refs, URLs and separators, then see
    # whether anything assertable is left.
    # A short label followed by a URL is a citation line ("Title: https://..."), not a
    # statement: there is nothing in it that a source could support or contradict.
    tail = re.search(r"https?://\S+$", text)
    if tail:
        prefix = text[: tail.start()]
        label = re.sub(r"\[(?:E\d+|\d+)\]", " ", prefix)
        label = label.strip(" \-—–:：|,，.。;；()（）[]")
        # A clause separator means there is prose before the link; a bare title does
        # not. An inline citation ("... 2024-09-26 (https://...)") is provenance
        # attached to a complete assertion, not a label, even when it is short.
        inline = bool(re.search(r"[(（]\s*$", prefix))
        if len(label) <= 40 and not inline and not re.search(r"[，,。；;！!]", label):
            return False
    stripped = re.sub(r"\[(?:E\d+|\d+)\]", " ", text)
    stripped = re.sub(r"https?://\S+", " ", stripped)
    stripped = re.sub(r"[\s\-—–:：|,，.。;；()（）\[\]]+", "", stripped)
    return len(stripped) >= 12


_CITATION_PREFIX = re.compile(
    r"^\s*(?:sources?|url|link|reference(?:s)?|source\s+url|来源|参考|链接|出处|原文|资料)\s*[:：]\s*",
    re.IGNORECASE,
)
_QUOTED_SPAN = re.compile(r'"[^"]*"|“[^”]*”|「[^」]*」|『[^』]*』', re.DOTALL)
_ABBREVIATED_NUMBER = re.compile(r"\b(?:[A-Z]|App(?:endix)?|Section|Sec|Fig|No)\.\d+\.")
_EXPLICIT_REF = re.compile(
    r"\[(E\d+)\]|\((?:evidence\s*)?(E\d+)\)|(?:evidence|摘录|来源)\s*[:：]?\s*(E\d+)",
    re.IGNORECASE,
)


def _split_claim_sentences(text: str) -> list[str]:
    """Split a block into sentences without breaking headings or quoted spans.

    A naive split on ``.`` turned ``## E.12. Release 17 / Release date: 2024-09-26``
    into a claim that ended at ``## E.12.``; the actual date lived in the next line.
    Quoted spans (which is where extractors keep verbatim source text) and appendix
    style numbering are masked before splitting and restored afterwards.
    """
    placeholders: Dict[str, str] = {}

    def stash(value: str) -> str:
        key = f"\x01{len(placeholders)}\x01"
        placeholders[key] = value
        return key

    masked = _QUOTED_SPAN.sub(lambda match: stash(match.group(0)), text)
    masked = _ABBREVIATED_NUMBER.sub(lambda match: stash(match.group(0)), masked)
    parts = [part for part in re.split(r"(?<=[.!?。！？])\s+|\n+", masked) if part.strip()]
    restored = [part.strip() for part in parts]
    for key, value in placeholders.items():
        restored = [part.replace(key, value) for part in restored]
    return [part for part in restored if part]


def _is_citation_line(line: str) -> bool:
    """Return True for a bare source/URL line that belongs to the previous fact."""
    stripped = line.strip(" -*\t#>")
    if not stripped:
        return False
    has_ref = "http://" in stripped or "https://" in stripped or bool(re.search(r"\[(?:E\d+|\d+)\]", stripped))
    if not has_ref:
        return False
    if _CITATION_PREFIX.match(stripped):
        return True
    return bool(re.fullmatch(r"(?:\[(?:E\d+|\d+)\][\s,，;；—–-]*)*https?://\S+", stripped))


def _claim_units(text: str) -> list[str]:
    """Group a compressed result into statement units without losing the source line.

    ``Fact sentence`` on one line and ``Source: [E2] URL`` on the next are one claim:
    the old splitter dropped the fact (no marker on that line) and filtered the source
    line, so a correct answer yielded zero claims.
    """
    blocks: list[str] = []
    current = ""
    placeholders: Dict[str, str] = {}

    def stash_quote(match: re.Match) -> str:
        key = f"\x02{len(placeholders)}\x02"
        placeholders[key] = match.group(0)
        return key

    # A verbatim quote may contain blank lines and headings. Mask it before the line
    # loop so the date inside the quote stays attached to the sentence that cites it.
    masked_text = _QUOTED_SPAN.sub(stash_quote, text)
    for raw in masked_text.splitlines():
        line = raw.strip()
        if not line:
            if current:
                blocks.append(current)
                current = ""
            continue
        if not current:
            current = line
            continue
        if _is_citation_line(line) or not _asserts_something(line):
            current = current + "\n" + line
            continue
        blocks.append(current)
        current = line
    if current:
        blocks.append(current)

    units: list[str] = []
    for block in blocks:
        assembled = ""
        assembled_asserts = False
        for sentence in _split_claim_sentences(block):
            if not assembled:
                assembled = sentence
                assembled_asserts = _asserts_something(sentence)
            elif _is_citation_line(sentence):
                assembled = assembled + "\n" + sentence
            elif _asserts_something(sentence):
                if assembled_asserts:
                    units.append(assembled)
                assembled = sentence
                assembled_asserts = True
            else:
                assembled = assembled + " " + sentence
        # A block that never asserted anything (a label, a source list, a plan) is
        # dropped as a whole instead of being published under the first URL it saw.
        if assembled and assembled_asserts:
            units.append(assembled)
    restored: list[str] = []
    for unit in units:
        for key, value in placeholders.items():
            unit = unit.replace(key, value)
        restored.append(unit)
    return restored


def _statement_without_citation(unit: str) -> str:
    """Keep the asserted prose; citation-only lines are provenance, not the claim."""
    parts = [line.strip() for line in unit.splitlines() if line.strip() and not _is_citation_line(line)]
    return " ".join(parts).strip()


def _normalize_url(url: str) -> str:
    value = str(url or "").strip()
    if not value:
        return ""
    value = value.split("#", 1)[0]
    return value.rstrip("/")


def evidence_url_map(evidence_index: Optional[List[Dict[str, Any]]]) -> Dict[str, str]:
    """Map an exact archived URL to its evidence id (never a positional guess)."""
    mapping: Dict[str, str] = {}
    for row in evidence_index or []:
        eid = str(row.get("evidence_id") or "")
        if not eid:
            continue
        for key in ("url", "original_url", "canonical_url"):
            url = _normalize_url(row.get(key) or "")
            if url and url not in mapping:
                mapping[url] = eid
    return mapping


def claim_records_from_text(
    text: str,
    evidence_ids: List[str],
    task_id: str,
    *,
    limit: int = 40,
    evidence_index: Optional[List[Dict[str, Any]]] = None,
) -> list[ClaimRecord]:
    """Extract atomic-looking claims from compressed research (structural, not semantic).

    A claim is recorded only when the text cites something the host can resolve: an
    explicit ``[E<n>]`` id that exists, or a URL that exactly matches an archived
    source. Unknown explicit ids are kept as an *unbound* claim (``evidence_refs=[]``)
    instead of being silently re-pointed at the first evidence of the task; there is
    no positional fallback. Support status stays ``unchecked``: a locatable marker is
    not semantic support, which is the Reviewer's call.
    """
    if not text:
        return []
    from open_deep_research.artifacts import sha256_text

    known = set(evidence_ids or [])
    url_map = evidence_url_map(evidence_index)
    claims: list[ClaimRecord] = []
    for unit in _claim_units(text):
        explicit = list(
            dict.fromkeys(next(group for group in match if group) for match in _EXPLICIT_REF.findall(unit))
        )
        urls = list(dict.fromkeys(re.findall(r"https?://[^\s\]\)>,，。；;\"']+", unit)))
        if not explicit and not urls:
            continue
        statement = _statement_without_citation(unit)
        if not statement or not _asserts_something(statement):
            continue
        refs = [eid for eid in explicit if eid in known]
        unknown = [eid for eid in explicit if eid not in known]
        if unknown:
            # An explicit citation the host cannot resolve makes the claim unbound:
            # falling back to a URL in the same sentence would silently replace the
            # source the extractor named (the E999 -> E1 re-pointing defect).
            refs = []
        else:
            for url in urls:
                mapped = url_map.get(_normalize_url(url))
                if mapped and mapped in known and mapped not in refs:
                    refs.append(mapped)
        digest = sha256_text(f"{task_id}:{statement[:200]}")[:10]
        claims.append(
            ClaimRecord(
                claim_id=f"C-{digest}",
                statement=statement[:800],
                kind=ClaimKind.FACT,
                evidence_refs=refs,
                support_status=SupportStatus.UNCHECKED,
                source_urls=urls[:4],
                unknown_refs=unknown,
                unbound_reason=(
                    "unknown evidence id(s): " + ", ".join(unknown) if unknown and not refs else ""
                ),
            )
        )
        if len(claims) >= limit:
            break
    return claims
