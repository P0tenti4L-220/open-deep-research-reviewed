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
    agent_instance_id: str = ""
    role: str = ""
    model: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
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


class WorkingMemory(_Contract):
    """Compact working state of one agent invocation (not the full history)."""

    task_id: str = ""
    confirmed_claim_refs: List[str] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    conflicts: List[str] = Field(default_factory=list)
    failed_approaches: List[str] = Field(default_factory=list)
    next_action: str = ""
    source_event_ids: List[str] = Field(default_factory=list)


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
    claim_id: str = ""
    issue_type: str = ""
    severity: Literal["blocking", "major", "minor"] = "major"
    evidence_refs: List[str] = Field(default_factory=list)
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
    error: str = ""


class RawReviewOutput(BaseModel):
    """Structured output requested from the Reviewer model (untrusted until checked)."""

    verdict: Literal["pass", "revise", "insufficient_evidence", "stop"] = Field(
        description="Overall verdict for the reviewed proposal, draft or completion request."
    )
    summary: str = Field(description="One or two sentences describing what was checked.")
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


def claim_records_from_text(text: str, evidence_ids: List[str], task_id: str, *, limit: int = 40) -> list[ClaimRecord]:
    """Extract atomic-looking claims from compressed research (structural, not semantic).

    This is deliberately conservative: claims are recorded only where there is a URL or
    an ``[E<n>]`` reference nearby, and their support status stays ``unchecked``. The
    Reviewer and the evaluator decide semantic support; this function never does.
    """
    if not text:
        return []
    from open_deep_research.artifacts import sha256_text

    claims: list[ClaimRecord] = []
    sentences = [s.strip(" -*\t#") for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s and len(s.strip()) > 20]
    cited: list[str] = []
    for sentence in sentences:
        explicit = re.findall(r"\[(E\d+)\]", sentence)
        has_url = "http://" in sentence or "https://" in sentence
        if not explicit and not has_url:
            continue
        refs = sorted({eid for eid in explicit if eid in evidence_ids}) or list(evidence_ids[:1])
        if not refs:
            continue
        cited.extend(refs)
        digest = sha256_text(f"{task_id}:{sentence[:200]}")[:10]
        claims.append(
            ClaimRecord(
                claim_id=f"C-{digest}",
                statement=sentence[:500],
                kind=ClaimKind.FACT,
                evidence_refs=refs,
                support_status=SupportStatus.UNCHECKED,
            )
        )
        if len(claims) >= limit:
            break
    return claims
