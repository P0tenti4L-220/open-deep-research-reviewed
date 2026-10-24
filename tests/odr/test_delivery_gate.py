"""R1 / R2 regression tests: a report is delivered as verified only when the whole
deliverable was actually reviewed.

These are the acceptance scenarios of ``ROUND1_ACCEPTANCE_REVIEW_2026-10-23.md``:

* R1 - two required claims with only one checked, a claim citing evidence that was
  never archived, and an empty claim list with factual assertions all fail;
  a fully checked report with archived evidence still passes.
* R2 - an unsupported statement placed after character 6000, in the 25th section,
  or added by the writer, is either shown to the reviewer (in full) or the delivery
  is explicitly partial. Nothing is silently cut.
"""

from __future__ import annotations

import pytest

from open_deep_research.artifacts import ArtifactStore, Identity
from open_deep_research.reviewer import (
    REVIEW_MAX_SEGMENTS,
    REVIEWER_BUDGET_CHARS,
    _segments,
    apply_review,
    build_review_packet,
    render_packet,
    validate_decision,
)
from open_deep_research.state import (
    RawReviewOutput,
    ResearchBrief,
    Requirement,
    ReviewFinding,
    ReviewStage,
    Verdict,
)

BRIEF = ResearchBrief(
    objective="Compare v1.0 and v2.0 release dates",
    requirements=[Requirement(requirement_id="req-1", statement="Find both release dates")],
)


def _store(tmp_path, run_id: str = "gate") -> ArtifactStore:
    return ArtifactStore(root=tmp_path, run_id=run_id)


def _evidence(store: ArtifactStore, *, text: str = "Version 2.0 was released on 2024-05-01; version 1.0 on 2020-02-02."):
    identity = Identity(run_id=store.run_id, agent_instance_id="researcher-01", role="researcher", task_id="task-1")
    return store.add_evidence(identity, content=text, excerpt=text[:200], source_url="https://example.com/release-notes", public=True)


def _claims(evidence_ids):
    return [
        {"claim_id": "C1", "statement": "Version 2.0 was released on 2024-05-01.", "evidence_refs": [evidence_ids[0]], "requirement_ids": ["req-1"]},
        {"claim_id": "C2", "statement": "Version 1.0 was released on 2020-02-02.", "evidence_refs": [evidence_ids[0]], "requirement_ids": ["req-1"]},
    ]


DRAFT = "# Report\n\nVersion 2.0 was released on 2024-05-01 [E1] (claim C1). Version 1.0 was released on 2020-02-02 [E1] (claim C2)."


def _pass(**kwargs) -> RawReviewOutput:
    return RawReviewOutput(verdict="pass", summary="checked", findings=[], **kwargs)


def _route(decision, packet) -> str:
    return apply_review(
        decision,
        current_hash=packet.draft_hash,
        remaining_budget={"open": True, "iterations_left": 2},
        revision_rounds=0,
        max_revision_rounds=2,
        stage=ReviewStage.REPORT,
    )


# --------------------------------------------------------------------------- #
# R1
# --------------------------------------------------------------------------- #
def test_a_checked_claim_plus_a_missing_source_cannot_pass(tmp_path):
    """Two required claims, only one checked, and the claims cite unarchived evidence."""
    store = _store(tmp_path)
    _evidence(store)
    claims = [{"claim_id": cid, "statement": cid, "evidence_refs": ["E-missing"], "requirement_ids": ["req-1"]} for cid in ("C1", "C2")]
    packet = build_review_packet(ReviewStage.REPORT, review_id="r1", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=claims, draft=DRAFT, draft_hash="h")

    assert packet.missing_evidence_ids == ["E-missing"]
    decision = validate_decision(_pass(checked_claim_ids=["C1"], checked_segment_ids=["S1"]), packet)
    assert decision.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert decision.review_status == "inconclusive"
    assert "E-missing" in decision.error
    assert decision.unchecked_claim_ids == ["C2"]
    assert _route(decision, packet) != "finalize"


def test_every_required_claim_must_be_checked(tmp_path):
    """One checked claim out of two is a partial review, never a pass."""
    store = _store(tmp_path)
    evidence = _evidence(store)
    packet = build_review_packet(ReviewStage.REPORT, review_id="r2", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=_claims([evidence.evidence_id]), draft=DRAFT, draft_hash="h")

    partial = validate_decision(_pass(checked_claim_ids=["C1"], checked_segment_ids=["S1"]), packet)
    assert partial.review_status == "inconclusive"
    assert partial.unchecked_claim_ids == ["C2"]
    assert "claim(s) were not checked" in partial.error

    full = validate_decision(
        _pass(checked_claim_ids=["C1", "C2"], checked_requirement_ids=["req-1"], checked_segment_ids=["S1"]),
        packet,
    )
    assert full.verdict == Verdict.PASS
    assert full.review_status == "passed"
    assert _route(full, packet) == "finalize"


def test_an_empty_claim_list_with_factual_assertions_cannot_pass(tmp_path):
    """No claim list does not mean "no facts to check"."""
    store = _store(tmp_path)
    evidence = _evidence(store)
    packet = build_review_packet(ReviewStage.REPORT, review_id="r3", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=[], evidence_ids=[evidence.evidence_id], draft=DRAFT, draft_hash="h")

    decision = validate_decision(_pass(checked_segment_ids=["S1"], checked_requirement_ids=["req-1"]), packet)
    assert decision.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert "no claim list" in decision.error
    # The refusal carries a concrete, checkable required action for the writer.
    assert any("archived evidence id" in finding.required_action for finding in decision.findings)
    assert _route(decision, packet) != "finalize"


def test_an_unbound_assertion_blocks_delivery(tmp_path):
    """A section that asserts facts but cites nothing was not verifiable."""
    store = _store(tmp_path)
    evidence = _evidence(store)
    draft = DRAFT + "\n\n## Outlook\n\nThe version 2.0 ecosystem will dominate the market within two years for sure."
    packet = build_review_packet(ReviewStage.REPORT, review_id="r4", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=_claims([evidence.evidence_id]), draft=draft, draft_hash="h")

    assert packet.unbound_assertions
    decision = validate_decision(
        _pass(checked_claim_ids=["C1", "C2"], checked_requirement_ids=["req-1"], checked_segment_ids=[seg["segment_id"] for seg in packet.report_segments]),
        packet,
    )
    assert decision.review_status == "inconclusive"
    assert decision.unbound_assertion_count == 1
    assert _route(decision, packet) != "finalize"


def test_apply_review_never_finalizes_a_partial_decision(tmp_path):
    """Defence in depth: a passed decision that still records gaps cannot finalize."""
    store = _store(tmp_path)
    evidence = _evidence(store)
    packet = build_review_packet(ReviewStage.REPORT, review_id="r5", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=_claims([evidence.evidence_id]), draft=DRAFT, draft_hash="h")
    decision = validate_decision(_pass(checked_claim_ids=["C1", "C2"], checked_segment_ids=["S1"]), packet)
    assert decision.review_status == "passed"
    forged = decision.model_copy(update={"unreviewed_segment_ids": ["S2"]})
    assert _route(forged, packet) == "finalize_partial"


# --------------------------------------------------------------------------- #
# R2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "draft",
    [
        # The unsupported marker sits after character 6000 of one huge section.
        "# Report\n\n" + "Version 2.0 was released on 2024-05-01 [E1]. " * 150 + "UNSUPPORTED_TAIL_ASSERTION",
        # The marker sits in the 25th section.
        "# Report\n\n" + "\n\n".join(f"## Section {i}\n\nVersion 2.0 was released on 2024-05-01 [E1]." for i in range(1, 26)) + "\n\n## Section 26\n\nUNSUPPORTED_TAIL_ASSERTION",
        # The marker was added by the writer after the research findings.
        DRAFT + "\n\n## Writer addition\n\nUNSUPPORTED_TAIL_ASSERTION was confirmed by everyone.",
    ],
)
def test_the_whole_draft_reaches_the_reviewer(tmp_path, draft):
    """No character, section or writer addition is cut before review."""
    store = _store(tmp_path)
    evidence = _evidence(store)
    packet = build_review_packet(ReviewStage.REPORT, review_id="r6", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=_claims([evidence.evidence_id]), draft=draft, draft_hash="h")

    rendered = render_packet(packet)
    assert "UNSUPPORTED_TAIL_ASSERTION" in rendered
    # No elision markers: the reviewer either sees the text or sees it flagged.
    assert "[... " not in rendered and "chars=6000" not in rendered
    # The segments are contiguous and reproduce the draft exactly.
    assert "".join(seg["text"] for seg in packet.report_segments) == draft


def test_segments_split_long_drafts_without_losing_text():
    draft = "\n".join(f"Paragraph {i}: version 2.0 was released in 2024 & more text here." for i in range(400))
    segments = _segments(draft, budget_chars=2000, max_segments=6)
    assert 2 <= len(segments)
    assert all(seg["within_budget"] is False for seg in segments[6:])
    assert "".join(seg["text"] for seg in segments) == draft
    assert [seg["segment_id"] for seg in segments] == [f"S{i}" for i in range(1, len(segments) + 1)]
    assert segments[0]["char_start"] == 0 and segments[-1]["char_end"] == len(draft)


def test_a_pass_needs_every_segment_read(tmp_path):
    store = _store(tmp_path)
    evidence = _evidence(store)
    draft = DRAFT + "\n\n" + "\n".join(f"Paragraph {i}: version 2.0 was released on 2024-05-01 [E1] {i}." * 4 for i in range(300))
    packet = build_review_packet(ReviewStage.REPORT, review_id="r7", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=_claims([evidence.evidence_id]), draft=draft, draft_hash="h")
    ids = [seg["segment_id"] for seg in packet.report_segments]
    assert len(ids) >= 2

    partial = validate_decision(_pass(checked_claim_ids=["C1", "C2"], checked_requirement_ids=["req-1"], checked_segment_ids=ids[:-1]), packet)
    assert partial.review_status == "inconclusive"
    assert partial.unreviewed_segment_ids == ids[-1:]
    assert _route(partial, packet) != "finalize"

    full = validate_decision(_pass(checked_claim_ids=["C1", "C2"], checked_requirement_ids=["req-1"], checked_segment_ids=ids), packet)
    assert full.review_status == "passed"
    assert _route(full, packet) == "finalize"


def test_unreviewed_requirement_needs_an_explicit_acknowledgement(tmp_path):
    """A mandatory requirement without evidence may pass only as a stated unknown."""
    store = _store(tmp_path)
    evidence = _evidence(store)
    brief = ResearchBrief(
        objective="Compare versions",
        requirements=[
            Requirement(requirement_id="req-1", statement="Find both release dates"),
            Requirement(requirement_id="req-2", statement="Find the support policy end date"),
        ],
    )
    packet = build_review_packet(stage=ReviewStage.REPORT, review_id="r8", run_id="gate", brief=brief, proposal="review", store=store, claims=_claims([evidence.evidence_id]), draft=DRAFT, draft_hash="h")

    silent = validate_decision(_pass(checked_claim_ids=["C1", "C2"], checked_requirement_ids=["req-1"], checked_segment_ids=["S1"]), packet)
    assert silent.review_status == "inconclusive"
    assert silent.unchecked_requirement_ids == ["req-2"]

    stated = validate_decision(
        _pass(checked_claim_ids=["C1", "C2"], checked_requirement_ids=["req-1"], acknowledged_unknown_requirement_ids=["req-2"], checked_segment_ids=["S1"]),
        packet,
    )
    assert stated.review_status == "passed"


def test_the_reviewer_cannot_invent_segments_or_requirements(tmp_path):
    store = _store(tmp_path)
    evidence = _evidence(store)
    packet = build_review_packet(ReviewStage.REPORT, review_id="r9", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=_claims([evidence.evidence_id]), draft=DRAFT, draft_hash="h")
    decision = validate_decision(
        _pass(checked_claim_ids=["C1", "C2"], checked_requirement_ids=["req-1", "req-404"], checked_segment_ids=["S1", "S99"]),
        packet,
    )
    assert decision.review_status == "inconclusive"
    assert "unknown segment ids" in decision.error or "unknown requirement ids" in decision.error


def test_a_report_with_no_draft_cannot_pass(tmp_path):
    store = _store(tmp_path)
    evidence = _evidence(store)
    packet = build_review_packet(ReviewStage.REPORT, review_id="r10", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=_claims([evidence.evidence_id]), draft="", draft_hash="h")
    decision = validate_decision(_pass(checked_claim_ids=["C1", "C2"], checked_requirement_ids=["req-1"]), packet)
    assert decision.review_status == "inconclusive"
    assert any("empty" in issue for issue in packet.scope_issues)


def test_unknown_reviewer_output_still_cannot_pass(tmp_path):
    store = _store(tmp_path)
    packet = build_review_packet(ReviewStage.REPORT, review_id="r11", run_id="gate", brief=BRIEF, proposal="review", store=store, claims=[], draft=DRAFT, draft_hash="h")
    missing = validate_decision(None, packet, error="review timed out")
    assert missing.review_status == "unreviewed"
    assert missing.unreviewed_segment_ids == ["S1"]
    blocking = validate_decision(
        RawReviewOutput(verdict="pass", summary="x", findings=[ReviewFinding(severity="blocking", issue_type="unsupported_claim")]),
        packet,
    )
    assert blocking.verdict == Verdict.REVISE


def test_segment_limit_is_recorded_when_a_draft_is_pathological():
    """Even past the segment cap, the packet never claims the whole draft was shown."""
    draft = "x" * (REVIEW_MAX_SEGMENTS * REVIEWER_BUDGET_CHARS + 5000)
    segments = _segments(draft)
    assert len(segments) > REVIEW_MAX_SEGMENTS
    assert "".join(seg["text"] for seg in segments) == draft
    assert all(seg["within_budget"] is False for seg in segments[REVIEW_MAX_SEGMENTS:])
    # A pass therefore requires the missing tail to be addressed, not ignored.
    store_less = validate_decision(_pass(checked_segment_ids=[s["segment_id"] for s in segments]), build_review_packet(ReviewStage.REPORT, review_id="r12", run_id="gate", brief=BRIEF, proposal="", store=None, draft=draft, draft_hash="h"))
    assert store_less.review_status != "passed"


# --------------------------------------------------------------------------- #
# R1 x R3: a review that runs out of allowance is a recorded budget stop
# --------------------------------------------------------------------------- #
def test_a_review_stopped_by_the_budget_is_not_reported_as_a_provider_failure(tmp_path):
    """The refusal is recorded as a budget stop; attempts that ran stay in the ledger."""
    import asyncio

    from open_deep_research.artifacts import Identity
    from open_deep_research.configuration import Configuration
    from open_deep_research.reviewer import run_review
    from open_deep_research.state import UsageRecord

    class Failing:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            raise ValueError("provider hiccup")

    store = ArtifactStore(root=tmp_path, run_id="budget-review")
    identity = Identity(run_id="budget-review", agent_instance_id="reviewer-01", role="reviewer")
    packet = build_review_packet(ReviewStage.REPORT, review_id="budget-review-1", run_id="budget-review", brief=BRIEF, proposal="", store=store, claims=[], draft=DRAFT, draft_hash="h")
    store.reserve_model_call(request_id="warmup", attempt_id="warmup#a1", role="reviewer", agent_instance_id="x", max_total_model_calls=5, max_total_tokens=10**9)
    store.record_usage(UsageRecord(request_id="warmup", attempt_id="warmup#a1", reservation_id="warmup#a1", role="reviewer", input_tokens=1, output_tokens=1))
    for index in range(4):
        store.reserve_model_call(request_id=f"s{index}", attempt_id=f"s{index}#a1", role="supervisor", agent_instance_id="x", max_total_model_calls=5, max_total_tokens=10**9)

    decision = asyncio.run(
        run_review(packet=packet, reviewer_identity=identity, store=store, model=Failing(), config=Configuration(max_total_model_calls=5, max_structured_output_retries=3), model_name="fake")
    )
    assert "budget" in decision.error
    assert decision.review_status != "passed"
