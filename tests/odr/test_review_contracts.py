"""Contract tests for the program-owned review decisions and state reducers."""

from __future__ import annotations

import pytest

from open_deep_research.reviewer import apply_review, validate_decision
from open_deep_research.state import (
    RawReviewOutput,
    ReviewDecision,
    ReviewFinding,
    ReviewPacket,
    ReviewStage,
    Verdict,
    review_history_reducer,
)


def _decision(verdict: Verdict, status: str = "changes_requested", draft_hash: str = "h") -> ReviewDecision:
    return ReviewDecision(
        review_id="review-1",
        stage=ReviewStage.REPORT,
        verdict=verdict,
        review_status=status,
        draft_hash=draft_hash,
    )


def _act(
    verdict: Verdict,
    *,
    stage: ReviewStage = ReviewStage.REPORT,
    status: str = "changes_requested",
    budget_open: bool = True,
    revisions: int = 0,
    max_revisions: int = 2,
    iterations_left: int = 3,
    decision_hash: str = "h",
    current_hash: str = "h",
) -> str:
    return apply_review(
        _decision(verdict, status, decision_hash),
        current_hash=current_hash,
        remaining_budget={"open": budget_open, "iterations_left": iterations_left},
        revision_rounds=revisions,
        max_revision_rounds=max_revisions,
        stage=stage,
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(verdict=Verdict.PASS, status="passed"), "finalize"),
        (dict(verdict=Verdict.REVISE), "rewrite_report"),
        (dict(verdict=Verdict.INSUFFICIENT_EVIDENCE), "continue_research"),
        (dict(verdict=Verdict.INSUFFICIENT_EVIDENCE, iterations_left=0), "finalize_partial"),
        (dict(verdict=Verdict.STOP), "finalize_partial"),
        (dict(verdict=Verdict.REVISE, budget_open=False), "finalize_partial"),
        (dict(verdict=Verdict.REVISE, revisions=2, max_revisions=2), "finalize_partial"),
        (dict(verdict=Verdict.PASS, status="passed", budget_open=False), "finalize_partial"),
        # A stale review cannot authorize delivery, even when its verdict is pass.
        (dict(verdict=Verdict.PASS, status="passed", decision_hash="old"), "rewrite_report"),
    ],
)
def test_report_stage_actions(kwargs, expected):
    assert _act(**kwargs) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(verdict=Verdict.PASS, status="passed"), "finalize"),
        (dict(verdict=Verdict.REVISE), "continue_research"),
        (dict(verdict=Verdict.INSUFFICIENT_EVIDENCE), "continue_research"),
        (dict(verdict=Verdict.STOP), "finalize_partial"),
        (dict(verdict=Verdict.REVISE, budget_open=False), "finalize_partial"),
        (dict(verdict=Verdict.REVISE, revisions=2, max_revisions=2), "finalize_partial"),
        (dict(verdict=Verdict.REVISE, decision_hash="old"), "continue_research"),
        (dict(verdict=Verdict.REVISE, decision_hash="old", max_revisions=0), "finalize_partial"),
    ],
)
def test_research_stage_actions(kwargs, expected):
    assert _act(stage=ReviewStage.READINESS, **kwargs) == expected


def _packet() -> ReviewPacket:
    return ReviewPacket(
        review_id="review-1",
        stage=ReviewStage.REPORT,
        run_id="run-1",
        objective="answer the question",
        claim_ids=["C-1"],
        evidence_ids=["E1"],
    )


def test_validate_decision_never_turns_failure_into_pass():
    packet = _packet()
    missing = validate_decision(None, packet, error="review timed out")
    assert missing.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert missing.review_status == "unreviewed"
    assert "timed out" in missing.error

    unchecked = validate_decision(
        RawReviewOutput(verdict="pass", summary="nothing checked", checked_claim_ids=[]),
        packet,
    )
    assert unchecked.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert unchecked.review_status == "inconclusive"

    bogus = validate_decision(
        RawReviewOutput(
            verdict="pass",
            summary="checked with a made-up source",
            findings=[ReviewFinding(evidence_refs=["E404"], issue_type="unsupported_claim")],
            checked_claim_ids=["C-1"],
        ),
        packet,
    )
    assert bogus.verdict == Verdict.INSUFFICIENT_EVIDENCE
    assert "non-existent evidence" in bogus.error

    blocking = validate_decision(
        RawReviewOutput(
            verdict="pass",
            summary="pass with a blocking finding",
            findings=[ReviewFinding(severity="blocking", issue_type="unsupported_claim")],
            checked_claim_ids=["C-1"],
        ),
        packet,
    )
    assert blocking.verdict == Verdict.REVISE
    assert blocking.review_status == "changes_requested"

    good = validate_decision(
        RawReviewOutput(verdict="pass", summary="checked", checked_claim_ids=["C-1"]),
        packet,
    )
    assert good.verdict == Verdict.PASS
    assert good.review_status == "passed"


def test_review_history_reducer_deduplicates_events_and_replaces_open_issues():
    current = {
        "reviews": [{"review_id": "review-1", "stage": "plan"}],
        "revisions": [{"stage": "report", "review_id": "review-1"}],
        "open_issues": ["old issue"],
        "cache": {"reviews": {}},
    }
    new = {
        "reviews": [
            {"review_id": "review-1", "stage": "plan"},
            {"review_id": "review-2", "stage": "report"},
        ],
        "revisions": [
            {"stage": "report", "review_id": "review-1"},
            {"stage": "report", "review_id": "review-2"},
        ],
        "proposals": [
            {"proposal_id": "proposal-1", "kind": "dispatch"},
            {"proposal_id": "proposal-1", "kind": "dispatch"},
        ],
        "open_issues": ["new issue"],
    }
    merged = review_history_reducer(current, new)
    assert [r["review_id"] for r in merged["reviews"]] == ["review-1", "review-2"]
    assert [r["review_id"] for r in merged["revisions"]] == ["review-1", "review-2"]
    assert len(merged["proposals"]) == 1
    assert merged["open_issues"] == ["new issue"]
    assert merged["cache"] == {"reviews": {}}