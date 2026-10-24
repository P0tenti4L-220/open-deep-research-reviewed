"""Review routing flows: rewrite, return-to-research, stop and budget exhaustion."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from open_deep_research.state import RawReviewOutput, ReviewFinding
from tests.odr.conftest import ToolCalls, render
from tests.odr.harness import run_graph
from tests.odr.test_graph_runs import base_answer, requirement_ids, segment_ids, uncovered_requirement_ids


def _report_requests(runtime):
    return [r for r in runtime.requests if r["role"] == "reviewer" and '<ReviewPacket stage="report"' in r["text"]]


def _tool_name(tool) -> str:
    """Return the name of a LangChain tool, a pydantic schema class or a dict."""
    return str(getattr(tool, "name", None) or getattr(tool, "__name__", None) or tool)


def _raw(verdict: str, messages, *, requirement_id: str = "") -> RawReviewOutput:
    text = render(messages)
    checked = [line[2:].split(" ", 1)[0] for line in text.splitlines() if line.startswith("- C-")]
    if verdict == "pass":
        # A pass states exactly what was verified: the claims, the mandatory
        # requirement coverage and the draft segments the reviewer read.
        return RawReviewOutput(
            verdict="pass",
            summary="checked the packet; every mandatory requirement is covered by the cited evidence",
            findings=[],
            checked_claim_ids=checked,
            checked_requirement_ids=requirement_ids(text),
            acknowledged_unknown_requirement_ids=uncovered_requirement_ids(text),
            checked_segment_ids=segment_ids(text),
        )
    finding = ReviewFinding(
        requirement_id=requirement_id,
        issue_type="missing_requirement" if verdict == "insufficient_evidence" else "other",
        severity="blocking" if verdict in {"insufficient_evidence", "stop"} else "major",
        required_action={
            "revise": "Clarify the presentation without changing the evidence.",
            "insufficient_evidence": "Find the missing release-date evidence.",
            "stop": "Stop and deliver what is supported.",
        }[verdict],
    )
    return RawReviewOutput(verdict=verdict, summary=f"{verdict} from the scripted reviewer", findings=[finding])


def reviewer_script(report_verdicts, *, research_verdict: str = "pass"):
    """Return an answer function with a scripted report-review sequence."""
    pending = list(report_verdicts)
    calls = {"supervisor": 0, "researcher": 0}

    def answer(messages, schema, tools):
        text = render(messages)
        names = [_tool_name(t) for t in tools]
        if schema is RawReviewOutput:
            if '<ReviewPacket stage="report"' in text:
                verdict = pending.pop(0) if pending else "pass"
                return _raw(verdict, messages, requirement_id="req-1")
            return _raw(research_verdict, messages)
        if "ConductResearch" in names:
            calls["supervisor"] += 1
            if calls["supervisor"] == 1:
                return ToolCalls.call(
                    "ConductResearch",
                    {"research_topic": "When exactly was version 2.0 released, and when was version 1.0 released?"},
                )
            return ToolCalls.call("ResearchComplete", {})
        if "tavily_search" in names or "think_tool" in names:
            calls["researcher"] += 1
            return AIMessage(content="Enough context was gathered for this task.")
        if "Research transcript" in text:
            return AIMessage(content="Version 2.0 was released in 2024 [E1](https://example.com/a).")
        return base_answer(messages, schema, tools)

    return answer


def test_report_revision_rewrites_and_increments_version(scripted, artifact_root):
    """A revise decision rewrites the draft; the uncited example cannot pass (R1).

    The rewrite still cites no archived evidence, so the final delivery is limited
    by the revision cap instead of being marked as a verified pass.
    """
    answer = reviewer_script(["revise", "pass"])
    result = run_graph(scripted(answer), review_enabled=True, max_revision_rounds=2, max_review_events=8)
    state = result.state
    assert len((state["review_history"] or {}).get("revisions", [])) >= 1
    assert result.count("report_writer") == 3
    assert result.count("review_report") == 3
    assert result.count("finalize") == 0
    assert result.count("finalize_partial") == 1
    assert state["report_version"] == 3
    assert len(_report_requests(result.runtime)) == 3
    # R2: every report review saw the draft in full segments.
    for request in _report_requests(result.runtime):
        assert '<Segment id="S1"' in request["text"]
    assert (artifact_root / "run-test" / "reports" / "final.md").exists()


def test_revision_cap_forces_partial_delivery(scripted, artifact_root):
    """Regression: one revision round must mean two reviews, not three."""
    answer = reviewer_script(["revise", "revise", "pass"])
    result = run_graph(scripted(answer), review_enabled=True, max_revision_rounds=1, max_review_events=8)
    state = result.state
    assert state["run_status"] == "partial"
    assert state["review_status"] == "inconclusive"
    assert state["report_version"] == 2
    assert result.count("report_writer") == 2
    assert result.count("review_report") == 2
    assert result.count("finalize") == 0
    assert result.count("finalize_partial") == 1
    assert len(_report_requests(result.runtime)) == 2
    assert "Limitations" in state["final_report"]
    # The delivered text is the latest draft, not a mix of both drafts.
    assert "Version 2.0 was released in 2024" in state["final_report"]


def test_insufficient_evidence_returns_to_research_with_same_run(scripted, artifact_root):
    """A report review that finds missing evidence re-enters research in the same run."""
    answer = reviewer_script(["insufficient_evidence", "pass"])
    result = run_graph(
        scripted(answer),
        review_enabled=True,
        run_id="return-run",
        max_revision_rounds=2,
        max_review_events=8,
        max_researcher_iterations=5,
    )
    state = result.state
    assert state["run_id"] == "return-run"
    assert state["run_status"] == "partial"
    # The scripted research never archives evidence and the draft never cites any, so
    # every report review refuses a verified pass and the run ends limited (R1).
    assert result.count("review_report") == 3
    assert result.count("finalize") == 0
    assert result.count("finalize_partial") == 1
    assert "Limitations" in state["final_report"]


def test_report_review_reenters_research_in_the_same_run(scripted, artifact_root):
    """A report review that finds a missing citation sends the run back to research.

    The re-entry keeps the same run id, store, ledger and iteration counter (a subgraph
    switch cannot reset them). The scripted research never archives evidence, so the
    run still ends with a limited delivery rather than an unverified pass.
    """
    pending = ["insufficient_evidence", "pass"]
    calls = {"supervisor": 0, "researcher": 0}

    def answer(messages, schema, tools):
        text = render(messages)
        names = [_tool_name(t) for t in tools]
        if schema is RawReviewOutput:
            if '<ReviewPacket stage="report"' in text:
                verdict = pending.pop(0) if pending else "pass"
                return _raw(verdict, messages, requirement_id="req-1")
            return _raw("pass", messages)
        if "ConductResearch" in names:
            calls["supervisor"] += 1
            if calls["supervisor"] == 1:
                return ToolCalls.call("ConductResearch", {"research_topic": "When exactly was version 2.0 released?"})
            return ToolCalls.call("ResearchComplete", {})
        if "tavily_search" in names or "think_tool" in names:
            calls["researcher"] += 1
            return AIMessage(content="Enough context was gathered for this task.")
        if "Research transcript" in text:
            return AIMessage(content="Version 2.0 was released in 2024 [E1](https://example.com/a).")
        return base_answer(messages, schema, tools)

    result = run_graph(
        scripted(answer),
        review_enabled=True,
        run_id="return-run",
        max_revision_rounds=2,
        max_review_events=8,
        max_researcher_iterations=3,
    )
    state = result.state
    # The same run, the same store: one artifact directory and one final report.
    runs = [p for p in artifact_root.iterdir() if p.is_dir()]
    assert [p.name for p in runs] == ["return-run"]
    (artifact_root / "return-run" / "reports" / "final.md").exists()
    # The first report review asked for evidence, the run researched again in place
    # and wrote a second draft; it is still not verifiable, so delivery stays limited.
    assert result.count("supervisor_return") == 1
    assert result.count("research_supervisor") == 2
    assert state["research_iterations"] >= 2
    assert result.count("report_writer") == 2
    assert result.count("finalize") == 0
    assert result.count("finalize_partial") == 1
    assert state["run_status"] == "partial"
