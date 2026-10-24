"""End-to-end runs of both compiled graphs with a scripted model boundary.

Each test runs the graph exactly once; the final state is read from the same
checkpointer (see ``harness.py``). Real providers are never contacted.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from open_deep_research.state import BriefExtraction, RawReviewOutput, ResearchQuestion
from tests.odr.conftest import ToolCalls, render
from tests.odr.harness import run_graph


def requirement_ids(text: str):
    """Return the requirement ids listed in a rendered review packet."""
    import re

    return re.findall(r"^- (req-\d+) \[", text, re.M)


def uncovered_requirement_ids(text: str):
    """Return mandatory requirement ids that the packet marks as missing coverage."""
    import re

    found = re.findall(r"^- (req-\d+) \[mandatory\].*\(state=missing;", text, re.M)
    return found


def segment_ids(text: str):
    """Return the draft segment ids shown in a rendered review packet."""
    import re

    return re.findall(r'<Segment id="(S\d+)"', text)


def base_answer(messages, schema, tools, *, supervisor_script=None):
    """Answer brief / requirement / supervisor / writer calls deterministically.

    ``supervisor_script`` is an optional list of tool messages returned in order, so a
    test can drive a dispatch -> complete sequence.
    """
    if supervisor_script is not None:
        pass
    if schema is ResearchQuestion:
        return ResearchQuestion(research_brief="When was version 2.0 released and how does it compare with 1.0?")
    if schema is BriefExtraction:
        return BriefExtraction(requirements=["Find the release date", "Compare the two versions"], constraints=["answer in English"])
    if schema is RawReviewOutput:
        text = render(messages)
        checked = [line[2:].split(" ", 1)[0] for line in text.splitlines() if line.startswith("- C-")]
        # A reviewer that verifies coverage says so explicitly: the program requires
        # every mandatory requirement and every shown draft segment to be checked.
        return RawReviewOutput(
            verdict="pass",
            summary="checked the packet; every mandatory requirement is covered by the cited evidence",
            findings=[],
            checked_claim_ids=checked,
            checked_requirement_ids=requirement_ids(text),
            checked_segment_ids=segment_ids(text),
            # A requirement with no archived evidence may still pass when the draft
            # states it as an explicit unknown/limitation (the reviewer says so here).
            acknowledged_unknown_requirement_ids=uncovered_requirement_ids(text),
        )
    if tools:
        names = [getattr(t, "name", str(t)) for t in tools]
        if "ConductResearch" in names:
            return ToolCalls.call("ResearchComplete", {})
        return ToolCalls.call("ResearchComplete", {})
    text = render(messages)
    if "report writer" in text:
        # The example report cites no archived evidence id, so it may only be
        # delivered as a limited run (R1): a report whose sources do not exist cannot
        # pass a verifiable-delivery check.
        return AIMessage(content="# Report\n\nVersion 2.0 was released in 2024 (source: https://example.com/a).\n\n### Sources\n[1] Example: https://example.com/a")
    return AIMessage(content="done")


def test_baseline_policy_graph_keeps_original_topology(scripted, artifact_root):
    """Arm A: baseline context policy, no reviewer."""
    from open_deep_research import deep_researcher as dr

    nodes = set(dr.build_deep_researcher().get_graph().nodes)
    assert "final_report_generation" in nodes
    assert "review_report" not in nodes
    result = run_graph(scripted(base_answer), review_enabled=False, context_policy="baseline")
    assert result.state["final_report"]
    assert "final_report_generation" in result.nodes
    assert "review_report" not in result.nodes
    # Every model call was recorded: usage aggregate, request record and manifest.
    metrics = artifact_root / "run-test" / "metrics"
    assert (metrics / "usage.json").exists()
    calls = result.runtime.requests
    assert [c["role"] for c in calls][:2] == ["brief_writer", "brief_writer"]
    manifests = list((artifact_root / "run-test" / "agents").rglob("*.manifest.json"))
    assert manifests
    # The run-level channels are declared on the parent but stay unused without review.
    assert result.state.get("review_history") == {}
    assert result.state.get("max_iterations") == 6


def test_scoped_graph_keeps_original_topology(scripted, artifact_root):
    """Arm B: scoped context policy, no reviewer."""
    result = run_graph(scripted(base_answer), review_enabled=False, context_policy="scoped")
    assert result.state["final_report"]
    assert "final_report_generation" in result.nodes
    assert "review_report" not in result.nodes
    assert (artifact_root / "run-test" / "metrics" / "usage.json").exists()


def test_review_graph_delivers_only_after_pass(scripted, artifact_root):
    """Arm D: scoped context policy with the reviewer gate.

    The reviewer is the role that reads the draft: these assertions cover what the
    reviewer saw (evidence + draft segments) and that delivery followed a pass. The
    uncited example draft cannot be fully verified, so the run ends as a limited
    delivery after the revision cap (see test_review_flows for the untruncated case).
    """
    result = run_graph(scripted(base_answer), review_enabled=True)
    state = result.state
    assert result.count("review_report") == 3
    assert result.count("report_writer") == 3
    assert result.count("finalize") == 0
    assert result.count("finalize_partial") == 1
    assert state["run_status"] == "partial"
    assert state["review_status"] == "inconclusive"
    assert "Limitations" in state["final_report"]
    reviewer = [r for r in result.runtime.requests if r["role"] == "reviewer"]
    assert reviewer, "the report review must run"
    for request in reviewer:
        assert "You are a research supervisor" not in request["text"]
    # R2: the reviewer saw every draft segment, not a truncated head.
    report_reviews = [r for r in reviewer if '<ReviewPacket stage="report"' in r["text"]]
    assert report_reviews
    for request in report_reviews:
        assert '<Segment id="S1"' in request["text"]
        assert "### Sources" in request["text"]
    assert any(p.name.startswith("reviewer-") for p in (artifact_root / "run-test" / "agents").iterdir())
    assert (artifact_root / "run-test" / "reports" / "final.md").exists()


def test_baseline_policy_review_graph_delivers(scripted, artifact_root):
    """Arm C: baseline context policy with the reviewer gate.

    The baseline writer output cites no archived evidence id and the run archived no
    claims, so the report review must refuse a full pass (R1): the delivery is an
    explicit, limited one instead of a "passed" report built from unverifiable text.
    """
    result = run_graph(scripted(base_answer), review_enabled=True, context_policy="baseline")
    state = result.state
    assert state["final_report"]
    assert "review_report" in result.nodes
    assert result.count("review_report") == 3
    assert result.count("finalize") == 0
    assert result.count("finalize_partial") == 1
    assert state["run_status"] == "partial"
    assert state["review_status"] == "inconclusive"
    assert "Limitations" in state["final_report"]
    assert (artifact_root / "run-test" / "reports" / "final.md").exists()


def test_static_delivery_edges_are_not_duplicated():
    """A Command(goto=...) is the only route out of the review/delivery decisions."""
    from open_deep_research import deep_researcher as dr

    graph = dr.build_deep_researcher(review_enabled=True)
    static_sources = {source for source, _ in graph.builder.edges}
    assert "review_report" not in static_sources
    assert "supervisor_return" not in static_sources
    assert graph.channels["final_report"].__class__.__name__ == "LastValue"


def test_run_level_channels_exist_on_both_graphs():
    from open_deep_research import deep_researcher as dr

    parent = dr.build_deep_researcher(review_enabled=True)
    subgraph = dr.review_supervisor_subgraph
    for key in ("research_iterations", "max_iterations", "task_ledger", "review_history", "report_ready"):
        assert key in parent.channels, key
        assert key in subgraph.channels, key
    for key in ("report_version", "writing_constraints", "final_report_draft_hash"):
        assert key in parent.channels, key
        assert key not in subgraph.channels, key