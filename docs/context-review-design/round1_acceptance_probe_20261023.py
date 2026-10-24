"""Offline acceptance probes for the first round (no real provider is contacted).

Run from the repository root::

    .venv/bin/python docs/context-review-design/round1_acceptance_probe_20261023.py /tmp/round1-probe.json

Each probe records ``expected`` (the post-fix contract) and ``ok`` (observed behaviour
satisfies it), so the output is a pass/fail record. The file
``round1_acceptance_probe_20261023.json`` next to this script is the pre-fix history and
is never overwritten; write new results to a new path.
"""
import asyncio
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ["ARTIFACT_ROOT"] = tempfile.mkdtemp(prefix="odr-round1-probes-")
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import pytest  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402

from open_deep_research import deep_researcher as dr  # noqa: E402
from open_deep_research.artifacts import ArtifactStore, Identity  # noqa: E402
from open_deep_research.configuration import Configuration  # noqa: E402
from open_deep_research.context import ContextBuilder, ModelCallBudget, assemble  # noqa: E402
from open_deep_research.reviewer import (  # noqa: E402
    apply_review,
    build_review_packet,
    render_packet,
    validate_decision,
)
from open_deep_research.state import (  # noqa: E402
    RawReviewOutput,
    Requirement,
    ResearchBrief,
    ReviewStage,
    UsageRecord,
)
from tests.odr.conftest import ScriptedRuntime  # noqa: E402
from tests.odr.harness import run_graph  # noqa: E402
from tests.odr.test_graph_runs import base_answer  # noqa: E402

results = {}


def record(name, expected, observed, ok):
    """Store one probe outcome."""
    results[name] = {"expected": expected, "observed": observed, "ok": bool(ok)}


def _route(decision, packet, *, stage=ReviewStage.REPORT):
    return apply_review(
        decision,
        current_hash=packet.draft_hash,
        remaining_budget={"open": True, "iterations_left": 2},
        revision_rounds=0,
        max_revision_rounds=2,
        stage=stage,
    )


# --------------------------------------------------------------------------- #
# Shared fixture: one archived evidence item, two claims, one brief.
# --------------------------------------------------------------------------- #
brief = ResearchBrief(objective="Check both claims", requirements=[Requirement(requirement_id="R1", statement="Check both")])
store = ArtifactStore(root=os.environ["ARTIFACT_ROOT"], run_id="probe")
researcher = Identity(run_id="probe", agent_instance_id="researcher-probe", role="researcher", task_id="task-probe")
evidence = store.add_evidence(researcher, content="C1 and C2 are supported.", excerpt="C1 and C2 are supported.", source_url="https://example.invalid/offline-fixture", public=True)
claims = [{"claim_id": cid, "statement": cid, "evidence_refs": [evidence.evidence_id], "requirement_ids": ["R1"]} for cid in ("C1", "C2")]
draft = "# Report\n\nC1 and C2 are supported [E1] (claims C1, C2)."


# --------------------------------------------------------------------------- #
# R1: an unchecked claim / an unarchived citation cannot authorize delivery.
# --------------------------------------------------------------------------- #
dangling = [{**c, "evidence_refs": ["E-missing"]} for c in claims]
packet = build_review_packet(ReviewStage.REPORT, review_id="r1", run_id="probe", brief=brief, proposal="review", store=store, claims=dangling, draft=draft, draft_hash="draft-hash")
decision = validate_decision(RawReviewOutput(verdict="pass", summary="Only checked C1", checked_claim_ids=["C1"], checked_segment_ids=["S1"]), packet)
route = _route(decision, packet)
record(
    "R1_partial_and_missing_evidence",
    "a pass with an unchecked claim and an unarchived citation cannot finalize",
    {"verdict": decision.verdict.value, "status": decision.review_status, "unchecked": decision.unchecked_claim_ids, "missing_evidence": packet.missing_evidence_ids, "route": route},
    decision.review_status != "passed" and route != "finalize",
)


# --------------------------------------------------------------------------- #
# R2: the tail of a long draft reaches the reviewer (no 6000-char cut).
# --------------------------------------------------------------------------- #
marker = "UNSUPPORTED_FINAL_ASSERTION_987"
long_packet = build_review_packet(ReviewStage.REPORT, review_id="r2", run_id="probe", brief=brief, proposal="review", store=store, claims=claims, draft="# Report\n" + "x" * 6100 + "\n\n" + marker, draft_hash="long-draft")
long_decision = validate_decision(RawReviewOutput(verdict="pass", summary="Passed visible draft", checked_claim_ids=["C1", "C2"], checked_segment_ids=["S1"]), long_packet)
long_route = _route(long_decision, long_packet)
render = render_packet(long_packet)
record(
    "R2_draft_tail_coverage",
    "the whole draft is shown to the reviewer; an unbound tail assertion cannot finalize",
    {"marker_visible": marker in render, "segments": len(long_packet.report_segments), "segment_chars": [len(seg["text"]) for seg in long_packet.report_segments], "status": long_decision.review_status, "route": long_route},
    marker in render and long_route != "finalize",
)


# --------------------------------------------------------------------------- #
# R3 + R4: budget is reserved per attempt; every attempt is counted for its role.
# --------------------------------------------------------------------------- #
class FlakyModel:
    """Fails twice, then answers."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls < 3:
            raise ValueError("local scripted failure")
        return RawReviewOutput(verdict="pass", summary="OK")


retry_store = ArtifactStore(root=os.environ["ARTIFACT_ROOT"], run_id="retry")
retry_identity = Identity(run_id="retry", agent_instance_id="reviewer-probe", role="reviewer")
retry_packet = ContextBuilder(Configuration(reviewer_model_max_tokens=200), retry_store).build(retry_identity, "reviewer", [HumanMessage(content="offline review request")], response_schema=RawReviewOutput)


def _seed(store, calls, cap):
    """Spend ``calls`` of a ``cap``-call budget on another role."""
    for index in range(calls):
        attempt_id = f"warmup-{index}#a1"
        store.reserve_model_call(request_id=f"warmup-{index}", attempt_id=attempt_id, role="supervisor", agent_instance_id="warmup", max_total_model_calls=cap, max_total_tokens=10**9)
        store.record_usage(UsageRecord(request_id=f"warmup-{index}", attempt_id=attempt_id, reservation_id=attempt_id, role="supervisor", input_tokens=1, output_tokens=1))


_seed(retry_store, 1, 3)
flaky = FlakyModel()
try:
    asyncio.run(assemble(flaky, retry_packet, store=retry_store, model_name="fake:offline", retries=3, budget=ModelCallBudget(retry_store, Configuration(max_total_model_calls=3))))
    refusal = None
except Exception as exc:  # noqa: BLE001 - the probe records the refusal
    refusal = f"{type(exc).__name__}: {exc}"
ledger = retry_store.run_dir / "control" / "calls.jsonl"
observed = {
    "cap": 3,
    "used_before": 1,
    "attempts_sent": flaky.calls,
    "refusal": refusal,
    "total_calls": retry_store.totals()["model_calls"],
    "reviewer_calls": retry_store.calls_by_role().get("reviewer", 0),
    "ledger_rows": len(ledger.read_text().splitlines()) if ledger.exists() else 0,
}
record(
    "R3_R4_retry_accounting",
    "retries stop when the remaining allowance is spent; every attempt is a ledger row and counts for the role",
    observed,
    observed["attempts_sent"] == 2 and observed["reviewer_calls"] == 2 and observed["total_calls"] == 3 and observed["ledger_rows"] == 3 and refusal is not None and "budget" in refusal,
)


class SlowModel:
    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        await asyncio.sleep(10)


timeout_store = ArtifactStore(root=os.environ["ARTIFACT_ROOT"], run_id="timeout")
timeout_packet = ContextBuilder(Configuration(), timeout_store).build(retry_identity, "reviewer", [HumanMessage(content="offline review request")], response_schema=RawReviewOutput)


async def _timed_out():
    return await asyncio.wait_for(assemble(SlowModel(), timeout_packet, store=timeout_store, model_name="fake:offline"), timeout=1)


try:
    asyncio.run(_timed_out())
    timeout_error = None
except asyncio.TimeoutError:
    timeout_error = "TimeoutError"
observed = {"error": timeout_error, "total_calls": timeout_store.totals()["model_calls"], "reviewer_calls": timeout_store.calls_by_role().get("reviewer", 0), "unknown_usage_calls": timeout_store.totals()["unknown_usage_calls"]}
record(
    "R4_timeout_accounting",
    "a request that was sent and then timed out counts once with unknown usage",
    observed,
    observed["error"] == "TimeoutError" and observed["total_calls"] == 1 and observed["reviewer_calls"] == 1 and observed["unknown_usage_calls"] == 1,
)


# --------------------------------------------------------------------------- #
# R3: a full graph run with a cap of one model call sends exactly one request.
# --------------------------------------------------------------------------- #
mp = pytest.MonkeyPatch()
runtime = ScriptedRuntime(base_answer).install(mp)
try:
    result = run_graph(runtime, review_enabled=True, run_id="budget-one", max_total_model_calls=1)
    observed = {"cap": 1, "requests_sent": len(runtime.requests), "roles": [r["role"] for r in runtime.requests], "status": result.state.get("run_status")}
finally:
    mp.undo()
record("R3_full_graph_budget_one", "a cap of 1 model call sends exactly one request", observed, observed["requests_sent"] <= 1)


# --------------------------------------------------------------------------- #
# R5: the published entry accepts plain input (no client-built runtime context).
# --------------------------------------------------------------------------- #
mp = pytest.MonkeyPatch()
runtime = ScriptedRuntime(base_answer).install(mp)
try:
    try:
        asyncio.run(dr.deep_researcher.ainvoke({"messages": [HumanMessage(content="Offline entry probe")]}, {"configurable": {"allow_clarification": False, "context_policy": "scoped", "reviewer_enabled": True}}))
        entry_error = None
    except Exception as exc:  # noqa: BLE001 - the probe records the error type
        entry_error = f"{type(exc).__name__}: {exc}"
finally:
    mp.undo()
roles = [r["role"] for r in runtime.requests]
record(
    "R5_public_entry",
    "plain JSON input runs through the exported entry and selects the requested graph",
    {"error": entry_error, "roles": roles},
    entry_error is None and "reviewer" in roles,
)

summary = {"passed": sum(1 for value in results.values() if value["ok"]), "total": len(results)}
output = json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2)
if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text(output + "\n", encoding="utf-8")
print(output)
