"""Model-boundary tests for the real ``assemble`` implementation.

The graph tests replace ``assemble`` wholesale, so they cannot catch a wrapper that
cannot bind tools or a response schema. These tests exercise the real binding order
and the per-attempt usage accounting with a local fake model (no provider).
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from open_deep_research.artifacts import Identity, get_store
from open_deep_research.configuration import Configuration
from open_deep_research.context import ContextBuilder, assemble
from open_deep_research.state import ResearchQuestion
from open_deep_research.utils import think_tool
from tests.odr.harness import run_graph
from tests.odr.test_graph_runs import base_answer


class FakeChatModel:
    """Minimal model with LangChain's binding surface and scripted failures."""

    def __init__(self, failures: int = 0):
        self.events = []
        self.failures = failures
        self.schema = None

    def bind_tools(self, tools):
        self.events.append(("bind_tools", tuple(sorted(str(getattr(t, "name", t)) for t in tools))))
        return self

    def with_structured_output(self, schema):
        self.events.append(("with_structured_output", schema.__name__))
        self.schema = schema
        return self

    async def ainvoke(self, messages):
        self.events.append(("ainvoke", self.schema.__name__ if self.schema else None))
        if self.failures:
            self.failures -= 1
            raise ValueError("transient provider failure")
        return ResearchQuestion(research_brief="assembled")


def _packet(store):
    identity = Identity(run_id=store.run_id, agent_instance_id="brief-01", role="brief_writer")
    builder = ContextBuilder(Configuration(artifacts_enabled=True), store)
    return builder.build(
        identity,
        "brief_writer",
        [HumanMessage(content="offline request")],
        tools=[think_tool],
        response_schema=ResearchQuestion,
    )


def test_assemble_binds_tools_and_schema_before_the_retry(artifact_root):
    import asyncio

    store = get_store("binding-run")
    packet = _packet(store)
    model = FakeChatModel(failures=2)
    result = asyncio.run(assemble(model, packet, store=store, model_name="fake:model", retries=3))

    assert result.research_brief == "assembled"
    assert model.events[0][0] == "bind_tools"
    assert model.events[1] == ("with_structured_output", "ResearchQuestion")
    assert [event[0] for event in model.events[2:]] == ["ainvoke", "ainvoke", "ainvoke"]

    # Every attempt (including the two failures) is a counted model call, and each
    # attempt has its own ledger row: retries never overwrite each other.
    assert store.totals()["model_calls"] == 3
    assert store.calls_by_role().get("brief_writer") == 3
    rows = [json.loads(line) for line in (store.run_dir / "control" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["counting_method"] for row in rows] == ["failed_call", "failed_call", "unknown"]
    assert [row["attempt_id"].rsplit("#a", 1)[-1] for row in rows] == ["1", "2", "3"]
    assert len({row["attempt_id"] for row in rows}) == 3
    events = [json.loads(line) for line in (store.run_dir / "control" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    failed = [e for e in events if e.get("type") == "model_call_failed"]
    assert [e["attempt_id"].rsplit("#a", 1)[-1] for e in failed] == ["1", "2"]


def test_assemble_stops_after_the_configured_attempts(artifact_root):
    import asyncio

    store = get_store("binding-fail-run")
    packet = _packet(store)
    model = FakeChatModel(failures=5)
    with pytest.raises(ValueError, match="transient provider failure"):
        asyncio.run(assemble(model, packet, store=store, model_name="fake:model", retries=2))
    assert store.totals()["model_calls"] == 2


def test_assemble_rejects_a_retry_wrapper(artifact_root):
    """The failure mode the audit reproduced: a wrapper cannot bind a schema."""
    import asyncio

    class RetryWrapper:
        async def ainvoke(self, messages):  # pragma: no cover - must not be reached
            raise AssertionError("should not be invoked")

    store = get_store("wrapper-run")
    packet = _packet(store)
    with pytest.raises(TypeError, match="retry wrapper"):
        asyncio.run(assemble(RetryWrapper(), packet, store=store))


def test_every_graph_model_call_passes_a_bindable_model(scripted, artifact_root):
    result = run_graph(scripted(base_answer), review_enabled=True)
    assert result.runtime.requests
    for request in result.runtime.requests:
        assert request["bindable"], (request["role"], type(request["model_object"]))
    roles = {request["role"] for request in result.runtime.requests}
    assert {"brief_writer", "supervisor", "reviewer", "writer"} <= roles
    # Structured calls request a bounded retry from the boundary, not a wrapper.
    structured = [r for r in result.runtime.requests if r["schema"] is not None]
    assert structured
    assert all(r["retries"] >= 1 for r in structured)

# --------------------------------------------------------------------------- #
# R3: the run-wide model-call budget is reserved per attempt, atomically
# --------------------------------------------------------------------------- #
def test_assemble_never_exceeds_the_run_budget(artifact_root):
    """A retrying call and a parallel burst cannot spend more than the cap."""
    import asyncio

    from open_deep_research.context import BudgetExhausted, ModelCallBudget

    store = get_store("budget-run")
    packet = _packet(store)
    budget = ModelCallBudget(store, max_calls=1, max_tokens=10_000_000)
    model = FakeChatModel(failures=3)
    with pytest.raises(BudgetExhausted) as excinfo:
        asyncio.run(assemble(model, packet, store=store, model_name="fake:model", retries=3, budget=budget))
    assert "budget" in str(excinfo.value)
    # Exactly one attempt was authorised and sent; the retry was refused, not sent.
    assert [e[0] for e in model.events[2:]] == ["ainvoke"]
    assert store.totals()["model_calls"] == 1
    assert model.failures == 2  # only the first (authorised) attempt reached the model


def test_budget_reservations_are_atomic_under_concurrency(artifact_root):
    """With one allowance left, exactly one of N concurrent reservations wins."""
    import asyncio

    from open_deep_research.context import BudgetExhausted, ModelCallBudget

    store = get_store("budget-race")
    budget = ModelCallBudget(store, max_calls=3, max_tokens=10_000_000)

    def reserve(i: int) -> bool:
        try:
            budget.reserve(request_id=f"req-{i}", attempt_id=f"req-{i}#a1", role="researcher", estimated_input_tokens=10, reserved_output_tokens=10)
            return True
        except BudgetExhausted:
            return False

    async def gather():
        # Each worker reserves in its own task; the store lock serialises the check.
        return await asyncio.gather(*(asyncio.to_thread(reserve, i) for i in range(10)))

    granted = asyncio.run(gather())
    assert sum(granted) == 3
    reservations = store.reservations()
    assert len(reservations) == 3
    assert all(entry["settled"] is False for entry in reservations)
    # Every blocked attempt is traceable in the event ledger.
    events = [json.loads(line) for line in (store.run_dir / "control" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len([e for e in events if e.get("type") == "model_call_blocked"]) == 7


def test_full_graph_respects_a_one_call_budget(scripted, artifact_root):
    """A full graph run with a cap of 1 model call sends exactly one request."""
    runtime = scripted(base_answer)
    result = run_graph(runtime, review_enabled=True, max_total_model_calls=1)
    assert len(runtime.requests) == 1
    assert result.state["run_status"] == "partial"
    assert result.count("finalize_partial") == 1
    store = get_store("run-test")
    assert store.totals()["model_calls"] == 1


def test_unknown_usage_is_charged_conservatively(artifact_root):
    """A call whose provider reports no usage still consumes a reserved allowance."""
    store = get_store("budget-unknown")
    from open_deep_research.context import ModelCallBudget

    budget = ModelCallBudget(store, max_calls=2, max_tokens=1_000)
    budget.reserve(request_id="r1", attempt_id="r1#a1", role="reviewer", estimated_input_tokens=900, reserved_output_tokens=4000)
    from open_deep_research.state import UsageRecord

    store.record_usage(UsageRecord(request_id="r1", attempt_id="r1#a1", reservation_id="r1#a1", role="reviewer", counting_method="unknown"))
    # The token budget is now consumed by the conservative output estimate.
    import pytest as _pytest

    from open_deep_research.context import BudgetExhausted

    with _pytest.raises(BudgetExhausted):
        budget.reserve(request_id="r2", attempt_id="r2#a1", role="reviewer", estimated_input_tokens=900, reserved_output_tokens=4000)


# --------------------------------------------------------------------------- #
# R4: one ledger row per attempt; timeouts are counted, cancellations propagate
# --------------------------------------------------------------------------- #
def test_each_retry_is_its_own_role_counted_attempt(artifact_root):
    import asyncio

    from open_deep_research.state import RawReviewOutput

    class ReviewModel:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls < 3:
                raise ValueError("transient provider failure")
            return RawReviewOutput(verdict="pass", summary="ok")

    store = get_store("retry-run")
    identity = Identity(run_id="retry-run", agent_instance_id="reviewer-01", role="reviewer")
    config = Configuration(reviewer_model_max_tokens=100)
    from open_deep_research.context import ContextBuilder

    builder = ContextBuilder(config, store)
    packet = builder.build(identity, "reviewer", [HumanMessage(content="review this")], response_schema=RawReviewOutput)
    model = ReviewModel()
    asyncio.run(assemble(model, packet, store=store, model_name="fake:model", retries=3))

    assert model.calls == 3
    assert store.totals()["model_calls"] == 3
    assert store.calls_by_role()["reviewer"] == 3
    rows = [json.loads(line) for line in (store.run_dir / "control" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert len({row["attempt_id"] for row in rows}) == 3
    assert all(row["request_id"] == packet.request_id for row in rows)


def test_a_timed_out_call_is_counted_once_with_unknown_usage(artifact_root):
    """The provider may already have been charged: the attempt must be in the ledger."""
    import asyncio

    from open_deep_research.state import RawReviewOutput

    class SlowModel:
        def with_structured_output(self, schema):
            return self

        async def ainvoke(self, messages):
            await asyncio.sleep(10)

    store = get_store("timeout-run")
    identity = Identity(run_id="timeout-run", agent_instance_id="reviewer-01", role="reviewer")
    from open_deep_research.context import ContextBuilder

    packet = ContextBuilder(Configuration(), store).build(identity, "reviewer", [HumanMessage(content="review")], response_schema=RawReviewOutput)

    async def run():
        return await asyncio.wait_for(assemble(SlowModel(), packet, store=store, model_name="fake:model"), timeout=0.2)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(run())
    assert store.totals()["model_calls"] == 1
    assert store.calls_by_role()["reviewer"] == 1
    rows = [json.loads(line) for line in (store.run_dir / "control" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["counting_method"] == "timed_out"
    assert store.totals()["unknown_usage_calls"] == 1
    # The granted reservation is settled by the timeout row; no phantom call remains.
    (reservation,) = store.reservations()
    assert reservation["settled"] is True


def test_role_and_call_counts_survive_a_restart(artifact_root):
    """Reloading the run metadata from disk keeps role counts consistent."""
    from open_deep_research.artifacts import reset_registry, _STORES

    store = get_store("restart-run")
    for i in range(1, 4):
        store.record_usage(
            __import__("open_deep_research.state", fromlist=["UsageRecord"]).UsageRecord(
                request_id="req-1", attempt_id=f"req-1#a{i}", role="reviewer", input_tokens=10, output_tokens=5
            )
        )
    reset_registry()
    _STORES.clear()
    reloaded = get_store("restart-run")
    assert reloaded.totals()["model_calls"] == 3
    assert reloaded.calls_by_role()["reviewer"] == 3
    assert reloaded.totals()["input_tokens"] == 30
