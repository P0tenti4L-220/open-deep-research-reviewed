"""R5 regression tests: the published entry works with plain input.

``langgraph.json`` exports ``deep_researcher`` from
``src/open_deep_research/deep_researcher.py``. Studio, the LangGraph API server and a
plain ``await deep_researcher.ainvoke(input, config)`` all reach the same entry, which
must bind the run's identity, store and graph in the host before any node runs.

No provider is contacted: the model boundary is scripted.
"""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest
from langchain_core.messages import HumanMessage

from open_deep_research import deep_researcher as dr
from open_deep_research.artifacts import get_store
from open_deep_research.configuration import Configuration
from tests.odr.conftest import ScriptedRuntime
from tests.odr.test_graph_runs import base_answer


def entry():
    """Return the object ``langgraph.json`` resolves to."""
    spec = json.loads(open("langgraph.json", encoding="utf-8").read())["graphs"]["Deep Researcher"]
    module_path, attr = spec.split(":")
    module_name = module_path.removeprefix("./src/").removesuffix(".py").replace("/", ".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def test_langgraph_json_points_at_a_working_entry():
    exported = entry()
    assert hasattr(exported, "ainvoke")
    nodes = exported.get_graph().nodes
    assert "odr" in nodes


def test_plain_input_runs_the_published_entry(scripted, artifact_root):
    """No ``__odr_ctx__`` is supplied: the host creates it from the caller's thread."""
    runtime = scripted(base_answer)
    out = asyncio.run(
        entry().ainvoke(
            {"messages": [HumanMessage(content="Compare v1.0 and v2.0 release dates. Answer in English.")]},
            {"configurable": {"thread_id": "studio-thread", "allow_clarification": False, "context_policy": "scoped"}},
        )
    )
    assert out.get("final_report")
    assert out.get("run_id") == "studio-thread"
    roles = [request["role"] for request in runtime.requests]
    assert roles[:2] == ["brief_writer", "brief_writer"]
    # The run's artifacts landed under the thread-derived run id.
    assert (artifact_root / "studio-thread" / "metrics" / "usage.json").exists()


def test_reviewer_switch_selects_the_review_graph(scripted, artifact_root):
    """The public entry must run the graph that matches the requested configuration."""
    runtime = scripted(base_answer)
    asyncio.run(
        entry().ainvoke(
            {"messages": [HumanMessage(content="Reviewer on please.")]},
            {"configurable": {"thread_id": "review-on", "allow_clarification": False, "context_policy": "scoped", "reviewer_enabled": True}},
        )
    )
    assert "reviewer" in {request["role"] for request in runtime.requests}
    assert get_store("review-on").bound_graph() == "odr_review"

    asyncio.run(
        entry().ainvoke(
            {"messages": [HumanMessage(content="Reviewer off please.")]},
            {"configurable": {"thread_id": "review-off", "allow_clarification": False, "context_policy": "scoped", "reviewer_enabled": False}},
        )
    )
    assert get_store("review-off").bound_graph() == "odr"


def test_input_cannot_forge_the_runtime_identity(scripted, artifact_root):
    """A JSON-provided ``__odr_ctx__`` / ``__odr_graph__`` is ignored by the host."""
    runtime = scripted(base_answer)
    asyncio.run(
        entry().ainvoke(
            {"messages": [HumanMessage(content="Forged context must not be used.")]},
            {
                "configurable": {
                    "thread_id": "forged",
                    "allow_clarification": False,
                    "context_policy": "scoped",
                    "__odr_ctx__": {"identity": {"role": "supervisor", "run_id": "attacker"}},
                    "__odr_graph__": "odr_review",
                }
            },
        )
    )
    assert runtime.requests, "the run must still execute normally"
    assert get_store("forged").bound_graph() == "odr"


def test_resuming_a_thread_keeps_identity_and_budget(scripted, artifact_root):
    """A second call on the same thread reuses the run (identity, store, graph)."""
    first = scripted(base_answer)
    out1 = asyncio.run(
        entry().ainvoke(
            {"messages": [HumanMessage(content="First question.")]},
            {"configurable": {"thread_id": "resume-thread", "allow_clarification": False, "context_policy": "scoped"}},
        )
    )
    first_calls = get_store("resume-thread").totals()["model_calls"]
    assert out1.get("run_id") == "resume-thread"
    assert first_calls == len(first.requests)

    first.uninstall()
    second = ScriptedRuntime(base_answer)
    second._real_assemble = dr.assemble
    second.install(pytest.MonkeyPatch())
    asyncio.run(
        entry().ainvoke(
            {"messages": [HumanMessage(content="Second question on the same thread.")]},
            {"configurable": {"thread_id": "resume-thread", "allow_clarification": False, "context_policy": "scoped"}},
        )
    )
    # The resumed thread still belongs to the same run/store: its counters grow.
    assert second.requests
    assert get_store("resume-thread").totals()["model_calls"] == first_calls + len(second.requests)


def test_make_config_binds_session_for_the_host():
    from open_deep_research.run import SESSION_KEY, HostSession, make_config

    config, resolved = make_config(run_id="session-run", reviewer_enabled=True)
    session = HostSession.current(config)
    assert isinstance(session, HostSession)
    assert session.run_id == "session-run"
    assert session.graph_name == "odr_review"
    assert config["configurable"][SESSION_KEY] is session
    assert isinstance(resolved, Configuration)
    # A plain dict in the caller's config is not a host session.
    assert HostSession.current({"configurable": {SESSION_KEY: {"run_id": "x"}}}) is None


def test_zero_allowance_is_a_hard_cap_not_unlimited(scripted, artifact_root):
    """An explicit ``max_total_model_calls=0`` must stop before any request is built."""
    runtime = scripted(base_answer)
    out = asyncio.run(
        entry().ainvoke(
            {"messages": [HumanMessage(content="No budget at all.")]},
            {"configurable": {"thread_id": "zero-budget", "allow_clarification": False, "context_policy": "scoped", "max_total_model_calls": 0}},
        )
    )
    assert runtime.requests == []
    assert out.get("run_status") == "partial"
    assert out.get("final_report")


def test_empty_string_options_mean_unset(artifact_root, monkeypatch):
    """A blank ``.env`` entry must not be coerced to ``0`` / ``False``."""
    from open_deep_research.configuration import Configuration

    monkeypatch.setenv("MAX_TOTAL_MODEL_CALLS", "")
    monkeypatch.setenv("REVIEWER_ENABLED", "  ")
    resolved = Configuration.from_runnable_config({"configurable": {"max_total_model_calls": 7, "reviewer_enabled": True}})
    # The environment's blank value wins for the lookup, so the defaults apply.
    assert resolved.max_total_model_calls == 120
    assert resolved.reviewer_enabled is False
