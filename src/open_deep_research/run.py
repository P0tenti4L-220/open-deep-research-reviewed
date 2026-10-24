"""Host-bound entry point for one research run.

Usage::

    python -m open_deep_research.run "your research question"
    EXPERIMENT_ARM=D python -m open_deep_research.run --question "..." --json

The host (this module) is the only place that

* picks the artifact root and creates the run's :class:`~open_deep_research.artifacts.ArtifactStore`;
* binds the graph (baseline vs review-enabled DAG) to the run;
* binds the supervisor identity into the runnable config.

Model output never influences any of these values.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from typing import Any, Dict, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from open_deep_research.artifacts import get_store, safe_id
from open_deep_research.configuration import Configuration, ContextPolicy
from open_deep_research.deep_researcher import RuntimeContext, build_deep_researcher, _write_run_manifest
from open_deep_research.graph_switch import BASELINE_GRAPH, REVIEW_GRAPH, bind_gate, bind_graph

# Config key that carries the run's store/identity from the host to the adapter. It is
# never resolved from JSON input or model output: only ``HostSession.current()`` (the
# process that created the run) can hand it over.
SESSION_KEY = "__odr_session__"


class HostSession:
    """Host-owned facts for one run: its store, identity and bound graph name."""

    def __init__(self, run_id: str, store: Any, graph_name: str):
        self.run_id = run_id
        self.store = store
        self.graph_name = graph_name

    def context(self) -> RuntimeContext:
        """Return the supervisor identity bound to this run's store."""
        return RuntimeContext(run_id=self.run_id, agent_instance_id="supervisor-01", role="supervisor", store=self.store)

    @classmethod
    def current(cls, config: Optional[RunnableConfig] = None) -> Optional["HostSession"]:
        """Return the session the *host* put into ``config`` (None from a public caller).

        A dict is rejected on purpose: JSON input from Studio or a request body could
        carry a fabricated identity, and the host must never trust one.
        """
        value = (config or {}).get("configurable", {}).get(SESSION_KEY) if config else None
        return value if isinstance(value, cls) else None


def make_config(
    *,
    run_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    **configurable: Any,
) -> tuple[RunnableConfig, Configuration]:
    """Build the runnable config for one run, bound to a store, a graph and an identity."""
    run_id = safe_id(run_id or f"run-{uuid.uuid4().hex[:12]}")
    configurable = {k: v for k, v in configurable.items() if v is not None}
    configurable["__odr_run_id__"] = run_id
    base: RunnableConfig = {
        "configurable": {**configurable, "thread_id": thread_id or run_id},
        "recursion_limit": int(os.environ.get("ODR_RECURSION_LIMIT", "400")),
    }
    config = Configuration.from_runnable_config(base).apply_experiment_arm()
    store = get_store(run_id) if config.artifacts_enabled else None
    graph_name = REVIEW_GRAPH if config.reviewer_enabled else BASELINE_GRAPH
    bound: RunnableConfig = bind_graph(base, graph_name)
    bound = bind_gate(bound, config.reviewer_enabled)
    # ``Configuration.from_runnable_config`` reads the environment first; mirror the
    # resolved factors back so the graph and the config cannot disagree.
    bound["configurable"]["context_policy"] = config.context_policy.value
    bound["configurable"]["reviewer_enabled"] = config.reviewer_enabled
    bound["configurable"]["__odr_ctx__"] = RuntimeContext(
        run_id=run_id,
        agent_instance_id="supervisor-01",
        role="supervisor",
        store=store,
    )
    if store is not None:
        _write_run_manifest(store, config, run_id)
        # The graph is bound to the run once; a resumed thread reads the same binding.
        store.bind_graph(graph_name)
        bound["configurable"][SESSION_KEY] = HostSession(run_id, store, graph_name)
    return bound, config


class PublicEntry:
    """Run adapter for the published graph (``langgraph.json`` -> ``deep_researcher``).

    Studio, the API server, ``graph.ainvoke(json_input, config)`` and the CLI all end
    here. The adapter is the *host*: it derives the run id from the caller's thread
    (never from input), lets ``make_config`` create the store / identity / graph
    bindings, and hands the input to the compiled graph that the caller asked for.

    Input JSON can never supply the runtime context: a ``__odr_ctx__`` or
    ``__odr_graph__`` value in the caller's ``configurable`` is discarded.
    """

    def __init__(self, graphs: Dict[str, Any]):
        self._graphs = dict(graphs)

    def _graph(self, graph_name: str) -> Any:
        try:
            return self._graphs[graph_name]
        except KeyError as exc:  # pragma: no cover - programming error
            raise ValueError(f"Unknown graph {graph_name!r}") from exc

    async def __call__(self, state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        caller = dict((config or {}).get("configurable") or {})
        # The run id comes from the caller's thread, never from the run input.
        run_id = str(caller.get("__odr_run_id__") or caller.get("thread_id") or f"run-{uuid.uuid4().hex[:12]}")
        # Dunder keys (including a client-forged ``__odr_ctx__``) never reach the graph.
        options = {k: v for k, v in caller.items() if not str(k).startswith("__odr_") and k != "thread_id"}
        bound, resolved = make_config(run_id=run_id, thread_id=caller.get("thread_id"), **options)
        graph_name = REVIEW_GRAPH if resolved.reviewer_enabled else BASELINE_GRAPH
        return await self._graph(graph_name).ainvoke(state, bound)


# Channels of the inner run state that the public entry exposes to the caller.
ENTRY_STATE_CHANNELS = {
    "brief",
    "coverage_matrix",
    "evidence_manifest_ref",
    "final_report",
    "final_report_draft",
    "final_report_draft_hash",
    "max_iterations",
    "messages",
    "metrics_ref",
    "notes",
    "raw_notes",
    "report_ready",
    "report_review_summary",
    "report_version",
    "research_brief",
    "research_iterations",
    "review_history",
    "review_status",
    "run_id",
    "run_status",
    "supervisor_messages",
    "task_ledger",
    "todo_write_notice",
    "writing_constraints",
}


class EntryState(TypedDict, total=False):
    """State of the public entry: the run state the inner graph returns.

    Every inner node updates a channel of this mapping; declaring the channels here
    keeps LangGraph's merge semantics (a partial update is merged, not substituted).
    """


for _channel in sorted(ENTRY_STATE_CHANNELS):
    EntryState.__annotations__[_channel] = Any


def build_public_graph(*, checkpointer: Any = None):
    """Compile the entry that ``langgraph.json`` exports (``deep_researcher``).

    One compiled adapter routes each invocation to the graph its own host session was
    bound to, so the published entry keeps working with plain JSON input while a run
    still never switches graphs between invocations.
    """
    builder = StateGraph(EntryState, config_schema=Configuration)
    builder.add_node(
        "odr",
        PublicEntry({BASELINE_GRAPH: build_deep_researcher(review_enabled=False, name=BASELINE_GRAPH), REVIEW_GRAPH: build_deep_researcher(review_enabled=True, name=REVIEW_GRAPH)}),
    )
    builder.add_edge(START, "odr")
    builder.add_edge("odr", END)
    return builder.compile(name="odr_entry", checkpointer=checkpointer)


def graph_for(config: Configuration, checkpointer: Any = None):
    """Return the compiled main graph allowed by the resolved configuration."""
    if config.reviewer_enabled:
        return build_deep_researcher(review_enabled=True, name=REVIEW_GRAPH, checkpointer=checkpointer)
    return build_deep_researcher(review_enabled=False, name=BASELINE_GRAPH, checkpointer=checkpointer)


async def run(
    question: str,
    *,
    run_id: Optional[str] = None,
    checkpointer: Any = None,
    stream: bool = False,
    **configurable: Any,
) -> Dict[str, Any]:
    """Run one research request and return the final state."""
    config, resolved = make_config(run_id=run_id, **configurable)
    graph = graph_for(resolved, checkpointer=checkpointer or MemorySaver())
    inputs = {"messages": [HumanMessage(content=question)]}
    if stream:
        final: Dict[str, Any] = {}
        async for chunk in graph.astream(inputs, config, stream_mode="values"):
            final = chunk
        return final
    return await graph.ainvoke(inputs, config)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run one Open Deep Research task with host-bound identity and storage.")
    parser.add_argument("question", nargs="?", help="research question")
    parser.add_argument("--question", "-q", dest="question_opt", help="research question (alternative to the positional argument)")
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument("--arm", default=None, choices=list("ABCD"), help="experiment arm: A/B/C/D")
    parser.add_argument("--context-policy", default=None, choices=["baseline", "scoped"])
    parser.add_argument("--reviewer", default=None, choices=["on", "off"])
    parser.add_argument("--json", action="store_true", help="print the full result state as JSON")
    args = parser.parse_args(argv)
    question = args.question_opt or args.question
    if not question:
        parser.error("a research question is required")
    if args.arm:
        os.environ["EXPERIMENT_ARM"] = args.arm
    overrides: Dict[str, Any] = {}
    if args.context_policy:
        overrides["context_policy"] = args.context_policy
    if args.reviewer:
        overrides["reviewer_enabled"] = args.reviewer == "on"
    state = asyncio.run(run(question, run_id=args.run_id, **overrides))
    if args.json:
        printable = {k: v for k, v in state.items() if k not in {"messages", "supervisor_messages"}}
        print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    else:
        print(state.get("final_report", ""))
        print("\n---")
        print(f"run_status={state.get('run_status')} review_status={state.get('review_status')} run_id={state.get('run_id')}")
    return 0 if state.get("run_status") != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
