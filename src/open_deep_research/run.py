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
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from open_deep_research import observability as obs
from open_deep_research.artifacts import get_store, safe_id
from open_deep_research.configuration import RESOLVED_CONFIG_KEY, Configuration, ContextPolicy
from open_deep_research.deep_researcher import RuntimeContext, build_deep_researcher, _write_run_manifest
from open_deep_research.graph_switch import BASELINE_GRAPH, REVIEW_GRAPH, bind_gate, bind_graph
from open_deep_research.preflight import run_preflight

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


def _run_directory(config: Configuration, run_id: str) -> Path:
    """Return the run directory the host writes to (single source of truth)."""
    root = (
        os.environ.get("ARTIFACT_ROOT")
        or getattr(config, "artifact_root", None)
        or ".odr_runs"
    )
    return Path(str(root)).expanduser().resolve() / safe_id(run_id)


def _merge_host_config(
    config: RunnableConfig,
    *,
    callbacks: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    run_name: Optional[str] = None,
) -> RunnableConfig:
    """Preserve host callbacks/metadata/tags without putting them in ``configurable``.

    LangChain treats ``callbacks``, ``metadata``, ``tags`` and ``run_name`` as
    RunnableConfig keys, not model settings. Merging them here keeps an existing
    sentinel handler (and its parent chain) alive across the adapter.
    """
    merged: Dict[str, Any] = dict(config)
    if callbacks is not None:
        merged["callbacks"] = callbacks
    if metadata:
        merged["metadata"] = {**dict(merged.get("metadata") or {}), **metadata}
    if tags:
        existing = list(merged.get("tags") or [])
        for tag in tags:
            if tag and tag not in existing:
                existing.append(tag)
        merged["tags"] = existing
    if run_name:
        merged["run_name"] = run_name
    return merged


def make_config(
    *,
    run_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    callbacks: Any = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list] = None,
    run_name: Optional[str] = None,
    dataset_task_id: Optional[str] = None,
    experiment_id: Optional[str] = None,
    repeat: Optional[int] = None,
    arm: Optional[str] = None,
    model_mode: Optional[str] = None,
    retrieval_mode: Optional[str] = None,
    previous_execution_id: Optional[str] = None,
    previous_execution_reason: Optional[str] = None,
    source_snapshot_hash: Optional[str] = None,
    corpus_hash: Optional[str] = None,
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
    # The host resolves the configuration once and binds that snapshot. Nodes read it
    # from the runnable config, so a later environment change (another test, a dotenv
    # load, a sibling run) cannot rewrite this run's model, factors or limits.
    base["configurable"][RESOLVED_CONFIG_KEY] = config
    # A new execution must not silently append to an old run directory: mixing two
    # executions under one id made a "39-call run" that was really 5 + 34 calls.
    if str(os.environ.get("ODR_REQUIRE_FRESH_RUN") or "") == "1":
        run_path = _run_directory(config, run_id)
        if run_path.exists() and any(run_path.iterdir()):
            raise RuntimeError(
                f"run directory {run_path} already has artifacts; choose a new run_id instead of reusing an old execution"
            )
    store = get_store(run_id) if config.artifacts_enabled else None
    graph_name = REVIEW_GRAPH if config.reviewer_enabled else BASELINE_GRAPH
    bound: RunnableConfig = bind_graph(base, graph_name)
    bound = bind_gate(bound, config.reviewer_enabled)
    # Mirror the resolved factors into the plain configurable keys as well, so callers
    # that inspect them (and code that predates the snapshot) cannot disagree.
    bound["configurable"]["context_policy"] = config.context_policy.value
    bound["configurable"]["reviewer_enabled"] = config.reviewer_enabled
    bound["configurable"]["__odr_ctx__"] = RuntimeContext(
        run_id=run_id,
        agent_instance_id="supervisor-01",
        role="supervisor",
        store=store,
    )
    bound = _merge_host_config(bound, callbacks=callbacks, metadata=metadata, tags=tags, run_name=run_name)
    if store is not None:
        _write_run_manifest(store, config, run_id)
        # Metering prices this run's usage with the same frozen table the host checked.
        store.pricing_path = config.pricing_path
        # The graph is bound to the run once; a resumed thread reads the same binding.
        store.bind_graph(graph_name)
        bound["configurable"][SESSION_KEY] = HostSession(run_id, store, graph_name)
    # Observability is opened by the host, not by a graph node: every entry point
    # gets the same execution/scoping, and ``none`` stays a true no-op.
    if str(config.observability_backend or "none").lower() != "none":
        spec = obs.RunSpec.from_config(
            config,
            run_id=run_id,
            thread_id=thread_id or run_id,
            dataset_task_id=dataset_task_id,
            experiment_id=experiment_id,
            repeat=repeat,
            arm=arm,
            model_mode=model_mode,
            retrieval_mode=retrieval_mode,
            previous_execution_id=previous_execution_id,
            previous_execution_reason=previous_execution_reason,
            source_snapshot_hash=source_snapshot_hash,
            corpus_hash=corpus_hash,
        )
        # The store's own run directory is authoritative: the usage ledger and the
        # observation records must live under the same run, or reconciliation can
        # never join them.
        run_dir = store.run_dir if store is not None else _run_directory(config, run_id)
        session = obs.RunTelemetry.open(spec, run_dir=run_dir, store=store)
        bound = session.attach_root(bound, metadata=metadata, tags=tags, run_name=run_name)
    return bound, config


def _finish_session(config: RunnableConfig, *, product_status: str, error: Optional[BaseException] = None) -> None:
    """Finish and close the run's telemetry session (bounded, never raises)."""
    session = obs.session_from_config(config)
    if session is None or not session.enabled:
        return
    # Observation must never replace the research result: both steps are best-effort.
    try:
        session.finish(product_status, error=error)
    except Exception:  # noqa: BLE001
        pass
    try:
        session.close()
    except Exception:  # noqa: BLE001 - export must not replace a research result
        pass


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
        # Caller-supplied callbacks/metadata/tags are standard RunnableConfig keys
        # (not identity); they are preserved, while a JSON body can never contain a
        # live callback object, an exporter or a key.
        bound, resolved = make_config(
            run_id=run_id,
            thread_id=caller.get("thread_id"),
            callbacks=(config or {}).get("callbacks"),
            metadata=(config or {}).get("metadata"),
            tags=(config or {}).get("tags"),
            run_name=(config or {}).get("run_name"),
            **options,
        )
        graph_name = REVIEW_GRAPH if resolved.reviewer_enabled else BASELINE_GRAPH
        session = obs.session_from_config(bound)
        if session is None or not session.enabled:
            # "none"/"local" runs must not inherit a platform exporter from the parent
            # process: without a session there is no scope to own it, so the adapter
            # opens the tracing-disabled scope around the whole invocation.
            if obs.is_local_scope(resolved.observability_backend):
                with obs.suppress_inherited_tracing():
                    return await self._graph(graph_name).ainvoke(state, bound)
            return await self._graph(graph_name).ainvoke(state, bound)
        error: Optional[BaseException] = None
        result: Dict[str, Any] = {}
        try:
            with session.invocation():
                result = await self._graph(graph_name).ainvoke(state, bound)
        except BaseException as exc:
            error = exc
            raise
        finally:
            status = "failed" if error is not None else str((result or {}).get("run_status") or "complete")
            _finish_session(bound, product_status=status, error=error)
        return result


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
    session = obs.session_from_config(config)
    if session is not None and session.enabled:
        error: Optional[BaseException] = None
        final: Dict[str, Any] = {}
        try:
            with session.invocation():
                if stream:
                    async for chunk in graph.astream(inputs, config, stream_mode="values"):
                        final = chunk
                else:
                    final = await graph.ainvoke(inputs, config)
        except BaseException as exc:
            error = exc
            raise
        finally:
            _finish_session(config, product_status="failed" if error is not None else str(final.get("run_status") or "complete"), error=error)
        return final
    if obs.is_local_scope(resolved.observability_backend):
        with obs.suppress_inherited_tracing():
            return await _invoke(graph, inputs, config, stream)
    return await _invoke(graph, inputs, config, stream)


async def _invoke(graph: Any, inputs: Dict[str, Any], config: RunnableConfig, stream: bool) -> Dict[str, Any]:
    """Invoke one graph, streaming or not (keeps the two call sites identical)."""
    if stream:
        final: Dict[str, Any] = {}
        async for chunk in graph.astream(inputs, config, stream_mode="values"):
            final = chunk
        return final
    return await graph.ainvoke(inputs, config)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(description="Run one Open Deep Research task with host-bound identity and storage.")
    parser.add_argument("question", nargs="?", help="research question")
    parser.add_argument("--question", "-q", dest="question_opt", help="research question (alternative to the positional argument)")
    parser.add_argument("--run-id", default=None, help="run id (default: generated)")
    parser.add_argument("--arm", default=None, choices=list("ABCD"), help="experiment arm: A/B/C/D")
    parser.add_argument("--context-policy", default=None, choices=["baseline", "scoped"])
    parser.add_argument("--reviewer", default=None, choices=["once", "on", "off"], help="once: review and optionally revise once (daily default); on: strict review gates; off: no reviewer")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="hard amount cap for this run's model usage (required for a paid run)")
    parser.add_argument("--max-total-tokens", type=int, default=None, help="max total tokens (further capped by --max-cost-usd)")
    parser.add_argument("--json", action="store_true", help="print the full result state as JSON")
    args = parser.parse_args(argv)
    question = args.question_opt or args.question
    if not question:
        parser.error("a research question is required")
    overrides: Dict[str, Any] = {}
    if args.arm:
        # Passed as an explicit value, never written into the process environment: the
        # CLI must not decide for any other run in the same process.
        overrides["experiment_arm"] = args.arm
    if args.context_policy:
        overrides["context_policy"] = args.context_policy
    reviewer = args.reviewer
    if reviewer is None and not args.arm and not os.getenv("EXPERIMENT_ARM"):
        reviewer = "once"
    if reviewer:
        overrides["reviewer_enabled"] = reviewer == "on"
        overrides["reviewer_once"] = reviewer == "once"
    if reviewer == "once":
        overrides["allow_clarification"] = False
    if args.max_cost_usd is not None:
        overrides["max_cost_usd"] = args.max_cost_usd
    if args.max_total_tokens is not None:
        overrides["max_total_tokens"] = args.max_total_tokens

    # Fail before the graph starts: authorization, amount cap, frozen prices, model
    # capabilities and credentials are checked against the configuration this run will
    # actually use. No store is created for a refused run.
    planned = Configuration.from_runnable_config({"configurable": dict(overrides)}).apply_experiment_arm()
    report = run_preflight(planned)
    if not report.ok:
        for error in report.errors:
            print(f"preflight refused: {error}", file=sys.stderr)
        return 3
    if report.applied_max_total_tokens:
        # The amount cap is enforced as a worst-case token ceiling.
        overrides["max_total_tokens"] = report.applied_max_total_tokens

    run_id = args.run_id or f"run-{uuid.uuid4().hex[:12]}"
    state = asyncio.run(run(question, run_id=run_id, **overrides))
    _write_preflight_record(planned, run_id, report)
    if args.json:
        printable = {k: v for k, v in state.items() if k not in {"messages", "supervisor_messages"}}
        print(json.dumps(printable, ensure_ascii=False, indent=2, default=str))
    else:
        print(state.get("final_report", ""))
        print("\n---")
        print(f"run_status={state.get('run_status')} review_status={state.get('review_status')} run_id={state.get('run_id')}")
    return 0 if state.get("run_status") != "failed" else 1


def _write_preflight_record(config: Configuration, run_id: str, report: Any) -> None:
    """Write the pre-flight verdict next to the run's metrics (best effort)."""
    try:
        directory = _run_directory(config, run_id) / "metrics"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "preflight.json").write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    except Exception:  # noqa: BLE001 - the record is evidence, not a reason to lose the run
        pass


if __name__ == "__main__":
    sys.exit(main())
