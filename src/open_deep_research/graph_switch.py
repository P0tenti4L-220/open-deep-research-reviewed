"""Run-wide graph switch.

Two compiled graphs exist:

* ``odr`` - baseline topology (A/B): ``research_supervisor -> final_report_generation``;
* ``odr_review`` - review-enabled DAG (C/D): every report draft passes ``review_report``.

A single run must never hop between them (a subgraph switch would reset counters,
starve the checkpointer or bypass a gate), so the host binds the graph to the run at
entry and every nested call reads it back from the runnable config. The binding is
written by the host (``python -m open_deep_research.run``), never by a model, and it
lives under a dunder key that is not a ``Configuration`` field.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.runnables import RunnableConfig

GRAPH_SWITCH_KEY = "__odr_graph__"
GATE_SWITCH_KEY = "__odr_gate__"
CONTEXT_SWITCH_KEY = "__odr_ctx__"

BASELINE_GRAPH = "odr"
REVIEW_GRAPH = "odr_review"


def bind_graph(config: Optional[RunnableConfig], name: str) -> RunnableConfig:
    """Return a copy of ``config`` whose configurable section binds the graph."""
    if name not in {BASELINE_GRAPH, REVIEW_GRAPH}:
        raise ValueError(f"Unknown graph name {name!r}")
    config = dict(config or {})
    configurable = dict(config.get("configurable") or {})
    configurable[GRAPH_SWITCH_KEY] = name
    config["configurable"] = configurable
    return config


def get_active_graph(config: Optional[RunnableConfig]) -> str:
    """Return the graph bound to the run, or the default baseline graph."""
    configurable = (config or {}).get("configurable") or {}
    return str(configurable.get(GRAPH_SWITCH_KEY) or BASELINE_GRAPH)


def bind_gate(config: Optional[RunnableConfig], enabled: bool) -> RunnableConfig:
    """Bind whether the run's supervisor uses the review gate (the C/D graph)."""
    config = dict(config or {})
    configurable = dict(config.get("configurable") or {})
    configurable[GATE_SWITCH_KEY] = bool(enabled)
    config["configurable"] = configurable
    return config


def gate_enabled(config: Optional[RunnableConfig]) -> bool:
    """Return True when the active supervisor subgraph has a review gate.

    This is the graph topology, not a policy preference: the gate exists only in the
    review-enabled subgraph. The switch is written by the host, never by a model.
    """
    configurable = (config or {}).get("configurable") or {}
    return bool(configurable.get(GATE_SWITCH_KEY, False))
