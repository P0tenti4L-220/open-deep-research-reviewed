"""One-run graph harness shared by the offline tests.

The previous helper executed every graph twice with the same run id/store (once for
the update order, once for the final state). Every model call, review and report was
therefore counted twice, which made budget assertions meaningless. This harness runs
the graph once in ``updates`` mode and reads the final state from the same
checkpointer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from open_deep_research import deep_researcher as dr
from open_deep_research.run import make_config
from tests.odr.conftest import ScriptedRuntime


@dataclass
class RunResult:
    """Final state plus the top-level node order of one graph run."""

    state: Dict[str, Any]
    order: List[Tuple[str, Any]] = field(default_factory=list)
    runtime: ScriptedRuntime | None = None

    @property
    def nodes(self) -> List[str]:
        """Return only the node names, in execution order."""
        return [node for node, _ in self.order]

    def count(self, node: str) -> int:
        """Return how many times ``node`` ran."""
        return self.nodes.count(node)


async def _astream(graph, cfg, question: str) -> List[Tuple[str, Any]]:
    order: List[Tuple[str, Any]] = []
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=question)]},
        cfg,
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            keys = sorted(update) if isinstance(update, dict) else getattr(update, "goto", "")
            order.append((node, keys))
    return order


async def _final_state(graph, cfg) -> Dict[str, Any]:
    snapshot = await graph.aget_state(cfg)
    return dict(snapshot.values)


def run_graph(
    runtime: ScriptedRuntime | None,
    *,
    review_enabled: bool,
    run_id: str = "run-test",
    question: str = "Compare v1.0 and v2.0 release dates. Answer in English.",
    **overrides: Any,
) -> RunResult:
    """Run the selected graph once and return its final state and node order."""
    config = {
        "allow_clarification": False,
        "context_policy": "scoped",
        "reviewer_enabled": review_enabled,
        "max_total_model_calls": 80,
    }
    config.update(overrides)
    cfg, _ = make_config(run_id=run_id, **config)
    name = "odr_review" if review_enabled else "odr"
    graph = dr.build_deep_researcher(review_enabled=review_enabled, name=name, checkpointer=MemorySaver())
    order = asyncio.run(_astream(graph, cfg, question))
    state = asyncio.run(_final_state(graph, cfg))
    return RunResult(state=state, order=order, runtime=runtime)