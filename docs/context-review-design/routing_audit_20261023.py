"""Offline audit probes; never modify production source or call real providers.

Run from the repository root with .venv/bin/python. The routing-only variant
changes an in-memory builder, not deep_researcher.py. Results are diagnostic,
not an assertion that the broader implementation is complete.

Pass an output path as the first argument. The default is the post-fix snapshot
``routing_audit_20261023_after_fix.json``; the pre-fix snapshot
``routing_audit_20261023.json`` is kept as the historical evidence referenced by
``ROUTING_AUDIT_2026-10-23.md`` and must not be overwritten by a re-run.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import sys
import tempfile
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
NETWORK_ATTEMPTS = []
_connect = socket.socket.connect


def offline_connect(self, address, *args):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        NETWORK_ATTEMPTS.append(str(address))
        raise RuntimeError("audit: external network disabled")
    return _connect(self, address, *args)


socket.socket.connect = offline_connect

import pytest
from importlib.metadata import version
from langchain_core.messages import HumanMessage
from open_deep_research import artifacts, context, deep_researcher as dr
from open_deep_research.reviewer import apply_review
from open_deep_research.run import make_config
from open_deep_research.state import (
    RawReviewOutput, ResearchQuestion, ReviewDecision, ReviewFinding,
    ReviewStage, Verdict,
)
from tests.odr.conftest import ScriptedRuntime, render
from tests.odr.test_graph_runs import base_answer


def routing_only(graph):
    graph.builder.edges.difference_update(
        {("review_report", dest) for dest in (
            "finalize", "report_writer", "supervisor_return", "finalize_partial"
        )} | {("supervisor_return", "research_supervisor")}
    )
    return graph.builder.compile(name="odr_review")


async def graph_probe(label, remove_duplicate_routes=False, revise_twice=False):
    artifacts.reset_registry()
    artifacts._STORES.clear()
    nodes = []
    draft_versions = []
    report_review_ids = []
    report_review_count = 0

    def answer(messages, schema, tools):
        nonlocal report_review_count
        text = render(messages)
        if schema is RawReviewOutput and '<ReviewPacket stage="report"' in text:
            report_review_count += 1
            if revise_twice and report_review_count <= 2:
                return RawReviewOutput(
                    verdict="revise", summary="Please clarify the presentation.",
                    findings=[ReviewFinding(issue_type="other", severity="major",
                        required_action="Clarify the presentation without new evidence.")],
                )
        return base_answer(messages, schema, tools)

    with tempfile.TemporaryDirectory(prefix="odr-routing-audit-") as root:
        with pytest.MonkeyPatch.context() as mp:
            mp.setenv("ARTIFACT_ROOT", root)
            runtime = ScriptedRuntime(answer).install(mp)
            cfg, _ = make_config(
                run_id=label, allow_clarification=False, context_policy="scoped",
                reviewer_enabled=True, max_total_model_calls=30,
                max_review_events=4 if revise_twice else 20,
                max_revision_rounds=1 if revise_twice else 3,
            )
            cfg["recursion_limit"] = 60
            graph = dr.build_deep_researcher(review_enabled=True, name="odr_review")
            if remove_duplicate_routes:
                graph = routing_only(graph)
            result = {"label": label, "nodes": nodes, "draft_versions": draft_versions,
                      "report_review_ids": report_review_ids}
            try:
                async for chunk in graph.astream(
                    {"messages": [HumanMessage(content="Compare v1.0 and v2.0 release dates. Answer in English.")]},
                    cfg, stream_mode="updates",
                ):
                    for node, update in chunk.items():
                        nodes.append({"node": node, "keys": sorted(update) if isinstance(update, dict) else []})
                        if node == "report_writer":
                            draft_versions.append(update.get("report_version"))
                        if node == "review_report":
                            report_review_ids.extend(x["review_id"] for x in update.get("review_history", {}).get("reviews", []))
                result["status"] = "completed"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["model_calls"] = len(runtime.requests)
            result["report_review_model_calls"] = report_review_count
            result["limits"] = {"max_review_events": 4, "max_revision_rounds": 1} if revise_twice else None
            return result


async def probes():
    graph = dr.build_deep_researcher(review_enabled=True)
    keys = ["research_iterations", "max_iterations", "review_history", "task_ledger",
            "report_version", "report_ready", "review_status", "run_status", "final_report"]
    result = {
        "versions": {n: version(n) for n in ("langgraph", "langchain-core", "pytest")},
        "channels": {k: {"parent": k in graph.channels, "subgraph": k in dr.review_supervisor_subgraph.channels} for k in keys},
        "final_report_channel": type(graph.channels["final_report"]).__name__,
        "report_actions": {},
    }
    for verdict in Verdict:
        decision = ReviewDecision(review_id="audit", stage=ReviewStage.REPORT,
            verdict=verdict, review_status="passed" if verdict == Verdict.PASS else "changes_requested",
            draft_hash="h")
        result["report_actions"][verdict.value] = apply_review(
            decision, current_hash="h", remaining_budget={"open": True, "iterations_left": 3},
            revision_rounds=0, max_revision_rounds=3, stage=ReviewStage.REPORT)
    model = dr.configurable_model.with_retry(stop_after_attempt=3).with_config(
        {"model": "openai:gpt-4.1", "api_key": "offline-placeholder"})
    packet = context.ContextPacket(request_id="audit", role="brief_writer",
        identity=artifacts.Identity(run_id="audit", agent_instance_id="brief-01", role="brief_writer"),
        messages=[HumanMessage(content="offline")], response_schema=ResearchQuestion)
    try:
        await context.assemble(model, packet, store=None)
        result["real_assemble_wrapped_schema"] = "unexpected success"
    except Exception as exc:
        result["real_assemble_wrapped_schema"] = f"{type(exc).__name__}: {exc}"
    result["graph_runs"] = [
        await graph_probe("original"),
        await graph_probe("routing-only", remove_duplicate_routes=True),
        await graph_probe("revision-caps", remove_duplicate_routes=True, revise_twice=True),
    ]
    result["external_network_attempts"] = len(NETWORK_ATTEMPTS)
    result["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT / "src/open_deep_research/deep_researcher.py",
                  ROOT / "src/open_deep_research/state.py",
                  ROOT / "src/open_deep_research/reviewer.py",
                  ROOT / "src/open_deep_research/context.py",
                  ROOT / "tests/odr/test_graph_runs.py",
                  ROOT / "tests/odr/conftest.py")
    }
    return result


if __name__ == "__main__":
    output = asyncio.run(probes())
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("routing_audit_20261023_after_fix.json")
    )
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))
