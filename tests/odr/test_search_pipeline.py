"""Offline search -> archive -> evidence -> compression -> review -> report test.

The Tavily transport is replaced at its lowest local seam (``tavily_search_async``).
The real evidence-aware search wrapper, archiver, extraction loop, ledger, writer and
review packet all run. No provider or network is contacted.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from open_deep_research.state import RawReviewOutput, Summary
from tests.odr.conftest import ToolCalls, render
from tests.odr.harness import run_graph
from tests.odr.test_graph_runs import base_answer, requirement_ids, segment_ids, uncovered_requirement_ids


class _ScriptedModel:
    """Stand-in for init_chat_model: only the binding surface is used."""

    def with_structured_output(self, schema):
        return self

    def with_retry(self, **kwargs):
        return self

    def with_config(self, *args, **kwargs):
        return self


def _tool_name(tool) -> str:
    return str(getattr(tool, "name", None) or getattr(tool, "__name__", None) or tool)


def test_mock_tavily_pipeline(scripted, artifact_root, monkeypatch):
    from open_deep_research import utils

    calls = []

    async def fake_search(queries, max_results=5, topic="general", include_raw_content=True, config=None):
        calls.append(list(queries))
        return [
            {
                "query": query,
                "results": [
                    {
                        "url": "https://example.com/release-notes",
                        "title": "Release notes",
                        "content": "Version 2.0 was released on 2024-05-01.",
                        "raw_content": (
                            "Release notes. Version 2.0 was released on 2024-05-01. "
                            "Version 1.0 was released on 2020-02-02."
                        ),
                    }
                ],
            }
            for query in queries
        ]

    def fail_client(**kwargs):  # pragma: no cover - must not be constructed
        raise AssertionError("real Tavily client must not be constructed")

    monkeypatch.setattr(utils, "tavily_search_async", fake_search)
    monkeypatch.setattr(utils, "AsyncTavilyClient", fail_client)
    monkeypatch.setattr(utils, "init_chat_model", lambda *args, **kwargs: _ScriptedModel())

    counters = {"supervisor": 0, "researcher": 0}

    def answer(messages, schema, tools):
        text = render(messages)
        names = [_tool_name(t) for t in tools]
        if schema is Summary or "summarizing the raw content of a webpage" in text:
            return Summary(
                summary="Version 2.0 was released on 2024-05-01 [E1].",
                key_excerpts="Version 2.0 was released on 2024-05-01.",
            )
        if schema is RawReviewOutput:
            # Check exactly the claims rendered into the packet, and state the
            # requirement coverage / draft segments the reviewer actually verified.
            checked = re.findall(r"^- (C-\S+) ", text, flags=re.MULTILINE)
            return RawReviewOutput(
                verdict="pass",
                summary="checked every claim against the archived release notes",
                findings=[],
                checked_claim_ids=checked,
                checked_requirement_ids=requirement_ids(text),
                acknowledged_unknown_requirement_ids=uncovered_requirement_ids(text),
                checked_segment_ids=segment_ids(text),
            )
        if "ConductResearch" in names:
            counters["supervisor"] += 1
            if counters["supervisor"] == 1:
                return ToolCalls.call(
                    "ConductResearch",
                    {"research_topic": "When exactly was version 2.0 released, and when was version 1.0 released?"},
                )
            return ToolCalls.call("ResearchComplete", {})
        if "tavily_search" in names:
            counters["researcher"] += 1
            if counters["researcher"] == 1:
                return ToolCalls.call("tavily_search", {"queries": ["version 2.0 release date"]})
            return AIMessage(content="The evidence is sufficient for this task.")
        if "Research transcript" in text:
            return AIMessage(content="Version 2.0 was released on 2024-05-01 [E1].")
        if "report writer" in text:
            return AIMessage(
                content=(
                    "# Report\n\nVersion 2.0 was released on 2024-05-01 "
                    "[E1](https://example.com/release-notes).\n\n"
                    "### Sources\n[1] Release notes: https://example.com/release-notes"
                )
            )
        return base_answer(messages, schema, tools)

    result = run_graph(scripted(answer), review_enabled=True, run_id="search-run", max_total_model_calls=80)
    state = result.state

    # 1. The search tool really ran, once per requested query.
    assert calls == [["version 2.0 release date"]]
    # 2. The original page was archived before extraction and indexed as E1.
    run_dir = artifact_root / "search-run"
    assert (run_dir / "evidence" / "records" / "E1.json").exists()
    assert list((run_dir / "evidence" / "objects").glob("*.txt"))
    # 3. Claims written by the compressor are persisted with the evidence reference.
    claim_files = list((run_dir / "published" / "claims").glob("*/*.json"))
    assert claim_files
    assert "E1" in claim_files[0].read_text(encoding="utf-8")
    assert state["task_ledger"], "the dispatched task must be in the ledger"
    assert any("E1" in (entry.get("evidence_refs") or []) for entry in state["task_ledger"].values())
    # 4. The reviewer really saw the evidence and claim, and delivery followed a pass.
    report_requests = [r for r in result.runtime.requests if r["role"] == "reviewer" and 'stage="report"' in r["text"]]
    assert report_requests
    assert "<Evidence>" in report_requests[-1]["text"]
    assert "https://example.com/release-notes" in report_requests[-1]["text"]
    assert state["run_status"] == "complete"
    assert state["review_status"] == "passed"
    assert state["final_report"].startswith("# Report")
    assert "2024-05-01" in state["final_report"]
    assert (run_dir / "reports" / "final.md").exists()


def test_mock_tavily_timeout_and_empty_results(artifact_root, monkeypatch):
    """Transport failures and empty result sets stay structured, never exceptions."""
    import asyncio

    from open_deep_research import utils
    from open_deep_research.artifacts import Identity, get_store

    store = get_store("search-errors")
    identity = Identity(run_id="search-errors", agent_instance_id="researcher-01", role="researcher", task_id="task-1")
    config = {"configurable": {"__odr_run_id__": "search-errors"}}

    async def timeout_search(queries, **kwargs):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(utils, "tavily_search_async", timeout_search)
    timed_out = asyncio.run(utils.evidence_aware_web_search(["q"], identity=identity, store=store, config=config))
    assert timed_out.status == "error"
    assert timed_out.error_class == "timeout"
    assert "does not mean the information does not exist" in timed_out.content

    async def empty_search(queries, **kwargs):
        return [{"query": query, "results": []} for query in queries]

    monkeypatch.setattr(utils, "tavily_search_async", empty_search)
    monkeypatch.setattr(utils, "init_chat_model", lambda *args, **kwargs: _ScriptedModel())
    empty = asyncio.run(utils.evidence_aware_web_search(["q"], identity=identity, store=store, config=config))
    assert empty.status == "ok"
    assert empty.evidence_ids == []
    assert "No valid search results found" in empty.content