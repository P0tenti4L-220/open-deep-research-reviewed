"""Shared fixtures for the context / review tests.

The real chat models are replaced at the single boundary every model call goes
through (``context.assemble``), not by a fake model class. That keeps the production
request assembly, manifest recording and structured-output handling under test while
the script sees exactly the request that would be sent.

No network and no real model is used.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from open_deep_research.state import BriefExtraction, RawReviewOutput, ResearchQuestion, UsageRecord

_IDEMPOTENCY = itertools.count(1)


def render(messages: List[BaseMessage]) -> str:
    """Flatten messages into one searchable string."""
    parts: List[str] = []
    for message in messages:
        parts.append(str(message.content))
        for call in getattr(message, "tool_calls", None) or []:
            parts.append(str(call.get("name")))
            parts.append(str(call.get("args")))
    return "\n".join(parts)


class ScriptedRuntime:
    """Answers every model request with ``answer(messages, schema, tools)``.

    The scripted answer goes through the real request/manifest/usage recording, so a
    test can assert on ``.odr_runs`` exactly like a real run.
    """

    def __init__(self, answer: Callable[[List[BaseMessage], Any, Any], Any]):
        self.answer = answer
        self.requests: List[Dict[str, Any]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "ScriptedRuntime":
        self._monkeypatch = monkeypatch
        import open_deep_research.context as context

        real_assemble = context.assemble
        self._real_assemble = real_assemble
        runtime = self

        async def scripted_assemble(model, packet, *, store, record_requests=True, max_tokens=0, model_name="", retries=1, budget=None, role_cap=None):
            messages = list(packet.messages)
            if store is not None:
                try:
                    store.write_request(
                        packet.identity,
                        packet.request_id,
                        record_requests,
                        {"request_id": packet.request_id, "role": packet.role, "model": model_name, "schema": getattr(packet.response_schema, "__name__", None)},
                        packet.manifest,
                    )
                except Exception:  # noqa: BLE001 - mirrors assemble's best-effort recording
                    pass
            # The scripted model goes through the same atomic budget reservation and
            # per-attempt usage accounting as a real call; only the provider is fake.
            budget = budget if budget is not None else context.ModelCallBudget(store)
            attempt_id = f"{packet.request_id}#a1"
            reservation = budget.reserve(
                request_id=packet.request_id,
                attempt_id=attempt_id,
                role=packet.role,
                agent_instance_id=packet.identity.agent_instance_id,
                estimated_input_tokens=packet.estimated_input_tokens,
                reserved_output_tokens=max_tokens,
                role_cap=role_cap,
            )
            runtime.requests.append(
                {
                    "role": packet.role,
                    "model": model_name,
                    "model_object": model,
                    "bindable": hasattr(model, "bind_tools") and hasattr(model, "with_structured_output"),
                    "schema": packet.response_schema,
                    "tools": list(packet.tools or []),
                    "packet": packet,
                    "retries": retries,
                    "text": render(messages),
                    "attempt_id": attempt_id,
                }
            )
            result = runtime.answer(messages, packet.response_schema, list(packet.tools or []))
            if isinstance(result, str):
                result = AIMessage(content=result)
            if store is not None:
                try:
                    store.record_usage(
                        UsageRecord(
                            request_id=packet.request_id,
                            attempt_id=attempt_id,
                            reservation_id=(reservation or {}).get("id", ""),
                            agent_instance_id=packet.identity.agent_instance_id,
                            role=packet.role,
                            model=model_name,
                            input_tokens=packet.estimated_input_tokens or max(1, len(render(messages)) // 4),
                            output_tokens=42,
                            latency_s=0.0,
                            counting_method="scripted",
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass
            return result

        # ``deep_researcher`` imports ``assemble`` by name, so patch both bindings.
        import open_deep_research.deep_researcher as dr

        monkeypatch.setattr(context, "assemble", scripted_assemble)
        monkeypatch.setattr(dr, "assemble", scripted_assemble)
        return self

    def uninstall(self) -> None:
        """Restore the real ``assemble`` (a test that scripts several calls needs this).

        ``monkeypatch.undo()`` would also drop the ``artifact_root`` fixture's patches,
        so the two bindings are restored directly from the function this runtime saved.
        """
        import open_deep_research.context as context

        import open_deep_research.deep_researcher as dr

        real = getattr(self, "_real_assemble", None)
        if real is not None:
            context.assemble = real
            dr.assemble = real


@dataclass
class ToolCalls:
    """Build AIMessages with unique tool call ids."""

    @staticmethod
    def call(name: str, args: Dict[str, Any], **kwargs: Any) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_IDEMPOTENCY)}", "type": "tool_call"}], **kwargs)


@pytest.fixture()
def artifact_root(tmp_path, monkeypatch):
    """Fresh artifact root and in-memory registry per test."""
    from open_deep_research import artifacts

    root = tmp_path / "runs"
    monkeypatch.setenv("ARTIFACT_ROOT", str(root))
    artifacts.reset_registry()
    artifacts._STORES.clear()  # type: ignore[attr-defined]
    yield root
    artifacts.reset_registry()
    artifacts._STORES.clear()  # type: ignore[attr-defined]


@pytest.fixture()
def scripted(monkeypatch):
    """Return a factory ``install(answer) -> ScriptedRuntime``."""

    def install(answer: Callable[[List[BaseMessage], Any, Any], Any]) -> ScriptedRuntime:
        return ScriptedRuntime(answer).install(monkeypatch)

    return install
