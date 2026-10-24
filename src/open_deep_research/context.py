"""统一上下文组装器（ContextBuilder）。

设计依据：``docs/context-review-design/CONTEXT_AND_STORAGE.md`` §6 与
``DYNAMIC_CONTEXT.md``。

每次模型调用遵守同一流程：

```text
身份与权限检查
→ 选择角色规则和任务必需信息
→ 读取获准任务状态及证据
→ 选择本次工具与对应说明
→ 去重、移出旧工具载荷、必要时生成带来源的工作摘要
→ 计算完整输入预算，保留输出与协议余量
→ 校验消息协议及关键约束完整性
→ 保存本次请求清单
→ 调用模型
→ 将用量、输出和工具结果写入受控记录
```

本模块只负责前六步。它返回 :class:`ContextPacket`；调用方必须把它原样传给
模型，不得再次拼接历史。:func:`assemble` 负责“记录清单 + 调用模型 + 记录用量”。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    get_buffer_string,
)
from pydantic import BaseModel

from open_deep_research.artifacts import (
    ArtifactStore,
    Identity,
    hash_obj,
    redact_secrets,
    sha256_text,
    utc_now,
    validate_id,
)
from open_deep_research.configuration import Configuration
from open_deep_research.state import ContextManifest, EvictionRecord, UsageRecord

LOGGER = logging.getLogger(__name__)

POLICY_DIR = Path(__file__).parent / "context_policy"
CONTEXT_POLICY_VERSION = "context-v1"

# Auxiliary model calls (requirement extraction, ...) only run while this many model
# calls remain in the run budget: the last reserve belongs to the supervisor, the
# reviewer and the writer, which produce the deliverable.
AUXILIARY_CALL_RESERVE = 3

# Rough output-token / protocol safety margins per role (tokens).
_OUTPUT_RESERVE = {"supervisor": 1500, "researcher": 2000, "extractor": 800, "compressor": 1500, "reviewer": 1500, "writer": 2000, "brief_writer": 1500}

# Long tool results are replaced by an excerpt plus a pointer once they exceed this
# fraction of a role's input soft cap.
_OFFLOAD_FRACTION = 0.35


# --------------------------------------------------------------------------- #
# Rules and profiles
# --------------------------------------------------------------------------- #
class PolicyError(RuntimeError):
    """Raised when the static rule files are missing or inconsistent."""


def _read_policy_file(relpath: str) -> str:
    path = POLICY_DIR / relpath
    if not path.exists():
        raise PolicyError(f"Missing context policy resource: {relpath}")
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def load_profiles() -> Dict[str, Any]:
    """Load and verify ``profiles.json`` plus referenced rule files."""
    raw = _read_policy_file("profiles.json")
    profiles = json.loads(raw)
    for role, spec in profiles.get("roles", {}).items():
        for rule in spec.get("rules", []):
            _read_policy_file(f"roles/{rule}")
    for capability, guides in profiles.get("tool_guides", {}).items():
        for guide in guides:
            _read_policy_file(f"tool_guides/{guide}")
    return profiles


def role_rules(role: str, **fmt: Any) -> str:
    """Return the role-specific rule text (common rules + role rules)."""
    profiles = load_profiles()
    spec = profiles["roles"].get(role)
    if not spec:
        raise PolicyError(f"Unknown role for context policy: {role}")
    common = _read_policy_file("common.md")
    parts = [common.strip()]
    for rule in spec.get("rules", []):
        raw = _read_policy_file(f"roles/{rule}")
        parts.append(raw.replace("{tool_list}", fmt.get("tool_list", "")).replace("{date}", fmt.get("date", "")).replace("{context_policy}", fmt.get("context_policy", "scoped")))
    return "\n\n".join(p for p in parts if p)


def tool_guides_for(capability_tags: Iterable[str]) -> List[str]:
    """Return guide texts for the capabilities enabled in this request."""
    profiles = load_profiles()
    out = []
    for tag in capability_tags:
        for guide in profiles["tool_guides"].get(tag, []):
            text = _read_policy_file(f"tool_guides/{guide}").strip()
            if text:
                out.append(text)
    return out


def _role_budget(config: Configuration, role: str) -> int:
    key = load_profiles()["roles"].get(role, {}).get("budget_key", "")
    return int(getattr(config, key, config.researcher_context_budget))


# --------------------------------------------------------------------------- #
# Counting
# --------------------------------------------------------------------------- #
try:  # tiktoken is optional; the heuristic fallback is always labelled as such.
    import tiktoken  # type: ignore

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001
    _ENCODER = None


def count_tokens(text: str) -> int:
    """Estimate tokens for a text (tiktoken when available)."""
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))
        except Exception:  # noqa: BLE001
            pass
    # Heuristic: CJK characters ~1 token each, other characters ~4 per token.
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff")
    other = len(text) - cjk
    return cjk + int(math.ceil(other / 4))


def counting_method() -> str:
    """Name the tokenizer in use (recorded in every manifest)."""
    return "tiktoken:cl100k_base" if _ENCODER is not None else "heuristic:cjk+chars/4"


def message_tokens(messages: Sequence[BaseMessage]) -> int:
    """Estimate the token size of a message list including protocol overhead."""
    total = 0
    for message in messages:
        total += count_tokens(_message_text(message)) + 6
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                total += count_tokens(json.dumps(call.get("args", {}), ensure_ascii=False)) + 12
    return total


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def tool_tokens(tools: Sequence[Any]) -> int:
    """Estimate the token size of tool JSON schemas."""
    total = 0
    for tool in tools or []:
        schema = getattr(tool, "args_schema", None) or getattr(tool, "args", None)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            payload = schema.model_json_schema()
        elif schema is not None:
            payload = getattr(schema, "__doc__", "") or ""
        else:
            payload = getattr(tool, "description", "") or ""
        name = getattr(tool, "name", None) or (tool.get("name", "") if isinstance(tool, dict) else "")
        total += count_tokens(str(name)) + count_tokens(str(payload)) + 20
    return total


def tool_schema_hashes(tools: Sequence[Any]) -> Dict[str, str]:
    """Return ``{tool_name: sha256(schema)}`` for the manifest."""
    out: Dict[str, str] = {}
    for tool in tools or []:
        name = getattr(tool, "name", None) or (tool.get("name", "") if isinstance(tool, dict) else "unknown")
        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            payload = schema.model_json_schema()
        else:
            payload = str(schema or getattr(tool, "description", ""))
        out[str(name)] = sha256_text(json.dumps(payload, sort_keys=True, default=str))
    return out


def _model_window(model: Optional[str]) -> Optional[int]:
    if not model:
        return None
    from open_deep_research.utils import get_model_token_limit

    try:
        return get_model_token_limit(model)
    except Exception:  # noqa: BLE001
        return None


def effective_budget(role: str, config: Configuration, model: Optional[str], declared_max_tokens: int = 0) -> int:
    """Return ``min(role soft cap, model window - output reserve - margin)``."""
    soft = _role_budget(config, role)
    window = _model_window(model)
    if not window:
        # Unknown window must be explicit, not silently mapped to another model.
        LOGGER.warning("Unknown model window for %s; using role soft cap %s", model, soft)
        return soft
    reserve = max(_OUTPUT_RESERVE.get(role, 1500), int(declared_max_tokens or 0))
    hard = window - reserve - 500
    return max(1000, min(soft, hard))


# --------------------------------------------------------------------------- #
# Packet
# --------------------------------------------------------------------------- #
@dataclass
class ContextPacket:
    """Exactly what one model call will send."""

    request_id: str
    role: str
    identity: Identity
    messages: List[BaseMessage]
    tools: List[Any] = field(default_factory=list)
    response_schema: Optional[type] = None
    manifest: Optional[ContextManifest] = None
    estimated_input_tokens: int = 0
    budget_tokens: int = 0
    evictions: List[EvictionRecord] = field(default_factory=list)
    tool_names: List[str] = field(default_factory=list)


def new_request_id(role: str) -> str:
    """Return a unique, path-safe request id."""
    return f"req-{role}-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Message assembly helpers
# --------------------------------------------------------------------------- #
def to_interaction_groups(messages: Sequence[Any]) -> List[List[BaseMessage]]:
    """Split a message list into protocol-complete interaction units.

    A group is an AIMessage with tool calls plus *all* of its ToolMessages. Dividing
    units that way keeps every ``tool_call_id`` paired - never a dangling call or a
    result without a call.

    Raises:
        ValueError: when a tool result has no preceding call (corrupt history).
    """
    converted = convert_to_messages(list(messages))
    # Re-attach tool calls if the driver passed dicts without them (defensive).
    groups: List[List[BaseMessage]] = []
    open_calls: Dict[str, List[BaseMessage]] = {}
    for message in converted:
        if isinstance(message, ToolMessage):
            pending = open_calls.get(message.tool_call_id)
            if pending is None:
                raise ValueError(f"ToolMessage {message.tool_call_id} has no matching tool call")
            pending.append(message)
            continue
        if isinstance(message, AIMessage) and message.tool_calls:
            group: List[BaseMessage] = [message]
            groups.append(group)
            for call in message.tool_calls:
                open_calls[call["id"]] = group
            continue
        groups.append([message])
    missing = [cid for cid, group in open_calls.items() if not any(isinstance(m, ToolMessage) and m.tool_call_id == cid for m in group)]
    if missing and groups:
        raise ValueError(f"Tool calls without results: {missing}")
    return groups


def render_groups_as_text(groups: Sequence[Sequence[BaseMessage]]) -> str:
    """Render interaction units as readable plain text (for the compressor)."""
    lines: List[str] = []
    for group in groups:
        for message in group:
            if isinstance(message, AIMessage):
                text = _message_text(message).strip()
                if text:
                    lines.append(f"[assistant] {text}")
                for call in message.tool_calls or []:
                    lines.append(f"[tool call:{call.get('name')}] {json.dumps(call.get('args', {}), ensure_ascii=False)[:1200]}")
            elif isinstance(message, ToolMessage):
                lines.append(f"[tool result:{message.name or message.tool_call_id}] {_message_text(message)}")
            else:
                lines.append(f"[{message.type}] {_message_text(message)}")
    return "\n\n".join(lines)


@dataclass
class ToolResultSnapshot:
    """What a tool execution produced before model-facing trimming."""

    call_id: str
    name: str
    status: str
    text: str
    evidence_ids: List[str] = field(default_factory=list)
    artifact_ref: str = ""
    error_class: str = ""


def offload_long_tool_results(
    identity: Identity,
    snapshots: Sequence[ToolResultSnapshot],
    *,
    store: Optional[ArtifactStore],
    head_chars: int,
    tail_chars: int,
) -> Tuple[List[ToolMessage], List[EvictionRecord]]:
    """Archive long tool results and return excerpt messages plus eviction records.

    The original is stored *first*; if storing fails, the full text is kept so that
    no irreversible trimming happens without a recovery pointer.
    """
    messages: List[ToolMessage] = []
    evictions: List[EvictionRecord] = []
    for snap in snapshots:
        text = snap.text
        if snap.status != "ok":
            messages.append(
                ToolMessage(
                    content=f"[tool_error:{snap.error_class or 'unknown'}] {text}"[: max(400, head_chars)],
                    name=snap.name,
                    tool_call_id=snap.call_id,
                    status="error",
                )
            )
            continue
        if len(text) <= head_chars + tail_chars:
            messages.append(ToolMessage(content=text, name=snap.name, tool_call_id=snap.call_id))
            continue
        artifact_ref = snap.artifact_ref
        if store is not None and snap.evidence_ids:
            artifact_ref = f"evidence://{snap.evidence_ids[0]}"
        if store is not None and not artifact_ref:
            try:
                doc = store.archive_document(identity, text, url="", title=f"tool result {snap.call_id}")
                artifact_ref = f"artifact://tool-results/{doc['content_hash'][:16]}"
            except Exception:  # noqa: BLE001
                LOGGER.warning("Could not archive tool result; keeping full text", exc_info=True)
                artifact_ref = ""
        if not artifact_ref:
            messages.append(ToolMessage(content=text, name=snap.name, tool_call_id=snap.call_id))
            continue
        pointer = (
            f"\n\n[... {len(text) - head_chars - tail_chars} characters omitted from the model view. "
            f"Full original archived as {artifact_ref} (source ids: {', '.join(snap.evidence_ids) or 'none'}). "
            f"Read a specific passage with the request_context tool before quoting.]\n\n"
        )
        truncated = text[:head_chars] + pointer + text[-tail_chars:]
        messages.append(ToolMessage(content=truncated, name=snap.name, tool_call_id=snap.call_id))
        evictions.append(EvictionRecord(artifact_id=artifact_ref, reason="offloaded", recoverable=True))
    return messages, evictions


def trim_groups_to_budget(
    groups: Sequence[Sequence[BaseMessage]],
    *,
    base_tokens: int,
    budget: int,
) -> Tuple[List[BaseMessage], List[EvictionRecord], int]:
    """Keep the newest complete interaction units until the budget is met.

    Whole units are dropped; the assistant's last textual plan is kept first. No
    requirement or evidence text is dropped here: callers put non-negotiable material
    in ``base_tokens``.
    """
    evictions: List[EvictionRecord] = []
    kept: List[Sequence[BaseMessage]] = list(groups)
    used = base_tokens + sum(message_tokens(g) for g in kept)
    while kept and used > budget:
        dropped = kept.pop(0)
        used -= message_tokens(dropped)
        evictions.append(
            EvictionRecord(
                artifact_id=f"messages/{id(dropped)}",
                reason="history_trimmed",
                recoverable=True,
            )
        )
    return [m for group in kept for m in group], evictions, used


def _section(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"## {title}\n{body}"


# --------------------------------------------------------------------------- #
# ContextBuilder
# --------------------------------------------------------------------------- #
class ContextBuilder:
    """Assemble one model request for one role, identity and phase."""

    def __init__(self, config: Configuration, store: Optional[ArtifactStore] = None):
        self.config = config
        self.store = store

    # -- scoped projections -------------------------------------------------
    def supervisor_messages(self, identity: Identity, state: Dict[str, Any]) -> Tuple[List[BaseMessage], List[Dict[str, Any]]]:
        """Build the supervisor request from a ledger projection (no raw history)."""
        runtime_turn = (
            state.get("runtime_turn") == "start"
            and not state.get("pending_review")
            and state.get("dispatch_count", 0) == 0
        )
        if runtime_turn:
            return self._supervisor_runtime_messages(identity, state)
        return self._supervisor_projection_messages(identity, state)

    def _supervisor_runtime_messages(self, identity: Identity, state: Dict[str, Any]) -> Tuple[List[BaseMessage], List[Dict[str, Any]]]:
        from open_deep_research.utils import get_today_str

        items: List[Dict[str, Any]] = []
        tools = state.get("tools") or []
        tool_list = "\n".join(f"- {getattr(t, 'name', str(t))}: {getattr(t, 'description', '')}" for t in tools)
        rules = role_rules("supervisor", date=get_today_str(), tool_list=tool_list, context_policy=state.get("context_policy", "scoped"), max_concurrent_research_units=self.config.max_concurrent_research_units, max_researcher_iterations=state.get("max_iterations") or self.config.max_researcher_iterations)
        items.append({"kind": "role", "ref": "roles/supervisor.md", "reason": "role_policy"})
        for guide in tool_guides_for(state.get("capabilities", [])):
            rules += "\n\n" + guide
        brief = state.get("brief")
        brief_text = brief.objective if brief is not None else state.get("research_brief", "")
        constraints = "\n".join(f"- {c}" for c in (brief.constraints if brief is not None else []))
        items.append({"kind": "task", "ref": "research_brief", "reason": "mandatory_constraints"})
        body = _section("Research objective", brief_text)
        body += "\n\n" + _section("Non-negotiable constraints (keep every one)", constraints) if constraints else ""
        body += "\n\n" + _section("Run policy", f"context_policy={state.get('context_policy', 'scoped')}; reviewer_enabled={str(bool(self.config.reviewer_enabled)).lower()}; remaining supervisor iterations: {max(0, (state.get('max_iterations') or 0) - state.get('iteration', 0))}")
        prompt = f"{rules}\n\n{body}"
        messages: List[BaseMessage] = [SystemMessage(content=prompt)]
        pending = state.get("pending_review")
        if pending:
            messages.append(HumanMessage(content=pending))
            items.append({"kind": "review", "ref": "pending_review", "reason": "unresolved_review"})
        user_original = brief.user_messages if brief is not None else ""
        if user_original:
            messages.append(HumanMessage(content=f"<User request>\n{user_original[: self.budget_tokens_for('supervisor') * 2]}\n</User request>"))
        return messages, items

    def budget_tokens_for(self, role: str, model: Optional[str] = None, declared_max_tokens: int = 0) -> int:
        """Return the effective input budget for a role."""
        return effective_budget(role, self.config, model, declared_max_tokens)

    def _supervisor_projection_messages(self, identity: Identity, state: Dict[str, Any]) -> Tuple[List[BaseMessage], List[Dict[str, Any]]]:
        from open_deep_research.utils import get_today_str

        items: List[Dict[str, Any]] = [
            {"kind": "role", "ref": "roles/supervisor.md", "reason": "role_policy"},
            {"kind": "task", "ref": "research_brief", "reason": "mandatory_constraints"},
        ]
        tools = state.get("tools") or []
        tool_list = "\n".join(f"- {getattr(t, 'name', str(t))}: {getattr(t, 'description', '')}" for t in tools)
        header = role_rules(
            "supervisor",
            date=get_today_str(),
            tool_list=tool_list,
            context_policy=state.get("context_policy", "scoped"),
            max_concurrent_research_units=self.config.max_concurrent_research_units,
            max_researcher_iterations=state.get("max_iterations") or self.config.max_researcher_iterations,
        )
        for guide in tool_guides_for(state.get("capabilities", [])):
            header += "\n\n" + guide

        brief = state.get("brief")
        constraints = "\n".join(f"- {c}" for c in (getattr(brief, "constraints", []) or []))
        reqs = getattr(brief, "requirements", []) or []
        requirement_lines = "\n".join(f"- {r.requirement_id} [{'mandatory' if r.mandatory else 'optional'}]: {r.statement}" for r in reqs)

        ledger: Dict[str, Any] = state.get("task_ledger") or {}
        task_lines = []
        for task_id, entry in ledger.items():
            task_lines.append(
                f"- {task_id} [{entry.get('status', 'pending')}] objective: {entry.get('objective', '')[:200]}; "
                f"evidence: {len(entry.get('evidence_refs', []))}; gaps: {entry.get('gaps', [])[:3]}"
            )
        results = state.get("published") or []
        result_lines = []
        for result in results:
            result_lines.append(
                f"- {result.get('task_id')} v{result.get('version', 1)} [{result.get('status')}]: {result.get('summary', '')[:500]}"
                + (f"\n  gaps: {result.get('gaps')}" if result.get("gaps") else "")
            )
        reviewed = state.get("reviewed_actions") or []
        review_lines = [f"- {r.get('stage')}: {r.get('verdict')} ({r.get('summary', '')[:300]})" for r in reviewed]
        open_issues = state.get("open_issues") or []

        body = _section("Research objective", getattr(brief, "objective", None) or state.get("research_brief", ""))
        body += "\n\n" + _section("Requirements to satisfy", requirement_lines)
        if constraints:
            body += "\n\n" + _section("Non-negotiable constraints (keep every one)", constraints)
        body += "\n\n" + _section("Task progress", "\n".join(task_lines) or "(no tasks yet)")
        body += "\n\n" + _section("Published result summaries", "\n".join(result_lines) or "(none yet)")
        body += "\n\n" + _section("Completed reviews", "\n".join(review_lines) or "(none yet)")
        body += "\n\n" + _section("Open issues from reviews (must be addressed)", "\n".join(f"- {i}" for i in open_issues) or "(none)")
        body += "\n\n" + _section(
            "Budget",
            f"remaining supervisor iterations: {max(0, (state.get('max_iterations') or 0) - state.get('iteration', 0))}; "
            f"model calls: {state.get('usage', {}).get('model_calls', 0)}/{self.config.max_total_model_calls}",
        )
        pending = state.get("pending_review")
        if pending:
            body += "\n\n" + _section("Reviewer feedback on your last proposal", pending)
            items.append({"kind": "review", "ref": "pending_review", "reason": "unresolved_review"})
        messages: List[BaseMessage] = [SystemMessage(content=header), HumanMessage(content=body)]
        if state.get("research_brief"):
            messages.append(HumanMessage(content=f"<Research brief>\n{state['research_brief']}\n</Research brief>"))
        return messages, items

    def researcher_messages(self, identity: Identity, state: Dict[str, Any], *, include_history: bool = True) -> Tuple[List[BaseMessage], List[Dict[str, Any]], List[EvictionRecord]]:
        """Project a researcher state into a minimal sufficient request."""
        task_card = state.get("task_card")
        items: List[Dict[str, Any]] = [
            {"kind": "role", "ref": "roles/researcher.md", "reason": "role_policy"},
            {"kind": "task", "ref": getattr(task_card, "task_id", "research_topic"), "reason": "current_task"},
        ]
        tools = state.get("tools") or []
        tool_list = ", ".join(f"{getattr(t, 'name', '?')} ({getattr(t, 'description', '')[:90]})" for t in tools)
        rules = role_rules("researcher", date=state.get("date", ""), tool_list=tool_list or "(none bound in this turn)")
        for guide in tool_guides_for(state.get("capabilities", [])):
            rules += "\n\n" + guide
        background = getattr(task_card, "necessary_background", "") if task_card is not None else ""
        base_prompt = rules + ("\n\n## Task card\n" + (task_card.model_dump_json(indent=2) if task_card is not None else json.dumps({"objective": state.get("research_topic", "")}, ensure_ascii=False)))
        if background and background not in base_prompt:
            base_prompt += f"\n\n## Background (from the supervisor, may be an assumption)\n{background[:2000]}"
        working = state.get("working_memory")
        if working is not None:
            base_prompt += "\n\n## Your working summary so far\n" + json.dumps(working.model_dump(mode="json"), ensure_ascii=False)[:1500]
            items.append({"kind": "memory", "ref": "working_memory", "reason": "own_state"})

        history = list(state.get("messages") or []) if include_history else []
        groups = to_interaction_groups(history) if history else []
        system_tokens = count_tokens(base_prompt) + 8
        tools_tk = tool_tokens(tools)
        max_tool_result_tokens = int(self.budget_tokens_for("researcher") * _OFFLOAD_FRACTION)
        # Offload oversized single tool results before measuring.
        history_messages, evictions = self._offload_groups(identity, groups, max_tool_result_tokens)
        history_tokens = message_tokens(history_messages)
        budget = self.budget_tokens_for("researcher")
        trimmed, extra_evictions, _ = trim_groups_to_budget(
            to_interaction_groups(history_messages) if history_messages else [],
            base_tokens=system_tokens + tools_tk,
            budget=budget,
        )
        evictions.extend(extra_evictions)
        if len(trimmed) < len(history_messages):
            base_prompt += "\n\n## Note on history\nThe oldest turns were moved out of this request to stay within budget. Their tool results are archived; the task card and your working summary above are complete. Use request_context with a topic to recover a specific passage."
        messages = [SystemMessage(content=base_prompt)] + trimmed
        return messages, items, evictions

    def _offload_groups(self, identity: Identity, groups: Sequence[Sequence[BaseMessage]], max_result_tokens: int) -> Tuple[List[BaseMessage], List[EvictionRecord]]:
        out: List[BaseMessage] = []
        evictions: List[EvictionRecord] = []
        for group in groups:
            for message in group:
                if isinstance(message, ToolMessage) and count_tokens(_message_text(message)) > max_result_tokens:
                    # Already archived at execution time; the pointer is in the content.
                    if "archived as" not in _message_text(message).lower() and "request_context" not in _message_text(message):
                        text = _message_text(message)
                        ref = ""
                        if self.store is not None:
                            try:
                                doc = self.store.archive_document(identity, text, title=f"tool result {message.tool_call_id}")
                                ref = f"artifact://tool-results/{doc['content_hash'][:16]}"
                            except Exception:  # noqa: BLE001
                                ref = ""
                        if ref:
                            head = text[: max(400, int(max_result_tokens * 2))]
                            out.append(
                                ToolMessage(
                                    content=f"{head}\n\n[... trimmed; full original archived as {ref}; use request_context to read a specific passage.]",
                                    name=message.name,
                                    tool_call_id=message.tool_call_id,
                                )
                            )
                            evictions.append(EvictionRecord(artifact_id=ref, reason="offloaded", recoverable=True))
                            continue
                    out.append(message)
                else:
                    out.append(message)
        return out, evictions

    def compressor_messages(self, identity: Identity, state: Dict[str, Any], *, compact: bool = False) -> Tuple[List[BaseMessage], List[Dict[str, Any]]]:
        """Build the compression request from evidence/work state, never by appending."""
        from open_deep_research.utils import get_today_str

        task_card = state.get("task_card")
        items: List[Dict[str, Any]] = [{"kind": "role", "ref": "roles/compressor.md", "reason": "role_policy"}]
        rules = role_rules("compressor", date=get_today_str())
        history = list(state.get("messages") or [])
        groups = to_interaction_groups(history) if history else []
        rendered = render_groups_as_text(groups)
        task_text = task_card.model_dump_json(indent=2) if task_card is not None else json.dumps({"objective": state.get("research_topic", "")}, ensure_ascii=False)
        items.append({"kind": "task", "ref": getattr(task_card, "task_id", "research_topic"), "reason": "current_task"})
        budget = self.budget_tokens_for("compressor")
        base = count_tokens(rules) + count_tokens(task_text) + 300
        allowed_chars = max(2000, (budget - base) * 4)
        if len(rendered) > allowed_chars:
            # Keep the most recent material (where the confirmed findings usually are)
            # and say explicitly what was omitted so the model cannot pretend it read it.
            omitted = len(rendered) - allowed_chars
            rendered = f"[... {omitted} characters of older tool results omitted from this request; they remain archived and can be re-read if needed ...]\n\n" + rendered[-allowed_chars:]
            items.append({"kind": "note", "ref": "history_window", "reason": "offloaded"})
        human = (
            "All above messages are about research conducted by an AI Researcher. Please clean up these findings.\n\n"
            "DO NOT summarize the information. I want the raw information returned, just in a cleaner format. "
            "Make sure all relevant information is preserved - you can rewrite findings verbatim.\n\n"
            "<Task card>\n" + task_text + "\n</Task card>\n\n"
            "<Research transcript>\n" + rendered + "\n</Research transcript>"
        )
        return [SystemMessage(content=rules), HumanMessage(content=human)], items

    def extractor_messages(self, url: str, title: str, content: str) -> Tuple[List[BaseMessage], List[Dict[str, Any]]]:
        """Build one webpage-extraction request (evidence id bound by the caller)."""
        from open_deep_research.prompts import summarize_webpage_prompt
        from open_deep_research.utils import get_today_str

        budget = self.budget_tokens_for("extractor")
        char_cap = max(2000, budget * 4)
        clipped = content[:char_cap]
        items = [
            {"kind": "role", "ref": "roles/extractor.md", "reason": "role_policy"},
            {"kind": "source", "ref": url or title or "unknown-source", "reason": "document_under_extraction"},
        ]
        prompt = summarize_webpage_prompt.format(webpage_content=clipped, date=get_today_str())
        return [HumanMessage(content=prompt)], items

    def writer_messages(self, identity: Identity, state: Dict[str, Any], *, findings: str) -> Tuple[List[BaseMessage], List[Dict[str, Any]]]:
        """Build the report-writing request from requirements, material and review limits."""
        from open_deep_research.utils import get_today_str

        rules = role_rules("writer", date=get_today_str())
        items: List[Dict[str, Any]] = [
            {"kind": "role", "ref": "roles/writer.md", "reason": "role_policy"},
            {"kind": "task", "ref": "research_brief", "reason": "user_requirements"},
        ]
        brief = state.get("brief")
        requirements = "\n".join(f"- {r.requirement_id}: {r.statement}" for r in (getattr(brief, "requirements", []) or []))
        constraints = "\n".join(f"- {c}" for c in (getattr(brief, "constraints", []) or []))
        user_messages = get_buffer_string(state.get("messages", []))
        review_notes = state.get("writing_constraints") or ""
        budget = self.budget_tokens_for("writer")
        head = (
            f"{rules}\n\n"
            f"## Research brief\n{getattr(brief, 'objective', None) or state.get('research_brief', '')}\n\n"
            f"## Requirements\n{requirements or '(none enumerated)'}\n\n"
            + (f"## Constraints\n{constraints}\n\n" if constraints else "")
            + (f"## Reviewer constraints for this draft\n{review_notes}\n\n" if review_notes else "")
            + f"## User messages (language and format)\n{user_messages[:4000]}\n\n"
        )
        allowed = max(3000, (budget - count_tokens(head) - 200) * 4)
        if len(findings) > allowed:
            findings = findings[:allowed] + "\n\n[... remaining findings omitted for budget; uncovered requirements must be listed under open questions ...]"
            items.append({"kind": "note", "ref": "findings_trimmed", "reason": "budget"})
        items.append({"kind": "evidence", "ref": "findings_bundle", "reason": "report_material"})
        return [HumanMessage(content=head + "## Research findings\n" + findings)], items

    # -- generic builder ----------------------------------------------------
    def build(
        self,
        identity: Identity,
        role: str,
        messages: Sequence[BaseMessage],
        *,
        tools: Optional[List[Any]] = None,
        response_schema: Optional[type] = None,
        items: Optional[List[Dict[str, Any]]] = None,
        evictions: Optional[List[EvictionRecord]] = None,
        model: Optional[str] = None,
        declared_max_tokens: int = 0,
        policy_version: str = CONTEXT_POLICY_VERSION,
    ) -> ContextPacket:
        """Measure, manifest and freeze one request."""
        tools = list(tools or [])
        budget = self.budget_tokens_for(role, model, declared_max_tokens)
        estimated = message_tokens(messages) + tool_tokens(tools)
        request_id = new_request_id(role)
        manifest_items = list(items or [{"kind": "messages", "ref": role, "reason": "assembled"}])
        manifest = ContextManifest(
            request_id=request_id,
            agent_instance_id=identity.agent_instance_id,
            role=role,
            policy_version=policy_version,
            ordered_items=manifest_items,
            tool_names=[str(getattr(t, "name", t.get("name", "")) if isinstance(t, dict) else getattr(t, "name", "")) for t in tools],
            tool_schema_hashes=tool_schema_hashes(tools),
            dropped_items=list(evictions or []),
            estimated_input_tokens=estimated,
            budget_tokens=budget,
            counting_method=counting_method(),
            request_hash=hash_obj([_message_text(m) for m in messages] + [getattr(t, "name", "") for t in tools]),
            model=model or "",
            observability="full" if tools else "none",
        )
        if estimated > budget:
            LOGGER.warning("Request %s for %s exceeds budget (%s > %s)", request_id, role, estimated, budget)
        return ContextPacket(
            request_id=request_id,
            role=role,
            identity=identity,
            messages=list(messages),
            tools=tools,
            response_schema=response_schema,
            manifest=manifest,
            estimated_input_tokens=estimated,
            budget_tokens=budget,
            evictions=list(evictions or []),
            tool_names=manifest.tool_names,
        )


# --------------------------------------------------------------------------- #
# Model call wrapper (records requests, manifests and usage for every role)
# --------------------------------------------------------------------------- #
class BudgetExhausted(RuntimeError):
    """Raised when a model call may not start because the run budget is spent.

    This is a controlled stop, not a provider failure: callers must not retry it
    (retrying would be exactly the over-spend it prevents).
    """

    def __init__(self, reason: str, *, role: str = "", request_id: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.role = role
        self.request_id = request_id


class ModelCallBudget:
    """Run-wide, atomic model-call budget used by every model call.

    Every attempt (first try and each retry) takes one reservation before the request
    is sent. The grant check and the write share the store's metadata lock, so
    concurrent reservations - parallel researchers, a retrying review and an
    auxiliary extraction - cannot both spend the last remaining call.

    Without a store there is no execution-time metering: the run cannot observe what
    it spends, so it is *not* claimed to be strictly capped (``strict`` is False and
    budget failures never raise). The review-enabled graph always runs with a store,
    which is where the caps matter.
    """

    def __init__(self, store: Optional[ArtifactStore], config: Optional[Configuration] = None, *, max_calls: Optional[int] = None, max_tokens: Optional[int] = None):
        self.store = store
        self.max_calls = int(max_calls if max_calls is not None else (getattr(config, "max_total_model_calls", 120) or 0))
        self.max_tokens = int(max_tokens if max_tokens is not None else (getattr(config, "max_total_tokens", 2_000_000) or 0))
        self._lock = threading.RLock()

    @property
    def strict(self) -> bool:
        """Return True when the budget is enforced against a real ledger."""
        return self.store is not None

    def available(self, role: Optional[str] = None, role_cap: Optional[int] = None) -> bool:
        """Return True when at least one more attempt may be granted (no reservation).

        Cheaper than :meth:`reserve` and safe to call before binding a model or a tool
        schema, which may construct a provider client.
        """
        if self.store is None:
            return True
        totals = self.store.totals()
        # ``is not None``: an explicit 0 means "no call may start", while None means
        # the budget was not configured here (the store still has the real cap).
        if self.max_calls is not None and totals["model_calls"] >= self.max_calls:
            return False
        if self.max_tokens is not None and (totals["input_tokens"] + totals["output_tokens"]) >= self.max_tokens:
            return False
        if role and role_cap is not None and self.store.calls_by_role().get(role, 0) >= int(role_cap):
            return False
        return True

    def reserve(
        self,
        *,
        request_id: str,
        attempt_id: str,
        role: str,
        agent_instance_id: str = "",
        estimated_input_tokens: int = 0,
        reserved_output_tokens: int = 0,
        role_cap: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Take one attempt's reservation, or raise :class:`BudgetExhausted`."""
        if self.store is None:
            # No ledger, no metering: the run cannot enforce (or claim) a hard cap, so
            # this is a no-op reservation and ``strict`` is False.
            return None
        granted, reason, reservation = self.store.reserve_model_call(
            request_id=request_id,
            attempt_id=attempt_id,
            role=role,
            agent_instance_id=agent_instance_id,
            max_total_model_calls=self.max_calls,
            max_total_tokens=self.max_tokens,
            estimated_input_tokens=estimated_input_tokens,
            unknown_output_tokens=max(1, int(reserved_output_tokens or 0)),
            role_cap=role_cap,
        )
        if not granted:
            self.store.append_event({"type": "model_call_blocked", "request_id": request_id, "attempt_id": attempt_id, "role": role, "reason": reason})
            raise BudgetExhausted(reason, role=role, request_id=request_id)
        return reservation


def _request_payload(packet: ContextPacket, model_name: str, max_tokens: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "request_id": packet.request_id,
        "role": packet.role,
        "agent_instance_id": packet.identity.agent_instance_id,
        "model": model_name,
        "max_tokens": max_tokens,
        "messages": [_serialize_message(m) for m in packet.messages],
        "tools": packet.manifest.tool_names if packet.manifest else [],
        "tool_schema_hashes": packet.manifest.tool_schema_hashes if packet.manifest else {},
        "response_schema": getattr(packet.response_schema, "__name__", None),
        "context_policy": "scoped",
        "recorded_at": utc_now(),
    }
    return redact_secrets(payload)


def _serialize_message(message: BaseMessage) -> Dict[str, Any]:
    data: Dict[str, Any] = {"type": message.type, "content": message.content}
    if isinstance(message, AIMessage) and message.tool_calls:
        data["tool_calls"] = [{"id": c.get("id"), "name": c.get("name"), "args": c.get("args")} for c in message.tool_calls]
    if isinstance(message, ToolMessage):
        data["tool_call_id"] = message.tool_call_id
        data["name"] = message.name
    if isinstance(message, SystemMessage):
        data.pop("type", None)
        data["type"] = "system"
    return data


def _usage_from_response(response: Any, *, request_id: str, identity: Identity, role: str, model_name: str, latency: float, attempt_id: str = "", reservation_id: str = "") -> UsageRecord:
    usage = getattr(response, "usage_metadata", None) or {}
    meta = getattr(response, "response_metadata", {}) or {}
    input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    cached = None
    if isinstance(usage, dict):
        details = usage.get("input_token_details") or {}
        cached = details.get("cache_read") if isinstance(details, dict) else None
    if input_tokens is None:
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(token_usage, dict):
            input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
            output_tokens = token_usage.get("completion_tokens", token_usage.get("output_tokens"))
    return UsageRecord(
        request_id=request_id,
        attempt_id=attempt_id,
        reservation_id=reservation_id,
        agent_instance_id=identity.agent_instance_id,
        role=role,
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached,
        cost_usd=None,
        latency_s=round(latency, 3),
        counting_method="provider_usage" if input_tokens is not None else "unknown",
    )


def _strip_unsupported(messages: Sequence[BaseMessage], model: Any) -> List[BaseMessage]:
    """Remove tool call ids the bound model rejects (e.g. seeded placeholder ids)."""
    try:
        kwargs = model._get_ls_params() if hasattr(model, "_get_ls_params") else {}
        name = str(kwargs.get("ls_model_name") or "")
    except Exception:  # noqa: BLE001
        name = ""
    if not any(marker in name.lower() for marker in ("gemini", "google", "vertex")):
        return list(messages)
    out: List[BaseMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id.startswith("call_seed_"):
            out.append(HumanMessage(content=f"[tool result:{message.name or ''}] {_message_text(message)}"))
        elif isinstance(message, AIMessage) and message.tool_calls and any(c.get("id", "").startswith("call_seed_") for c in message.tool_calls):
            out.append(HumanMessage(content=f"[assistant note] {_message_text(message)}"))
        else:
            out.append(message)
    return out


async def assemble(
    model: Any,
    packet: ContextPacket,
    *,
    store: Optional[ArtifactStore],
    record_requests: bool = True,
    max_tokens: int = 0,
    model_name: str = "",
    retries: int = 1,
    budget: Optional["ModelCallBudget"] = None,
    role_cap: Optional[int] = None,
) -> Any:
    """Send one frozen packet and record request, manifest and usage.

    The packet is the only source of messages and tools; the caller must not append
    anything after this call.

    Binding order is fixed here: configure the concrete model, bind the packet's tools
    and response schema, then apply the bounded retry. A ``with_retry`` wrapper does
    not expose ``bind_tools`` / ``with_structured_output``, so passing one in is a
    programming error rather than a supported call path.

    Every attempt - including each retry - first takes an atomic budget reservation
    (:class:`ModelCallBudget`). When the remaining allowance is gone the call is not
    sent at all and :class:`BudgetExhausted` is raised; the caller must turn that into
    a controlled stop, never retry around it.
    """
    if budget is None:
        # Direct callers (evaluation harnesses, tests) get a budget bound to the same
        # store: the accounting path is never optional.
        budget = ModelCallBudget(store)
    if not budget.available(packet.role, role_cap):
        # Check before binding: ``with_structured_output`` / ``bind_tools`` may build a
        # provider client, and a call that will not be authorised must not touch one.
        budget.reserve(
            request_id=packet.request_id,
            attempt_id=f"{packet.request_id}#a1",
            role=packet.role,
            agent_instance_id=packet.identity.agent_instance_id,
            role_cap=role_cap,
        )
    invoke_model = model
    if packet.tools:
        binder = getattr(model, "bind_tools", None)
        if binder is None:
            raise TypeError("ContextPacket carries tools but the model cannot bind tools; pass the configured chat model, not a retry wrapper")
        invoke_model = binder(packet.tools)
    if packet.response_schema is not None:
        structured = getattr(invoke_model, "with_structured_output", None)
        if structured is None:
            raise TypeError("ContextPacket carries a response schema but the model cannot produce structured output; pass the configured chat model, not a retry wrapper")
        invoke_model = structured(packet.response_schema)
    if store is not None:
        try:
            store.write_request(packet.identity, packet.request_id, record_requests, _request_payload(packet, model_name, max_tokens), packet.manifest)
        except Exception:  # noqa: BLE001 - observation must never break a run
            LOGGER.warning("Failed to record request %s", packet.request_id, exc_info=True)
    messages = _strip_unsupported(packet.messages, invoke_model if packet.tools else model)
    attempts = max(1, int(retries or 1))
    output_reserve = max(0, int(max_tokens or 0))
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        attempt_id = f"{packet.request_id}#a{attempt}"
        reservation = budget.reserve(
            request_id=packet.request_id,
            attempt_id=attempt_id,
            role=packet.role,
            agent_instance_id=packet.identity.agent_instance_id,
            estimated_input_tokens=packet.estimated_input_tokens,
            reserved_output_tokens=output_reserve,
            role_cap=role_cap,
        )
        started = time.perf_counter()
        try:
            response = await invoke_model.ainvoke(messages)
        except asyncio.CancelledError as exc:
            # Timeout / cancellation: the request may already have consumed provider
            # resources, so it is recorded as an attempt with unknown usage and the
            # cancellation is propagated untouched (never swallowed).
            last_error = exc
            _record_attempt(store, packet, attempt_id=attempt_id, reservation=reservation, model_name=model_name, latency=time.perf_counter() - started, counting_method="timed_out", error=exc)
            raise
        except Exception as exc:
            last_error = exc
            _record_attempt(store, packet, attempt_id=attempt_id, reservation=reservation, model_name=model_name, latency=time.perf_counter() - started, counting_method="failed_call", error=exc)
            if attempt >= attempts:
                raise
            continue
        latency = time.perf_counter() - started
        if store is not None:
            try:
                store.record_usage(_usage_from_response(response, request_id=packet.request_id, identity=packet.identity, role=packet.role, model_name=model_name, latency=latency, attempt_id=attempt_id, reservation_id=(reservation or {}).get("id", "")))
            except Exception:  # noqa: BLE001
                LOGGER.warning("Failed to record usage for %s", packet.request_id, exc_info=True)
        return response
    raise last_error  # pragma: no cover - the loop either returns or raises


def _record_attempt(
    store: Optional[ArtifactStore],
    packet: ContextPacket,
    *,
    attempt_id: str,
    reservation: Optional[Dict[str, Any]],
    model_name: str,
    latency: float,
    counting_method: str,
    error: BaseException,
) -> None:
    """Record one attempt that did not return a response (never raises)."""
    if store is None:
        return
    try:
        store.record_usage(
            UsageRecord(
                request_id=packet.request_id,
                attempt_id=attempt_id,
                reservation_id=(reservation or {}).get("id", ""),
                agent_instance_id=packet.identity.agent_instance_id,
                role=packet.role,
                model=model_name,
                latency_s=round(latency, 3),
                counting_method=counting_method,
            )
        )
        store.append_event(
            {
                "type": "model_call_timed_out" if counting_method == "timed_out" else "model_call_failed",
                "request_id": packet.request_id,
                "attempt_id": attempt_id,
                "role": packet.role,
                "error_class": type(error).__name__,
            }
        )
    except Exception:  # noqa: BLE001 - observation must never replace the model error
        LOGGER.warning("Failed to record attempt %s", attempt_id, exc_info=True)


def baseline_request(packet: ContextPacket) -> Tuple[List[BaseMessage], List[Any], Optional[type]]:
    """Return the raw messages/tools/schema for baseline (unscoped) calls."""
    return list(packet.messages), list(packet.tools), packet.response_schema


def dedupe_tool_messages(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    """Drop exact duplicate tool results that appear in later turns (keep the first)."""
    seen: Dict[str, int] = {}
    keep: List[BaseMessage] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            digest = hashlib.sha256(_message_text(message).encode("utf-8")).hexdigest()
            if digest in seen:
                seen[digest] += 1
                continue
            seen[digest] = 1
        keep.append(message)
    return keep


def is_within(path: str | os.PathLike, root: str | os.PathLike) -> bool:
    """Return True when ``path`` (resolved) is inside ``root`` (resolved)."""
    try:
        Path(path).expanduser().resolve().relative_to(Path(root).expanduser().resolve())
        return True
    except ValueError:
        return False


def evidence_pointer(evidence_ids: Iterable[str]) -> str:
    """Render a stable pointer string for evidence ids."""
    ids = [validate_id(e, "evidence id") for e in evidence_ids]
    return ", ".join(ids)
