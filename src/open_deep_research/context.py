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
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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

from open_deep_research import observability as obs
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


# --------------------------------------------------------------------------- #
# Budget clipping that does not silently drop the *conditions* of an answer
# --------------------------------------------------------------------------- #
# A plain head (or tail) cut is the cheapest way to lose the one sentence that
# decides the answer: an exception, a negation, a version or a date near the other
# end of the document. These markers do not prove relevance, but keeping a window
# around them is strictly better than keeping only one side, and the omission count
# stays explicit so the model knows what it did not see.
PROTECTED_MARKERS = (
    "exception",
    "except",
    "unless",
    "however",
    "but ",
    "not ",
    "never",
    "only",
    "before ",
    "after ",
    "deadline",
    "as of",
    "deprecated",
    "caveat",
    "warning",
    "erratum",
    "correction",
    "例外",
    "但是",
    "然而",
    "除非",
    "仅",
    "只",
    "不得",
    "不能",
    "截至",
    "版本",
    "注意",
    "警告",
    "相反",
    "撤回",
    "勘误",
)


@dataclass
class ClippedText:
    """A bounded view of a longer text plus what was left out."""

    text: str
    omitted_chars: int = 0
    protected_spans: int = 0


def _merge_intervals(intervals: List[tuple], limit: int) -> List[tuple]:
    merged: List[tuple] = []
    for start, end in sorted(intervals):
        start = max(0, start)
        end = min(limit, end)
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def clip_preserving_spans(
    text: str,
    limit_chars: int,
    *,
    markers: Sequence[str] = PROTECTED_MARKERS,
    head_ratio: float = 0.45,
    tail_ratio: float = 0.2,
    max_windows: int = 3,
) -> ClippedText:
    """Clip ``text`` to ``limit_chars`` while keeping head, tail and marker windows.

    The result length never exceeds ``limit_chars`` (the omission markers are paid for
    out of the budget). ``omitted_chars`` counts only source characters that are not
    shown, so a caller can record exactly how much of the material was moved out.
    """
    text = text or ""
    limit_chars = max(200, int(limit_chars or 0))
    if len(text) <= limit_chars:
        return ClippedText(text=text)
    # Reserve space for up to max_windows omission markers.
    marker_reserve = 120 * (max_windows + 1)
    budget = max(120, limit_chars - marker_reserve)
    head = int(budget * head_ratio)
    tail = max(40, int(budget * tail_ratio))
    window_budget = max(0, budget - head - tail)
    # At a very tight cap the marker windows become narrower than a sentence: the
    # clipper then keeps only fragments and retains *less* than a plain head cut.
    # Degrade to a clean head+tail view instead of advertising protection it cannot
    # provide (measured: 3/24 vs 8/24 on the 800-char stress set).
    if window_budget < 240:
        tail = max(40, int(limit_chars * 0.12))
        head = max(80, limit_chars - tail - 48)
        omitted = max(0, len(text) - head - tail)
        body = text[:head] + f"\n[... {omitted} characters omitted ...]\n" + text[-tail:]
        if len(body) > limit_chars:
            body = body[:limit_chars]
        return ClippedText(text=body, omitted_chars=omitted, protected_spans=0)
    middle_start, middle_end = head, len(text) - tail
    windows: List[tuple] = []
    if window_budget and middle_end > middle_start:
        lowered = text.lower()
        hits: List[int] = []
        for marker in markers:
            start = middle_start
            while len(hits) < 40:
                idx = lowered.find(marker.lower(), start, middle_end)
                if idx < 0:
                    break
                hits.append(idx)
                start = idx + max(1, len(marker))
        hits = sorted(set(hits))
        if hits:
            # Spread the windows over the most separated hits instead of stacking them.
            chunk = max(1, window_budget // min(max_windows, len(hits)))
            chosen: List[int] = []
            for hit in hits:
                if all(abs(hit - other) > chunk for other in chosen):
                    chosen.append(hit)
                if len(chosen) >= max_windows:
                    break
            for hit in chosen:
                half = chunk // 2
                windows.append((hit - half, hit + half))
    intervals = _merge_intervals(windows + [(0, head), (len(text) - tail, len(text))], len(text))
    pieces: List[str] = []
    omitted = 0
    previous = 0
    for start, end in intervals:
        if start > previous:
            omitted += start - previous
        pieces.append(text[start:end])
        previous = end
    if previous < len(text):
        omitted += len(text) - previous
    body = ""
    cursor = 0
    for start, end in intervals:
        if start > cursor:
            body += f"\n[... {start - cursor} characters omitted ...]\n"
        body += text[start:end]
        cursor = end
    if cursor < len(text):
        body += f"\n[... {len(text) - cursor} characters omitted ...]\n"
    if len(body) > limit_chars:
        body = body[:limit_chars]
    return ClippedText(text=body, omitted_chars=omitted, protected_spans=len(windows))


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
                doc = store.archive_tool_result(identity, text, title=f"tool result {snap.call_id}", call_id=snap.call_id)
                artifact_ref = doc["artifact_id"]
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


def _group_text(group: Sequence[BaseMessage]) -> str:
    return "\n".join(_message_text(message) for message in group)


def select_groups_to_budget(
    groups: Sequence[Sequence[BaseMessage]],
    *,
    base_tokens: int,
    budget: int,
    focus_terms: Sequence[str] = (),
) -> Tuple[List[BaseMessage], List[EvictionRecord], int, int]:
    """Select complete interaction units by task focus first, recency second.

    Unlike a plain oldest-first trim, this keeps a turn that mentions a requirement
    the task still has to cover, and drops an older turn that mentions nothing the
    current task needs. The newest unit is always kept so the model sees the immediate
    result it has to react to. Only whole units are kept: no dangling tool call and no
    orphan tool result can be produced.

    Returns:
        ``(messages, evictions, used_tokens, kept_units)``.
    """
    terms = [str(term).lower() for term in focus_terms if term and len(str(term)) > 1]
    scored: List[Tuple[int, int, int]] = []
    for index, group in enumerate(groups):
        text = _group_text(group).lower()
        overlap = sum(1 for term in terms if term in text)
        scored.append((overlap, index, message_tokens(group)))
    keep: List[int] = [len(groups) - 1] if groups else []
    used = base_tokens + (scored[-1][2] if scored else 0)
    order = sorted(scored, key=lambda item: (-item[0], -item[1]))
    for overlap, index, tokens in order:
        if index in keep:
            continue
        if used + tokens > budget:
            continue
        keep.append(index)
        used += tokens
    keep.sort()
    kept_ids = set(keep)
    evictions: List[EvictionRecord] = []
    for index, group in enumerate(groups):
        if index in kept_ids:
            continue
        evictions.append(
            EvictionRecord(
                artifact_id=f"messages/{index}",
                reason="history_trimmed_focus",
                recoverable=True,
            )
        )
    messages = [message for index in keep for message in groups[index]]
    return messages, evictions, used, len(keep)


def _section(title: str, body: str) -> str:
    body = (body or "").strip()
    if not body:
        return ""
    return f"## {title}\n{body}"


_FOCUS_WORD_RE = re.compile(r"[A-Za-z0-9_./-]{3,}|[\u4e00-\u9fff]{2,}")


def focus_terms(*sources: Any) -> List[str]:
    """Return the searchable terms of the current task, in first-seen order.

    The terms come from the task card, its requirement statements and the working
    memory's open gaps. They are used to decide which already-executed interaction
    units stay in the request when the history does not fit; they never invent new
    material and never replace the mandatory prompt sections.
    """
    terms: List[str] = []
    seen: set[str] = set()

    def add(text: Any) -> None:
        if not text:
            return
        for match in _FOCUS_WORD_RE.findall(str(text)):
            lowered = match.lower()
            if lowered in seen or lowered.isdigit() and len(lowered) < 4:
                continue
            seen.add(lowered)
            terms.append(lowered)

    for source in sources:
        if source is None:
            continue
        if isinstance(source, str):
            add(source)
            continue
        add(getattr(source, "objective", ""))
        add(getattr(source, "statement", ""))
        add(getattr(source, "next_action", ""))
        for attribute in ("requirements", "unknowns", "needs_revalidation", "active_requirement_ids"):
            for item in getattr(source, attribute, []) or []:
                add(getattr(item, "statement", item))
    return terms[:30]


def render_working_memory(memory: Any, *, limit_chars: int = 2200) -> str:
    """Render the compact work state as named fields instead of a raw JSON blob.

    Named fields make it visible what is still open; a truncated JSON dump hides the
    exact line the next action depends on (usually ``next_action`` at the end).
    """
    if memory is None:
        return ""
    data = memory.model_dump(mode="json") if hasattr(memory, "model_dump") else dict(memory)
    lines: List[str] = []
    fields = (
        ("active_requirement_ids", "requirements this task must cover"),
        ("covered_requirement_ids", "requirements with evidence"),
        ("unknowns", "unknowns / gaps"),
        ("conflicts", "conflicts"),
        ("failed_approaches", "approaches that failed (do not repeat)"),
        ("recovery_refs", "material restored via request_context"),
    )
    for key, label in fields:
        values = data.get(key) or []
        if values:
            lines.append(f"- {label}: " + "; ".join(str(v)[:220] for v in values[:8]))
    if data.get("needs_revalidation"):
        lines.append("- needs revalidation (do not restate as confirmed): " + ", ".join(str(v) for v in data["needs_revalidation"]))
    if data.get("next_action"):
        lines.append(f"- next action: {str(data['next_action'])[:300]}")
    text = "\n".join(lines)
    return text[:limit_chars]


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
        body += "\n\n" + _section("Run policy", f"context_policy={state.get('context_policy', 'scoped')}; reviewer_enabled={str(bool(self.config.reviewer_enabled)).lower()}; remaining supervisor turns: {max(0, (state.get('iteration_allowance') or state.get('max_iterations') or 0) - state.get('iteration', 0))}")
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
            f"remaining supervisor turns: {max(0, (state.get('iteration_allowance') or state.get('max_iterations') or 0) - state.get('iteration', 0))}"
            + (" (this is a repair turn granted by a review)" if state.get("repair_turn") else "")
            + f"; model calls: {state.get('usage', {}).get('model_calls', 0)}/{self.config.max_total_model_calls}",
        )
        repair_claims = state.get("pending_claims") or []
        repair_feedback = state.get("repair_feedback") or []
        if repair_feedback:
            body += "\n\n" + _section(
                "Host results of earlier claim repairs (must address)",
                "\n".join(json.dumps({"applied": row.get("applied", []), "rejected": row.get("rejected", [])}, ensure_ascii=True) for row in repair_feedback),
            )
            items.append({"kind": "tool_result", "ref": "repair_feedback", "reason": "host_rejection"})
        if repair_claims:
            claim_lines = []
            for claim in repair_claims:
                claim_lines.append(
                    f"- {claim.get('claim_id')} v{claim.get('version', 1)} [{claim.get('status', 'active')}] "
                    f"evidence={claim.get('evidence_refs')} requirements={claim.get('requirement_ids')}\n"
                    f"  statement: {str(claim.get('statement') or '')[:400]}"
                    + (f"\n  unbound: {claim.get('unbound_reason')}" if claim.get("unbound_reason") else "")
                    + ("\n  Current archived support quote (JSON, preserve escapes): " + json.dumps(claim["support_span"]["quote"], ensure_ascii=True) if (claim.get("support_span") or {}).get("quote") else "")
                )
            body += "\n\n" + _section(
                "Claims a review asked you to correct",
                "\n".join(claim_lines)
                + "\n\nUse ReviseClaims with the claim id, a complete replacement statement, exactly one archived evidence id and a "
                "verbatim quote from that evidence. Withdraw a claim (action=withdraw) when no archived source supports it; never delete evidence.",
                # Quotes retain NBSP and Markdown characters. Copy the JSON string
                # exactly, or narrow/withdraw the claim instead of normalizing it.
            )
            body += "\nCopy quote characters exactly, including Unicode whitespace and Markdown links. JSON \\u escapes are allowed. If a host rejected a quote, correct it or withdraw the unsupported/off-topic claim; do not repeat the rejected proposal."
            items.append({"kind": "claim", "ref": "pending_claims", "reason": "repair_target"})
        repair_evidence = state.get("repair_evidence") or []
        if repair_evidence:
            body += "\n\n" + _section(
                "Archived excerpts for the claims above",
                "\n".join(f"- [{row.get('evidence_id')}] <{row.get('url', '')}>: {row.get('excerpt', '')}" for row in repair_evidence),
            )
            items.append({"kind": "evidence", "ref": "repair_evidence", "reason": "repair_target"})
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
        brief = state.get("brief")
        constraints = [str(item) for item in (getattr(brief, "constraints", []) or [])]
        if constraints:
            # Every hard constraint is non-negotiable: the earlier ``[:8]`` silently
            # dropped the ninth and later ones with no notice to the model.
            base_prompt += "\n\n## Non-negotiable constraints (keep every one)\n" + "\n".join(f"- {item}" for item in constraints)
        if background and background not in base_prompt:
            base_prompt += f"\n\n## Background (from the supervisor, may be an assumption)\n{background[:2000]}"
        working = state.get("working_memory")
        if working is not None:
            rendered_memory = render_working_memory(working)
            if rendered_memory:
                base_prompt += "\n\n## Your working summary so far\n" + rendered_memory
                items.append({"kind": "memory", "ref": "working_memory", "reason": "own_state"})
        # The demand of *this* turn: what the task still owes and what the model was
        # about to do. It is rebuilt from the ledger every turn, not copied from the
        # supervisor's original plan, so a tool failure or a new gap changes it.
        demand_lines: List[str] = []
        active_ids = list(getattr(working, "active_requirement_ids", []) or []) or list(getattr(task_card, "requirement_ids", []) or [])
        if active_ids:
            demand_lines.append("- requirement ids in scope: " + ", ".join(active_ids[:12]))
        missing = list(getattr(working, "unknowns", []) or [])
        if missing:
            demand_lines.append("- still missing: " + "; ".join(str(item)[:180] for item in missing[:5]))
        if demand_lines:
            base_prompt += "\n\n## Current demand\n" + "\n".join(demand_lines)

        history = list(state.get("messages") or []) if include_history else []
        groups = to_interaction_groups(history) if history else []
        system_tokens = count_tokens(base_prompt) + 8
        tools_tk = tool_tokens(tools)
        max_tool_result_tokens = int(self.budget_tokens_for("researcher") * _OFFLOAD_FRACTION)
        # Offload oversized single tool results before measuring.
        history_messages, evictions = self._offload_groups(identity, groups, max_tool_result_tokens)
        budget = self.budget_tokens_for("researcher")
        kept_groups = to_interaction_groups(history_messages) if history_messages else []
        terms = focus_terms(task_card, working)
        trimmed, extra_evictions, _, kept_units = select_groups_to_budget(
            kept_groups,
            base_tokens=system_tokens + tools_tk,
            budget=budget,
            focus_terms=terms,
        )
        evictions.extend(extra_evictions)
        if kept_units < len(kept_groups):
            base_prompt += (
                "\n\n## Note on history\n"
                f"{len(kept_groups) - kept_units} older interaction unit(s) were moved out of this request to stay within budget; "
                "units that mention the current requirements were kept first. Their tool results are archived and the task card "
                "plus this working summary are complete. Use request_context with a topic to recover a specific passage."
            )
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
                                doc = self.store.archive_tool_result(identity, text, title=f"tool result {message.tool_call_id}", call_id=message.tool_call_id)
                                ref = doc["artifact_id"]
                            except Exception:  # noqa: BLE001
                                ref = ""
                        if ref:
                            limit_chars = max(400, int(max_result_tokens * 2))
                            clipped = clip_preserving_spans(text, limit_chars, head_ratio=0.45, tail_ratio=0.3)
                            out.append(
                                ToolMessage(
                                    content=(
                                        f"{clipped.text}\n\n[... {clipped.omitted_chars} characters trimmed; "
                                        f"full original archived as {ref}; use request_context to read a specific passage.]"
                                    ),
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
        archive_rows = []
        if self.store is not None:
            for eid in list(dict.fromkeys(state.get("evidence_refs") or []))[:8]:
                if not self.store.can_read_evidence(identity, eid):
                    continue
                row = self.store.read_excerpt(identity, eid, max_tokens=min(300, max(60, budget // 16)))
                candidate = json.dumps(archive_rows + [row], ensure_ascii=False).replace("\u00a0", "\\u00a0")
                if count_tokens(candidate) <= budget // 2:
                    archive_rows.append(row)
        source_text = json.dumps(archive_rows, ensure_ascii=False).replace("\u00a0", "\\u00a0")
        if archive_rows:
            items.append({"kind": "evidence", "ref": "archived_source_excerpts", "reason": "verbatim_fact_binding"})
        base = count_tokens(rules) + count_tokens(task_text) + count_tokens(source_text) + 300
        allowed_chars = max(2000, (budget - base) * 4)
        if len(rendered) > allowed_chars:
            # Keep the head, the tail and windows around exception/negation markers.
            # Keeping only the newest suffix lost the early definitions; keeping only a
            # prefix lost the late corrections. Both are conditions of the answer.
            clipped = clip_preserving_spans(rendered, allowed_chars, head_ratio=0.35, tail_ratio=0.45)
            rendered = clipped.text
            items.append(
                {
                    "kind": "note",
                    "ref": "history_window",
                    "reason": "offloaded",
                    "omitted_chars": clipped.omitted_chars,
                }
            )
        human = (
            "All above messages are about research conducted by an AI Researcher. Please clean up these findings.\n\n"
            "DO NOT summarize the information. I want the raw information returned, just in a cleaner format. "
            "Make sure all relevant information is preserved - you can rewrite findings verbatim.\n\n"
            "<Task card>\n" + task_text + "\n</Task card>\n\n"
            "<Archived source excerpts (JSON)>\n" + source_text + "\n</Archived source excerpts>\n"
            "Copy fact quotes exactly from readable archived text; preserve JSON escapes and whitespace. Summaries are not replacement source quotes.\n\n"
            "<Research transcript>\n" + rendered + "\n</Research transcript>"
        )
        return [SystemMessage(content=rules), HumanMessage(content=human)], items

    def extractor_messages(self, url: str, title: str, content: str, *, query: str = "", objective: str = "", char_budget: Optional[int] = None) -> Tuple[List[BaseMessage], List[Dict[str, Any]]]:
        """Build one webpage-extraction request (evidence id bound by the caller).

        The extractor must see the part of the page the task is about, not only the
        page's opening. The head and tail are kept, and the rest of the budget goes to
        the window that best matches the search query / task objective; omitted ranges
        are recorded in the manifest with their offsets.
        """
        from open_deep_research.prompts import summarize_webpage_prompt
        from open_deep_research.utils import compose_relevant_excerpt, get_today_str

        budget = self.budget_tokens_for("extractor")
        char_cap = max(2000, budget * 4)
        if char_budget is not None:
            char_cap = min(char_cap, max(200, int(char_budget)))
        items = [
            {"kind": "role", "ref": "roles/extractor.md", "reason": "role_policy"},
            {"kind": "source", "ref": url or title or "unknown-source", "reason": "document_under_extraction"},
        ]
        if query or objective:
            items.append({"kind": "task", "ref": (query or objective)[:120], "reason": "extraction_focus"})
        # The document was archived in full by the caller. Clip only what the extractor
        # sees, and keep the task-relevant middle as well as the opening definitions and
        # the trailing exceptions.
        focused, spans = compose_relevant_excerpt(content, query=query, objective=objective, char_budget=char_cap)
        if len(focused) < len(content):
            kept = sum(span["end"] - span["start"] for span in spans)
            items.append({"kind": "note", "ref": "extraction_window", "reason": "offloaded", "omitted_chars": max(0, len(content) - kept), "spans": spans})
        prompt = summarize_webpage_prompt.format(webpage_content=focused, date=get_today_str())
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
        requirement_list = list(getattr(brief, "requirements", []) or [])
        requirements = "\n".join(f"- {r.requirement_id}: {r.statement}" for r in requirement_list)
        constraints = "\n".join(f"- {c}" for c in (getattr(brief, "constraints", []) or []))
        user_messages = get_buffer_string(state.get("messages", []))
        review_notes = state.get("writing_constraints") or ""
        budget = self.budget_tokens_for("writer")
        # A single-requirement brief is answered, not expanded into a survey. Stating
        # the answer scope here (rather than only in the role prompt) keeps the writer
        # from turning the claim inventory into candidate tables and negative sweeps:
        # the archived L-D report grew that way and never converged.
        scope_note = ""
        if len(requirement_list) == 1:
            only = requirement_list[0]
            scope_note = (
                "## Answer scope\n"
                f"- Required answer: {only.requirement_id}: {only.statement.strip()}\n"
                f"- Required constraints: {', '.join(str(c) for c in (getattr(brief, 'constraints', []) or [])) or '(none)'}\n"
                "Write the shortest report that answers this. Add a section only when the requirement or a constraint needs it. "
                "The answer sentence itself carries its citation; an uncited conclusion line is not a delivered answer. "
                "Do not claim that a value was not found, or that no other source differs, unless an archived claim states it.\n\n"
            )
        head = (
            f"{rules}\n\n"
            f"## Research brief\n{getattr(brief, 'objective', None) or state.get('research_brief', '')}\n\n"
            f"## Requirements\n{requirements or '(none enumerated)'}\n\n"
            + (f"## Constraints\n{constraints}\n\n" if constraints else "")
            + scope_note
            + (f"## Reviewer constraints for this draft\n{review_notes}\n\n" if review_notes else "")
            + f"## User messages (language and format)\n{user_messages[:4000]}\n\n"
        )
        allowed = max(3000, (budget - count_tokens(head) - 200) * 4)
        if len(findings) > allowed:
            clipped = clip_preserving_spans(findings, allowed, head_ratio=0.5, tail_ratio=0.3)
            findings = (
                clipped.text
                + "\n\n[... remaining findings omitted for budget; uncovered requirements must be listed under open questions; "
                "the archived originals stay readable ...]"
            )
            items.append({"kind": "note", "ref": "findings_trimmed", "reason": "budget", "omitted_chars": clipped.omitted_chars})
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


class ContextBudgetExceeded(BudgetExhausted):
    """Raised when the exact request does not fit the model's input budget.

    A controlled stop, like :class:`BudgetExhausted`: the request is never sent and no
    attempt is reserved. It subclasses ``BudgetExhausted`` so every graph node already
    routes a refused call to its controlled partial path instead of crashing.
    """

    def __init__(self, *, role: str, request_id: str, estimated: int, budget: int):
        super().__init__(
            f"context budget exceeded for {role}: {estimated} estimated input tokens > {budget}",
            role=role,
            request_id=request_id,
        )
        self.estimated = int(estimated)
        self.budget = int(budget)


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

    def __init__(self, store: Optional[ArtifactStore], config: Optional[Configuration] = None, *, max_calls: Optional[int] = None, max_tokens: Optional[int] = None, execution_id: str = ""):
        self.store = store
        self.max_calls = int(max_calls if max_calls is not None else (getattr(config, "max_total_model_calls", 120) or 0))
        self.max_tokens = int(max_tokens if max_tokens is not None else (getattr(config, "max_total_tokens", 2_000_000) or 0))
        self.execution_id = str(execution_id or "")
        self._lock = threading.RLock()

    def _execution_id(self) -> str:
        """Return the ledger execution this budget spends against.

        The host opens the observability session before any node runs, so the active
        session is the authoritative execution. An explicit id wins for direct
        callers (tests, evaluation harnesses) that have no session.
        """
        if self.execution_id:
            return self.execution_id
        session = obs.current_session()
        if session is not None:
            return str(session.spec.execution_id)
        return ""

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
            execution_id=self._execution_id(),
        )
        if not granted:
            self.store.append_event({"type": "model_call_blocked", "request_id": request_id, "attempt_id": attempt_id, "role": role, "reason": reason})
            # Refused before a request is sent: an event, never a fabricated
            # zero-token provider call or a phantom attempt span.
            obs.record_event(
                "budget_denied",
                request_id=request_id,
                attempt_id=attempt_id,
                role=role,
                agent_instance_id=agent_instance_id,
                reason_code=reason[:120],
                budget_denied=True,
            )
            raise BudgetExhausted(reason, role=role, request_id=request_id)
        return reservation


@dataclass
class ModelCallContext:
    """The one resolved call context shared by every auxiliary model invocation.

    It carries the *actual* run configuration, the same run-wide budget/ledger and the
    execution id. Auxiliary calls (page summarisation/extraction) must not rebuild a
    default Configuration from an empty dict: that silently gives them a different
    model, output cap and budget than the run they belong to.
    """

    config: Configuration
    budget: "ModelCallBudget"
    execution_id: str = ""
    store: Optional[ArtifactStore] = None
    role: str = "summarization"


def call_context_for(
    config: Optional[Mapping[str, Any]],
    *,
    role: str = "summarization",
    store: Optional[ArtifactStore] = None,
) -> ModelCallContext:
    """Resolve one read-only call context from the runnable config.

    ``config`` is the LangGraph config of the calling node/tool. The runtime context in
    ``configurable["__odr_ctx__"]`` is the host-owned store/identity; a public caller
    cannot fabricate it, and its absence simply means "no ledger" (tests, direct use).
    """
    configurable = Configuration.from_runnable_config(dict(config or {}))
    ctx = (config or {}).get("configurable", {}).get("__odr_ctx__") if isinstance(config, Mapping) else None
    resolved_store = store if store is not None else getattr(ctx, "store", None)
    session = obs.current_session()
    execution_id = str(session.spec.execution_id) if session is not None else ""
    budget = ModelCallBudget(resolved_store, configurable, execution_id=execution_id)
    return ModelCallContext(config=configurable, budget=budget, execution_id=execution_id, store=resolved_store, role=role)


def auxiliary_identity(context: ModelCallContext, role: Optional[str] = None) -> Identity:
    """Return a host-derived identity for an auxiliary call (never model supplied)."""
    runtime = None
    try:
        runtime = getattr(context.store, "runtime", None)
    except Exception:  # noqa: BLE001
        runtime = None
    run_id = getattr(runtime, "run_id", "") or "aux-run"
    return Identity(run_id=str(run_id), agent_instance_id=f"aux-{role or context.role}-01", role=role or context.role)


def auxiliary_packet(
    context: ModelCallContext,
    messages: Sequence[Any],
    *,
    response_schema: Optional[type] = None,
    max_output_tokens: int = 0,
    role: Optional[str] = None,
    identity: Optional[Identity] = None,
    estimated_input_tokens: Optional[int] = None,
) -> ContextPacket:
    """Wrap already-built messages for :func:`assemble` without rewriting them.

    This is the baseline path: the prompt/messages/schema are exactly the caller's, so
    routing the call through the ledger cannot turn baseline into scoped trimming.
    """
    role_name = role or context.role
    tokens = estimated_input_tokens
    if tokens is None:
        tokens = sum(count_tokens(_message_text(message)) for message in messages)
    model_name = str(getattr(context.config, "summarization_model", "") or "")
    return ContextPacket(
        request_id=new_request_id(role_name),
        role=role_name,
        identity=identity or auxiliary_identity(context, role_name),
        messages=list(messages),
        response_schema=response_schema,
        estimated_input_tokens=tokens,
        # A real window, not ``input + output`` (which always "fit"): an auxiliary
        # request that cannot fit is refused by the same final check as the graph.
        budget_tokens=effective_budget(role_name, context.config, model_name, max_output_tokens),
    )


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
        # The label must describe the request that was actually assembled; a baseline
        # packet recorded as "scoped" would corrupt any per-arm analysis.
        "context_policy": (packet.manifest.policy_version if packet.manifest else CONTEXT_POLICY_VERSION) or CONTEXT_POLICY_VERSION,
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


def _usage_from_response(response: Any, *, request_id: str, identity: Identity, role: str, model_name: str, latency: float, attempt_id: str = "", reservation_id: str = "", execution_id: str = "") -> UsageRecord:
    usage = getattr(response, "usage_metadata", None) or {}
    meta = getattr(response, "response_metadata", {}) or {}
    input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    cached = None
    if isinstance(usage, dict):
        details = usage.get("input_token_details") or {}
        cached = details.get("cache_read") if isinstance(details, dict) else None
    actual_model = str(meta.get("model_name") or meta.get("model") or "")
    if input_tokens is None:
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        if isinstance(token_usage, dict):
            input_tokens = token_usage.get("prompt_tokens", token_usage.get("input_tokens"))
            output_tokens = token_usage.get("completion_tokens", token_usage.get("output_tokens"))
    return UsageRecord(
        request_id=request_id,
        attempt_id=attempt_id,
        reservation_id=reservation_id,
        execution_id=execution_id,
        agent_instance_id=identity.agent_instance_id,
        role=role,
        model=model_name,
        actual_model=actual_model,
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


def _request_timeout_seconds() -> float:
    """Bound one provider request (``ODR_MODEL_TIMEOUT_SECONDS``, default 300s)."""
    try:
        value = float(os.environ.get("ODR_MODEL_TIMEOUT_SECONDS", "300") or 300)
    except ValueError:
        value = 300.0
    return value if value > 0 else 300.0


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
    request_timeout: Optional[float] = None,
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
    execution_id = budget._execution_id()
    # Whether a native generation must appear for this attempt is a capability of the
    # model object, not a guess: a plain async double cannot emit framework callbacks.
    try:
        from langchain_core.language_models.chat_models import BaseChatModel

        native_callback_expected = isinstance(model, BaseChatModel)
    except Exception:  # noqa: BLE001
        native_callback_expected = False
    # The final send check: messages + tool schemas were already measured into
    # ``estimated_input_tokens``; ``budget_tokens`` is the model window minus the output
    # reserve. An over-budget request is refused here, before any provider client is
    # built and before an attempt is reserved, so it can never be sent.
    if packet.budget_tokens and packet.estimated_input_tokens > packet.budget_tokens:
        obs.record_event(
            "context_budget_exceeded",
            request_id=packet.request_id,
            role=packet.role,
            agent_instance_id=packet.identity.agent_instance_id,
            estimated_input_tokens=packet.estimated_input_tokens,
            budget_tokens=packet.budget_tokens,
            reason_code="request_exceeds_context_budget",
        )
        if store is not None:
            store.append_event(
                {
                    "type": "context_budget_exceeded",
                    "request_id": packet.request_id,
                    "role": packet.role,
                    "estimated_input_tokens": packet.estimated_input_tokens,
                    "budget_tokens": packet.budget_tokens,
                }
            )
        raise ContextBudgetExceeded(
            role=packet.role,
            request_id=packet.request_id,
            estimated=packet.estimated_input_tokens,
            budget=packet.budget_tokens,
        )
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
    structured_raw = False
    if packet.response_schema is not None:
        structured = getattr(invoke_model, "with_structured_output", None)
        if structured is None:
            raise TypeError("ContextPacket carries a response schema but the model cannot produce structured output; pass the configured chat model, not a retry wrapper")
        try:
            # Keep the raw AIMessage alongside the parsed object so provider token
            # usage is still visible after structured-output parsing. Older/fake
            # models that do not accept the keyword keep working unchanged.
            invoke_model = structured(packet.response_schema, include_raw=True)
            structured_raw = True
        except TypeError:
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
        # A denied reservation raises *before* the attempt span exists: nothing was
        # sent, so there is no attempt to time or bill.
        reservation = budget.reserve(
            request_id=packet.request_id,
            attempt_id=attempt_id,
            role=packet.role,
            agent_instance_id=packet.identity.agent_instance_id,
            estimated_input_tokens=packet.estimated_input_tokens,
            reserved_output_tokens=output_reserve,
            role_cap=role_cap,
        )
        attempt_span = obs.span(
            "llm.attempt",
            kind=obs.SPAN_MODEL_ATTEMPT,
            attributes={
                "attempt_id": attempt_id,
                "request_id": packet.request_id,
                "attempt_number": attempt,
                "role": packet.role,
                "agent_instance_id": packet.identity.agent_instance_id,
                "model": model_name,
                "estimated_input_tokens": packet.estimated_input_tokens,
                "budget_tokens": packet.budget_tokens,
                "transport_attempts": "unknown",
                "native_callback_expected": native_callback_expected,
            },
        )
        async with attempt_span:
            started = time.perf_counter()
            try:
                with obs.attempt_context(attempt_id, packet.request_id):
                    # A provider connection can hang forever (observed: a compressor
                    # request sat for 27+ minutes with no response and no error). An
                    # unbounded await turns that into a stuck run; the timeout makes it
                    # a recorded, retryable failed attempt instead.
                    timeout = _request_timeout_seconds() if request_timeout is None else float(request_timeout)
                    response = await asyncio.wait_for(invoke_model.ainvoke(messages), timeout=timeout)
            except asyncio.TimeoutError as exc:
                last_error = exc
                attempt_span.set(metric_status=obs.METRIC_UNKNOWN, missing_reason="timeout_before_usage")
                _record_attempt(store, packet, attempt_id=attempt_id, reservation=reservation, model_name=model_name, latency=time.perf_counter() - started, counting_method="timed_out", error=exc, execution_id=execution_id)
                attempt_span.fail(error_class=type(exc).__name__)
                if attempt >= attempts:
                    raise
                continue
            except asyncio.CancelledError as exc:
                # Timeout / cancellation: the request may already have consumed provider
                # resources, so it is recorded as an attempt with unknown usage and the
                # cancellation is propagated untouched (never swallowed).
                last_error = exc
                attempt_span.set(metric_status=obs.METRIC_UNKNOWN, missing_reason="cancelled_before_usage")
                _record_attempt(store, packet, attempt_id=attempt_id, reservation=reservation, model_name=model_name, latency=time.perf_counter() - started, counting_method="timed_out", error=exc, execution_id=execution_id)
                raise
            except Exception as exc:
                last_error = exc
                attempt_span.set(metric_status=obs.METRIC_UNKNOWN, missing_reason="failed_call_before_usage")
                _record_attempt(store, packet, attempt_id=attempt_id, reservation=reservation, model_name=model_name, latency=time.perf_counter() - started, counting_method="failed_call", error=exc, execution_id=execution_id)
                attempt_span.fail(error_class=type(exc).__name__)
                if attempt >= attempts:
                    raise
                continue
            latency = time.perf_counter() - started
            parsed = response
            usage_source = response
            if structured_raw and isinstance(response, dict):
                usage_source = response.get("raw") or response
                parsed = response.get("parsed")
                if response.get("parsing_error") is not None:
                    # The provider answered but the structured contract did not:
                    # this is a failed attempt, not a successful model delivery.
                    attempt_span.set(status_detail="structured_parse_failed")
                    attempt_span.fail(error_class=type(response["parsing_error"]).__name__)
            recorded = _usage_from_response(usage_source, request_id=packet.request_id, identity=packet.identity, role=packet.role, model_name=model_name, latency=latency, attempt_id=attempt_id, reservation_id=(reservation or {}).get("id", ""), execution_id=execution_id)
            _price_usage(recorded, pricing_path=getattr(store, "pricing_path", None))
            attempt_span.set(
                input_tokens=recorded.input_tokens,
                output_tokens=recorded.output_tokens,
                cache_read_tokens=recorded.cached_tokens,
                cost_usd=recorded.cost_usd,
                cost_method=recorded.cost_method,
                price_snapshot_hash=recorded.price_snapshot_hash,
                metric_status=obs.METRIC_MEASURED if recorded.input_tokens is not None else obs.METRIC_UNKNOWN,
                missing_reason=None if recorded.input_tokens is not None else "provider_returned_no_usage",
            )
            if store is not None:
                try:
                    store.record_usage(recorded)
                except Exception:  # noqa: BLE001
                    LOGGER.warning("Failed to record usage for %s", packet.request_id, exc_info=True)
            return parsed
    raise last_error  # pragma: no cover - the loop either returns or raises


def _price_usage(recorded: UsageRecord, pricing_path: Optional[str] = None) -> None:
    """Attach a frozen-price estimate to one usage row (never raises).

    The estimate is recomputable from the ledger plus ``price_snapshot_hash``. A
    null price stays null and is labelled ``unknown``: a missing price is not free.
    """
    try:
        from open_deep_research.pricing import estimate_cost, load_table

        table = load_table(pricing_path) if pricing_path else load_table()
        cost, method, detail = estimate_cost(
            table,
            model=recorded.model,
            input_tokens=recorded.input_tokens,
            output_tokens=recorded.output_tokens,
            cache_read_tokens=recorded.cached_tokens,
            cache_write_tokens=recorded.cache_write_tokens,
            reasoning_tokens=recorded.reasoning_tokens,
        )
        recorded.cost_usd = cost
        recorded.estimated_model_cost_usd = cost
        recorded.cost_method = method
        recorded.price_snapshot_hash = table.snapshot_hash
    except Exception:  # noqa: BLE001 - pricing must never abort a model call
        LOGGER.warning("Could not price usage for %s", recorded.model, exc_info=True)


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
    execution_id: str = "",
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
                execution_id=execution_id,
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
