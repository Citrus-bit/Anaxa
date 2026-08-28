from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from medrix_flow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from medrix_flow.runtime.runs import RunRecord
from medrix_flow.runtime.serialization import extract_text

MAX_SUMMARY_CHARS = 320
MAX_DETAIL_CHARS = 4000
SENSITIVE_KEY_RE = re.compile(
    r"(^token$|api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|secret|password|authorization|credential)",
    re.IGNORECASE,
)
HIDDEN_REASONING_KEY_RE = re.compile(
    r"(^reasoning_content$|^reasoning_details$|^thinking$|^thoughts?$|chain[_-]?of[_-]?thought)",
    re.IGNORECASE,
)
THINK_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
DECISION_TOOL_NAME = "record_decision"

NodeKind = Literal["user", "agent", "decision", "subagent", "tool", "artifact", "checkpoint", "final", "error", "event"]
NodeStatus = Literal["pending", "running", "success", "error", "interrupted"]


class WorkflowRun(BaseModel):
    run_id: str
    thread_id: str
    assistant_id: str | None = None
    status: str
    error: str | None = None
    created_at: str
    updated_at: str
    last_event_at: str | None = None


class WorkflowNode(BaseModel):
    id: str
    kind: NodeKind
    label: str
    status: NodeStatus = "success"
    summary: str = ""
    caller: str | None = None
    tool_name: str | None = None
    artifact_path: str | None = None
    seq: int | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class WorkflowEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str | None = None


class WorkflowEvent(BaseModel):
    seq: int
    run_id: str
    thread_id: str
    event_type: str
    caller: str
    summary: str
    content: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class WorkflowArtifact(BaseModel):
    filepath: str
    filename: str
    size: int | None = None
    modified_at: str | None = None


class WorkflowUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class WorkflowSnapshot(BaseModel):
    run: WorkflowRun
    nodes: list[WorkflowNode]
    edges: list[WorkflowEdge]
    events: list[WorkflowEvent]
    artifacts: list[WorkflowArtifact]
    usage: WorkflowUsage
    has_more: bool = False


def _truncate(value: str, limit: int = MAX_SUMMARY_CHARS) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if HIDDEN_REASONING_KEY_RE.search(str(key)):
                redacted[str(key)] = "[hidden]"
            elif SENSITIVE_KEY_RE.search(str(key)):
                redacted[str(key)] = "[redacted]"
            else:
                redacted[str(key)] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value[:100]]
    if isinstance(value, str):
        value = THINK_TAG_RE.sub("[hidden]", value)
        return value if len(value) <= MAX_DETAIL_CHARS else value[:MAX_DETAIL_CHARS] + "..."
    return _safe_json(value)


def _artifact_node_id(path: str) -> str:
    digest = sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"artifact-{digest}"


def normalize_stream_event(event_type: str, data: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Convert an in-flight stream chunk into a storable run event."""
    safe_data = _redact(data)
    message_type = safe_data.get("type") if isinstance(safe_data, dict) else None

    if event_type == "messages-tuple" and message_type == "tool":
        caller = str(safe_data.get("name") or "tool")
        return "tool_message", caller, safe_data

    if event_type == "messages-tuple" and message_type == "ai":
        tool_calls = safe_data.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return "ai_tool_calls", "assistant", safe_data
        return "ai_message", "assistant", safe_data

    if event_type == "values":
        return "state_snapshot", "checkpoint", safe_data

    if event_type == "error":
        return "error", "gateway", safe_data

    if event_type == "end":
        return "run_end", "gateway", safe_data

    return event_type, "gateway", safe_data


def _content_summary(content: dict[str, Any]) -> str:
    if "summary" in content and isinstance(content["summary"], str):
        return _truncate(content["summary"])
    if "message" in content and isinstance(content["message"], str):
        return _truncate(content["message"])
    if "content" in content:
        return _truncate(extract_text(content["content"]))
    if "title" in content and content["title"]:
        return _truncate(str(content["title"]))
    if "usage" in content:
        return "Token usage updated."
    if "artifacts" in content and isinstance(content["artifacts"], list):
        return f"{len(content['artifacts'])} artifact(s) available."
    return _truncate(json.dumps(_redact(content), ensure_ascii=False, default=str))


def _node_kind(event_type: str, caller: str, content: dict[str, Any]) -> NodeKind:
    if event_type == "human_message" or content.get("type") == "human":
        return "user"
    if event_type in {"tool_message", "ai_tool_calls"} or content.get("type") == "tool":
        tool_name = str(content.get("name") or caller or "")
        if tool_name == "task" or tool_name.startswith("task"):
            return "subagent"
        return "tool"
    if event_type == "error":
        return "error"
    if event_type == "state_snapshot":
        return "checkpoint"
    if event_type == "run_end":
        return "final"
    if event_type == "ai_message" or content.get("type") == "ai":
        return "agent"
    return "event"


def _node_label(kind: NodeKind, event_type: str, caller: str, content: dict[str, Any]) -> str:
    if kind == "user":
        return "User request"
    if kind == "subagent":
        return str(content.get("name") or caller or "Sub-agent")
    if kind == "tool":
        tool_calls = content.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            names = [str(item.get("name")) for item in tool_calls if isinstance(item, dict) and item.get("name")]
            return ", ".join(names[:3]) if names else "Tool call"
        return str(content.get("name") or caller or "Tool")
    if kind == "checkpoint":
        return "State checkpoint"
    if kind == "final":
        return "Run finished"
    if kind == "error":
        return "Run error"
    if kind == "agent":
        return str(caller) if caller and caller != "assistant" else "Assistant"
    return event_type.replace("_", " ").title()


def _node_status(kind: NodeKind, run_status: str) -> NodeStatus:
    if kind == "error":
        return "error"
    if kind == "final":
        if run_status == "interrupted":
            return "interrupted"
        if run_status == "error":
            return "error"
    return "success"


def _decision_status(status: str | None, run_status: str) -> NodeStatus:
    normalized = str(status or "").strip().lower()
    if normalized in {"pending", "planned", "queued", "awaiting"}:
        return "pending"
    if normalized in {"running", "executing", "in_progress", "active"}:
        return "running"
    if normalized in {"error", "failed", "blocked"}:
        return "error"
    if normalized in {"interrupted", "cancelled", "canceled"}:
        return "interrupted"
    if run_status == "error" and normalized in {"", "unknown"}:
        return "error"
    return "success"


def _tool_name_from_content(caller: str, content: dict[str, Any]) -> str:
    name = content.get("name")
    if isinstance(name, str) and name:
        return name
    return caller or ""


def _is_decision_tool_name(name: str | None) -> bool:
    return str(name or "").strip() == DECISION_TOOL_NAME


def _try_parse_json(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return None


def _find_decision_payload(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        parsed = _try_parse_json(value)
        return _find_decision_payload(parsed) if parsed is not None else None

    if isinstance(value, list):
        for item in value:
            found = _find_decision_payload(item)
            if found is not None:
                return found
        return None

    if not isinstance(value, dict):
        return None

    if isinstance(value.get("title"), str) and (isinstance(value.get("rationale"), str) or isinstance(value.get("next_step"), str) or isinstance(value.get("decision_type"), str)):
        return value

    for key in ("decision", "args", "input", "output", "content", "payload", "result"):
        if key in value:
            found = _find_decision_payload(value.get(key))
            if found is not None:
                return found

    for item in value.values():
        found = _find_decision_payload(item)
        if found is not None:
            return found
    return None


def _normalize_decision_payload(payload: dict[str, Any], *, tool_call_id: str | None = None) -> dict[str, Any]:
    alternatives = payload.get("alternatives")
    if not isinstance(alternatives, list):
        alternatives = []
    return {
        "title": _truncate(str(payload.get("title") or "Decision"), 160),
        "decision_type": _truncate(str(payload.get("decision_type") or "decision"), 80),
        "rationale": _truncate(str(payload.get("rationale") or ""), 1200),
        "next_step": _truncate(str(payload.get("next_step") or ""), 1200),
        "status": _truncate(str(payload.get("status") or "success"), 80),
        "alternatives": [_truncate(str(item), 240) for item in alternatives[:8] if str(item).strip()],
        "related_tool": _truncate(str(payload.get("related_tool") or ""), 160) or None,
        "outcome": _truncate(str(payload.get("outcome") or ""), 1200) or None,
        "tool_call_id": tool_call_id,
    }


def _decision_payloads_from_event(event: WorkflowEvent, content: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    tool_calls = content.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict) or not _is_decision_tool_name(call.get("name")):
                continue
            args = call.get("args")
            if not isinstance(args, dict):
                continue
            payloads.append(_normalize_decision_payload(args, tool_call_id=str(call.get("id") or "") or None))

    if event.event_type == "tool_message" and _is_decision_tool_name(_tool_name_from_content(event.caller, content)):
        found = _find_decision_payload(content)
        if found is not None:
            payloads.append(_normalize_decision_payload(found, tool_call_id=content.get("tool_call_id") if isinstance(content.get("tool_call_id"), str) else None))

    return payloads


def _content_without_decision_tool_calls(content: dict[str, Any]) -> dict[str, Any]:
    tool_calls = content.get("tool_calls")
    if not isinstance(tool_calls, list):
        return content
    kept = [call for call in tool_calls if not (isinstance(call, dict) and _is_decision_tool_name(call.get("name")))]
    if len(kept) == len(tool_calls):
        return content
    updated = dict(content)
    updated["tool_calls"] = kept
    return updated


def _extract_artifacts_from_event(content: dict[str, Any]) -> list[str]:
    found: list[str] = []
    artifacts = content.get("artifacts")
    if isinstance(artifacts, list):
        found.extend(str(item) for item in artifacts if isinstance(item, str))
    tool_calls = content.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            args = call.get("args")
            if not isinstance(args, dict):
                continue
            for key in ("paths", "filepaths", "artifacts"):
                value = args.get(key)
                if isinstance(value, list):
                    found.extend(str(item) for item in value if isinstance(item, str))
                elif isinstance(value, str):
                    found.append(value)
    return [path for path in found if path.startswith(VIRTUAL_PATH_PREFIX)]


def _usage_from_event(content: dict[str, Any]) -> WorkflowUsage:
    usage = content.get("usage_metadata") or content.get("usage") or {}
    if not isinstance(usage, dict):
        return WorkflowUsage()
    return WorkflowUsage(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
    )


def _event_node(
    *,
    event: WorkflowEvent,
    content: dict[str, Any],
    kind: NodeKind,
    run_status: str,
) -> WorkflowNode:
    return WorkflowNode(
        id=f"event-{event.seq}",
        kind=kind,
        label=_node_label(kind, event.event_type, event.caller, content),
        status=_node_status(kind, run_status),
        summary=_content_summary(content),
        caller=event.caller,
        tool_name=content.get("name") if isinstance(content.get("name"), str) else None,
        seq=event.seq,
        created_at=event.created_at,
        metadata={"event_type": event.event_type},
    )


def _artifact_node(*, artifact_path: str, event: WorkflowEvent) -> WorkflowNode:
    return WorkflowNode(
        id=_artifact_node_id(artifact_path),
        kind="artifact",
        label=Path(artifact_path).name,
        status="success",
        summary=artifact_path,
        artifact_path=artifact_path,
        seq=event.seq,
        created_at=event.created_at,
    )


def _append_artifact_nodes(
    *,
    parent_id: str | None,
    event: WorkflowEvent,
    content: dict[str, Any],
    nodes: list[WorkflowNode],
    edges: list[WorkflowEdge],
    seen_artifact_nodes: set[str],
) -> None:
    if parent_id is None:
        return
    for artifact_path in _extract_artifacts_from_event(content):
        if artifact_path in seen_artifact_nodes:
            continue
        seen_artifact_nodes.add(artifact_path)
        artifact_node = _artifact_node(artifact_path=artifact_path, event=event)
        nodes.append(artifact_node)
        edges.append(
            WorkflowEdge(
                id=f"edge-{parent_id}-{artifact_node.id}",
                source=parent_id,
                target=artifact_node.id,
                label="created",
            )
        )


def _build_event_tree_nodes(
    *,
    record: RunRecord,
    event_items: list[tuple[WorkflowEvent, dict[str, Any]]],
) -> tuple[list[WorkflowNode], list[WorkflowEdge]]:
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []
    seen_artifact_nodes: set[str] = set()
    previous_node_id: str | None = None

    for event, content in event_items:
        kind = _node_kind(event.event_type, event.caller, content)
        node = _event_node(event=event, content=content, kind=kind, run_status=record.status.value)
        nodes.append(node)
        if previous_node_id is not None:
            edges.append(WorkflowEdge(id=f"edge-{previous_node_id}-{node.id}", source=previous_node_id, target=node.id))
        previous_node_id = node.id

        _append_artifact_nodes(
            parent_id=node.id,
            event=event,
            content=content,
            nodes=nodes,
            edges=edges,
            seen_artifact_nodes=seen_artifact_nodes,
        )

    return nodes, edges


def _decision_node(
    *,
    event: WorkflowEvent,
    payload: dict[str, Any],
    index: int,
    run_status: str,
) -> WorkflowNode:
    node_id = f"decision-{event.seq}-{index}"
    rationale = str(payload.get("rationale") or "")
    next_step = str(payload.get("next_step") or "")
    outcome = str(payload.get("outcome") or "")
    summary = rationale or next_step or outcome
    return WorkflowNode(
        id=node_id,
        kind="decision",
        label=str(payload.get("title") or "Decision"),
        status=_decision_status(payload.get("status") if isinstance(payload.get("status"), str) else None, run_status),
        summary=summary,
        caller=event.caller,
        tool_name=DECISION_TOOL_NAME,
        seq=event.seq,
        created_at=event.created_at,
        metadata={
            "event_type": event.event_type,
            "decision_type": payload.get("decision_type") or "decision",
            "rationale": rationale,
            "next_step": next_step,
            "decision_status": payload.get("status") or "success",
            "alternatives": payload.get("alternatives") or [],
            "related_tool": payload.get("related_tool"),
            "outcome": payload.get("outcome"),
            "tool_call_id": payload.get("tool_call_id"),
        },
    )


def _build_decision_tree_nodes(
    *,
    record: RunRecord,
    event_items: list[tuple[WorkflowEvent, dict[str, Any]]],
) -> tuple[list[WorkflowNode], list[WorkflowEdge]]:
    nodes: list[WorkflowNode] = []
    edges: list[WorkflowEdge] = []
    seen_artifact_nodes: set[str] = set()
    decision_tool_call_ids: set[str] = set()
    latest_decision_id: str | None = None
    previous_decision_id: str | None = None

    for event, original_content in event_items:
        decision_payloads = _decision_payloads_from_event(event, original_content)
        for index, payload in enumerate(decision_payloads):
            tool_call_id = payload.get("tool_call_id")
            if isinstance(tool_call_id, str) and tool_call_id:
                if event.event_type == "tool_message" and tool_call_id in decision_tool_call_ids:
                    continue
                decision_tool_call_ids.add(tool_call_id)

            node = _decision_node(
                event=event,
                payload=payload,
                index=index,
                run_status=record.status.value,
            )
            nodes.append(node)
            if previous_decision_id is not None:
                edges.append(
                    WorkflowEdge(
                        id=f"edge-{previous_decision_id}-{node.id}",
                        source=previous_decision_id,
                        target=node.id,
                        label="next",
                    )
                )
            previous_decision_id = node.id
            latest_decision_id = node.id

        if event.event_type == "tool_message" and _is_decision_tool_name(_tool_name_from_content(event.caller, original_content)):
            continue

        content = _content_without_decision_tool_calls(original_content)
        if event.event_type == "ai_tool_calls" and not content.get("tool_calls"):
            continue

        kind = _node_kind(event.event_type, event.caller, content)
        if kind in {"user", "agent", "checkpoint", "event"}:
            _append_artifact_nodes(
                parent_id=latest_decision_id,
                event=event,
                content=content,
                nodes=nodes,
                edges=edges,
                seen_artifact_nodes=seen_artifact_nodes,
            )
            continue

        node = _event_node(event=event, content=content, kind=kind, run_status=record.status.value)
        nodes.append(node)
        if latest_decision_id is not None:
            edges.append(
                WorkflowEdge(
                    id=f"edge-{latest_decision_id}-{node.id}",
                    source=latest_decision_id,
                    target=node.id,
                )
            )

        _append_artifact_nodes(
            parent_id=node.id if latest_decision_id is not None else None,
            event=event,
            content=content,
            nodes=nodes,
            edges=edges,
            seen_artifact_nodes=seen_artifact_nodes,
        )

    return nodes, edges


def _list_artifacts(thread_id: str) -> list[WorkflowArtifact]:
    try:
        outputs_dir = get_paths().sandbox_outputs_dir(thread_id)
    except ValueError:
        return []
    if not outputs_dir.exists():
        return []

    items: list[WorkflowArtifact] = []
    for path in outputs_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(outputs_dir)
        if any(part.startswith(".") for part in relative.parts):
            continue
        stat = path.stat()
        items.append(
            WorkflowArtifact(
                filepath=f"{VIRTUAL_PATH_PREFIX}/outputs/{relative.as_posix()}",
                filename=path.name,
                size=stat.st_size,
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
            )
        )
    return sorted(items, key=lambda item: item.modified_at or "", reverse=True)


def build_workflow_snapshot(
    *,
    record: RunRecord,
    event_rows: list[dict[str, Any]],
    has_more: bool,
) -> WorkflowSnapshot:
    events: list[WorkflowEvent] = []
    usage = WorkflowUsage()
    event_items: list[tuple[WorkflowEvent, dict[str, Any]]] = []

    for row in event_rows:
        content = _redact(row.get("content") or {})
        summary = _content_summary(content)
        event = WorkflowEvent(
            seq=int(row["seq"]),
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            event_type=row["event_type"],
            caller=row["caller"],
            summary=summary,
            content=content,
            created_at=row["created_at"],
        )
        events.append(event)
        event_items.append((event, content))

        event_usage = _usage_from_event(content)
        usage.input_tokens += event_usage.input_tokens
        usage.output_tokens += event_usage.output_tokens
        usage.total_tokens += event_usage.total_tokens

    has_decisions = any(_decision_payloads_from_event(event, content) for event, content in event_items)
    if has_decisions:
        nodes, edges = _build_decision_tree_nodes(record=record, event_items=event_items)
    else:
        nodes, edges = _build_event_tree_nodes(record=record, event_items=event_items)

    artifacts = _list_artifacts(record.thread_id)

    last_event_at = events[-1].created_at if events else None
    return WorkflowSnapshot(
        run=WorkflowRun(
            run_id=record.run_id,
            thread_id=record.thread_id,
            assistant_id=record.assistant_id,
            status=record.status.value,
            error=record.error,
            created_at=record.created_at,
            updated_at=record.updated_at,
            last_event_at=last_event_at,
        ),
        nodes=nodes[-200:],
        edges=[edge for edge in edges if edge.source in {node.id for node in nodes[-200:]} and edge.target in {node.id for node in nodes[-200:]}],
        events=events[-200:],
        artifacts=artifacts,
        usage=usage,
        has_more=has_more,
    )
