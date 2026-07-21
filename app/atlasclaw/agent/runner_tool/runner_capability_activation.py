# -*- coding: utf-8 -*-
# Copyright 2026  Qianyun, Inc., www.cloudchef.io, All rights reserved.

"""Request-scoped capability activation for planner-free routing."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
import unicodedata
from typing import Any

from pydantic_ai import RunContext

from app.atlasclaw.agent.tool_gate_models import ToolIntentAction, ToolIntentPlan
from app.atlasclaw.core.deps import SkillDeps
from app.atlasclaw.tools.catalog import STANDARD_SKILL_RUNTIME_TOOL_NAMES
from app.atlasclaw.tools.base import ToolResult


CAPABILITY_ACTIVATION_TOOL_NAME = "activate_capability"
CAPABILITY_INDEX_KEY = "_authorized_capability_index"
CAPABILITY_TOOLS_KEY = "_authorized_capability_tools"
DYNAMIC_TOOL_FILTER_MARKER = "_atlasclaw_dynamic_tool_filter"


def _unique(values: Any) -> list[str]:
    raw_values = values if isinstance(values, list) else [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        normalized = str(value or "").strip()
        if not normalized or normalized.lower() in seen:
            continue
        seen.add(normalized.lower())
        result.append(normalized)
    return result


def _routing_tokens(value: Any) -> set[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    latin_tokens = set(re.findall(r"[a-z0-9][a-z0-9_.:-]*", text))
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", text)
    cjk_tokens: set[str] = set()
    for run in cjk_runs:
        if len(run) == 1:
            cjk_tokens.add(run)
        else:
            cjk_tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return latin_tokens | cjk_tokens


def rank_capability_index_for_request(
    capability_index: list[dict[str, Any]],
    user_message: str,
    *,
    max_count: int,
) -> list[dict[str, Any]]:
    """Compress capability candidates using only declared metadata overlap."""
    request_tokens = _routing_tokens(user_message)
    if not request_tokens:
        return []
    prepared: list[tuple[int, dict[str, Any], set[str]]] = []
    document_frequency: Counter[str] = Counter()
    for index, entry in enumerate(capability_index):
        if not isinstance(entry, dict):
            continue
        metadata_text = " ".join(
            str(value or "")
            for value in (
                entry.get("capability_id"),
                entry.get("name"),
                entry.get("description"),
                entry.get("routing_terms"),
                entry.get("artifact_types"),
                entry.get("target_capability_classes"),
                entry.get("declared_tool_names"),
            )
        )
        metadata_tokens = _routing_tokens(metadata_text)
        prepared.append((index, dict(entry), metadata_tokens))
        document_frequency.update(metadata_tokens)

    entry_count = max(1, len(prepared))
    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, entry, metadata_tokens in prepared:
        overlap = request_tokens.intersection(metadata_tokens)
        partial_matches = {
            request_token
            for request_token in request_tokens
            if request_token not in overlap
            and len(request_token) >= 4
            and any(request_token in metadata_token for metadata_token in metadata_tokens)
        }
        score = sum(
            max(1, entry_count // max(1, document_frequency[token]))
            for token in overlap
        ) * 2 + len(partial_matches)
        if score:
            ranked.append((score, index, dict(entry)))
        elif str(entry.get("routing_visibility", "") or "").strip().lower() == "general":
            ranked.append((0, index, dict(entry)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [entry for _score, _index, entry in ranked[: max(1, int(max_count))]]


def capability_entry_to_intent_plan(entry: dict[str, Any]) -> ToolIntentPlan | None:
    """Convert one server-authorized capability entry into an execution plan."""
    capability_id = str(entry.get("capability_id", "") or "").strip()
    if not capability_id or ":" not in capability_id:
        return None
    prefix, raw_name = capability_id.split(":", 1)
    prefix = prefix.strip().lower()
    raw_name = raw_name.strip()
    if not raw_name or prefix not in {"tool", "skill", "provider_skill"}:
        return None

    target_provider_instances = _unique(entry.get("target_provider_instances"))
    target_provider_types = _unique(entry.get("target_provider_types"))
    target_provider_skill_names = _unique(entry.get("target_provider_skill_names"))
    target_skill_names = _unique(entry.get("target_skill_names"))
    target_capability_classes = _unique(entry.get("target_capability_classes"))
    target_tool_names = _unique(
        entry.get("target_tool_names") or entry.get("declared_tool_names")
    )

    if prefix == "provider_skill":
        if not (
            target_provider_instances
            and target_provider_types
            and target_provider_skill_names
        ):
            return None
        target_skill_names = []
    elif prefix == "skill":
        if target_provider_instances or target_provider_types:
            return None
        if not target_skill_names:
            target_skill_names = [raw_name]
    else:
        if target_provider_instances or target_provider_types:
            return None
        if not target_tool_names:
            target_tool_names = [raw_name]

    return ToolIntentPlan(
        action=ToolIntentAction.USE_TOOLS,
        target_provider_instances=target_provider_instances,
        target_provider_types=target_provider_types,
        target_provider_skill_names=target_provider_skill_names,
        target_skill_names=target_skill_names,
        target_capability_classes=target_capability_classes,
        target_tool_names=target_tool_names,
        reason=f"Main model activated authorized capability '{capability_id}'.",
    )


def prepare_runtime_tools(ctx: Any, tool_definitions: list[Any]) -> list[Any]:
    """Filter tools from request state before every model request."""
    deps = getattr(ctx, "deps", None)
    extra = getattr(deps, "extra", None)
    if not isinstance(extra, dict):
        return tool_definitions
    raw_allowed = extra.get("runtime_allowed_tool_names")
    if raw_allowed is None:
        return tool_definitions
    allowed = {
        str(name or "").strip()
        for name in (raw_allowed if isinstance(raw_allowed, list) else [raw_allowed])
        if str(name or "").strip()
    }
    return [tool for tool in tool_definitions if str(getattr(tool, "name", "")) in allowed]


def _load_skill_instructions(entry: dict[str, Any], *, max_file_bytes: int) -> str:
    locator = str(entry.get("locator", "") or "").strip()
    if not locator:
        return ""
    path = Path(locator)
    try:
        if not path.is_file() or path.stat().st_size > max_file_bytes:
            return ""
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return re.sub(r"^---\s.*?---\s*", "", text, count=1, flags=re.DOTALL).strip()


async def activate_capability_tool(
    ctx: "RunContext[SkillDeps]",
    capability_id: str,
) -> dict[str, Any]:
    """Activate one exact capability ID from the current authorized capability index."""
    deps = getattr(ctx, "deps", None)
    extra = getattr(deps, "extra", None)
    if not isinstance(extra, dict):
        return ToolResult.error("Capability activation context is unavailable.").to_dict()

    normalized_id = str(capability_id or "").strip()
    capability_index = extra.get(CAPABILITY_INDEX_KEY)
    if not normalized_id or not isinstance(capability_index, list):
        return ToolResult.error("Capability ID is not available in this request.").to_dict()
    entry = next(
        (
            item
            for item in capability_index
            if isinstance(item, dict)
            and str(item.get("capability_id", "") or "").strip() == normalized_id
        ),
        None,
    )
    if entry is None:
        return ToolResult.error("Capability ID is not authorized in this request.").to_dict()

    plan = capability_entry_to_intent_plan(entry)
    if plan is None:
        return ToolResult.error("Capability metadata is incomplete or non-executable.").to_dict()

    # Import lazily to avoid a registration-time dependency cycle.
    from app.atlasclaw.agent.runner_tool.runner_execution_prepare import (
        apply_provider_instance_selection_policy,
        persist_provider_instance_targets_from_intent_plan,
    )
    from app.atlasclaw.agent.runner_tool.runner_tool_projection import (
        project_minimal_toolset,
    )

    plan, _trace = apply_provider_instance_selection_policy(deps=deps, intent_plan=plan)
    await persist_provider_instance_targets_from_intent_plan(deps=deps, intent_plan=plan)

    all_tools = extra.get(CAPABILITY_TOOLS_KEY)
    if not isinstance(all_tools, list):
        all_tools = []
    projected_tools, projection_trace = project_minimal_toolset(
        allowed_tools=all_tools,
        intent_plan=plan,
    )
    allowed_tool_names = [
        str(tool.get("name", "") or "").strip()
        for tool in projected_tools
        if isinstance(tool, dict) and str(tool.get("name", "") or "").strip()
    ]

    instructions = ""
    if str(entry.get("kind", "") or "").strip().lower() in {"md_skill", "skill", "provider_skill"}:
        max_file_bytes = int(extra.get("md_skills_max_file_bytes", 262144) or 262144)
        instructions = _load_skill_instructions(entry, max_file_bytes=max_file_bytes)
        if instructions and not allowed_tool_names:
            available_names = {
                str(tool.get("name", "") or "").strip()
                for tool in all_tools
                if isinstance(tool, dict)
            }
            allowed_tool_names = [
                name
                for name in sorted(STANDARD_SKILL_RUNTIME_TOOL_NAMES)
                if name in available_names
            ]

    if not allowed_tool_names and not instructions:
        return ToolResult.error(
            "The selected capability has no executable tools in this request."
        ).to_dict()

    if instructions and not allowed_tool_names:
        plan = plan.model_copy(
            update={
                "action": ToolIntentAction.DIRECT_ANSWER,
                "reason": (
                    f"Main model activated instruction-only capability '{normalized_id}'."
                ),
            }
        )

    extra["tool_intent_plan"] = plan.model_dump(mode="python")
    extra["runtime_allowed_tool_names"] = allowed_tool_names
    extra["tools_snapshot"] = [
        dict(tool)
        for tool in all_tools
        if isinstance(tool, dict)
        and str(tool.get("name", "") or "").strip() in set(allowed_tool_names)
    ]
    extra["tool_projection_trace"] = dict(projection_trace)
    extra["activated_capability_id"] = normalized_id
    if instructions:
        extra["target_md_skill"] = {
            "qualified_name": str(entry.get("name", "") or "").strip(),
            "provider_skill_name": str(entry.get("provider_skill_name", "") or "").strip(),
            "file_path": str(entry.get("locator", "") or "").strip(),
            "instructions": instructions,
        }

    result_text = (
        f"Activated capability `{normalized_id}`. Continue this turn using only the newly "
        "available tools; do not treat activation itself as completion evidence."
    )
    if instructions:
        result_text += f"\n\nSelected skill instructions:\n{instructions}"
    return ToolResult.text(
        result_text,
        details={
            "capability_id": normalized_id,
            "allowed_tool_names": allowed_tool_names,
            "coordination_only": True,
        },
    ).to_dict()
