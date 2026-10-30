"""Official Apple ToolSandbox backend for the SkillRL adapter.

The imports from ``tool_sandbox`` are deliberately lazy.  This keeps the normal
SkillRL import path usable when the optional official package is not installed.
"""

from __future__ import annotations

import copy
import importlib
from collections import OrderedDict
from typing import Any, Mapping


def _load_object(spec: str):
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"Expected a module:function spec, got {spec!r}")
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def _call_factory(factory, record: Mapping[str, Any]):
    """Call a scenario factory while supporting common local wrappers."""
    name = record.get("scenario_name") or record.get("name") or record.get("id")
    attempts = []
    if name is not None:
        attempts.append(((), {"scenario_name": name}))
        attempts.append(((name,), {}))
    attempts.extend([((record,), {}), ((), {})])
    last_error = None
    for args, kwargs in attempts:
        try:
            return factory(*args, **kwargs)
        except (TypeError, KeyError) as exc:
            last_error = exc
    raise TypeError(f"Could not call scenario factory {factory!r}: {last_error}")


def _resolve_scenario(record: Mapping[str, Any]):
    factory = _load_object(
        record.get("scenario_factory", "tool_sandbox.scenarios:named_scenarios")
    )
    try:
        from tool_sandbox.common.tool_discovery import ToolBackend
        preferred = record.get("preferred_tool_backend", "DEFAULT")
        if isinstance(preferred, str):
            preferred = ToolBackend[preferred.upper()]
    except Exception:
        preferred = None

    if getattr(factory, "__name__", "") == "named_scenarios" and preferred is not None:
        scenarios = factory(preferred_tool_backend=preferred)
    else:
        try:
            scenarios = _call_factory(factory, record)
        except TypeError:
            if preferred is None:
                raise
            scenarios = factory(preferred_tool_backend=preferred)
    if isinstance(scenarios, Mapping):
        name = record.get("scenario_name") or record.get("name") or record.get("id")
        if name is None:
            raise ValueError("ToolSandbox scenario record needs scenario_name, name, or id")
        if name not in scenarios:
            raise KeyError(f"Scenario {name!r} not found; available examples include {list(scenarios)[:5]}")
        return scenarios[name]
    return scenarios


def _plain(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, OrderedDict):
        return {str(k): _plain(v) for k, v in value.items()}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


class OfficialToolSandboxBackend:
    def __init__(self, scenario_record: Mapping[str, Any]):
        self.record = dict(scenario_record)
        if not any(self.record.get(key) for key in ("scenario_name", "name", "id")):
            raise ValueError("ToolSandbox scenario record needs scenario_name, name, or id")
        self.scenario = _resolve_scenario(self.record)
        self.context = None
        self._done = False
        self._step = 0
        self._last_evaluation = None

    def _imports(self):
        from tool_sandbox.common.execution_context import (
            DatabaseNamespace,
            RoleType,
            new_context,
        )
        from tool_sandbox.common.message_conversion import Message
        from tool_sandbox.roles.base_role import BaseRole
        from tool_sandbox.roles.execution_environment import ExecutionEnvironment
        return DatabaseNamespace, RoleType, new_context, Message, BaseRole, ExecutionEnvironment

    def _db_rows(self):
        DatabaseNamespace, _, _, _, _, _ = self._imports()
        return self.context.get_database(
            DatabaseNamespace.SANDBOX,
            get_all_history_snapshots=True,
            drop_sandbox_message_index=False,
        ).to_dicts()

    def _latest_content(self, sender, recipient):
        for row in reversed(self._db_rows()):
            if row.get("sender") == sender and row.get("recipient") == recipient:
                return row.get("content", "")
        return ""

    def _available_tools(self):
        return self.context.get_available_tools(scrambling_allowed=True)

    def _initialize_context(self):
        _, RoleType, new_context, _, _, ExecutionEnvironment = self._imports()
        with new_context(self.context):
            rows = self._db_rows()
            max_index = self.context.max_sandbox_message_index
            for index in range(max_index + 1):
                row = next((r for r in rows if r.get("sandbox_message_index") == index), None)
                if row and row.get("sender") == RoleType.SYSTEM and row.get("recipient") == RoleType.EXECUTION_ENVIRONMENT:
                    ExecutionEnvironment().respond(ending_index=index)

    def reset(self):
        _, RoleType, new_context, _, _, _ = self._imports()
        self.context = copy.deepcopy(self.scenario.starting_context)
        self._done = False
        self._step = 0
        self._last_evaluation = None
        self._initialize_context()
        with new_context(self.context):
            observation = self._latest_content(RoleType.USER, RoleType.AGENT)
            tools = sorted(self._available_tools())
            try:
                from tool_sandbox.common.tool_conversion import convert_to_openai_tools
                tool_schemas = convert_to_openai_tools(self._available_tools())
            except Exception:
                tool_schemas = []
            info = {
                "scenario_id": self.record.get("id") or self.record.get("scenario_name"),
                # The official Scenario stores the initial user message in the
                # sandbox database rather than a separate goal field.
                "user_goal": observation,
                "available_tools": tools,
                "tool_schemas": _plain(tool_schemas),
                "tool_allow_list": _plain(self.context.tool_allow_list),
            }
        return observation, info

    def _evaluate(self):
        if self._last_evaluation is not None:
            return self._last_evaluation
        result = self.scenario.evaluation.evaluate(
            execution_context=self.context,
            max_turn_count=int(self.record.get("max_messages", self.scenario.max_messages)),
        )
        self._last_evaluation = result
        return result

    def _evaluation_info(self):
        result = self._evaluate()
        similarity = float(result.similarity)
        minefield = float(result.minefield_similarity)
        return {
            "won": bool(similarity > 0.5 and minefield == 0),
            "success": bool(similarity > 0.5 and minefield == 0),
            "similarity": similarity,
            "milestone_score": float(result.milestone_similarity),
            "milestone_similarity": float(result.milestone_similarity),
            "minefield_similarity": minefield,
            "minefield_violation": bool(minefield > 0),
            "turn_count": int(result.turn_count),
            "milestone_mapping": _plain(result.milestone_mapping),
            "minefield_mapping": _plain(result.minefield_mapping),
        }

    def step(self, action):
        DatabaseNamespace, RoleType, new_context, Message, BaseRole, ExecutionEnvironment = self._imports()
        if self._done:
            return "", 0.0, True, self._evaluation_info()
        action = action if isinstance(action, Mapping) else {"type": "invalid", "raw": str(action)}
        self._step += 1
        with new_context(self.context):
            if action.get("type") == "final":
                BaseRole.add_messages([Message(
                    sender=RoleType.AGENT,
                    recipient=RoleType.USER,
                    content=str(action.get("content", "")),
                )])
                self._done = True
                return str(action.get("content", "")), self._evaluation_info()["similarity"], True, self._evaluation_info()

            if action.get("type") != "tool_call" and not action.get("name"):
                return str(action.get("raw", "Invalid action")), -0.1, False, {"invalid_action": True}
            name = str(action.get("name"))
            arguments = action.get("arguments", {})
            available = self._available_tools()
            if name not in available:
                return f"Unknown or unavailable tool: {name}", -0.1, False, {"invalid_action": True, "available_tools": sorted(available)}
            execution_name = self.context.get_execution_facing_tool_name(name)
            code = (
                f"call_{self._step}_parameters = {repr(dict(arguments))}\n"
                f"call_{self._step}_response = {execution_name}(**call_{self._step}_parameters)\n"
                f"print(repr(call_{self._step}_response))"
            )
            BaseRole.add_messages([Message(
                sender=RoleType.AGENT,
                recipient=RoleType.EXECUTION_ENVIRONMENT,
                content=code,
                openai_tool_call_id=f"call_{self._step}",
                openai_function_name=name,
            )])
            ExecutionEnvironment().respond()
            rows = self._db_rows()
            response = next((r.get("content", "") for r in reversed(rows)
                             if r.get("sender") == RoleType.EXECUTION_ENVIRONMENT
                             and r.get("recipient") == RoleType.AGENT), "")
            latest = next((r for r in reversed(rows)
                           if r.get("sender") == RoleType.EXECUTION_ENVIRONMENT
                           and r.get("recipient") == RoleType.AGENT), {})
            exception = latest.get("tool_call_exception")
            info = {"tool_call_exception": exception} if exception else {}
            return response, -0.1 if exception else 0.0, False, info

    def finalize(self):
        if self._done:
            return self._evaluation_info()
        self._done = True
        return self._evaluation_info()

    def close(self):
        self.context = None


def make_backend(scenario_record):
    return OfficialToolSandboxBackend(scenario_record)

