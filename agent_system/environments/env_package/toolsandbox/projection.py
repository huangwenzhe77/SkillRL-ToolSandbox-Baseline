"""Projection from model text to ToolSandbox actions."""

import json
import re
from typing import Any, Dict, List, Tuple


_TOOL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.I | re.S)
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.I | re.S)


def _decode_object(text: str) -> Any:
    candidates = [text.strip()]
    tagged = _TOOL_TAG.search(text)
    if tagged:
        candidates.insert(0, tagged.group(1).strip())
    block = _JSON_BLOCK.search(text)
    if block:
        candidates.insert(0, block.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, ValueError):
            continue
    return None


def toolsandbox_projection(actions: List[str]) -> Tuple[List[Dict[str, Any]], List[int]]:
    """Parse model responses into backend actions.

    Accepted forms are ``<tool_call>{...}</tool_call>``, a JSON object, and a
    final response object with ``{"type": "final", "content": ...}``.  The
    returned validity flag is independent from the extracted action so failed
    generations remain inspectable in trajectory logs.
    """

    projected: List[Dict[str, Any]] = []
    valids: List[int] = []
    for raw in actions:
        obj = _decode_object(raw)
        if isinstance(obj, dict) and isinstance(obj.get("tool_call"), dict):
            obj = obj["tool_call"]
        if isinstance(obj, dict):
            if "name" in obj and "arguments" not in obj:
                obj["arguments"] = obj.pop("args", {})
            valid = int((obj.get("type") == "final") or (bool(obj.get("name")) and isinstance(obj.get("arguments", {}), dict)))
            projected.append(obj)
            valids.append(valid)
        else:
            projected.append({"type": "invalid", "raw": raw})
            valids.append(0)
    return projected, valids
