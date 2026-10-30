"""Run a tiny official ToolSandbox integration smoke test.

This intentionally bypasses SkillRL/Ray and GPU code.  It verifies that the
official package, the scenario resolver, tool execution, and evaluation all
work together in the same Python environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.toolsandbox.official_backend import make_backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario-file",
        default=str(Path(__file__).with_name("scenarios") / "cellular_off.jsonl"),
    )
    args = parser.parse_args()

    record = json.loads(Path(args.scenario_file).read_text(encoding="utf-8").splitlines()[0])
    try:
        backend = make_backend(record)
    except ModuleNotFoundError as exc:
        if exc.name == "tool_sandbox":
            raise SystemExit(
                "Official ToolSandbox is not installed. Run "
                "pip install git+https://github.com/apple-aiml-research/ToolSandbox.git "
                "in the same environment as SkillRL."
            ) from exc
        raise
    observation, info = backend.reset()
    print("initial observation:", observation)
    print("available tools:", info.get("available_tools"))

    if "set_cellular_service_status" not in info.get("available_tools", []):
        raise RuntimeError("The cellular_off scenario did not expose the expected tool")

    observation, reward, done, info = backend.step(
        {
            "type": "tool_call",
            "name": "set_cellular_service_status",
            "arguments": {"on": False},
        }
    )
    print("tool observation:", observation)
    print("tool info:", info)
    if info.get("tool_call_exception"):
        raise RuntimeError(f"Official tool call failed: {info['tool_call_exception']}")

    _, _, done, evaluation = backend.step(
        {"type": "final", "content": "Cellular service is turned off"}
    )
    print("evaluation:", evaluation)
    if not done or not evaluation.get("won"):
        raise RuntimeError(f"Smoke test did not win the scenario: {evaluation}")
    print("ToolSandbox official backend smoke test passed")


if __name__ == "__main__":
    main()
