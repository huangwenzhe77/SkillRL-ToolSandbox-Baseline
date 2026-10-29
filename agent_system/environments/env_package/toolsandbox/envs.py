"""Vector wrapper for ToolSandbox scenarios.

The wrapper deliberately does not hard-code a particular ToolSandbox release.
Pass ``env.toolsandbox.backend_factory`` as ``module:function``.  The factory
receives one scenario and must return an object implementing ``reset()`` and
``step(action)`` (plus optional ``close()``).  A backend may also expose
``success_evaluator()`` or return milestone/minefield metadata in ``info``.
"""

import importlib
import json
import os
from copy import deepcopy
from typing import Any, Dict, List

import numpy as np


def _load_factory(spec: str):
    if not spec or ":" not in spec:
        raise ValueError("env.toolsandbox.backend_factory must be 'module:function'")
    module_name, function_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def load_scenarios(path: str) -> List[Dict[str, Any]]:
    """Load ToolSandbox scenarios from JSON, JSONL, or a directory of either."""
    if not path:
        raise ValueError("ToolSandbox requires env.toolsandbox.scenario_path")
    paths = [path]
    if os.path.isdir(path):
        paths = sorted(
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.lower().endswith((".json", ".jsonl"))
        )
    scenarios: List[Dict[str, Any]] = []
    for filename in paths:
        with open(filename, "r", encoding="utf-8") as handle:
            if filename.lower().endswith(".jsonl"):
                records = [json.loads(line) for line in handle if line.strip()]
            else:
                payload = json.load(handle)
                records = payload if isinstance(payload, list) else payload.get("scenarios", [payload])
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Scenario in {filename} is not an object")
            scenarios.append(record)
    if not scenarios:
        raise ValueError(f"No scenarios found at {path}")
    return scenarios


class ToolSandboxEnvs:
    """A synchronous vector environment backed by one ToolSandbox per slot."""

    def __init__(self, scenarios, env_num, group_n, backend_factory, max_steps, seed=0):
        self.scenarios = scenarios
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.max_steps = max_steps
        self.rng = np.random.RandomState(seed)
        self.backend_factory = backend_factory
        self.backends = [None] * self.batch_size
        self.current_steps = np.zeros(self.batch_size, dtype=np.int32)
        self.current_scenarios = [None] * self.batch_size

    @staticmethod
    def _reset_backend(backend):
        result = backend.reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}

    def reset(self):
        replace = self.env_num > len(self.scenarios)
        selected = self.rng.choice(len(self.scenarios), self.env_num, replace=replace)
        selected = np.repeat(selected, self.group_n).tolist()
        observations, infos = [], []
        for slot, index in enumerate(selected):
            scenario = deepcopy(self.scenarios[index])
            backend = self.backend_factory(scenario)
            self.backends[slot] = backend
            self.current_scenarios[slot] = scenario
            self.current_steps[slot] = 0
            observation, info = self._reset_backend(backend)
            info = dict(info or {})
            info.setdefault("scenario_id", scenario.get("id", str(index)))
            observations.append(observation)
            infos.append(info)
        return observations, infos

    def step(self, actions):
        if len(actions) != self.batch_size:
            raise ValueError(f"Expected {self.batch_size} actions, got {len(actions)}")
        results = []
        for slot, action in enumerate(actions):
            backend = self.backends[slot]
            if backend is None:
                raise RuntimeError("ToolSandbox environment must be reset before step")
            self.current_steps[slot] += 1
            result = backend.step(action)
            if len(result) == 4:
                observation, reward, done, info = result
            elif len(result) == 3:
                observation, reward, done = result
                info = {}
            else:
                raise ValueError("ToolSandbox backend step must return 3 or 4 values")
            done = bool(done or self.current_steps[slot] >= self.max_steps)
            info = dict(info or {})
            info.setdefault("won", bool(info.get("success", False) and done))
            info.setdefault("tool_calling", int(action.get("type") == "tool_call" or "name" in action))
            results.append((observation, float(reward), done, info))
        return tuple(map(list, zip(*results)))

    def close(self):
        for backend in self.backends:
            if backend is not None and hasattr(backend, "close"):
                backend.close()


def build_toolsandbox_envs(scenario_path, backend_factory, seed=0, env_num=1, group_n=1, is_train=True, max_steps=20):
    scenarios = load_scenarios(scenario_path)
    factory = _load_factory(backend_factory) if isinstance(backend_factory, str) else backend_factory
    return ToolSandboxEnvs(scenarios, env_num, group_n, factory, max_steps=max_steps, seed=seed)
