# ToolSandbox baseline adapter

SkillRL can run ToolSandbox scenarios through `env.env_name=toolsandbox`.
The adapter keeps execution backend specific code outside SkillRL. Configure a
factory using `module:function`; it receives one scenario dictionary and must
return an object with:

```python
reset() -> observation | (observation, info)
step(action_dict) -> (observation, reward, done, info)
```

Example overrides for an existing trainer command:

```text
env.env_name=toolsandbox \
env.max_steps=20 \
env.history_length=4 \
env.toolsandbox.scenario_path=/data/toolsandbox/scenarios.jsonl \
env.toolsandbox.backend_factory=my_toolsandbox_backend:make_backend
```

JSON and JSONL files are accepted. A JSON object may contain a `scenarios`
array. Each scenario can use `user_goal`, `task`, or `user_utterance` for the
goal and may include `tools`/`available_tools` for prompt rendering.

The model response is projected from either
`<tool_call>{...}</tool_call>` or a JSON object. Tool calls should contain
`name` and `arguments`; a final response can use `{"type":"final", ...}`.
The backend should return `won`, `milestone_score`, and
`minefield_violation` in `info` when those metrics are available. SkillRL
reports `success_rate`, `milestone_score`, and
`minefield_violation_rate` per rollout.

The official ToolSandbox package is intentionally optional. Install and wrap
the version used by your dataset in `my_toolsandbox_backend:make_backend`, so
changes in its Python API do not leak into the training loop.
