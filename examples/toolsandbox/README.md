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

## Run with Apple's official ToolSandbox

This repository includes `official_backend.py`, which resolves a scenario from
the official package and executes its real `ExecutionContext`, tools, and
`Evaluation`. The backend must run in the same Python process as SkillRL, so
install both projects in the environment used by the trainer. Use Python 3.10
or 3.11 for the combined environment; the official package pins NumPy,
Polars, Transformers, and related dependencies and does not target Python 3.14.

```bash
# SkillRL environment
pip install -e .

# Official ToolSandbox package (Linux/macOS)
pip install git+https://github.com/apple-aiml-research/ToolSandbox.git
```

On Windows run the same `pip install` from the activated SkillRL environment.
If the pinned dependencies conflict, create a dedicated combined environment
for this experiment rather than installing ToolSandbox into the production
environment. Verify the scenario name against the installed revision:

```bash
python -c "from tool_sandbox.scenarios import named_scenarios; from tool_sandbox.common.tool_discovery import ToolBackend; print(sorted(named_scenarios(ToolBackend.DEFAULT))[:20])"
```

The included example is `scenarios/cellular_off.jsonl`. A server run can use:

```bash
python -m verl.trainer.main_ppo \
  data.train_files=/path/to/toolsandbox_data/train.parquet \
  data.val_files=/path/to/toolsandbox_data/val.parquet \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  actor_rollout_ref.model.path=/path/to/model \
  trainer.n_gpus_per_node=1 \
  env.env_name=toolsandbox \
  env.max_steps=30 \
  env.rollout.n=1 \
  env.toolsandbox.scenario_path=/path/to/scenarios/cellular_off.jsonl \
  env.toolsandbox.backend_factory=examples.toolsandbox.official_backend:make_backend
```

Before launching a trainer, verify the combined environment without a GPU:

```bash
python examples/toolsandbox/smoke_test.py
```

The smoke test performs the `cellular_off` tool call and checks the official
milestone and minefield evaluation. It should be run from the SkillRL root.

The trainer also expects parquet rows. Generate them from scenario files with:

```bash
python examples/toolsandbox/prepare_dataset.py \
  --train /path/to/train_scenarios.jsonl \
  --val /path/to/val_scenarios.jsonl \
  --output-dir /path/to/toolsandbox_data
```

Then include these required data overrides in the trainer command:

```bash
  data.train_files=/path/to/toolsandbox_data/train.parquet \
  data.val_files=/path/to/toolsandbox_data/val.parquet \
  data.train_batch_size=8 \
  data.val_batch_size=8
```

For a dataset, replace the JSONL records with your scenario names and set
`train_path`/`val_path` separately if needed. The initial prompt in the parquet
file is only a data-loader placeholder; the manager uses the ToolSandbox user
message and current execution result as the actual multi-turn observation.
