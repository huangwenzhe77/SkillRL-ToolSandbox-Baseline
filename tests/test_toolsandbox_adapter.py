import json

from agent_system.environments.env_package.toolsandbox.envs import ToolSandboxEnvs, load_scenarios
from agent_system.environments.env_package.toolsandbox.projection import toolsandbox_projection


def test_projection_accepts_tool_and_final_actions():
    actions, valids = toolsandbox_projection([
        '<tool_call>{"name":"search_contacts","arguments":{"query":"Ada"}}</tool_call>',
        '{"type":"final","content":"done"}',
        'not json',
    ])
    assert actions[0]["name"] == "search_contacts"
    assert actions[1]["type"] == "final"
    assert valids == [1, 1, 0]


def test_jsonl_loader_and_vector_backend(tmp_path):
    path = tmp_path / "scenarios.jsonl"
    path.write_text(json.dumps({"id": "one", "user_goal": "finish"}) + "\n", encoding="utf-8")
    assert load_scenarios(str(path))[0]["id"] == "one"

    class Backend:
        def reset(self):
            return "ready", {}

        def step(self, action):
            return "done", 1.0, True, {"won": action["type"] == "final"}

    env = ToolSandboxEnvs(load_scenarios(str(path)), 1, 1, lambda scenario: Backend(), max_steps=2)
    assert env.reset()[0] == ["ready"]
    assert env.step([{"type": "final"}])[1:] == ([1.0], [True], [{"won": True, "tool_calling": 0}])
