"""Create the minimal parquet index expected by SkillRL's trainer.

ToolSandbox scenarios live in JSON/JSONL and are selected by the environment.
SkillRL still requires a parquet row for every rollout item, so this utility
creates placeholder user prompts while preserving scenario ids in metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("scenarios", [payload])


def write_split(source: Path, output: Path, split: str) -> None:
    records = load_records(source)
    if not records:
        raise ValueError(f"No scenarios found in {source}")
    rows = []
    for index, record in enumerate(records):
        scenario_id = record.get("id") or record.get("scenario_name") or str(index)
        rows.append(
            {
                "data_source": "toolsandbox",
                "prompt": [{"role": "user", "content": "ToolSandbox task"}],
                "ability": "agent",
                "extra_info": {"split": split, "index": index, "scenario_id": scenario_id},
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"wrote {len(rows)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path, help="training scenario JSON/JSONL")
    parser.add_argument("--val", required=True, type=Path, help="validation scenario JSON/JSONL")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    write_split(args.train, args.output_dir / "train.parquet", "train")
    write_split(args.val, args.output_dir / "val.parquet", "val")


if __name__ == "__main__":
    main()
