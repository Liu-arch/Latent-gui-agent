from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic fixed-size LaRA GUI train/test subsets.")
    parser.add_argument("--train-input", required=True)
    parser.add_argument("--test-input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=100)
    parser.add_argument("--test-samples", type=int, default=100)
    parser.add_argument("--prefix", default="lara_gui_clean_s100")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_prefix(path: Path, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"Missing sample_id in {path} at line {line_number}.")
            rows.append(row)
            if len(rows) >= count:
                break
    if len(rows) != count:
        raise RuntimeError(f"Requested {count} rows from {path}, found only {len(rows)}.")
    return rows


def trajectory_key(row: dict[str, Any]) -> str:
    for key in ("task_id", "trajectory_key", "sample_group_id"):
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    sample_id = str(row["sample_id"])
    return sample_id.split("_step_", 1)[0]


def canonical_digest(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        existing: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            existing = [json.loads(line) for line in handle if line.strip()]
        if canonical_digest(existing) != canonical_digest(rows):
            raise RuntimeError(f"Refusing to overwrite a different fixed split: {path}. Use --force intentionally.")
        return
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    train_input = Path(args.train_input)
    test_input = Path(args.test_input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_prefix(train_input, max(1, int(args.train_samples)))
    test_rows = read_prefix(test_input, max(1, int(args.test_samples)))

    train_ids = {str(row["sample_id"]) for row in train_rows}
    test_ids = {str(row["sample_id"]) for row in test_rows}
    train_trajectories = {trajectory_key(row) for row in train_rows}
    test_trajectories = {trajectory_key(row) for row in test_rows}
    if train_ids & test_ids:
        raise RuntimeError("Fixed train/test subsets contain overlapping sample_id values.")
    if train_trajectories & test_trajectories:
        raise RuntimeError("Fixed train/test subsets contain overlapping trajectories.")

    train_out = out_dir / f"{args.prefix}.train.jsonl"
    test_out = out_dir / f"{args.prefix}.test.jsonl"
    manifest_out = out_dir / f"{args.prefix}.manifest.json"
    write_jsonl(train_out, train_rows, bool(args.force))
    write_jsonl(test_out, test_rows, bool(args.force))

    manifest = {
        "train_input": str(train_input),
        "test_input": str(test_input),
        "train_out": str(train_out),
        "test_out": str(test_out),
        "train_sample_count": len(train_rows),
        "test_sample_count": len(test_rows),
        "train_trajectory_count": len(train_trajectories),
        "test_trajectory_count": len(test_trajectories),
        "train_sha256": canonical_digest(train_rows),
        "test_sha256": canonical_digest(test_rows),
        "train_sample_ids": [str(row["sample_id"]) for row in train_rows],
        "test_sample_ids": [str(row["sample_id"]) for row in test_rows],
    }
    manifest_temp = manifest_out.with_name(manifest_out.name + ".tmp")
    manifest_temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(manifest_temp, manifest_out)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
