from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select deterministic, trajectory-disjoint structural rows before vLLM "
            "enrichment. The trajectory shuffle matches finalize_agentnet_lara_clean.py."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="agentnet_lara_clean_s2000_t200")
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            yield row


def trajectory_key(row: dict[str, Any]) -> str:
    value = str(row.get("trajectory_key", "") or "").strip()
    if value:
        return value
    task_id = str(row.get("task_id", "") or "").strip()
    if task_id:
        return f"task_id:{task_id}"
    sample_id = str(row.get("sample_id", "") or "").strip()
    return sample_id.rsplit("_step_", 1)[0]


def canonical_digest(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]], force: bool) -> None:
    if path.exists() and not force:
        existing = list(iter_jsonl(path))
        if canonical_digest(existing) != canonical_digest(rows):
            raise RuntimeError(
                f"Refusing to replace a different fixed subset: {path}. Pass --force intentionally."
            )
        return
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def action_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        action = row.get("gold_action") or row.get("parsed_action") or {}
        counts[str(action.get("type", "unknown") or "unknown")] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    ratios = (float(args.train_ratio), float(args.val_ratio), float(args.test_ratio))
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")
    train_samples = max(1, int(args.train_samples))
    test_samples = max(1, int(args.test_samples))

    input_path = Path(args.input).resolve()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectory_counts: Counter[str] = Counter()
    source_row_count = 0
    for row in iter_jsonl(input_path):
        sample_id = str(row.get("sample_id", "") or "").strip()
        key = trajectory_key(row)
        if not sample_id or not key:
            raise ValueError(f"Row {source_row_count + 1} has no sample_id or trajectory key")
        trajectory_counts[key] += 1
        source_row_count += 1

    keys = list(trajectory_counts)
    random.Random(int(args.seed)).shuffle(keys)
    train_cut = int(len(keys) * ratios[0])
    val_cut = train_cut + int(len(keys) * ratios[1])
    train_keys = set(keys[:train_cut])
    test_keys = set(keys[val_cut:])
    if train_keys & test_keys:
        raise AssertionError("Trajectory split overlap")

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for row in iter_jsonl(input_path):
        sample_id = str(row["sample_id"])
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate sample_id in structural source: {sample_id}")
        seen_samples.add(sample_id)
        key = trajectory_key(row)
        if key in train_keys and len(train_rows) < train_samples:
            train_rows.append(row)
        elif key in test_keys and len(test_rows) < test_samples:
            test_rows.append(row)
        if len(train_rows) == train_samples and len(test_rows) == test_samples:
            break

    if len(train_rows) != train_samples or len(test_rows) != test_samples:
        raise RuntimeError(
            f"Insufficient rows after trajectory split: train={len(train_rows)}/{train_samples}, "
            f"test={len(test_rows)}/{test_samples}"
        )

    train_ids = {str(row["sample_id"]) for row in train_rows}
    test_ids = {str(row["sample_id"]) for row in test_rows}
    selected_train_keys = {trajectory_key(row) for row in train_rows}
    selected_test_keys = {trajectory_key(row) for row in test_rows}
    if train_ids & test_ids:
        raise AssertionError("Selected train/test sample overlap")
    if selected_train_keys & selected_test_keys:
        raise AssertionError("Selected train/test trajectory overlap")

    train_path = output_dir / f"{args.prefix}.structural.train.jsonl"
    test_path = output_dir / f"{args.prefix}.structural.test.jsonl"
    manifest_path = output_dir / f"{args.prefix}.selection_manifest.json"
    write_jsonl(train_path, train_rows, bool(args.force))
    write_jsonl(test_path, test_rows, bool(args.force))

    manifest = {
        "source": str(input_path),
        "source_row_count": source_row_count,
        "source_trajectory_count": len(keys),
        "seed": int(args.seed),
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "train": {
            "path": str(train_path),
            "row_count": len(train_rows),
            "trajectory_count": len(selected_train_keys),
            "sha256": canonical_digest(train_rows),
            "sample_ids": [str(row["sample_id"]) for row in train_rows],
            "action_type_counts": action_counts(train_rows),
        },
        "test": {
            "path": str(test_path),
            "row_count": len(test_rows),
            "trajectory_count": len(selected_test_keys),
            "sha256": canonical_digest(test_rows),
            "sample_ids": [str(row["sample_id"]) for row in test_rows],
            "action_type_counts": action_counts(test_rows),
        },
        "quality_checks": {
            "sample_overlap": 0,
            "trajectory_overlap": 0,
            "exact_train_size": len(train_rows) == train_samples,
            "exact_test_size": len(test_rows) == test_samples,
        },
    }
    temp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temp_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temp_manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
