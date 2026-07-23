from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator


POINTER_ACTIONS = {"click", "double_click", "right_click"}
FINAL_SCHEMA_VERSION = "agentnet_lara_clean_stage1_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate enriched AgentNet LaRA rows, emit a compact final dataset, and split "
            "it by trajectory without leakage."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="agentnet_ubuntu_lara_clean")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-next-count", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=0, help="<=0 means all rows")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path, max_rows: int = 0) -> Iterator[dict[str, Any]]:
    yielded = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            yield payload
            yielded += 1
            if max_rows > 0 and yielded >= max_rows:
                break


def trajectory_key(row: dict[str, Any]) -> str:
    value = str(row.get("trajectory_key", "") or "").strip()
    if value:
        return value
    task_id = str(row.get("task_id", "") or "").strip()
    if task_id:
        return f"task_id:{task_id}"
    sample_id = str(row.get("sample_id", "") or "").strip()
    return sample_id.rsplit("_step_", 1)[0]


def resolve_image(dataset_root: Path, image_name: str) -> Path:
    normalized = str(image_name or "").strip()
    candidates = (
        dataset_root / "ubuntu_images" / normalized,
        dataset_root / normalized,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing referenced image: {normalized}")


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def length_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "mean": 0.0, "p95": 0.0, "max": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


def validate_row(
    row: dict[str, Any],
    dataset_root: Path,
    img_next_count: int,
) -> tuple[str, str]:
    sample_id = str(row.get("sample_id", "") or "").strip()
    if not sample_id:
        raise ValueError("Row has no sample_id")
    if row.get("enrich_status") != "ok":
        raise ValueError(f"{sample_id}: enrichment status is not ok: {row.get('enrich_error')}")
    for field in ("instruction", "actual_task", "thought", "reflection"):
        if not str(row.get(field, "") or "").strip():
            raise ValueError(f"{sample_id}: empty field {field}")
    resolve_image(dataset_root, str(row.get("before_screenshot", "")))
    resolve_image(dataset_root, str(row.get("after_screenshot", "")))

    explicit = str(row.get("explicit_reasoning", "") or "")
    if re.search(r"(?im)^\s*bbox\s*:", explicit):
        raise ValueError(f"{sample_id}: bbox leaked into explicit reasoning")
    for field in ("actual_task", "thought", "reflection"):
        if f"{field}:" not in explicit:
            raise ValueError(f"{sample_id}: explicit reasoning is missing {field}")
    if explicit.count("<img next>") != img_next_count:
        raise ValueError(
            f"{sample_id}: expected {img_next_count} <img next> tokens, "
            f"got {explicit.count('<img next>')}"
        )

    action = row.get("gold_action")
    if not isinstance(action, dict):
        raise ValueError(f"{sample_id}: gold_action is not an object")
    action_type = str(action.get("type", "") or "")
    if not action_type:
        raise ValueError(f"{sample_id}: gold_action has no type")
    if action_type in POINTER_ACTIONS:
        x = action.get("x_norm")
        y = action.get("y_norm")
        if x is None or y is None:
            raise ValueError(f"{sample_id}: pointer action has no normalized coordinates")
        if not (0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0):
            raise ValueError(f"{sample_id}: normalized coordinates are outside [0,1]")
    return sample_id, action_type


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    output = dict(row)
    output["schema_version"] = FINAL_SCHEMA_VERSION
    output.pop("enrich_raw_response_text", None)
    output.pop("enrich_error", None)
    output["trajectory_key"] = trajectory_key(output)
    return output


def main() -> None:
    args = parse_args()
    ratio_sum = float(args.train_ratio) + float(args.val_ratio) + float(args.test_ratio)
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    input_path = Path(args.input).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_rows = max(0, int(args.max_rows))
    img_next_count = max(0, int(args.img_next_count))

    seen_samples: set[str] = set()
    trajectory_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    field_lengths: dict[str, list[int]] = defaultdict(list)
    row_count = 0

    for row in iter_jsonl(input_path, max_rows):
        sample_id, action_type = validate_row(row, dataset_root, img_next_count)
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate sample_id: {sample_id}")
        seen_samples.add(sample_id)
        key = trajectory_key(row)
        if not key:
            raise ValueError(f"{sample_id}: empty trajectory key")
        trajectory_counts[key] += 1
        action_counts[action_type] += 1
        for field in ("instruction", "actual_task", "thought", "reflection"):
            field_lengths[field].append(len(str(row.get(field, "") or "").split()))
        row_count += 1

    if row_count == 0:
        raise RuntimeError("No rows found")

    keys = list(trajectory_counts)
    random.Random(int(args.seed)).shuffle(keys)
    train_cut = int(len(keys) * float(args.train_ratio))
    val_cut = train_cut + int(len(keys) * float(args.val_ratio))
    split_keys = {
        "train": set(keys[:train_cut]),
        "val": set(keys[train_cut:val_cut]),
        "test": set(keys[val_cut:]),
    }
    key_to_split = {
        key: split_name for split_name, split_set in split_keys.items() for key in split_set
    }

    output_paths = {
        "full": output_dir / f"{args.prefix}.stage1_full.jsonl",
        "train": output_dir / f"{args.prefix}.train.jsonl",
        "val": output_dir / f"{args.prefix}.val.jsonl",
        "test": output_dir / f"{args.prefix}.test.jsonl",
    }
    manifest_path = output_dir / f"{args.prefix}.manifest.json"
    for path in [*output_paths.values(), manifest_path]:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite to replace it: {path}")

    temp_paths = {name: path.with_name(path.name + ".tmp") for name, path in output_paths.items()}
    for path in temp_paths.values():
        if path.exists():
            path.unlink()

    split_row_counts: Counter[str] = Counter()
    split_action_counts: dict[str, Counter[str]] = defaultdict(Counter)
    handles = {name: path.open("w", encoding="utf-8") for name, path in temp_paths.items()}
    try:
        for row in iter_jsonl(input_path, max_rows):
            compact = compact_row(row)
            line = json.dumps(compact, ensure_ascii=False) + "\n"
            handles["full"].write(line)
            split_name = key_to_split[trajectory_key(row)]
            handles[split_name].write(line)
            split_row_counts[split_name] += 1
            split_action_counts[split_name][str(compact["gold_action"]["type"])] += 1
    finally:
        for handle in handles.values():
            handle.close()

    for name, temp_path in temp_paths.items():
        os.replace(temp_path, output_paths[name])

    manifest = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "source": str(input_path),
        "dataset_root": str(dataset_root),
        "row_count": row_count,
        "trajectory_count": len(keys),
        "seed": int(args.seed),
        "ratios": {
            "train": float(args.train_ratio),
            "val": float(args.val_ratio),
            "test": float(args.test_ratio),
        },
        "reasoning_fields": ["actual_task", "thought", "reflection"],
        "bbox_in_reasoning": False,
        "img_next_count": img_next_count,
        "action_type_counts": dict(sorted(action_counts.items())),
        "field_word_lengths": {
            field: length_summary(values) for field, values in sorted(field_lengths.items())
        },
        "outputs": {
            name: {
                "path": str(path),
                "row_count": row_count if name == "full" else int(split_row_counts[name]),
                "trajectory_count": len(keys) if name == "full" else len(split_keys[name]),
                "action_type_counts": (
                    dict(sorted(action_counts.items()))
                    if name == "full"
                    else dict(sorted(split_action_counts[name].items()))
                ),
            }
            for name, path in output_paths.items()
        },
        "quality_checks": {
            "duplicate_sample_ids": 0,
            "missing_images": 0,
            "enrichment_errors": 0,
            "trajectory_overlap_between_splits": 0,
            "invalid_pointer_coordinates": 0,
        },
    }
    temp_manifest = manifest_path.with_name(manifest_path.name + ".tmp")
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_manifest, manifest_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
