from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from finalize_agentnet_lara_clean import compact_row, trajectory_key, validate_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge enriched shards and validate a fixed trajectory-disjoint train/test subset."
    )
    parser.add_argument("--train-shards", nargs="+", required=True)
    parser.add_argument("--test-shards", nargs="+", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prefix", default="agentnet_lara_clean_s2000_t200")
    parser.add_argument("--img-next-count", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield row


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def load_and_validate(
    shard_paths: list[Path], dataset_root: Path, img_next_count: int
) -> tuple[list[dict[str, Any]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    seen: set[str] = set()
    for path in shard_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing enriched shard: {path}")
        for row in iter_jsonl(path):
            sample_id, action_type = validate_row(row, dataset_root, img_next_count)
            if sample_id in seen:
                raise ValueError(f"Duplicate sample_id across shards: {sample_id}")
            seen.add(sample_id)
            rows.append(compact_row(row))
            action_counts[action_type] += 1
    return rows, action_counts


def main() -> None:
    args = parse_args()
    train_shards = [Path(value).resolve() for value in args.train_shards]
    test_shards = [Path(value).resolve() for value in args.test_shards]
    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.selection_manifest).resolve()
    output_dir = Path(args.out_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_train_ids = [str(value) for value in selection["train"]["sample_ids"]]
    expected_test_ids = [str(value) for value in selection["test"]["sample_ids"]]
    train_rows, train_actions = load_and_validate(
        train_shards, dataset_root, int(args.img_next_count)
    )
    test_rows, test_actions = load_and_validate(
        test_shards, dataset_root, int(args.img_next_count)
    )

    actual_train_ids = [str(row["sample_id"]) for row in train_rows]
    actual_test_ids = [str(row["sample_id"]) for row in test_rows]
    if actual_train_ids != expected_train_ids:
        raise RuntimeError("Merged train rows do not exactly match the fixed selection manifest")
    if actual_test_ids != expected_test_ids:
        raise RuntimeError("Merged test rows do not exactly match the fixed selection manifest")

    train_keys = {trajectory_key(row) for row in train_rows}
    test_keys = {trajectory_key(row) for row in test_rows}
    if set(actual_train_ids) & set(actual_test_ids):
        raise RuntimeError("Train/test sample overlap after enrichment")
    if train_keys & test_keys:
        raise RuntimeError("Train/test trajectory overlap after enrichment")

    outputs = {
        "train": output_dir / f"{args.prefix}.train.jsonl",
        "test": output_dir / f"{args.prefix}.test.jsonl",
        "full": output_dir / f"{args.prefix}.stage1_full.jsonl",
    }
    report_path = output_dir / f"{args.prefix}.manifest.json"
    ready_path = output_dir / "READY"
    existing = [*outputs.values(), report_path, ready_path]
    if any(path.exists() for path in existing) and not bool(args.overwrite):
        raise FileExistsError("Final outputs already exist; pass --overwrite to rebuild them")

    atomic_write_jsonl(outputs["train"], train_rows)
    atomic_write_jsonl(outputs["test"], test_rows)
    atomic_write_jsonl(outputs["full"], train_rows + test_rows)

    report = {
        "selection_manifest": str(manifest_path),
        "reasoning_fields": ["actual_task", "thought", "reflection"],
        "bbox_in_reasoning": False,
        "img_next_count": int(args.img_next_count),
        "train": {
            "path": str(outputs["train"]),
            "row_count": len(train_rows),
            "trajectory_count": len(train_keys),
            "action_type_counts": dict(sorted(train_actions.items())),
            "sha256": file_sha256(outputs["train"]),
        },
        "test": {
            "path": str(outputs["test"]),
            "row_count": len(test_rows),
            "trajectory_count": len(test_keys),
            "action_type_counts": dict(sorted(test_actions.items())),
            "sha256": file_sha256(outputs["test"]),
        },
        "quality_checks": {
            "enrichment_errors": 0,
            "missing_images": 0,
            "duplicate_sample_ids": 0,
            "invalid_pointer_coordinates": 0,
            "sample_overlap": 0,
            "trajectory_overlap": 0,
            "selection_order_exact": True,
        },
    }
    temp_report = report_path.with_name(report_path.name + ".tmp")
    temp_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_report, report_path)
    temp_ready = ready_path.with_name(ready_path.name + ".tmp")
    temp_ready.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_ready, ready_path)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
