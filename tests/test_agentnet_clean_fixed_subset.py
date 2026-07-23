from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_overlong_refined_fields_are_trimmed_deterministically() -> None:
    from enrich_agentnet_lara_clean_with_vllm import FIELD_WORD_LIMITS, validate_refined_fields

    payload = {
        "actual_task": "one two three four five six seven eight nine ten eleven twelve thirteen",
        "thought": "Choose the visible target because it directly advances this exact GUI task now.",
        "reflection": "The expected panel opened successfully.",
    }
    refined = validate_refined_fields(payload)
    assert len(refined["actual_task"].split()) == FIELD_WORD_LIMITS["actual_task"]
    assert len(refined["thought"].split()) <= FIELD_WORD_LIMITS["thought"]


def test_fixed_subset_selection_and_merge(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_root = dataset_root / "ubuntu_images"
    image_root.mkdir(parents=True)
    source_rows: list[dict] = []
    for trajectory_index in range(30):
        for step_index in range(4):
            sample_id = f"trajectory-{trajectory_index:02d}_step_{step_index:04d}"
            before = f"{sample_id}-before.png"
            after = f"{sample_id}-after.png"
            (image_root / before).write_bytes(b"before")
            (image_root / after).write_bytes(b"after")
            source_rows.append(
                {
                    "sample_id": sample_id,
                    "trajectory_key": f"trajectory:{trajectory_index:02d}",
                    "task_id": f"task-{trajectory_index:02d}",
                    "instruction": "Complete the GUI task.",
                    "before_screenshot": before,
                    "after_screenshot": after,
                    "gold_action": {"type": "click", "x_norm": 0.25, "y_norm": 0.75},
                    "img_next": ["<img next>"] * 16,
                }
            )

    source = tmp_path / "structural.jsonl"
    structural_dir = tmp_path / "structural"
    write_jsonl(source, source_rows)
    prefix = "fixed"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "prepare_agentnet_clean_fixed_subset.py"),
            "--input",
            str(source),
            "--out-dir",
            str(structural_dir),
            "--prefix",
            prefix,
            "--train-samples",
            "20",
            "--test-samples",
            "8",
            "--seed",
            "42",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    manifest_path = structural_dir / f"{prefix}.selection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["train"]["row_count"] == 20
    assert manifest["test"]["row_count"] == 8
    assert manifest["quality_checks"]["trajectory_overlap"] == 0

    def enrich(path: Path) -> list[dict]:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row.update(
                {
                    "actual_task": "Click the target control.",
                    "thought": "The target advances the task.",
                    "reflection": "The target interface opened.",
                    "explicit_reasoning": "\n".join(
                        [
                            "actual_task: Click the target control.",
                            "thought: The target advances the task.",
                            "reflection: The target interface opened.",
                            " ".join(["<img next>"] * 16),
                        ]
                    ),
                    "enrich_status": "ok",
                    "enrich_error": None,
                }
            )
        return rows

    train_rows = enrich(structural_dir / f"{prefix}.structural.train.jsonl")
    test_rows = enrich(structural_dir / f"{prefix}.structural.test.jsonl")
    shard_dir = tmp_path / "shards"
    train_shards = [shard_dir / "train0.jsonl", shard_dir / "train1.jsonl"]
    test_shards = [shard_dir / "test0.jsonl", shard_dir / "test1.jsonl"]
    write_jsonl(train_shards[0], train_rows[:10])
    write_jsonl(train_shards[1], train_rows[10:])
    write_jsonl(test_shards[0], test_rows[:4])
    write_jsonl(test_shards[1], test_rows[4:])

    final_dir = tmp_path / "final"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "merge_agentnet_clean_fixed_subset.py"),
            "--train-shards",
            *map(str, train_shards),
            "--test-shards",
            *map(str, test_shards),
            "--selection-manifest",
            str(manifest_path),
            "--dataset-root",
            str(dataset_root),
            "--out-dir",
            str(final_dir),
            "--prefix",
            prefix,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    final_manifest = json.loads((final_dir / f"{prefix}.manifest.json").read_text("utf-8"))
    assert final_manifest["train"]["row_count"] == 20
    assert final_manifest["test"]["row_count"] == 8
    assert final_manifest["quality_checks"]["trajectory_overlap"] == 0
    assert (final_dir / "READY").is_file()
