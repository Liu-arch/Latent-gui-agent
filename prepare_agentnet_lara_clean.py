from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from qwen3_gui_agent.rl.agentnet_adapter import _parse_action_code


POINTER_ACTIONS = {"click", "double_click", "right_click"}
SCHEMA_VERSION = "agentnet_lara_clean_structural_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand official AgentNet trajectories into deterministic step-level rows for "
            "three-field LaRA GUI reasoning. Coordinates stay exclusively in gold_action."
        )
    )
    parser.add_argument("--input", required=True, help="Official agentnet_ubuntu_5k.jsonl")
    parser.add_argument("--dataset-root", required=True, help="Directory containing ubuntu_images")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument(
        "--unsupported-out",
        default=None,
        help="Audit JSONL for valid source steps outside the current action space",
    )
    parser.add_argument("--max-tasks", type=int, default=0, help="<=0 means all trajectories")
    parser.add_argument("--max-steps", type=int, default=0, help="<=0 means all steps")
    parser.add_argument("--img-next-count", type=int, default=16)
    parser.add_argument("--log-every-tasks", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            yield payload


def first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not action:
        return None
    normalized: dict[str, Any] = {"type": str(action.get("type", "wait") or "wait")}
    for key in ("text", "status", "button"):
        if action.get(key) is not None:
            normalized[key] = str(action[key])
    if action.get("keys") is not None:
        normalized["keys"] = [str(item) for item in list(action.get("keys") or [])]
    if action.get("amount") is not None:
        normalized["amount"] = int(action["amount"])
    for key in ("x_norm", "y_norm"):
        if action.get(key) is not None:
            normalized[key] = round(float(action[key]), 6)
    for key in ("x", "y"):
        if action.get(key) is not None:
            normalized[key] = int(float(action[key]))
    return normalized


def validate_action(action: dict[str, Any]) -> str | None:
    action_type = str(action.get("type", ""))
    if action_type not in POINTER_ACTIONS:
        return None
    x_norm = action.get("x_norm")
    y_norm = action.get("y_norm")
    if x_norm is None or y_norm is None:
        return "pointer_missing_normalized_coordinates"
    if not (0.0 <= float(x_norm) <= 1.0 and 0.0 <= float(y_norm) <= 1.0):
        return "pointer_coordinates_out_of_range"
    return None


def recover_pointer_coordinates(
    action: dict[str, Any],
    image_path: Path,
    image_size_cache: dict[Path, tuple[int, int]],
) -> bool:
    if str(action.get("type", "")) not in POINTER_ACTIONS:
        return False
    if action.get("x_norm") is not None and action.get("y_norm") is not None:
        return False
    x = action.get("x")
    y = action.get("y")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return False
    if image_path not in image_size_cache:
        from PIL import Image

        with Image.open(image_path) as image:
            image_size_cache[image_path] = tuple(map(int, image.size))
    width, height = image_size_cache[image_path]
    if width <= 0 or height <= 0:
        return False
    x_norm = float(x) / float(width)
    y_norm = float(y) / float(height)
    if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
        return False
    action["x_norm"] = round(x_norm, 6)
    action["y_norm"] = round(y_norm, 6)
    return True


def action_code_pattern(code: str) -> str:
    calls = re.findall(r"(?:pyautogui|computer)\.([A-Za-z_]\w*)\s*\(", code)
    return "+".join(item.lower() for item in calls) if calls else "no_supported_call_syntax"


def resolve_image(dataset_root: Path, image_name: str) -> Path:
    candidates = (
        dataset_root / "ubuntu_images" / image_name,
        dataset_root / image_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve AgentNet image: {image_name}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.out).resolve()
    summary_path = (
        Path(args.summary_out).resolve()
        if args.summary_out
        else output_path.with_suffix(".summary.json")
    )
    unsupported_path = (
        Path(args.unsupported_out).resolve()
        if args.unsupported_out
        else output_path.with_suffix(".unsupported.jsonl")
    )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite to replace it: {output_path}")
    if unsupported_path.exists() and not args.overwrite:
        raise FileExistsError(
            "Unsupported-action audit exists; pass --overwrite to replace it: "
            f"{unsupported_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unsupported_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output_path.with_name(output_path.name + ".tmp")
    temp_unsupported = unsupported_path.with_name(unsupported_path.name + ".tmp")
    if temp_output.exists():
        temp_output.unlink()
    if temp_unsupported.exists():
        temp_unsupported.unlink()

    action_counts: Counter[str] = Counter()
    invalid_action_reasons: Counter[str] = Counter()
    invalid_code_patterns: Counter[str] = Counter()
    invalid_action_examples: list[dict[str, Any]] = []
    source_task_count = 0
    source_step_count = 0
    exported_step_count = 0
    invalid_action_count = 0
    skipped_step_count = 0
    pointer_count = 0
    recovered_pointer_coordinate_count = 0
    unique_images: set[str] = set()
    image_size_cache: dict[Path, tuple[int, int]] = {}

    try:
        with (
            temp_output.open("w", encoding="utf-8") as output_handle,
            temp_unsupported.open("w", encoding="utf-8") as unsupported_handle,
        ):
            for task_index, task_row in enumerate(iter_jsonl(input_path)):
                if int(args.max_tasks) > 0 and source_task_count >= int(args.max_tasks):
                    break
                source_task_count += 1

                trajectory = task_row.get("traj")
                if not isinstance(trajectory, list) or not trajectory:
                    continue

                task_id = first_non_empty(
                    task_row.get("task_id"),
                    task_row.get("sample_id"),
                    f"agentnet_task_{task_index:06d}",
                )
                instruction = first_non_empty(
                    task_row.get("instruction"),
                    task_row.get("natural_language_task"),
                    task_row.get("actual_task"),
                )
                natural_language_task = first_non_empty(task_row.get("natural_language_task"))
                episode_actual_task = first_non_empty(task_row.get("actual_task"))

                for step_index, trajectory_item in enumerate(trajectory):
                    if int(args.max_steps) > 0 and exported_step_count >= int(args.max_steps):
                        break
                    source_step_count += 1
                    if not isinstance(trajectory_item, dict):
                        skipped_step_count += 1
                        continue
                    value = trajectory_item.get("value")
                    if not isinstance(value, dict):
                        skipped_step_count += 1
                        continue

                    before_screenshot = first_non_empty(
                        trajectory_item.get("image"), trajectory_item.get("screenshot")
                    )
                    if not before_screenshot:
                        skipped_step_count += 1
                        continue
                    after_screenshot = before_screenshot
                    if step_index + 1 < len(trajectory) and isinstance(trajectory[step_index + 1], dict):
                        after_screenshot = first_non_empty(
                            trajectory[step_index + 1].get("image"),
                            trajectory[step_index + 1].get("screenshot"),
                            before_screenshot,
                        )

                    before_image_path = resolve_image(dataset_root, before_screenshot)
                    resolve_image(dataset_root, after_screenshot)
                    unique_images.update((before_screenshot, after_screenshot))

                    code = str(value.get("code", "") or "").strip()
                    parsed_action = normalize_action(_parse_action_code(code))
                    if parsed_action is not None and recover_pointer_coordinates(
                        parsed_action, before_image_path, image_size_cache
                    ):
                        recovered_pointer_coordinate_count += 1
                    invalid_reason = (
                        "action_parse_failed"
                        if parsed_action is None
                        else validate_action(parsed_action)
                    )
                    if invalid_reason is not None:
                        invalid_action_count += 1
                        skipped_step_count += 1
                        invalid_action_reasons[invalid_reason] += 1
                        invalid_code_patterns[action_code_pattern(code)] += 1
                        if len(invalid_action_examples) < 50:
                            invalid_action_examples.append(
                                {
                                    "sample_id": f"{task_id}_step_{step_index:04d}",
                                    "reason": invalid_reason,
                                    "code_pattern": action_code_pattern(code),
                                    "code": code,
                                    "action_text": str(value.get("action", "") or "").strip(),
                                    "parsed_action": parsed_action,
                                }
                            )
                        unsupported_handle.write(
                            json.dumps(
                                {
                                    "schema_version": "agentnet_lara_clean_unsupported_action_v1",
                                    "sample_id": f"{task_id}_step_{step_index:04d}",
                                    "task_id": task_id,
                                    "trajectory_key": f"task_id:{task_id}",
                                    "step_index": step_index,
                                    "traj_index": int(trajectory_item.get("index", step_index)),
                                    "instruction": instruction,
                                    "natural_language_task": natural_language_task,
                                    "episode_actual_task": episode_actual_task,
                                    "before_screenshot": before_screenshot,
                                    "after_screenshot": after_screenshot,
                                    "action_text_raw": str(value.get("action", "") or "").strip(),
                                    "thought_raw": str(value.get("thought", "") or "").strip(),
                                    "reflection_raw": str(value.get("reflection", "") or "").strip(),
                                    "code": code,
                                    "unsupported_reason": invalid_reason,
                                    "code_pattern": action_code_pattern(code),
                                    "parsed_action": parsed_action,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue

                    action_type = str(parsed_action["type"])
                    action_counts[action_type] += 1
                    if action_type in POINTER_ACTIONS:
                        pointer_count += 1

                    raw_action_text = str(value.get("action", "") or "").strip()
                    raw_thought = str(value.get("thought", "") or "").strip()
                    raw_reflection = str(value.get("reflection", "") or "").strip()
                    step_task_seed = first_non_empty(raw_action_text, raw_thought, episode_actual_task)
                    img_next = ["<img next>"] * max(0, int(args.img_next_count))

                    output_row = {
                        "schema_version": SCHEMA_VERSION,
                        "sample_id": f"{task_id}_step_{step_index:04d}",
                        "task_id": task_id,
                        "trajectory_key": f"task_id:{task_id}",
                        "step_index": step_index,
                        "traj_index": int(trajectory_item.get("index", step_index)),
                        "instruction": instruction,
                        "instruction_raw": str(task_row.get("instruction", "") or "").strip(),
                        "natural_language_task": natural_language_task,
                        "episode_actual_task": episode_actual_task,
                        "before_screenshot": before_screenshot,
                        "after_screenshot": after_screenshot,
                        "actual_task_seed": step_task_seed,
                        "action_text_raw": raw_action_text,
                        "thought_raw": raw_thought,
                        "reflection_raw": raw_reflection,
                        "observation_raw": str(value.get("observation", "") or "").strip(),
                        "code": code,
                        "gold_action": parsed_action,
                        "parsed_action": parsed_action,
                        "img_next": img_next,
                        "task_completed": bool(task_row.get("task_completed", False)),
                        "last_step_correct": value.get("last_step_correct"),
                        "last_step_redundant": value.get("last_step_redundant"),
                        "alignment_score": task_row.get("alignment_score"),
                        "efficiency_score": task_row.get("efficiency_score"),
                        "task_difficulty": task_row.get("task_difficulty"),
                    }
                    output_handle.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                    exported_step_count += 1

                if int(args.max_steps) > 0 and exported_step_count >= int(args.max_steps):
                    break
                if source_task_count % max(1, int(args.log_every_tasks)) == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "prepare_agentnet_lara_clean",
                                "source_tasks": source_task_count,
                                "exported_steps": exported_step_count,
                                "invalid_actions": invalid_action_count,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
        os.replace(temp_output, output_path)
        os.replace(temp_unsupported, unsupported_path)
    finally:
        if temp_output.exists():
            temp_output.unlink()
        if temp_unsupported.exists():
            temp_unsupported.unlink()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "input": str(input_path),
        "dataset_root": str(dataset_root),
        "output": str(output_path),
        "unsupported_output": str(unsupported_path),
        "source_task_count": source_task_count,
        "source_step_count": source_step_count,
        "exported_step_count": exported_step_count,
        "invalid_action_count": invalid_action_count,
        "invalid_action_reason_counts": dict(sorted(invalid_action_reasons.items())),
        "invalid_code_pattern_counts": dict(sorted(invalid_code_patterns.items())),
        "invalid_action_examples": invalid_action_examples,
        "skipped_step_count": skipped_step_count,
        "pointer_step_count": pointer_count,
        "recovered_pointer_coordinate_count": recovered_pointer_coordinate_count,
        "unique_referenced_image_count": len(unique_images),
        "action_type_counts": dict(sorted(action_counts.items())),
        "img_next_count": max(0, int(args.img_next_count)),
        "reasoning_fields": ["actual_task", "thought", "reflection"],
        "bbox_in_reasoning": False,
        "coordinates_only_in_gold_action": True,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
