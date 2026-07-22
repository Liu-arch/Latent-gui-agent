from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen3_gui_agent.rl.agentnet_adapter import _parse_action_code
from qwen3_gui_agent.rl.schema import append_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a LaRA-VLA-style GUI step dataset directly from raw AgentNet trajectories. "
            "This keeps primitive field names close to the original data: instruction / actual_task / "
            "action / thought / reflection / code, and adds GUI-specific bbox plus repeated <img next> slots."
        )
    )
    parser.add_argument("--input", required=True, help="Raw AgentNet JSONL path")
    parser.add_argument("--out", required=True, help="Output JSONL path")
    parser.add_argument("--summary-out", default=None, help="Optional summary JSON path")
    parser.add_argument("--max-tasks", type=int, default=0, help="<=0 means all tasks")
    parser.add_argument("--max-steps", type=int, default=0, help="<=0 means all steps")
    parser.add_argument(
        "--img-next-count",
        type=int,
        default=16,
        help="Number of repeated <img next> tokens to export per step",
    )
    parser.add_argument(
        "--drop-invalid-actions",
        action="store_true",
        help="Skip steps whose code cannot be parsed into a GUI action",
    )
    parser.add_argument(
        "--actual-task-source",
        choices=["actual_task", "natural_language_task", "instruction", "auto"],
        default="actual_task",
        help="Which top-level field to use for the exported actual_task key",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                row = json.loads(stripped)
                if isinstance(row, dict):
                    yield row


def first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_actual_task(row: dict[str, Any], source: str) -> str:
    if source == "actual_task":
        return first_non_empty(row.get("actual_task"))
    if source == "natural_language_task":
        return first_non_empty(row.get("natural_language_task"))
    if source == "instruction":
        return first_non_empty(row.get("instruction"))
    return first_non_empty(
        row.get("actual_task"),
        row.get("natural_language_task"),
        row.get("instruction"),
        row.get("task"),
    )


def normalize_action(action: dict[str, Any] | None) -> dict[str, Any] | None:
    if not action:
        return None
    normalized = dict(action)
    if normalized.get("x_norm") is not None:
        normalized["x_norm"] = round(float(normalized["x_norm"]), 4)
    if normalized.get("y_norm") is not None:
        normalized["y_norm"] = round(float(normalized["y_norm"]), 4)
    if normalized.get("x") is not None:
        normalized["x"] = int(float(normalized["x"]))
    if normalized.get("y") is not None:
        normalized["y"] = int(float(normalized["y"]))
    if normalized.get("amount") is not None:
        normalized["amount"] = int(normalized["amount"])
    if normalized.get("text") is not None:
        normalized["text"] = str(normalized["text"])
    if normalized.get("keys") is not None:
        normalized["keys"] = list(normalized["keys"])
    if normalized.get("status") is not None:
        normalized["status"] = str(normalized["status"])
    normalized["type"] = str(normalized.get("type", "wait"))
    return normalized


def build_bbox(action: dict[str, Any] | None) -> list[float] | None:
    if not action:
        return None
    if action.get("x_norm") is not None and action.get("y_norm") is not None:
        x = round(float(action["x_norm"]), 4)
        y = round(float(action["y_norm"]), 4)
        return [x, y, x, y]
    return None


def build_explicit_supervision_text(sample: dict[str, Any]) -> str:
    lines = [
        f"instruction: {str(sample.get('instruction', '')).strip()}",
        f"actual_task: {str(sample.get('actual_task', '')).strip()}",
    ]
    bbox = sample.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        lines.append(f"bbox: [{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}]")
    else:
        lines.append("bbox: []")
    lines.append(f"thought: {str(sample.get('thought', '')).strip()}")
    lines.append(f"reflection: {str(sample.get('reflection', '')).strip()}")
    img_next = sample.get("img_next", [])
    if isinstance(img_next, list) and img_next:
        lines.append(" ".join(str(token) for token in img_next))
    return "\n".join(lines).strip()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    processed_tasks = 0
    exported_steps = 0
    skipped_tasks = 0
    skipped_steps = 0
    invalid_actions = 0

    for row in iter_jsonl(input_path):
        processed_tasks += 1
        if args.max_tasks > 0 and processed_tasks > args.max_tasks:
            break

        traj = row.get("traj")
        if not isinstance(traj, list) or not traj:
            skipped_tasks += 1
            continue

        task_id = first_non_empty(row.get("task_id"), row.get("sample_id"), f"agentnet_task_{processed_tasks:06d}")
        instruction = first_non_empty(row.get("instruction"), row.get("natural_language_task"), row.get("actual_task"))
        actual_task = resolve_actual_task(row, args.actual_task_source)

        for step_index, traj_item in enumerate(traj):
            if args.max_steps > 0 and exported_steps >= args.max_steps:
                break
            if not isinstance(traj_item, dict):
                skipped_steps += 1
                continue
            value = traj_item.get("value", {})
            if not isinstance(value, dict):
                skipped_steps += 1
                continue

            before_screenshot = first_non_empty(traj_item.get("image"), traj_item.get("screenshot"))
            if not before_screenshot:
                skipped_steps += 1
                continue

            after_screenshot = before_screenshot
            if step_index + 1 < len(traj) and isinstance(traj[step_index + 1], dict):
                after_screenshot = first_non_empty(
                    traj[step_index + 1].get("image"),
                    traj[step_index + 1].get("screenshot"),
                    before_screenshot,
                )

            code = str(value.get("code", "")).strip()
            parsed_action = normalize_action(_parse_action_code(code))
            if parsed_action is None:
                invalid_actions += 1
                if args.drop_invalid_actions:
                    skipped_steps += 1
                    continue

            sample = {
                "sample_id": f"{task_id}_step_{step_index:04d}",
                "task_id": task_id,
                "step_index": step_index,
                "traj_index": int(traj_item.get("index", step_index)),
                "instruction": instruction,
                "actual_task": actual_task,
                "before_screenshot": before_screenshot,
                "after_screenshot": after_screenshot,
                "action": str(value.get("action", "")).strip(),
                "thought": str(value.get("thought", "")).strip(),
                "reflection": str(value.get("reflection", "")).strip(),
                "code": code,
                "bbox": build_bbox(parsed_action),
                "img_next": ["<img next>"] * max(0, int(args.img_next_count)),
                "observation": str(value.get("observation", "")).strip(),
                "parsed_action": parsed_action,
                "task_completed": bool(row.get("task_completed", False)),
                "alignment_score": row.get("alignment_score"),
                "efficiency_score": row.get("efficiency_score"),
                "task_difficulty": row.get("task_difficulty"),
            }
            sample["explicit_supervision"] = build_explicit_supervision_text(sample)
            append_jsonl(out_path, sample)
            exported_steps += 1

        if args.max_steps > 0 and exported_steps >= args.max_steps:
            break

        if processed_tasks % 100 == 0:
            print(
                json.dumps(
                    {
                        "stage": "build_lara_gui_dataset",
                        "processed_tasks": processed_tasks,
                        "exported_steps": exported_steps,
                        "invalid_actions": invalid_actions,
                    },
                    ensure_ascii=False,
                )
            )

    summary = {
        "input": str(input_path),
        "output": str(out_path),
        "summary_out": str(summary_path),
        "processed_tasks": processed_tasks if args.max_tasks <= 0 else min(processed_tasks, args.max_tasks),
        "exported_steps": exported_steps,
        "skipped_tasks": skipped_tasks,
        "skipped_steps": skipped_steps,
        "invalid_actions": invalid_actions,
        "img_next_count": int(args.img_next_count),
        "actual_task_source": str(args.actual_task_source),
        "drop_invalid_actions": bool(args.drop_invalid_actions),
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
