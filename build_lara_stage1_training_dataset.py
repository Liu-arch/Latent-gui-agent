from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert enriched LaRA-GUI step data into the final Stage-1 training format. "
            "The output stays compatible with train_lara_style_qwen3vl.py, but rewrites "
            "actual_task / thought / reflection / explicit_supervision into the compact "
            "LaRA-style supervision format."
        )
    )
    parser.add_argument("--input", required=True, help="Input enriched JSONL path, e.g. enriched_s50_v3.jsonl")
    parser.add_argument("--out", required=True, help="Output Stage-1 JSONL path")
    parser.add_argument("--summary-out", default=None, help="Optional summary JSON path")
    parser.add_argument("--max-samples", type=int, default=0, help="<=0 means all rows")
    parser.add_argument(
        "--img-next-count",
        type=int,
        default=16,
        help="Export this many public <img next> tokens per row",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    yield payload


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def bbox_text(value: Any) -> str:
    if isinstance(value, list) and len(value) == 4:
        try:
            return f"[{float(value[0]):.4f} {float(value[1]):.4f} {float(value[2]):.4f} {float(value[3]):.4f}]"
        except (TypeError, ValueError):
            return "[]"
    return "[]"


def normalize_action(action: dict[str, Any] | None) -> dict[str, Any]:
    action = dict(action or {})
    normalized: dict[str, Any] = {
        "type": str(action.get("type", "wait") or "wait"),
    }
    for key in ("text", "status", "button"):
        if action.get(key) is not None:
            normalized[key] = str(action[key])
    if action.get("keys") is not None:
        normalized["keys"] = [str(item) for item in list(action.get("keys") or [])]
    if action.get("amount") is not None:
        try:
            normalized["amount"] = int(action["amount"])
        except (TypeError, ValueError):
            pass
    for key in ("x_norm", "y_norm"):
        if action.get(key) is not None:
            try:
                normalized[key] = round(float(action[key]), 4)
            except (TypeError, ValueError):
                pass
    for key in ("x", "y"):
        if action.get(key) is not None:
            try:
                normalized[key] = int(float(action[key]))
            except (TypeError, ValueError):
                pass
    return normalized


def build_action_text_target(action: dict[str, Any]) -> str:
    action_type = str(action.get("type", "wait") or "wait")
    lines = [f"Action: {action_type}"]
    if action_type in {"click", "double_click", "right_click"}:
        if action.get("x_norm") is not None and action.get("y_norm") is not None:
            lines.append(f"Point: [{float(action['x_norm']):.4f} {float(action['y_norm']):.4f}]")
        elif action.get("x") is not None and action.get("y") is not None:
            lines.append(f"PointPx: [{int(action['x'])} {int(action['y'])}]")
    elif action_type == "type":
        lines.append(f'Text: "{str(action.get("text", ""))}"')
    elif action_type == "hotkey":
        keys = [str(item) for item in list(action.get("keys") or [])]
        lines.append(f"Keys: [{', '.join(keys)}]")
    elif action_type == "scroll":
        lines.append(f"Amount: {int(action.get('amount', 0) or 0)}")
    elif action_type in {"terminate", "wait"}:
        lines.append(f"Status: {str(action.get('status', 'success') or 'success')}")
    return "\n".join(lines)


def build_stage1_explicit_supervision(
    *,
    actual_task: str,
    bbox: Any,
    thought: str,
    reflection: str,
    img_next_tokens: list[str],
) -> str:
    lines = [
        f"actual_task: {actual_task.strip()}",
        f"bbox: {bbox_text(bbox)}",
        f"thought: {thought.strip()}",
        f"reflection: {reflection.strip()}",
        " ".join(img_next_tokens),
    ]
    return "\n".join(line for line in lines if line.strip()).strip()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_path = Path(args.out)
    summary_path = Path(args.summary_out) if args.summary_out else out_path.with_suffix(".summary.json")
    max_samples = int(args.max_samples)
    img_next_count = max(0, int(args.img_next_count))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    row_count = 0
    filled_actual_task_count = 0
    short_fallback_count = 0
    empty_actual_task_count = 0

    for row in iter_jsonl(input_path):
        if max_samples > 0 and row_count >= max_samples:
            break

        refined = row.get("refined_fields") or {}
        actual_task_raw = first_non_empty(row.get("actual_task_original"), row.get("actual_task"))
        actual_task_filled = first_non_empty(row.get("actual_task_filled"))
        actual_task_effective = first_non_empty(
            row.get("actual_task_effective"),
            refined.get("actual_task_short"),
            actual_task_filled,
            actual_task_raw,
        )
        thought_raw = first_non_empty(row.get("thought"))
        reflection_raw = first_non_empty(row.get("reflection"))
        thought_short = first_non_empty(refined.get("thought_short"), thought_raw)
        reflection_short = first_non_empty(refined.get("reflection_short"), reflection_raw)

        if actual_task_raw and actual_task_effective and actual_task_effective != actual_task_raw:
            filled_actual_task_count += 1
        if not actual_task_raw and actual_task_effective:
            short_fallback_count += 1
        if not actual_task_effective:
            empty_actual_task_count += 1

        img_next_tokens = ["<img next>"] * img_next_count
        parsed_action = normalize_action(row.get("parsed_action"))
        explicit_supervision = build_stage1_explicit_supervision(
            actual_task=actual_task_effective,
            bbox=row.get("bbox"),
            thought=thought_short,
            reflection=reflection_short,
            img_next_tokens=img_next_tokens,
        )
        stage1_teacher_response = f"Reasoning:\n{explicit_supervision}\n{build_action_text_target(parsed_action)}".strip()

        output_row = dict(row)
        output_row["actual_task_raw"] = actual_task_raw
        output_row["thought_raw"] = thought_raw
        output_row["reflection_raw"] = reflection_raw
        output_row["explicit_supervision_raw"] = row.get("explicit_supervision")
        output_row["actual_task"] = actual_task_effective
        output_row["thought"] = thought_short
        output_row["reflection"] = reflection_short
        output_row["img_next"] = img_next_tokens
        output_row["parsed_action"] = parsed_action
        output_row["task"] = str(output_row.get("instruction", "")).strip()
        output_row["current_subtask"] = actual_task_effective
        output_row["predicted_next_screen_desc"] = reflection_short
        output_row["expected_next_screen"] = reflection_short
        output_row["gold_action"] = parsed_action
        output_row["explicit_reasoning"] = explicit_supervision
        output_row["explicit_supervision"] = explicit_supervision
        output_row["stage1_teacher_response"] = stage1_teacher_response
        output_row["stage1_format"] = {
            "reasoning_fields": ["actual_task", "bbox", "thought", "reflection", "img_next"],
            "action_supervision": "separate_gold_action_text",
            "instruction_kept_in_user_prompt": True,
        }

        append_jsonl(out_path, output_row)
        row_count += 1

        if row_count % 20 == 0:
            print(
                json.dumps(
                    {
                        "stage": "build_lara_stage1_training_dataset",
                        "processed_rows": row_count,
                        "sample_id": output_row.get("sample_id"),
                    },
                    ensure_ascii=False,
                )
            )

    summary = {
        "input": str(input_path),
        "output": str(out_path),
        "summary_out": str(summary_path),
        "row_count": row_count,
        "img_next_count": img_next_count,
        "filled_or_compressed_actual_task_count": filled_actual_task_count,
        "missing_original_but_effective_present_count": short_fallback_count,
        "empty_actual_task_effective_count": empty_actual_task_count,
        "format_note": (
            "instruction stays as outer prompt input; explicit_supervision contains only "
            "actual_task / bbox / thought / reflection / repeated <img next>."
        ),
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
