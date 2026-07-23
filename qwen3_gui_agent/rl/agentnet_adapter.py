from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import re

from qwen3_gui_agent.rl.schema import append_jsonl, save_json


def adapt_agentnet_cli(input_path: Path, out_path: Path, limit: int | None = None) -> None:
    adapter = AgentNetReasoningAdapter(input_path=input_path, out_path=out_path, limit=limit)
    adapter.run()


class AgentNetReasoningAdapter:
    def __init__(self, input_path: Path, out_path: Path, limit: int | None = None) -> None:
        self.input_path = input_path
        self.out_path = out_path
        self.limit = limit
        self.summary_path = out_path.with_suffix(".summary.json")

    def run(self) -> None:
        if self.out_path.exists():
            self.out_path.unlink()

        converted = 0
        skipped = 0
        seen_rows = 0
        for row in self._iter_rows(self.input_path):
            seen_rows += 1
            if self.limit and seen_rows > self.limit:
                break
            samples = self._convert_row(row, converted + 1)
            if not samples:
                skipped += 1
                continue
            for sample in samples:
                append_jsonl(self.out_path, sample)
                converted += 1
            if seen_rows % 100 == 0:
                print(f"processed rows={seen_rows}, converted_steps={converted}, skipped_rows={skipped}")

        save_json(
            self.summary_path,
            {
                "input_path": str(self.input_path),
                "output_path": str(self.out_path),
                "source_rows": seen_rows,
                "converted": converted,
                "skipped": skipped,
                "note": (
                    "This adapter normalizes AgentNet-like reasoning trajectories into a simple step-level format. "
                    "It supports both flat rows and task rows with nested traj steps."
                ),
            },
        )
        print(f"adapted AgentNet-style rows: {converted}")
        print(f"skipped rows: {skipped}")
        print(f"saved adapted data: {self.out_path}")

    def _convert_row(self, row: dict[str, Any], index: int) -> list[dict[str, Any]]:
        traj = row.get("traj")
        if isinstance(traj, list) and traj:
            return self._convert_traj_row(row, index)
        flat = self._convert_flat_row(row, index)
        return [flat] if flat else []

    def _convert_flat_row(self, row: dict[str, Any], index: int) -> dict[str, Any] | None:
        task = _first_non_empty(
            row.get("task"),
            row.get("instruction"),
            _extract_instruction_from_messages(row.get("messages")),
            _extract_instruction_from_messages(row.get("conversation")),
        )
        reasoning = _first_non_empty(
            row.get("explicit_reasoning"),
            row.get("reasoning"),
            row.get("reasoning_content"),
            _extract_reasoning_from_messages(row.get("messages")),
            _extract_reasoning_from_messages(row.get("conversation")),
        )
        action = row.get("action") or row.get("gold_action") or row.get("target_action")
        before_screenshot = _first_non_empty(
            row.get("before_screenshot"),
            row.get("screenshot"),
            row.get("image"),
            _extract_first_image(row.get("images")),
        )
        after_screenshot = _first_non_empty(
            row.get("after_screenshot"),
            row.get("next_screenshot"),
            row.get("target_image"),
            before_screenshot,
        )

        if not task or not reasoning or not isinstance(action, dict):
            return None

        return {
            "sample_id": row.get("sample_id", f"agentnet_{index:06d}"),
            "task": task,
            "before_screenshot": before_screenshot,
            "after_screenshot": after_screenshot,
            "screen_size": row.get("screen_size") or row.get("resolution") or row.get("metadata", {}).get("others", {}).get("resolution"),
            "temporal_anchor": row.get("temporal_anchor", ""),
            "current_subtask": row.get("current_subtask", ""),
            "explicit_reasoning": reasoning,
            "predicted_next_screen_desc": row.get("predicted_next_screen_desc", ""),
            "semantic_anchors": row.get("semantic_anchors", []),
            "gold_action": action,
        }

    def _convert_traj_row(self, row: dict[str, Any], index: int) -> list[dict[str, Any]]:
        task = _first_non_empty(
            row.get("instruction"),
            row.get("natural_language_task"),
            row.get("actual_task"),
            row.get("task"),
        )
        task_id = _first_non_empty(row.get("task_id"), row.get("sample_id"), f"agentnet_{index:06d}")
        screen_size = row.get("screen_size") or row.get("resolution") or row.get("metadata", {}).get("others", {}).get("resolution")
        steps: list[dict[str, Any]] = []
        traj = row.get("traj", [])

        for step_index, traj_item in enumerate(traj):
            if not isinstance(traj_item, dict):
                continue
            value = traj_item.get("value", {})
            if not isinstance(value, dict):
                continue
            action = _parse_action_code(str(value.get("code", "")))
            if not action:
                continue
            before_image = _first_non_empty(traj_item.get("image"), traj_item.get("screenshot"))
            after_image = before_image
            if step_index + 1 < len(traj) and isinstance(traj[step_index + 1], dict):
                after_image = _first_non_empty(traj[step_index + 1].get("image"), traj[step_index + 1].get("screenshot"), before_image)
            explicit_reasoning = str(value.get("thought", "")).strip()
            predicted_next = _first_non_empty(
                str(value.get("reflection", "")).strip(),
                str(value.get("action", "")).strip(),
            )
            steps.append(
                {
                    "sample_id": f"{task_id}_step_{step_index:04d}",
                    "task": task,
                    "before_screenshot": before_image,
                    "after_screenshot": after_image,
                    "screen_size": screen_size,
                    "temporal_anchor": "",
                    "current_subtask": str(value.get("action", "")).strip(),
                    "explicit_reasoning": explicit_reasoning,
                    "predicted_next_screen_desc": predicted_next,
                    "semantic_anchors": [],
                    "gold_action": action,
                    "task_completed": bool(row.get("task_completed", False)),
                    "alignment_score": row.get("alignment_score"),
                    "efficiency_score": row.get("efficiency_score"),
                    "task_difficulty": row.get("task_difficulty"),
                }
            )
        return steps

    @staticmethod
    def _iter_rows(path: Path):
        with path.open("r", encoding="utf-8-sig") as handle:
            first_nonempty = ""
            while True:
                probe = handle.readline()
                if not probe:
                    return
                stripped = probe.strip()
                if stripped:
                    first_nonempty = stripped
                    break

            if first_nonempty.startswith("["):
                handle.seek(0)
                payload = json.load(handle)
                for row in payload:
                    if isinstance(row, dict):
                        yield row
                return

            buffer = first_nonempty
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(buffer)
                    if isinstance(row, dict):
                        yield row
                    buffer = line
                    continue
                except json.JSONDecodeError:
                    pass

                candidate = f"{buffer}\n{line}"
                try:
                    row = json.loads(candidate)
                    if isinstance(row, dict):
                        yield row
                    buffer = ""
                except json.JSONDecodeError:
                    buffer = candidate

            if buffer.strip():
                row = json.loads(buffer)
                if isinstance(row, dict):
                    yield row


def _extract_instruction_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        for content in message.get("content", []):
            if isinstance(content, dict) and content.get("type") == "text":
                text = str(content.get("text", "")).strip()
                if text:
                    return text
    return ""


def _extract_reasoning_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        value = str(message.get("reasoning_content", "")).strip()
        if value:
            return value
    return ""


def _extract_first_image(images: Any) -> str:
    if isinstance(images, list) and images:
        return str(images[0])
    return ""


def _first_non_empty(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _join_non_empty(*values: str) -> str:
    return " ".join(value for value in values if value)


def _parse_action_code(code: str) -> dict[str, Any] | None:
    code = code.strip()
    if not code:
        return None

    click_match = re.search(r"pyautogui\.(click|doubleClick|rightClick)\s*\((.*?)\)", code, flags=re.I)
    if click_match:
        action_type = {
            "click": "click",
            "doubleclick": "double_click",
            "rightclick": "right_click",
        }[click_match.group(1).lower()]
        args = _parse_named_args(click_match.group(2))
        action: dict[str, Any] = {"type": action_type, "button": args.get("button")}
        _attach_point_args(action, args)
        return action

    hotkey_match = re.search(r"pyautogui\.hotkey\s*\((.*?)\)", code, flags=re.I)
    if hotkey_match:
        raw = hotkey_match.group(1).strip()
        quoted = re.findall(r"['\"](.*?)['\"]", raw)
        if quoted:
            keys = quoted
        elif raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1]
            keys = [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
        else:
            keys = [item.strip().strip("'\"") for item in raw.split(",") if item.strip()]
        return {"type": "hotkey", "keys": keys}

    write_match = re.search(r"pyautogui\.(write|typewrite)\s*\((.*?)\)", code, flags=re.I)
    if write_match:
        args = _parse_named_args(write_match.group(2))
        if "message" in args:
            return {"type": "type", "text": str(args["message"])}
        text_match = re.search(r"['\"](.*?)['\"]", write_match.group(2))
        return {"type": "type", "text": text_match.group(1) if text_match else ""}

    press_match = re.search(r"pyautogui\.press\s*\((.*?)\)", code, flags=re.I)
    if press_match:
        key_match = re.search(r"['\"](.*?)['\"]", press_match.group(1))
        key = key_match.group(1) if key_match else ""
        return {"type": "hotkey", "keys": [key]} if key else None

    scroll_match = re.search(r"pyautogui\.scroll\s*\((.*?)\)", code, flags=re.I)
    if scroll_match:
        args = _parse_named_args(scroll_match.group(1))
        amount = args.get("_positional_0") or args.get("clicks") or args.get("amount")
        return {"type": "scroll", "amount": _maybe_int(amount)}

    wait_match = re.search(r"(?:computer|pyautogui)\.(?:wait|sleep)\s*\((.*?)\)", code, flags=re.I)
    if wait_match:
        return {"type": "wait", "status": "success"}

    terminate_match = re.search(r"computer\.terminate\s*\((.*?)\)", code, flags=re.I)
    if terminate_match:
        args = _parse_named_args(terminate_match.group(1))
        status = str(args.get("status", "success")).strip() or "success"
        return {"type": "terminate", "status": status}

    return None


def _parse_named_args(arg_text: str) -> dict[str, Any]:
    parts = [part.strip() for part in arg_text.split(",") if part.strip()]
    result: dict[str, Any] = {}
    positional_index = 0
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            result[key.strip()] = _parse_scalar(value.strip())
        else:
            result[f"_positional_{positional_index}"] = _parse_scalar(part)
            positional_index += 1
    return result


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except Exception:
        return value


def _attach_point_args(action: dict[str, Any], args: dict[str, Any]) -> None:
    x = args.get("x", args.get("_positional_0"))
    y = args.get("y", args.get("_positional_1"))
    if isinstance(x, (int, float)):
        if 0 <= float(x) <= 1:
            action["x_norm"] = float(x)
        else:
            action["x"] = int(float(x))
    if isinstance(y, (int, float)):
        if 0 <= float(y) <= 1:
            action["y_norm"] = float(y)
        else:
            action["y"] = int(float(y))


def _maybe_int(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    return None
