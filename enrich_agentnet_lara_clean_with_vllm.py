from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from openai import OpenAI

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


SCHEMA_VERSION = "agentnet_lara_clean_enriched_v1"
_CLIENT_LOCAL = threading.local()
FIELD_WORD_LIMITS = {"actual_task": 12, "thought": 18, "reflection": 18}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use an OpenAI-compatible Qwen3-VL server to refine AgentNet step-level "
            "actual_task, thought, and reflection fields."
        )
    )
    parser.add_argument("--steps", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen3-vl-8b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--progress-out", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0, help="<=0 means all selected rows")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=0, help="Exclusive; <=0 means end of input")
    parser.add_argument("--shard-count", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--no-json-mode",
        action="store_true",
        help="Disable the OpenAI-compatible JSON response constraint.",
    )
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=20)
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


def count_jsonl(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_slice(
    row_count: int,
    start_index: int,
    end_index: int,
    shard_count: int,
    shard_index: int,
) -> tuple[int, int]:
    start = max(0, int(start_index))
    end = row_count if int(end_index) <= 0 else min(row_count, int(end_index))
    if end < start:
        raise ValueError(f"Invalid slice: start={start}, end={end}")
    if int(shard_count) > 0:
        count = int(shard_count)
        index = int(shard_index)
        if index < 0 or index >= count:
            raise ValueError(f"shard_index must be in [0, {count - 1}]")
        span = end - start
        return start + span * index // count, start + span * (index + 1) // count
    return start, end


def normalize_base_url(base_url: str) -> str:
    normalized = str(base_url).rstrip("/")
    return normalized if normalized.endswith("/v1") else normalized + "/v1"


def get_client(args: argparse.Namespace) -> OpenAI:
    client = getattr(_CLIENT_LOCAL, "client", None)
    if client is None:
        client = OpenAI(
            api_key=str(args.api_key),
            base_url=normalize_base_url(str(args.base_url)),
            timeout=float(args.request_timeout),
            max_retries=0,
        )
        _CLIENT_LOCAL.client = client
    return client


def resolve_image(dataset_root: Path, image_name: str) -> Path:
    normalized = str(image_name or "").strip()
    candidates = (
        dataset_root / "ubuntu_images" / normalized,
        dataset_root / normalized,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve image: {normalized}")


def build_prompt(row: dict[str, Any], *, compact_retry: bool = False) -> str:
    source_thought = str(row.get("thought_raw", "") or "").strip()
    source_reflection = str(row.get("reflection_raw", "") or "").strip()
    if compact_retry:
        source_thought = "[omitted after an overlong or malformed response]"
        source_reflection = "[omitted after an overlong or malformed response]"
    return "\n".join(
        [
            "You are creating compact supervision for one GUI-agent trajectory step.",
            "Image 1 is the GUI before the action. Image 2 is the GUI after the action.",
            "Rewrite exactly three step-level reasoning fields while preserving the evidence.",
            "Return one JSON object only with this exact schema:",
            '{"actual_task":"...","thought":"...","reflection":"..."}',
            "Requirements:",
            "- actual_task: the concrete current GUI subtask, at most 12 words.",
            "- thought: the essential reason for choosing this action, at most 18 words.",
            "- reflection: the observed GUI change or action result, at most 18 words.",
            "- The word limits are mandatory. Summarize; never copy long source passages.",
            "- End immediately after the JSON object's closing brace.",
            "- Use the screenshots to correct unsupported or overly broad source text.",
            "- Preserve application names, menu labels, typed text, and completion status when relevant.",
            "- Do not include coordinates, point values, bounding boxes, bbox, action JSON, or code.",
            "- Do not shorten or rewrite the episode instruction; it is context only.",
            "- Do not invent UI elements or results not supported by the two screenshots.",
            (
                "- RETRY MODE: Return short phrases only; the previous response was "
                "overlong or malformed."
                if compact_retry
                else "- Use concise phrases rather than copying source sentences."
            ),
            "",
            f"episode_instruction: {str(row.get('instruction', '') or '').strip()}",
            f"episode_task_context: {str(row.get('episode_actual_task', '') or '').strip()}",
            f"current_step_seed: {str(row.get('actual_task_seed', '') or '').strip()}",
            f"source_action_description: {str(row.get('action_text_raw', '') or '').strip()}",
            f"source_thought: {source_thought}",
            f"source_reflection: {source_reflection}",
            f"source_action_code: {str(row.get('code', '') or '').strip()}",
        ]
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.I | re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
        for position, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(candidate[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    preview = re.sub(r"\s+", " ", stripped)[:500]
    raise ValueError(f"JSON object not found in model response; raw_preview={preview!r}")


def compact_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text.strip('"').strip()


def validate_refined_fields(payload: dict[str, Any]) -> dict[str, str]:
    fields = {
        "actual_task": compact_text(payload.get("actual_task")),
        "thought": compact_text(payload.get("thought")),
        "reflection": compact_text(payload.get("reflection")),
    }
    empty = [key for key, value in fields.items() if not value]
    if empty:
        raise ValueError(f"Empty refined fields: {empty}")
    forbidden = re.compile(r"\b(?:bbox|bounding box|x_norm|y_norm|point\s*:|pyautogui)\b", re.I)
    bad = [key for key, value in fields.items() if forbidden.search(value)]
    if bad:
        raise ValueError(f"Forbidden coordinate/action syntax in refined fields: {bad}")
    overlong = {
        key: len(value.split())
        for key, value in fields.items()
        if len(value.split()) > FIELD_WORD_LIMITS[key]
    }
    if overlong:
        raise ValueError(
            f"Refined fields exceed hard word limits: {overlong}; limits={FIELD_WORD_LIMITS}"
        )
    return fields


def action_text(action: dict[str, Any]) -> str:
    action_type = str(action.get("type", "wait") or "wait")
    lines = [f"Action: {action_type}"]
    if action_type in {"click", "double_click", "right_click"}:
        x = int(round(float(action.get("x_norm", 0.5)) * 1000.0))
        y = int(round(float(action.get("y_norm", 0.5)) * 1000.0))
        lines.append(f"Point: [{x} {y}]")
    elif action_type == "type":
        lines.append(f'Text: {json.dumps(str(action.get("text", "")), ensure_ascii=False)}')
    elif action_type == "hotkey":
        lines.append("Keys: [" + ", ".join(str(key) for key in action.get("keys", [])) + "]")
    elif action_type == "scroll":
        lines.append(f"Amount: {int(action.get('amount', 0) or 0)}")
    elif action_type in {"terminate", "wait"}:
        lines.append(f"Status: {str(action.get('status', 'success') or 'success')}")
    return "\n".join(lines)


def enrich_once(
    row: dict[str, Any],
    args: argparse.Namespace,
    dataset_root: Path,
    *,
    retry_index: int = 0,
) -> dict[str, Any]:
    before = resolve_image(dataset_root, str(row.get("before_screenshot", "")))
    after = resolve_image(dataset_root, str(row.get("after_screenshot", "")))
    content = [
        {"type": "image_url", "image_url": {"url": before.as_uri()}},
        {"type": "image_url", "image_url": {"url": after.as_uri()}},
        {"type": "text", "text": build_prompt(row, compact_retry=retry_index > 0)},
    ]
    request: dict[str, Any] = {
        "model": str(args.model),
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": int(args.max_new_tokens),
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    if not bool(args.no_json_mode):
        request["response_format"] = {"type": "json_object"}
    response = get_client(args).chat.completions.create(**request)
    raw_text = response.choices[0].message.content or ""
    refined = validate_refined_fields(extract_json_object(raw_text))

    output = dict(row)
    output["schema_version"] = SCHEMA_VERSION
    output["actual_task"] = refined["actual_task"]
    output["thought"] = refined["thought"]
    output["reflection"] = refined["reflection"]
    output["refined_fields"] = dict(refined)
    output["enrich_status"] = "ok"
    output["enrich_error"] = None
    output["enrich_raw_response_text"] = raw_text

    img_next = list(output.get("img_next") or [])
    explicit_reasoning = "\n".join(
        [
            f"actual_task: {refined['actual_task']}",
            f"thought: {refined['thought']}",
            f"reflection: {refined['reflection']}",
            " ".join(str(token) for token in img_next),
        ]
    ).strip()
    gold_action = dict(output.get("gold_action") or output.get("parsed_action") or {})
    output["task"] = str(output.get("instruction", "") or "").strip()
    output["current_subtask"] = refined["actual_task"]
    output["predicted_next_screen_desc"] = refined["reflection"]
    output["expected_next_screen"] = refined["reflection"]
    output["explicit_reasoning"] = explicit_reasoning
    output["explicit_supervision"] = explicit_reasoning
    output["stage1_teacher_response"] = (
        f"Reasoning:\n{explicit_reasoning}\n{action_text(gold_action)}"
    ).strip()
    output["stage1_format"] = {
        "reasoning_fields": ["actual_task", "thought", "reflection", "img_next"],
        "bbox_in_reasoning": False,
        "coordinates_only_in_gold_action": True,
        "instruction_kept_in_user_prompt": True,
    }
    return output


def enrich_with_retries(
    row: dict[str, Any], args: argparse.Namespace, dataset_root: Path
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(max(0, int(args.max_retries)) + 1):
        try:
            return enrich_once(row, args, dataset_root, retry_index=attempt)
        except Exception as exc:  # Network and format failures are retried together.
            last_error = exc
            if attempt < int(args.max_retries):
                time.sleep(min(30.0, 2.0**attempt))
    output = dict(row)
    output["schema_version"] = SCHEMA_VERSION
    output["enrich_status"] = "error"
    output["enrich_error"] = f"{type(last_error).__name__}: {last_error}"
    return output


def repair_existing_output(path: Path) -> tuple[int, int]:
    """Keep the valid ordered prefix so --resume retries the first bad row."""
    if not path.exists():
        return 0, 0
    valid_lines: list[str] = []
    repair_reason: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                repair_reason = f"invalid JSON at line {line_number}: {exc}"
                break
            if not isinstance(row, dict):
                repair_reason = f"non-object JSON at line {line_number}"
                break
            if row.get("enrich_status") != "ok":
                repair_reason = (
                    f"failed enrichment at line {line_number}: "
                    f"{row.get('enrich_error')}"
                )
                break
            valid_lines.append(stripped + "\n")

    if repair_reason is not None:
        temp_path = path.with_name(path.name + ".resume.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.writelines(valid_lines)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        print(
            json.dumps(
                {
                    "stage": "repair_enrichment_resume_prefix",
                    "path": str(path),
                    "kept_rows": len(valid_lines),
                    "reason": repair_reason,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return len(valid_lines), 0


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def save_progress(
    path: Path,
    args: argparse.Namespace,
    slice_start: int,
    slice_end: int,
    completed_rows: int,
    error_count: int,
    elapsed_seconds: float,
) -> None:
    atomic_json(
        path,
        {
            "steps": str(Path(args.steps).resolve()),
            "out": str(Path(args.out).resolve()),
            "slice_start": slice_start,
            "slice_end": slice_end,
            "completed_rows": completed_rows,
            "next_global_index": slice_start + completed_rows,
            "error_count": error_count,
            "elapsed_seconds": elapsed_seconds,
        },
    )


def main() -> None:
    args = parse_args()
    steps_path = Path(args.steps).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.out).resolve()
    summary_path = (
        Path(args.summary_out).resolve()
        if args.summary_out
        else output_path.with_suffix(".summary.json")
    )
    progress_path = (
        Path(args.progress_out).resolve()
        if args.progress_out
        else output_path.with_suffix(".progress.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_rows = count_jsonl(steps_path)
    slice_start, slice_end = resolve_slice(
        input_rows,
        int(args.start_index),
        int(args.end_index),
        int(args.shard_count),
        int(args.shard_index),
    )
    if int(args.max_samples) > 0:
        slice_end = min(slice_end, slice_start + int(args.max_samples))
    selected_rows = slice_end - slice_start
    if selected_rows <= 0:
        raise RuntimeError("No rows selected for enrichment")

    if not args.resume:
        if output_path.exists():
            raise FileExistsError(f"Output exists; use --resume or choose another path: {output_path}")
        if progress_path.exists():
            progress_path.unlink()
    completed_rows, error_count = (
        repair_existing_output(output_path) if args.resume else (0, 0)
    )
    if completed_rows > selected_rows:
        raise RuntimeError(
            f"Existing output has {completed_rows} rows but this slice has only {selected_rows}"
        )

    source_iterator = itertools.islice(
        iter_jsonl(steps_path),
        slice_start + completed_rows,
        slice_end,
    )
    progress_bar = (
        tqdm(
            total=selected_rows,
            initial=completed_rows,
            desc="enrich_agentnet_lara_clean",
            unit="step",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )
    start_time = time.time()
    mode = "a" if completed_rows > 0 else "w"
    concurrency = max(1, int(args.concurrency))

    with output_path.open(mode, encoding="utf-8") as output_handle, ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        while True:
            batch = list(itertools.islice(source_iterator, concurrency))
            if not batch:
                break
            results = list(
                executor.map(
                    lambda row: enrich_with_retries(row, args, dataset_root),
                    batch,
                )
            )
            batch_error_count = 0
            for result in results:
                output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                completed_rows += 1
                if result.get("enrich_status") != "ok":
                    error_count += 1
                    batch_error_count += 1
            output_handle.flush()

            elapsed = time.time() - start_time
            if progress_bar is not None:
                progress_bar.update(len(results))
                progress_bar.set_postfix(errors=error_count, workers=concurrency)
            if completed_rows % max(1, int(args.log_every)) < len(results):
                print(
                    json.dumps(
                        {
                            "stage": "enrich_agentnet_lara_clean",
                            "completed_rows": completed_rows,
                            "selected_rows": selected_rows,
                            "error_count": error_count,
                            "elapsed_seconds": elapsed,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if completed_rows % max(1, int(args.save_every)) < len(results):
                save_progress(
                    progress_path,
                    args,
                    slice_start,
                    slice_end,
                    completed_rows,
                    error_count,
                    elapsed,
                )
            if batch_error_count:
                print(
                    json.dumps(
                        {
                            "stage": "enrichment_stopped_for_retry",
                            "completed_rows": completed_rows,
                            "batch_error_count": batch_error_count,
                            "message": "Rerun with --resume to retry from the first failed row.",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break

    if progress_bar is not None:
        progress_bar.close()
    elapsed = time.time() - start_time
    save_progress(
        progress_path,
        args,
        slice_start,
        slice_end,
        completed_rows,
        error_count,
        elapsed,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "steps": str(steps_path),
        "dataset_root": str(dataset_root),
        "base_url": normalize_base_url(str(args.base_url)),
        "model": str(args.model),
        "output": str(output_path),
        "input_row_count": input_rows,
        "slice_start": slice_start,
        "slice_end": slice_end,
        "selected_row_count": selected_rows,
        "completed_row_count": completed_rows,
        "error_count": error_count,
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "reasoning_fields": ["actual_task", "thought", "reflection"],
        "bbox_in_reasoning": False,
    }
    atomic_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if error_count:
        raise SystemExit(f"Enrichment completed with {error_count} failed rows")


if __name__ == "__main__":
    main()
