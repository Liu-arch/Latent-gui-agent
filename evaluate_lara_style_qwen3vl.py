from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from qwen3_gui_agent.evaluation_utils import (
    build_step_analysis_row,
    coerce_norm_scalar,
    finalize_metric_accumulator,
    infer_trajectory_key,
    init_metric_accumulator,
    normalize_metric_accumulator_for_resume,
    normalize_action_dict,
    region_from_action,
    resolve_dataset_image,
    resolve_x_norm,
    resolve_y_norm,
    summarize_latencies,
    update_metric_accumulator,
)
from qwen3_gui_agent.lara_style_qwen3vl_agent import LaRAStyleQwen3VLAgent
from qwen3_gui_agent.rl.agentnet_adapter import _parse_action_code
from qwen3_gui_agent.rl.schema import read_jsonl
from qwen3_gui_agent.training_utils import resolve_device_map, resolve_torch_dtype

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LaRA-style Qwen3-VL latent reasoning agent with official-compatible metrics."
    )
    parser.add_argument("--steps", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--history-n", type=int, default=5)
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=0,
        help="Optional Qwen3-VL processor min_pixels override. 0 uses adapter metadata or model default.",
    )
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=0,
        help="Optional Qwen3-VL processor max_pixels override. 0 uses adapter metadata or model default.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--action-format",
        choices=["auto", "text", "action_tokens"],
        default="auto",
        help="Decode generated actions as legacy text or GUI action tokens. auto uses adapter metadata.",
    )
    parser.add_argument(
        "--action-model",
        choices=["auto", "unified", "flow_matching", "latent_two_way"],
        default="auto",
        help="Action expert used with --use-action-head. auto uses adapter metadata.",
    )
    parser.add_argument(
        "--flow-action-sample-steps",
        type=int,
        default=0,
        help="Flow integration steps. 0 inherits the adapter metadata.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--progress-out", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--step-out", default=None)
    parser.add_argument("--step-out-every", type=int, default=1)
    parser.add_argument("--report-out", default=None)
    parser.add_argument(
        "--use-action-head",
        action="store_true",
        help="Use the hidden-state action head instead of autoregressive text generation.",
    )
    return parser.parse_args()


_ADAPTER_EVAL_CONFIG_KEYS = (
    "history_n",
    "latent_slot_count",
    "pixel_prune_threshold",
    "pixel_prune_predictor_order",
    "pixel_temporal_reuse",
    "pixel_temporal_threshold",
    "image_min_pixels",
    "image_max_pixels",
    "training_stage",
    "stage2_target_format",
    "action_format",
    "action_model",
    "action_coord_bins",
    "lm_action_target",
    "flow_action_sample_steps",
    "flow_head_hidden_dim",
    "flow_head_depth",
    "two_way_hidden_dim",
    "two_way_depth",
    "two_way_num_heads",
    "two_way_location_queries",
    "two_way_dropout",
    "two_way_candidate_coord_loss_weight",
    "two_way_candidate_confidence_loss_weight",
    "flow_continuous_source",
    "flow_action_loss_weight",
    "flow_coord_loss_weight",
    "flow_coord_loss_scale",
    "flow_coord_loss_space",
    "flow_patch_loss_weight",
    "flow_patch_loss_mode",
    "flow_patch_gaussian_sigma",
    "flow_pointer_coord_source",
    "flow_patch_logit_temperature",
    "flow_patch_residual_scale",
    "action_hidden_source",
    "action_only_output",
    "use_action_head",
    "clean_observable_prompt",
    "include_current_subtask_in_prompt",
    "include_expected_next_screen_in_prompt",
    "latent_scaffolds_in_prompt",
    "use_lora",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
)


def _extract_adapter_eval_config(payload: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    extra_metadata = payload.get("extra_metadata")
    if isinstance(extra_metadata, dict):
        candidates.append(extra_metadata)

    extra_state = payload.get("extra_state")
    if isinstance(extra_state, dict):
        for key in ("extra_metadata", "adapter_extra_metadata", "metadata"):
            nested = extra_state.get(key)
            if isinstance(nested, dict):
                candidates.append(nested)
        candidates.append(extra_state)

    checkpoint_args = payload.get("args")
    if isinstance(checkpoint_args, argparse.Namespace):
        checkpoint_args = vars(checkpoint_args)
    if isinstance(checkpoint_args, dict):
        candidates.append(checkpoint_args)

    config: dict[str, Any] = {}
    for candidate in candidates:
        for key in _ADAPTER_EVAL_CONFIG_KEYS:
            if key in candidate and candidate[key] is not None:
                config[key] = candidate[key]
    return config


def load_adapter_eval_config(adapter_path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    meta_path = adapter_path.with_suffix(adapter_path.suffix + ".json")
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if isinstance(payload, dict):
            config.update(_extract_adapter_eval_config(payload))

    # Training checkpoints have a small sidecar json without full args. If the
    # sidecar did not contain runtime config, fall back to the actual checkpoint.
    try:
        import torch

        payload = torch.load(adapter_path, map_location="cpu")
    except Exception:
        return config
    if isinstance(payload, dict):
        config.update(_extract_adapter_eval_config(payload))
    return config


def resolve_progress_path(report_out: str | None, progress_out: str | None, resume_from: str | None) -> Path | None:
    if progress_out:
        return Path(progress_out)
    if resume_from:
        return Path(resume_from)
    if report_out:
        return Path(report_out).with_suffix(Path(report_out).suffix + ".progress.json")
    return None


def save_progress(
    *,
    progress_path: Path | None,
    args: argparse.Namespace,
    next_step_index: int,
    total_planned_steps: int,
    latencies: list[float],
    metrics: dict[str, Any],
    history: list[dict[str, Any]],
    parse_error_count: int,
    elapsed_seconds: float,
) -> None:
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "steps_path": str(args.steps),
        "adapter": str(args.adapter),
        "history_n": int(args.history_n),
        "image_min_pixels": int(args.image_min_pixels),
        "image_max_pixels": int(args.image_max_pixels),
        "max_new_tokens": int(args.max_new_tokens),
        "action_format": str(args.action_format),
        "action_model": str(args.action_model),
        "use_action_head": bool(args.use_action_head),
        "next_step_index": int(next_step_index),
        "total_planned_steps": int(total_planned_steps),
        "latencies": list(latencies),
        "metrics": metrics,
        "history": history,
        "parse_error_count": int(parse_error_count),
        "elapsed_seconds": float(elapsed_seconds),
    }
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_progress(progress_path: Path, args: argparse.Namespace, total_planned_steps: int) -> dict[str, Any]:
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    if str(payload.get("steps_path")) != str(args.steps):
        raise RuntimeError("Resume progress steps_path mismatch.")
    if str(payload.get("adapter")) != str(args.adapter):
        raise RuntimeError("Resume progress adapter mismatch.")
    if int(payload.get("history_n", -1)) != int(args.history_n):
        raise RuntimeError("Resume progress history_n mismatch.")
    if int(payload.get("image_min_pixels", 0)) != int(args.image_min_pixels):
        raise RuntimeError("Resume progress image_min_pixels mismatch.")
    if int(payload.get("image_max_pixels", 0)) != int(args.image_max_pixels):
        raise RuntimeError("Resume progress image_max_pixels mismatch.")
    if int(payload.get("max_new_tokens", -1)) != int(args.max_new_tokens):
        raise RuntimeError("Resume progress max_new_tokens mismatch.")
    if str(payload.get("action_format", "auto")) != str(args.action_format):
        raise RuntimeError("Resume progress action_format mismatch.")
    if str(payload.get("action_model", "auto")) != str(args.action_model):
        raise RuntimeError("Resume progress action_model mismatch.")
    if bool(payload.get("use_action_head", False)) != bool(args.use_action_head):
        raise RuntimeError("Resume progress use_action_head mismatch.")
    if int(payload.get("total_planned_steps", -1)) != int(total_planned_steps):
        raise RuntimeError("Resume progress total_planned_steps mismatch.")
    return payload


def build_flat_eval_samples(trajectories: list[dict[str, Any]], history_slots: int) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for trajectory_index, trajectory in enumerate(trajectories, start=1):
        history_paths: list[str] = []
        for row in trajectory["rows"]:
            flat.append(
                {
                    "trajectory_index": trajectory_index,
                    "trajectory_key": trajectory["trajectory_key"],
                    "row": row,
                    "history_image_paths": list(history_paths[-history_slots:]),
                }
            )
            history_paths.append(str(row["image_path"]))
    return flat


def extract_eval_action(step: dict[str, Any]) -> dict[str, Any]:
    parsed_action = step.get("parsed_action")
    if isinstance(parsed_action, dict) and parsed_action:
        return dict(parsed_action)
    gold_action = step.get("gold_action")
    if isinstance(gold_action, dict) and gold_action:
        return dict(gold_action)
    action = step.get("action")
    if isinstance(action, dict) and action:
        return dict(action)
    code = str(step.get("code", "") or "").strip()
    if code:
        parsed = _parse_action_code(code)
        if isinstance(parsed, dict) and parsed:
            return dict(parsed)
    bbox = step.get("bbox")
    if isinstance(bbox, list) and len(bbox) >= 2:
        x_value = coerce_norm_scalar(bbox[0])
        y_value = coerce_norm_scalar(bbox[1])
        if x_value is not None and y_value is not None:
            return {"type": "click", "x_norm": x_value, "y_norm": y_value}
    return {"type": "wait", "status": "success"}


def build_lara_eval_trajectories(
    *,
    steps: list[dict[str, Any]],
    dataset_root: Path,
    max_samples: int,
) -> list[dict[str, Any]]:
    trajectories: list[dict[str, Any]] = []
    current_rows: list[dict[str, Any]] = []
    current_key: str | None = None
    total_rows = 0

    for step in steps:
        image_name = str(step.get("before_screenshot", "") or "").strip()
        if not image_name:
            continue
        image_path = resolve_dataset_image(dataset_root, image_name)
        if image_path is None:
            continue

        trajectory_key = infer_trajectory_key(step)
        if current_key is None:
            current_key = trajectory_key
        elif trajectory_key != current_key:
            if current_rows:
                trajectories.append({"trajectory_key": current_key, "task": current_rows[0]["task"], "rows": current_rows})
            current_rows = []
            current_key = trajectory_key

        gold_action = extract_eval_action(step)
        region_name = region_from_action(gold_action)
        instruction = str(step.get("instruction", step.get("task", "")) or "").strip()
        actual_task = str(step.get("actual_task", step.get("current_subtask", "")) or "").strip()
        row = {
            "trajectory_key": trajectory_key,
            "image_path": image_path,
            "task": instruction,
            "current_subtask": actual_task,
            "expected_next_screen": None,
            "explicit_reasoning": str(step.get("explicit_reasoning", step.get("explicit_supervision_short", "")) or ""),
            "gold_action": gold_action,
            "teacher_action_type_name": str(gold_action.get("type", "wait")),
            "teacher_region_name": region_name,
            "teacher_terminate_status_name": str(gold_action.get("status", "success")),
            "has_pointer_target": gold_action.get("x_norm") is not None and gold_action.get("y_norm") is not None,
            "x_norm": resolve_x_norm(gold_action, region_name),
            "y_norm": resolve_y_norm(gold_action, region_name),
        }
        current_rows.append(row)
        total_rows += 1
        if total_rows >= max_samples:
            break

    if current_rows:
        trajectories.append({"trajectory_key": current_key, "task": current_rows[0]["task"], "rows": current_rows})
    return trajectories


def main() -> None:
    args = parse_args()
    dtype = resolve_torch_dtype(args.torch_dtype)
    device_map = resolve_device_map(args.device_map)
    adapter_config = load_adapter_eval_config(Path(args.adapter))
    resolved_history_n = int(adapter_config.get("history_n", args.history_n))
    resolved_latent_slot_count = int(adapter_config.get("latent_slot_count", 8))
    resolved_pixel_prune_threshold = float(adapter_config.get("pixel_prune_threshold", 0.0))
    resolved_pixel_prune_predictor_order = str(adapter_config.get("pixel_prune_predictor_order", "pred2d,left,up"))
    resolved_pixel_temporal_reuse = bool(adapter_config.get("pixel_temporal_reuse", False))
    resolved_pixel_temporal_threshold = float(adapter_config.get("pixel_temporal_threshold", 0.0))
    resolved_image_min_pixels = int(args.image_min_pixels) if int(args.image_min_pixels) > 0 else int(
        adapter_config.get("image_min_pixels", 0) or 0
    )
    resolved_image_max_pixels = int(args.image_max_pixels) if int(args.image_max_pixels) > 0 else int(
        adapter_config.get("image_max_pixels", 0) or 0
    )
    resolved_action_coord_bins = int(adapter_config.get("action_coord_bins", 1000))
    resolved_action_format = (
        str(adapter_config.get("action_format", "text") or "text")
        if str(args.action_format) == "auto"
        else str(args.action_format)
    )
    resolved_action_model = (
        str(adapter_config.get("action_model", "unified") or "unified")
        if str(args.action_model) == "auto"
        else str(args.action_model)
    )
    resolved_lm_action_target = str(adapter_config.get("lm_action_target", "include") or "include")
    resolved_flow_action_sample_steps = max(
        1,
        int(args.flow_action_sample_steps)
        if int(args.flow_action_sample_steps) > 0
        else int(adapter_config.get("flow_action_sample_steps", 8) or 8),
    )
    resolved_flow_head_hidden_dim = int(adapter_config.get("flow_head_hidden_dim", 0) or 0)
    resolved_flow_head_depth = max(1, int(adapter_config.get("flow_head_depth", 2) or 2))
    resolved_two_way_hidden_dim = max(64, int(adapter_config.get("two_way_hidden_dim", 512) or 512))
    resolved_two_way_depth = max(1, int(adapter_config.get("two_way_depth", 2) or 2))
    resolved_two_way_num_heads = max(1, int(adapter_config.get("two_way_num_heads", 8) or 8))
    resolved_two_way_location_queries = max(
        1,
        int(adapter_config.get("two_way_location_queries", 3) or 3),
    )
    resolved_two_way_dropout = max(0.0, float(adapter_config.get("two_way_dropout", 0.0) or 0.0))
    clean_observable_prompt = bool(adapter_config.get("clean_observable_prompt", False))
    resolved_include_current_subtask = bool(
        adapter_config.get("include_current_subtask_in_prompt", not clean_observable_prompt)
    )
    resolved_include_expected_next_screen = bool(
        adapter_config.get("include_expected_next_screen_in_prompt", not clean_observable_prompt)
    )
    resolved_latent_scaffolds_in_prompt = bool(
        adapter_config.get("latent_scaffolds_in_prompt", not clean_observable_prompt)
    )
    resolved_use_lora = bool(adapter_config.get("use_lora", False))
    resolved_lora_r = int(adapter_config.get("lora_r", 16) or 16)
    resolved_lora_alpha = int(adapter_config.get("lora_alpha", 32) or 32)
    resolved_lora_dropout = float(adapter_config.get("lora_dropout", 0.05) or 0.0)

    print(
        json.dumps(
            {
                "adapter": args.adapter,
                "resolved_history_n": resolved_history_n,
                "resolved_latent_slot_count": resolved_latent_slot_count,
                "resolved_pixel_prune_threshold": resolved_pixel_prune_threshold,
                "resolved_pixel_prune_predictor_order": resolved_pixel_prune_predictor_order,
                "resolved_pixel_temporal_reuse": resolved_pixel_temporal_reuse,
                "resolved_pixel_temporal_threshold": resolved_pixel_temporal_threshold,
                "resolved_image_min_pixels": resolved_image_min_pixels,
                "resolved_image_max_pixels": resolved_image_max_pixels,
                "resolved_action_format": resolved_action_format,
                "resolved_action_model": resolved_action_model,
                "resolved_lm_action_target": resolved_lm_action_target,
                "resolved_flow_action_sample_steps": resolved_flow_action_sample_steps,
                "resolved_flow_head_hidden_dim": resolved_flow_head_hidden_dim,
                "resolved_flow_head_depth": resolved_flow_head_depth,
                "resolved_two_way_hidden_dim": resolved_two_way_hidden_dim,
                "resolved_two_way_depth": resolved_two_way_depth,
                "resolved_two_way_num_heads": resolved_two_way_num_heads,
                "resolved_two_way_location_queries": resolved_two_way_location_queries,
                "resolved_two_way_dropout": resolved_two_way_dropout,
                "resolved_action_coord_bins": resolved_action_coord_bins,
                "resolved_clean_observable_prompt": clean_observable_prompt,
                "resolved_use_lora": resolved_use_lora,
                "adapter_extra_metadata": adapter_config,
            },
            ensure_ascii=False,
        )
    )

    agent = LaRAStyleQwen3VLAgent.from_pretrained(
        args.model,
        device_map=device_map,
        torch_dtype=dtype,
        latent_slot_count=resolved_latent_slot_count,
        pixel_prune_threshold=resolved_pixel_prune_threshold,
        pixel_prune_predictor_order=resolved_pixel_prune_predictor_order,
        pixel_temporal_reuse=resolved_pixel_temporal_reuse,
        pixel_temporal_threshold=resolved_pixel_temporal_threshold,
        action_coord_bins=resolved_action_coord_bins,
        action_model=resolved_action_model,
        flow_action_sample_steps=resolved_flow_action_sample_steps,
        flow_head_hidden_dim=resolved_flow_head_hidden_dim or None,
        flow_head_depth=resolved_flow_head_depth,
        two_way_hidden_dim=resolved_two_way_hidden_dim,
        two_way_depth=resolved_two_way_depth,
        two_way_num_heads=resolved_two_way_num_heads,
        two_way_location_queries=resolved_two_way_location_queries,
        two_way_dropout=resolved_two_way_dropout,
        image_min_pixels=resolved_image_min_pixels or None,
        image_max_pixels=resolved_image_max_pixels or None,
        include_current_subtask_in_prompt=resolved_include_current_subtask,
        include_expected_next_screen_in_prompt=resolved_include_expected_next_screen,
        latent_scaffolds_in_prompt=resolved_latent_scaffolds_in_prompt,
        use_lora=resolved_use_lora,
        lora_r=resolved_lora_r,
        lora_alpha=resolved_lora_alpha,
        lora_dropout=resolved_lora_dropout,
    )
    load_info = agent.load_adapter(args.adapter, strict=False)
    if load_info.get("skipped_shape_mismatch"):
        raise RuntimeError(
            "Evaluation adapter has incompatible tensor shapes; refusing to evaluate partially loaded weights: "
            + json.dumps(load_info["skipped_shape_mismatch"][:8], ensure_ascii=False)
        )
    agent.action_format = resolved_action_format
    agent.action_model = resolved_action_model
    agent.lm_action_target = resolved_lm_action_target
    agent.flow_action_sample_steps = resolved_flow_action_sample_steps
    agent.eval()
    print(json.dumps({"loaded_adapter": args.adapter, "load_info": load_info}, ensure_ascii=False))

    steps = read_jsonl(Path(args.steps))
    trajectories = build_lara_eval_trajectories(
        steps=steps,
        dataset_root=Path(args.dataset_root),
        max_samples=(args.max_samples if int(args.max_samples) > 0 else len(steps)),
    )
    if not trajectories:
        raise RuntimeError("No valid evaluation samples found.")

    history_slots = max(0, int(resolved_history_n) - 1)
    flat_samples = build_flat_eval_samples(trajectories, history_slots)
    total_planned_steps = len(flat_samples)
    progress_path = resolve_progress_path(args.report_out, args.progress_out, args.resume_from)
    step_out_path = Path(args.step_out) if args.step_out else None
    if step_out_path is not None and not args.resume_from:
        step_out_path.parent.mkdir(parents=True, exist_ok=True)
        step_out_path.write_text("", encoding="utf-8")

    latencies: list[float] = []
    metrics = init_metric_accumulator()
    history: list[dict[str, Any]] = []
    parse_error_count = 0
    total_steps = 0
    resumed_elapsed_seconds = 0.0

    if args.resume_from:
        payload = load_progress(Path(args.resume_from), args, total_planned_steps)
        total_steps = int(payload.get("next_step_index", 0))
        latencies = [float(value) for value in payload.get("latencies", [])]
        metrics = normalize_metric_accumulator_for_resume(payload.get("metrics", init_metric_accumulator()))
        history = list(payload.get("history", []))
        parse_error_count = int(payload.get("parse_error_count", 0))
        resumed_elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
        print(
            json.dumps(
                {
                    "stage": "resume_eval",
                    "resume_from": args.resume_from,
                    "next_step_index": total_steps,
                    "completed_steps": total_steps,
                    "total_planned_steps": total_planned_steps,
                },
                ensure_ascii=False,
            )
        )

    progress = (
        tqdm(total=total_planned_steps, desc="eval_lara_style", unit="step", initial=total_steps)
        if tqdm is not None
        else None
    )
    started = time.time()
    try:
        for sample in flat_samples[total_steps:]:
            row = sample["row"]
            trajectory_index = int(sample["trajectory_index"])
            trajectory_key = str(sample["trajectory_key"])
            image_paths = list(sample["history_image_paths"]) + [str(row["image_path"])]
            step_started = time.perf_counter()
            if args.use_action_head:
                result = agent.predict_action_with_head(
                    image_paths=image_paths,
                    task=row["task"],
                    history_frame_count=len(sample["history_image_paths"]),
                    current_subtask=row["current_subtask"],
                    expected_next_screen=row["expected_next_screen"],
                    temporal_sample_key=trajectory_key,
                )
            else:
                result = agent.generate_action(
                    image_paths=image_paths,
                    task=row["task"],
                    history_frame_count=len(sample["history_image_paths"]),
                    current_subtask=row["current_subtask"],
                    expected_next_screen=row["expected_next_screen"],
                    max_new_tokens=args.max_new_tokens,
                    temporal_sample_key=trajectory_key,
                )
            latency_seconds = time.perf_counter() - step_started
            latencies.append(float(latency_seconds))
            if result.get("_parse_error"):
                parse_error_count += 1
            normalized_action = normalize_action_dict(result.get("action", result))
            parse_error_value = result.get("_parse_error")
            if normalized_action.get("_normalize_error"):
                parse_error_value = parse_error_value or normalized_action["_normalize_error"]
                if not result.get("_parse_error"):
                    parse_error_count += 1
            update_metric_accumulator(
                metrics,
                pred_action=normalized_action,
                teacher_action_type=row["teacher_action_type_name"],
                teacher_region=row["teacher_region_name"],
                teacher_terminate=row["teacher_terminate_status_name"],
                has_pointer_target=row["has_pointer_target"],
                x_norm=row["x_norm"],
                y_norm=row["y_norm"],
            )
            step_row = {
                "global_step": total_steps + 1,
                "trajectory_index": trajectory_index,
                "trajectory_key": trajectory_key,
                "task": row["task"],
                "current_subtask": row["current_subtask"],
                "expected_next_screen": row["expected_next_screen"],
                "image_paths": image_paths,
                "history_frame_count": len(sample["history_image_paths"]),
                "teacher_action_type": row["teacher_action_type_name"],
                "teacher_region": row["teacher_region_name"],
                "teacher_terminate_status": row["teacher_terminate_status_name"],
                "teacher_has_pointer_target": row["has_pointer_target"],
                "teacher_x_norm": row["x_norm"],
                "teacher_y_norm": row["y_norm"],
                "pred_action": normalized_action,
                "latency_seconds": float(latency_seconds),
                "parse_error": parse_error_value,
                "raw_response_text": result.get("raw_text"),
            }
            step_row.update(
                build_step_analysis_row(
                    base_row={
                        "teacher_action_type": row["teacher_action_type_name"],
                        "teacher_region": row["teacher_region_name"],
                        "teacher_terminate_status": row["teacher_terminate_status_name"],
                        "teacher_has_pointer_target": row["has_pointer_target"],
                        "teacher_x_norm": row["x_norm"],
                        "teacher_y_norm": row["y_norm"],
                    },
                    normalized_action=normalized_action,
                )
            )
            history.append(step_row)
            total_steps += 1

            if progress is not None:
                live_metrics = finalize_metric_accumulator(metrics)
                progress.update(1)
                progress.set_postfix(
                    traj=trajectory_index,
                    hist=len(sample["history_image_paths"]),
                    lat=f"{latency_seconds:.2f}s",
                    avg_lat=(f"{sum(latencies) / len(latencies):.2f}s" if latencies else "-"),
                    aacc=(f"{live_metrics.get('action_type_accuracy', 0.0):.3f}"),
                    ex=(f"{live_metrics.get('action_exact_match_accuracy', 0.0):.3f}"),
                    pex=(f"{live_metrics.get('pointer_exact_match_accuracy', 0.0) or 0.0:.3f}"),
                    racc=(f"{live_metrics.get('region_accuracy', 0.0):.3f}"),
                )

            if step_out_path is not None and total_steps % max(1, int(args.step_out_every)) == 0:
                with step_out_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(step_row, ensure_ascii=False) + "\n")

            if total_steps % max(1, int(args.log_every)) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "eval_progress",
                            "completed_steps": total_steps,
                            "total_planned_steps": total_planned_steps,
                            "trajectory_index": trajectory_index,
                            "history_frame_count": len(sample["history_image_paths"]),
                            "avg_latency_seconds": (sum(latencies) / len(latencies) if latencies else None),
                            "live_metrics": finalize_metric_accumulator(metrics),
                        },
                        ensure_ascii=False,
                    )
                )

            if total_steps % max(1, int(args.save_every)) == 0:
                save_progress(
                    progress_path=progress_path,
                    args=args,
                    next_step_index=total_steps,
                    total_planned_steps=total_planned_steps,
                    latencies=latencies,
                    metrics=metrics,
                    history=history,
                    parse_error_count=parse_error_count,
                    elapsed_seconds=resumed_elapsed_seconds + (time.time() - started),
                )
    finally:
        if progress is not None:
            progress.close()

    elapsed = resumed_elapsed_seconds + (time.time() - started)
    report = {
        "adapter": args.adapter,
        "elapsed_seconds": elapsed,
        "sample_count": total_steps,
        "trajectory_count": len(trajectories),
        "history_n": resolved_history_n,
        "max_new_tokens": args.max_new_tokens,
        "action_format": resolved_action_format,
        "action_model": resolved_action_model,
        "lm_action_target": resolved_lm_action_target,
        "flow_action_sample_steps": resolved_flow_action_sample_steps,
        "flow_head_hidden_dim": resolved_flow_head_hidden_dim,
        "flow_head_depth": resolved_flow_head_depth,
        "action_coord_bins": resolved_action_coord_bins,
        "image_min_pixels": resolved_image_min_pixels,
        "image_max_pixels": resolved_image_max_pixels,
        "use_action_head": bool(args.use_action_head),
        "latency_summary": summarize_latencies(latencies),
        "metrics": finalize_metric_accumulator(metrics),
        "parse_error_count": parse_error_count,
        "history": history,
    }
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if progress_path is not None:
        save_progress(
            progress_path=progress_path,
            args=args,
            next_step_index=total_steps,
            total_planned_steps=total_planned_steps,
            latencies=latencies,
            metrics=metrics,
            history=history,
            parse_error_count=parse_error_count,
            elapsed_seconds=elapsed,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
