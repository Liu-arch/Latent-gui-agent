from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from evaluate_lara_style_qwen3vl import (
    build_flat_eval_samples,
    build_lara_eval_trajectories,
    load_adapter_eval_config,
    resolve_progress_path,
)
from qwen3_gui_agent.evaluation_utils import (
    build_step_analysis_row,
    finalize_metric_accumulator,
    init_metric_accumulator,
    normalize_action_dict,
    normalize_metric_accumulator_for_resume,
    summarize_latencies,
    update_metric_accumulator,
)
from qwen3_gui_agent.lara_style_qwen3vl_agent import LaRAStyleQwen3VLAgent
from qwen3_gui_agent.rl.schema import read_jsonl
from qwen3_gui_agent.typed_action_router import route_typed_hybrid_action

try:
    from qwen3_gui_agent.typed_action_router import route_selective_hybrid_action
except ImportError as exc:
    # Keep action_head/hybrid evaluation compatible with older router modules.
    route_selective_hybrid_action = None  # type: ignore[assignment]
    _SELECTIVE_HYBRID_IMPORT_ERROR: ImportError | None = exc
else:
    _SELECTIVE_HYBRID_IMPORT_ERROR = None
from qwen3_gui_agent.training_utils import resolve_device_map, resolve_torch_dtype
from train_lara_style_qwen3vl_active_batch import (
    advance_active_pool,
    make_state,
    skip_active_rows,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def extract_reasoning_text(raw_text: Any) -> str:
    text = str(raw_text or "").strip()
    if not text:
        return ""
    match = re.search(r"Reasoning:\s*(.*?)(?:\nAction:|\Z)", text, flags=re.S)
    if match:
        return match.group(1).strip()
    action_match = re.search(r"\nAction:", text)
    if action_match:
        return text[: action_match.start()].strip()
    return text


def summarize_click_branch_comparison(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare action-head and LM predictions on the same click GT rows."""

    click_rows: list[tuple[dict[str, Any], float, float]] = []
    skipped_invalid_gt = 0
    for row in history:
        if str(row.get("teacher_action_type", "") or "").strip().lower() != "click":
            continue
        try:
            gt_x = float(row.get("teacher_x_norm"))
            gt_y = float(row.get("teacher_y_norm"))
        except (TypeError, ValueError):
            skipped_invalid_gt += 1
            continue
        click_rows.append((row, gt_x, gt_y))

    if not click_rows:
        return {
            "available": False,
            "reason": "No click rows with valid ground-truth coordinates were evaluated.",
            "click_sample_count": 0,
            "skipped_invalid_gt": skipped_invalid_gt,
        }

    def summarize_branch(branch: str) -> dict[str, Any]:
        accumulator = init_metric_accumulator()
        available_count = 0
        missing_output_count = 0
        normalize_error_count = 0
        predicted_type_counts: dict[str, int] = {}

        for row, gt_x, gt_y in click_rows:
            if branch == "action_head":
                raw_action = row.get("action_head_action")
                if raw_action is None and str(row.get("eval_mode", "")) in {
                    "action_head",
                    "hybrid",
                    "selective_hybrid",
                }:
                    # Backward-compatible fallback for rows written before the
                    # raw action-head branch was saved explicitly.
                    raw_action = row.get("pred_action")
            else:
                raw_action = row.get("lm_generated_action")
                if raw_action is None and str(row.get("eval_mode", "")) == "generate":
                    raw_action = row.get("pred_action")

            if raw_action is None:
                missing_output_count += 1
            else:
                available_count += 1
            normalized = normalize_action_dict(raw_action)
            if normalized.get("_normalize_error"):
                normalize_error_count += 1
            pred_type = str(normalized.get("type", "") or "missing")
            predicted_type_counts[pred_type] = predicted_type_counts.get(pred_type, 0) + 1
            update_metric_accumulator(
                accumulator,
                pred_action=normalized,
                teacher_action_type="click",
                teacher_region=str(row.get("teacher_region", "") or ""),
                teacher_terminate=str(row.get("teacher_terminate_status", "success") or "success"),
                has_pointer_target=True,
                x_norm=gt_x,
                y_norm=gt_y,
            )

        return {
            "available": available_count > 0,
            "evaluated_click_count": len(click_rows),
            "available_output_count": available_count,
            "missing_output_count": missing_output_count,
            "normalize_error_count": normalize_error_count,
            "predicted_type_counts": predicted_type_counts,
            "metrics": finalize_metric_accumulator(accumulator),
        }

    action_head = summarize_branch("action_head")
    lm_generate = summarize_branch("lm_generate")
    comparison: dict[str, Any] = {
        "available": bool(action_head["available"] and lm_generate["available"]),
        "scope": "teacher_action_type == click, identical evaluated rows",
        "click_sample_count": len(click_rows),
        "skipped_invalid_gt": skipped_invalid_gt,
        "action_head": action_head,
        "lm_generate": lm_generate,
    }
    if comparison["available"]:
        accuracy_keys = (
            "action_type_accuracy",
            "region_accuracy",
            "coord_hit_accuracy@0p01",
            "coord_hit_accuracy@0p03",
            "coord_hit_accuracy@0p05",
            "action_exact_match_with_coord_accuracy@0p01",
            "action_exact_match_with_coord_accuracy@0p03",
            "action_exact_match_with_coord_accuracy@0p05",
        )
        error_keys = ("x_mae", "y_mae", "l1_mae", "l2_rmse_like")
        head_metrics = action_head["metrics"]
        lm_metrics = lm_generate["metrics"]
        comparison["head_minus_lm_accuracy"] = {
            key: float(head_metrics[key]) - float(lm_metrics[key])
            for key in accuracy_keys
            if head_metrics.get(key) is not None and lm_metrics.get(key) is not None
        }
        comparison["head_minus_lm_error"] = {
            key: float(head_metrics[key]) - float(lm_metrics[key])
            for key in error_keys
            if head_metrics.get(key) is not None and lm_metrics.get(key) is not None
        }
    else:
        comparison["reason"] = "Both action-head and LM branch outputs are required; run --eval-mode hybrid."
    return comparison


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _reconcile_step_output(path: Path, *, completed_steps: int, step_out_every: int) -> None:
    expected_rows = int(completed_steps) // max(1, int(step_out_every))
    if not path.is_file():
        if expected_rows > 0:
            raise RuntimeError(
                f"Resume progress confirms {expected_rows} saved step rows, but step output is missing: {path}"
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return

    temp_path = path.with_name(path.name + ".resume.tmp")
    row_count = 0
    with path.open("r", encoding="utf-8") as source, temp_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            row_count += 1
            if row_count <= expected_rows:
                target.write(line if line.endswith("\n") else line + "\n")
    if row_count < expected_rows:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Resume progress confirms {expected_rows} step rows, but {path} contains only {row_count}."
        )
    if row_count > expected_rows:
        os.replace(temp_path, path)
    else:
        temp_path.unlink(missing_ok=True)


def save_batched_progress(
    *,
    progress_path: Path | None,
    args: argparse.Namespace,
    next_step_index: int,
    total_planned_steps: int,
    latencies: list[float],
    batch_latencies: list[float],
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
        "dataset_root": str(args.dataset_root),
        "model": str(args.model),
        "adapter": str(args.adapter),
        "torch_dtype": str(args.torch_dtype),
        "history_n": int(args.history_n),
        "batch_size": int(args.batch_size),
        "batching_mode": str(args.batching_mode),
        "eval_mode": str(args.eval_mode),
        "max_new_tokens": int(args.max_new_tokens),
        "parameter_max_new_tokens": int(args.parameter_max_new_tokens),
        "selective_hard_action_policy": str(args.selective_hard_action_policy),
        "image_min_pixels": int(args.image_min_pixels),
        "image_max_pixels": int(args.image_max_pixels),
        "action_format": str(args.action_format),
        "action_model": str(args.action_model),
        "flow_continuous_source": str(args.flow_continuous_source),
        "flow_pointer_coord_source": str(args.flow_pointer_coord_source),
        "flow_patch_logit_temperature": float(args.flow_patch_logit_temperature),
        "flow_patch_residual_scale": float(args.flow_patch_residual_scale),
        "action_hidden_source": str(args.action_hidden_source),
        "two_way_query_mode": str(args.two_way_query_mode),
        "include_flow_alternatives": bool(args.include_flow_alternatives),
        "next_step_index": int(next_step_index),
        "total_planned_steps": int(total_planned_steps),
        "latencies": list(latencies),
        "batch_latencies": list(batch_latencies),
        "metrics": metrics,
        "history": history,
        "parse_error_count": int(parse_error_count),
        "elapsed_seconds": float(elapsed_seconds),
    }
    _atomic_write_json(progress_path, payload)


def load_batched_progress(progress_path: Path, args: argparse.Namespace, total_planned_steps: int) -> dict[str, Any]:
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    checks = {
        "steps_path": str(args.steps),
        "dataset_root": str(args.dataset_root),
        "model": str(args.model),
        "adapter": str(args.adapter),
        "torch_dtype": str(args.torch_dtype),
        "history_n": int(args.history_n),
        "batch_size": int(args.batch_size),
        "batching_mode": str(args.batching_mode),
        "eval_mode": str(args.eval_mode),
        "max_new_tokens": int(args.max_new_tokens),
        "parameter_max_new_tokens": int(args.parameter_max_new_tokens),
        "selective_hard_action_policy": str(args.selective_hard_action_policy),
        "image_min_pixels": int(args.image_min_pixels),
        "image_max_pixels": int(args.image_max_pixels),
        "action_format": str(args.action_format),
        "action_model": str(args.action_model),
        "flow_continuous_source": str(args.flow_continuous_source),
        "flow_pointer_coord_source": str(args.flow_pointer_coord_source),
        "flow_patch_logit_temperature": float(args.flow_patch_logit_temperature),
        "flow_patch_residual_scale": float(args.flow_patch_residual_scale),
        "action_hidden_source": str(args.action_hidden_source),
        "two_way_query_mode": str(args.two_way_query_mode),
        "include_flow_alternatives": bool(args.include_flow_alternatives),
        "total_planned_steps": int(total_planned_steps),
    }
    for key, expected in checks.items():
        # Older progress files predate selective hybrid generation. They are
        # compatible with the default because this value was not used there.
        compatible_defaults = {
            "parameter_max_new_tokens": int(args.parameter_max_new_tokens),
            "selective_hard_action_policy": "parameters_only",
            "two_way_query_mode": "auto",
        }
        actual = payload.get(key, compatible_defaults.get(key))
        if actual != expected:
            raise RuntimeError(f"Resume progress {key} mismatch: expected {expected!r}, got {actual!r}.")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batched action-head evaluation for LaRA-style Qwen3-VL GUI agent."
    )
    parser.add_argument("--steps", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--history-n", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--batching-mode",
        choices=["active_pool", "flat"],
        default="active_pool",
        help="active_pool matches training: parallel trajectories advance step by step. flat chunks flattened steps.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["action_head", "generate", "hybrid", "selective_hybrid"],
        default="action_head",
        help=(
            "action_head evaluates the action head only; generate evaluates parsed LM text action; "
            "hybrid generates reasoning for every sample; selective_hybrid runs the action head first "
            "and invokes the LM only for type/hotkey parameters."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--parameter-max-new-tokens",
        type=int,
        default=192,
        help="Generation limit used only for type/hotkey samples in selective_hybrid mode.",
    )
    parser.add_argument(
        "--selective-hard-action-policy",
        choices=["parameters_only", "full_action"],
        default="parameters_only",
        help=(
            "In selective_hybrid mode, parameters_only keeps the head's type and asks the LM only for "
            "text/keys; full_action lets the LM own the complete action for type/hotkey samples."
        ),
    )
    parser.add_argument("--image-min-pixels", type=int, default=0)
    parser.add_argument("--image-max-pixels", type=int, default=0)
    parser.add_argument(
        "--action-format",
        choices=["auto", "text", "action_tokens"],
        default="auto",
    )
    parser.add_argument(
        "--action-model",
        choices=["auto", "unified", "flow_matching", "latent_two_way"],
        default="auto",
    )
    parser.add_argument(
        "--flow-action-sample-steps",
        type=int,
        default=0,
        help="Flow integration steps. 0 inherits the adapter metadata.",
    )
    parser.add_argument(
        "--flow-continuous-source",
        choices=["auto", "direct", "sample"],
        default="auto",
        help=(
            "Continuous GUI action source for flow_matching. "
            "direct uses the directly supervised coordinate head; sample uses flow sampling."
        ),
    )
    parser.add_argument(
        "--flow-pointer-coord-source",
        choices=["auto", "mlp", "patch", "argmax_patch", "patch_residual"],
        default="auto",
        help="Coordinate source inside the direct pointer branch. auto loads it from the adapter metadata.",
    )
    parser.add_argument(
        "--flow-patch-logit-temperature",
        type=float,
        default=0.0,
        help="Override patch attention softargmax temperature. 0 loads it from adapter metadata.",
    )
    parser.add_argument(
        "--flow-patch-residual-scale",
        type=float,
        default=-1.0,
        help="Override patch residual scale. Negative value loads it from adapter metadata.",
    )
    parser.add_argument(
        "--include-flow-alternatives",
        action="store_true",
        help="For flow_matching action heads, also write both direct and sampled actions to step outputs.",
    )
    parser.add_argument(
        "--action-hidden-source",
        choices=["auto", "summary", "prompt_attn", "slot_attn", "prompt_slot_attn"],
        default="auto",
        help="Action-head hidden pooling source. auto loads it from adapter metadata.",
    )
    parser.add_argument(
        "--two-way-query-mode",
        choices=["auto", "semantic_pool", "latent_pos"],
        default="auto",
        help=(
            "Two-way action routing. latent_pos uses a <|POS|> router but keeps Stage-2 latent "
            "states as the direct grounding prompt; auto loads adapter metadata."
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--progress-out", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--step-out", default=None)
    parser.add_argument("--step-out-every", type=int, default=1)
    parser.add_argument("--report-out", default=None)
    return parser.parse_args()


def make_agent_sample(sample: dict[str, Any]) -> dict[str, Any]:
    row = sample["row"]
    image_paths = list(sample["history_image_paths"]) + [str(row["image_path"])]
    return {
        "image_paths": image_paths,
        "task": row["task"],
        "history_frame_count": len(sample["history_image_paths"]),
        "current_subtask": row["current_subtask"],
        "expected_next_screen": row["expected_next_screen"],
        "temporal_sample_key": str(sample["trajectory_key"]),
    }


def eval_sample_from_state(state: dict[str, Any], history_slots: int) -> dict[str, Any]:
    row = state["trajectory"]["rows"][state["row_index"]]
    history_image_paths = list(state["history_image_paths"])
    if history_slots <= 0:
        history_image_paths = []
    else:
        history_image_paths = history_image_paths[-history_slots:]
    return {
        "trajectory_index": int(state["trajectory"].get("trajectory_index", 0)),
        "trajectory_key": state["trajectory"]["trajectory_key"],
        "row": row,
        "history_image_paths": history_image_paths,
        "trajectory_step_index": int(state["row_index"]) + 1,
    }


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
    resolved_two_way_query_mode = (
        str(adapter_config.get("two_way_query_mode", "semantic_pool") or "semantic_pool")
        if str(args.two_way_query_mode) == "auto"
        else str(args.two_way_query_mode)
    )
    if resolved_two_way_query_mode not in {"semantic_pool", "latent_pos"}:
        resolved_two_way_query_mode = "semantic_pool"
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
    if str(args.flow_continuous_source) == "auto":
        resolved_flow_continuous_source = str(adapter_config.get("flow_continuous_source", "") or "")
        if resolved_flow_continuous_source not in {"direct", "sample"}:
            coord_weight = float(adapter_config.get("flow_coord_loss_weight", 0.0) or 0.0)
            resolved_flow_continuous_source = "direct" if coord_weight > 0.0 else "sample"
    else:
        resolved_flow_continuous_source = str(args.flow_continuous_source)
    if resolved_action_model == "latent_two_way":
        resolved_flow_continuous_source = "direct"
    if str(args.flow_pointer_coord_source) == "auto":
        resolved_flow_pointer_coord_source = str(
            adapter_config.get("flow_pointer_coord_source", "patch_residual") or "patch_residual"
        )
    else:
        resolved_flow_pointer_coord_source = str(args.flow_pointer_coord_source)
    resolved_flow_patch_logit_temperature = (
        float(args.flow_patch_logit_temperature)
        if float(args.flow_patch_logit_temperature) > 0.0
        else float(adapter_config.get("flow_patch_logit_temperature", 1.0) or 1.0)
    )
    resolved_flow_patch_residual_scale = (
        float(args.flow_patch_residual_scale)
        if float(args.flow_patch_residual_scale) >= 0.0
        else float(adapter_config.get("flow_patch_residual_scale", 1.0) or 1.0)
    )
    if str(args.action_hidden_source) == "auto":
        resolved_action_hidden_source = str(adapter_config.get("action_hidden_source", "summary") or "summary")
    else:
        resolved_action_hidden_source = str(args.action_hidden_source)
    if resolved_action_hidden_source not in {"summary", "prompt_attn", "slot_attn", "prompt_slot_attn"}:
        resolved_action_hidden_source = "summary"

    print(
        json.dumps(
            {
                "adapter": args.adapter,
                "batched_eval": True,
                "eval_mode": args.eval_mode,
                "batch_size": int(args.batch_size),
                "resolved_history_n": resolved_history_n,
                "resolved_latent_slot_count": resolved_latent_slot_count,
                "resolved_pixel_prune_threshold": resolved_pixel_prune_threshold,
                "resolved_pixel_temporal_reuse": resolved_pixel_temporal_reuse,
                "resolved_image_min_pixels": resolved_image_min_pixels,
                "resolved_image_max_pixels": resolved_image_max_pixels,
                "resolved_action_format": resolved_action_format,
                "resolved_action_model": resolved_action_model,
                "resolved_lm_action_target": resolved_lm_action_target,
                "resolved_flow_head_hidden_dim": resolved_flow_head_hidden_dim,
                "resolved_flow_head_depth": resolved_flow_head_depth,
                "resolved_two_way_hidden_dim": resolved_two_way_hidden_dim,
                "resolved_two_way_depth": resolved_two_way_depth,
                "resolved_two_way_num_heads": resolved_two_way_num_heads,
                "resolved_two_way_location_queries": resolved_two_way_location_queries,
                "resolved_two_way_dropout": resolved_two_way_dropout,
                "resolved_two_way_query_mode": resolved_two_way_query_mode,
                "resolved_flow_continuous_source": resolved_flow_continuous_source,
                "resolved_flow_pointer_coord_source": resolved_flow_pointer_coord_source,
                "resolved_flow_patch_logit_temperature": resolved_flow_patch_logit_temperature,
                "resolved_flow_patch_residual_scale": resolved_flow_patch_residual_scale,
                "resolved_action_hidden_source": resolved_action_hidden_source,
                "resolved_clean_observable_prompt": clean_observable_prompt,
                "resolved_use_lora": resolved_use_lora,
                "include_flow_alternatives": bool(args.include_flow_alternatives),
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
        two_way_query_mode=resolved_two_way_query_mode,
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
    hf_device_map = getattr(agent.model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        device_module_counts: dict[str, int] = {}
        for mapped_device in hf_device_map.values():
            device_name = str(mapped_device)
            device_module_counts[device_name] = device_module_counts.get(device_name, 0) + 1
        print(
            json.dumps(
                {
                    "stage": "eval_model_device_map",
                    "requested_device_map": args.device_map,
                    "device_module_counts": device_module_counts,
                },
                ensure_ascii=False,
            )
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
    agent.flow_continuous_source = resolved_flow_continuous_source
    agent.flow_pointer_coord_source = resolved_flow_pointer_coord_source
    agent.flow_patch_logit_temperature = resolved_flow_patch_logit_temperature
    agent.flow_patch_residual_scale = resolved_flow_patch_residual_scale
    agent.action_hidden_source = resolved_action_hidden_source
    agent.two_way_query_mode = resolved_two_way_query_mode
    agent.latent_two_way_action_head.query_mode = resolved_two_way_query_mode
    agent.include_flow_alternatives = bool(args.include_flow_alternatives)
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
    for trajectory_index, trajectory in enumerate(trajectories, start=1):
        trajectory["trajectory_index"] = trajectory_index
    flat_samples = build_flat_eval_samples(trajectories, history_slots)
    total_planned_steps = len(flat_samples)
    progress_path = resolve_progress_path(args.report_out, args.progress_out, args.resume_from)
    step_out_path = Path(args.step_out) if args.step_out else None
    if step_out_path is not None and not args.resume_from:
        step_out_path.parent.mkdir(parents=True, exist_ok=True)
        step_out_path.write_text("", encoding="utf-8")

    batch_size = max(1, int(args.batch_size))
    latencies: list[float] = []
    batch_latencies: list[float] = []
    metrics = init_metric_accumulator()
    history: list[dict[str, Any]] = []
    parse_error_count = 0
    total_steps = 0
    resumed_elapsed_seconds = 0.0

    if args.resume_from:
        payload = load_batched_progress(Path(args.resume_from), args, total_planned_steps)
        total_steps = int(payload.get("next_step_index", 0))
        latencies = [float(value) for value in payload.get("latencies", [])]
        batch_latencies = [float(value) for value in payload.get("batch_latencies", [])]
        metrics = normalize_metric_accumulator_for_resume(payload.get("metrics", init_metric_accumulator()))
        history = list(payload.get("history", []))
        parse_error_count = int(payload.get("parse_error_count", 0))
        resumed_elapsed_seconds = float(payload.get("elapsed_seconds", 0.0))
        if step_out_path is not None:
            _reconcile_step_output(
                step_out_path,
                completed_steps=total_steps,
                step_out_every=int(args.step_out_every),
            )
        print(
            json.dumps(
                {
                    "stage": "resume_batched_eval",
                    "resume_from": args.resume_from,
                    "next_step_index": total_steps,
                    "completed_steps": total_steps,
                    "total_planned_steps": total_planned_steps,
                },
                ensure_ascii=False,
            )
        )

    active_states: list[dict[str, Any]] = []
    next_trajectory_index = 0
    skipped_rows_for_active_pool = 0
    if str(args.batching_mode) == "active_pool":
        while next_trajectory_index < len(trajectories) and len(active_states) < batch_size:
            active_states.append(make_state(trajectories[next_trajectory_index]))
            next_trajectory_index += 1
        if total_steps > 0:
            active_states, next_trajectory_index, skipped_rows_for_active_pool = skip_active_rows(
                active_states=active_states,
                epoch_trajectories=trajectories,
                next_trajectory_index=next_trajectory_index,
                rows_to_skip=total_steps,
                batch_size=batch_size,
            )
            if int(skipped_rows_for_active_pool) != int(total_steps):
                raise RuntimeError(
                    "Active-pool resume could not reproduce the saved step boundary: "
                    f"requested={total_steps}, skipped={skipped_rows_for_active_pool}."
                )
            print(
                json.dumps(
                    {
                        "stage": "resume_active_pool_eval_fast_forward",
                        "requested_skip_rows": int(total_steps),
                        "actual_skipped_rows": int(skipped_rows_for_active_pool),
                        "active_batch_size": len(active_states),
                        "next_trajectory_index": int(next_trajectory_index),
                    },
                    ensure_ascii=False,
                )
            )

    progress = (
        tqdm(total=total_planned_steps, desc="eval_lara_style_batched", unit="step", initial=total_steps)
        if tqdm is not None
        else None
    )
    started = time.time()
    last_progress_save_steps = int(total_steps)
    try:
        while total_steps < total_planned_steps:
            if str(args.batching_mode) == "active_pool":
                batch_states = [state for state in active_states if state["row_index"] < len(state["trajectory"]["rows"])]
                if not batch_states:
                    break
                batch_samples = [eval_sample_from_state(state, history_slots) for state in batch_states]
            else:
                batch_states = []
                batch_samples = flat_samples[total_steps : min(total_steps + batch_size, total_planned_steps)]
            agent_samples = [make_agent_sample(sample) for sample in batch_samples]
            batch_started = time.perf_counter()
            if str(args.eval_mode) == "generate":
                generated_results = agent.generate_actions_batch(
                    agent_samples,
                    max_new_tokens=int(args.max_new_tokens),
                )
                results = []
                for generated_result in generated_results:
                    merged = dict(generated_result)
                    merged["lm_generated_action"] = generated_result.get("action")
                    results.append(merged)
            elif str(args.eval_mode) == "hybrid":
                reasoning_results = agent.generate_actions_batch(agent_samples, max_new_tokens=int(args.max_new_tokens))
                head_results = agent.predict_actions_with_head_batch(agent_samples)
                results = []
                for reasoning_result, head_result in zip(reasoning_results, head_results):
                    routed_action, routing_diagnostics = route_typed_hybrid_action(
                        head_action=head_result.get("action"),
                        lm_action=reasoning_result.get("action"),
                    )
                    merged = dict(head_result)
                    merged["action_head_action"] = head_result.get("action")
                    merged["action"] = routed_action
                    merged.update(routing_diagnostics)
                    merged["raw_text"] = reasoning_result.get("raw_text")
                    merged["pred_reasoning_text"] = extract_reasoning_text(reasoning_result.get("raw_text"))
                    merged["lm_generated_action"] = reasoning_result.get("action")
                    merged["lm_parse_error"] = reasoning_result.get("_parse_error")
                    merged["action_head_raw_text"] = head_result.get("raw_text")
                    merged["action_source"] = "typed_hybrid_router"
                    merged["reasoning_source"] = "generate"
                    if not routing_diagnostics["execution_allowed"]:
                        merged["_parse_error"] = routing_diagnostics["hybrid_parameter_error"]
                    results.append(merged)
            elif str(args.eval_mode) == "selective_hybrid":
                if route_selective_hybrid_action is None:
                    raise RuntimeError(
                        "selective_hybrid evaluation requires "
                        "qwen3_gui_agent.typed_action_router.route_selective_hybrid_action; "
                        "update typed_action_router.py to the same code version as this evaluator"
                    ) from _SELECTIVE_HYBRID_IMPORT_ERROR
                head_results = agent.predict_actions_with_head_batch(agent_samples)
                lm_indices = [
                    index
                    for index, head_result in enumerate(head_results)
                    if str((head_result.get("action") or {}).get("type", "") or "").strip().lower()
                    in {"type", "hotkey"}
                ]
                lm_results_by_index: dict[int, dict[str, Any]] = {}
                if lm_indices:
                    selected_action_types = [
                        str((head_results[index].get("action") or {}).get("type", "") or "").strip().lower()
                        for index in lm_indices
                    ]
                    if str(args.selective_hard_action_policy) == "full_action":
                        selected_lm_results = agent.generate_actions_batch(
                            [agent_samples[index] for index in lm_indices],
                            max_new_tokens=int(args.max_new_tokens),
                        )
                    else:
                        selected_lm_results = agent.generate_action_parameters_batch(
                            [agent_samples[index] for index in lm_indices],
                            action_types=selected_action_types,
                            max_new_tokens=int(args.parameter_max_new_tokens),
                        )
                    lm_results_by_index = dict(zip(lm_indices, selected_lm_results))

                results = []
                for index, head_result in enumerate(head_results):
                    lm_result = lm_results_by_index.get(index)
                    routed_action, routing_diagnostics = route_selective_hybrid_action(
                        head_action=head_result.get("action"),
                        lm_action=lm_result.get("action") if lm_result is not None else None,
                        hard_action_policy=str(args.selective_hard_action_policy),
                    )
                    lm_parse_error = lm_result.get("_parse_error") if lm_result is not None else None
                    if lm_parse_error:
                        routing_diagnostics["hybrid_parameter_valid"] = False
                        routing_diagnostics["hybrid_parameter_error"] = f"lm_parse_error:{lm_parse_error}"
                        routing_diagnostics["execution_allowed"] = False
                    merged = dict(head_result)
                    merged["action_head_action"] = head_result.get("action")
                    merged["action"] = routed_action
                    merged.update(routing_diagnostics)
                    merged["selective_lm_invoked"] = lm_result is not None
                    merged["raw_text"] = lm_result.get("raw_text") if lm_result is not None else None
                    merged["pred_reasoning_text"] = (
                        extract_reasoning_text(lm_result.get("raw_text")) if lm_result is not None else ""
                    )
                    merged["lm_generated_action"] = lm_result.get("action") if lm_result is not None else None
                    merged["lm_parse_error"] = lm_parse_error
                    merged["action_head_raw_text"] = head_result.get("raw_text")
                    merged["action_source"] = (
                        "lm_full_action_fallback"
                        if routing_diagnostics.get("selective_lm_owns_action")
                        else "head_first_selective_router"
                    )
                    merged["reasoning_source"] = "selective_generate" if lm_result is not None else "not_generated"
                    if not routing_diagnostics["execution_allowed"]:
                        merged["_parse_error"] = routing_diagnostics["hybrid_parameter_error"]
                    results.append(merged)
            else:
                head_results = agent.predict_actions_with_head_batch(agent_samples)
                results = []
                for head_result in head_results:
                    routed_action, routing_diagnostics = route_typed_hybrid_action(
                        head_action=head_result.get("action"),
                        lm_action=None,
                    )
                    merged = dict(head_result)
                    merged["action_head_action"] = head_result.get("action")
                    merged["action"] = routed_action
                    merged.update(routing_diagnostics)
                    merged["selective_lm_invoked"] = False
                    merged["action_source"] = "typed_action_head_router"
                    if not routing_diagnostics["execution_allowed"]:
                        merged["_parse_error"] = routing_diagnostics["hybrid_parameter_error"]
                    results.append(merged)
            batch_latency_seconds = time.perf_counter() - batch_started
            batch_latencies.append(float(batch_latency_seconds))
            amortized_latency = float(batch_latency_seconds / max(1, len(batch_samples)))

            for offset, (sample, result) in enumerate(zip(batch_samples, results), start=1):
                row = sample["row"]
                trajectory_index = int(sample["trajectory_index"])
                trajectory_key = str(sample["trajectory_key"])
                image_paths = list(sample["history_image_paths"]) + [str(row["image_path"])]
                latencies.append(amortized_latency)
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
                    "global_step": total_steps + offset,
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
                    "gt_reasoning": row.get("explicit_reasoning"),
                    "gold_action": row.get("gold_action"),
                    "pred_action": normalized_action,
                    "pred_action_source": result.get("action_source", str(args.eval_mode)),
                    "action_head_action": result.get("action_head_action"),
                    "flow_continuous_source": result.get("flow_continuous_source"),
                    "flow_direct_action": result.get("flow_direct_action"),
                    "flow_sample_action": result.get("flow_sample_action"),
                    "flow_patch_action": result.get("flow_patch_action"),
                    "flow_patch_argmax_action": result.get("flow_patch_argmax_action"),
                    "two_way_query_mode": result.get("two_way_query_mode"),
                    "pointer_grounding_required": result.get("pointer_grounding_required"),
                    "two_way_pos_latent_attention": result.get("two_way_pos_latent_attention"),
                    "two_way_pos_latent_attention_entropy": result.get(
                        "two_way_pos_latent_attention_entropy"
                    ),
                    "two_way_pos_latent_attention_max": result.get(
                        "two_way_pos_latent_attention_max"
                    ),
                    "pred_reasoning_text": result.get("pred_reasoning_text", extract_reasoning_text(result.get("raw_text"))),
                    "lm_generated_action": result.get("lm_generated_action"),
                    "lm_parse_error": result.get("lm_parse_error"),
                    "action_head_raw_text": result.get("action_head_raw_text"),
                    "hybrid_head_action_type": result.get("hybrid_head_action_type"),
                    "hybrid_lm_action_type": result.get("hybrid_lm_action_type"),
                    "hybrid_action_type_agreement": result.get("hybrid_action_type_agreement"),
                    "hybrid_parameter_source": result.get("hybrid_parameter_source"),
                    "hybrid_parameter_valid": result.get("hybrid_parameter_valid"),
                    "hybrid_parameter_error": result.get("hybrid_parameter_error"),
                    "execution_allowed": result.get("execution_allowed"),
                    "selective_lm_invoked": bool(result.get("selective_lm_invoked", False)),
                    "latency_seconds": amortized_latency,
                    "batch_latency_seconds": float(batch_latency_seconds),
                    "batch_size": len(batch_samples),
                    "eval_mode": str(args.eval_mode),
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
                step_row = _json_safe(step_row)
                history.append(step_row)
                if step_out_path is not None and (total_steps + offset) % max(1, int(args.step_out_every)) == 0:
                    with step_out_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(step_row, ensure_ascii=False) + "\n")

            total_steps += len(batch_samples)
            if str(args.batching_mode) == "active_pool":
                active_states, next_trajectory_index = advance_active_pool(
                    active_states=batch_states,
                    epoch_trajectories=trajectories,
                    next_trajectory_index=next_trajectory_index,
                    batch_size=batch_size,
                )
            live_metrics = finalize_metric_accumulator(metrics)
            if progress is not None:
                progress.update(len(batch_samples))
                progress.set_postfix(
                    batch=len(batch_samples),
                    batch_lat=f"{batch_latency_seconds:.2f}s",
                    avg_lat=(f"{sum(latencies) / len(latencies):.2f}s" if latencies else "-"),
                    aacc=(f"{live_metrics.get('action_type_accuracy', 0.0):.3f}"),
                    ex=(f"{live_metrics.get('action_exact_match_accuracy', 0.0):.3f}"),
                    pex=(f"{live_metrics.get('pointer_exact_match_accuracy', 0.0) or 0.0:.3f}"),
                    racc=(f"{live_metrics.get('region_accuracy', 0.0):.3f}"),
                )
            if total_steps % max(1, int(args.log_every)) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "batched_eval_progress",
                            "completed_steps": total_steps,
                            "total_planned_steps": total_planned_steps,
                            "batching_mode": str(args.batching_mode),
                            "batch_size": len(batch_samples),
                            "batch_latency_seconds": float(batch_latency_seconds),
                            "amortized_latency_seconds": amortized_latency,
                            "avg_latency_seconds": (sum(latencies) / len(latencies) if latencies else None),
                            "live_metrics": live_metrics,
                        },
                        ensure_ascii=False,
                    )
                )
            if total_steps - last_progress_save_steps >= max(1, int(args.save_every)):
                save_batched_progress(
                    progress_path=progress_path,
                    args=args,
                    next_step_index=total_steps,
                    total_planned_steps=total_planned_steps,
                    latencies=latencies,
                    batch_latencies=batch_latencies,
                    metrics=metrics,
                    history=history,
                    parse_error_count=parse_error_count,
                    elapsed_seconds=resumed_elapsed_seconds + (time.time() - started),
                )
                last_progress_save_steps = int(total_steps)
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
        "batch_size": int(args.batch_size),
        "batched_eval": True,
        "eval_mode": str(args.eval_mode),
        "max_new_tokens": int(args.max_new_tokens),
        "parameter_max_new_tokens": int(args.parameter_max_new_tokens),
        "selective_hard_action_policy": str(args.selective_hard_action_policy),
        "action_format": resolved_action_format,
        "action_model": resolved_action_model,
        "lm_action_target": resolved_lm_action_target,
        "flow_continuous_source": resolved_flow_continuous_source,
        "flow_pointer_coord_source": resolved_flow_pointer_coord_source,
        "flow_patch_logit_temperature": resolved_flow_patch_logit_temperature,
        "flow_patch_residual_scale": resolved_flow_patch_residual_scale,
        "action_hidden_source": resolved_action_hidden_source,
        "two_way_query_mode": resolved_two_way_query_mode,
        "include_flow_alternatives": bool(args.include_flow_alternatives),
        "flow_action_sample_steps": resolved_flow_action_sample_steps,
        "flow_head_hidden_dim": resolved_flow_head_hidden_dim,
        "flow_head_depth": resolved_flow_head_depth,
        "action_coord_bins": resolved_action_coord_bins,
        "image_min_pixels": resolved_image_min_pixels,
        "image_max_pixels": resolved_image_max_pixels,
        "latency_summary": summarize_latencies(latencies),
        "batch_latency_summary": summarize_latencies(batch_latencies),
        "metrics": finalize_metric_accumulator(metrics),
        "click_head_vs_lm": summarize_click_branch_comparison(history),
        "parse_error_count": parse_error_count,
        "selective_lm_call_count": sum(bool(row.get("selective_lm_invoked", False)) for row in history),
        "selective_lm_call_ratio": (
            sum(bool(row.get("selective_lm_invoked", False)) for row in history) / max(1, len(history))
        ),
        "selective_lm_full_action_count": sum(
            str(row.get("action_source", "")) == "lm_full_action_fallback" for row in history
        ),
        "history": history,
    }
    if args.report_out:
        _atomic_write_json(Path(args.report_out), report)
    if progress_path is not None:
        save_batched_progress(
            progress_path=progress_path,
            args=args,
            next_step_index=total_steps,
            total_planned_steps=total_planned_steps,
            latencies=latencies,
            batch_latencies=batch_latencies,
            metrics=metrics,
            history=history,
            parse_error_count=parse_error_count,
            elapsed_seconds=elapsed,
        )
    print(json.dumps(_json_safe(report), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
