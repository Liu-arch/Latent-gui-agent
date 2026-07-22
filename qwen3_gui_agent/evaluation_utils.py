from __future__ import annotations

import json
import statistics
from typing import Any

from qwen3_gui_agent.training_utils import infer_trajectory_key, resolve_dataset_image


COORD_THRESHOLDS = (0.01, 0.03, 0.05)


def coerce_norm_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = coerce_norm_scalar(item)
            if coerced is not None:
                return coerced
        return None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    if abs(parsed) > 1.0:
        parsed /= 1000.0
    return max(0.0, min(1.0, parsed))


def region_from_action(action: dict[str, Any]) -> str:
    x = coerce_norm_scalar(action.get("x_norm"))
    y = coerce_norm_scalar(action.get("y_norm"))
    if x is None or y is None:
        return "middle_center"
    horizontal = ("left", "center", "right")[min(2, max(0, int(x * 3)))]
    vertical = ("top", "middle", "bottom")[min(2, max(0, int(y * 3)))]
    return f"{vertical}_{horizontal}"


def region_center(region_name: str) -> tuple[float, float]:
    parts = region_name.split("_", 1)
    if len(parts) != 2:
        return 0.5, 0.5
    vertical, horizontal = parts
    x = {"left": 1 / 6, "center": 0.5, "right": 5 / 6}.get(horizontal, 0.5)
    y = {"top": 1 / 6, "middle": 0.5, "bottom": 5 / 6}.get(vertical, 0.5)
    return x, y


def resolve_x_norm(action: dict[str, Any], region_name: str) -> float:
    value = coerce_norm_scalar(action.get("x_norm"))
    return float(value) if value is not None else region_center(region_name)[0]


def resolve_y_norm(action: dict[str, Any], region_name: str) -> float:
    value = coerce_norm_scalar(action.get("y_norm"))
    return float(value) if value is not None else region_center(region_name)[1]


def normalize_action_dict(action: Any) -> dict[str, Any]:
    error: str | None = None
    if action is None:
        normalized: dict[str, Any] = {}
    elif isinstance(action, dict):
        normalized = dict(action)
    elif hasattr(action, "to_dict") and callable(action.to_dict):
        try:
            payload = action.to_dict()
            normalized = dict(payload) if isinstance(payload, dict) else {}
            if not isinstance(payload, dict):
                error = f"to_dict_returned_{type(payload).__name__}"
        except Exception as exc:  # pragma: no cover
            normalized = {}
            error = f"to_dict_failed:{type(exc).__name__}"
    elif isinstance(action, str):
        try:
            payload = json.loads(action.strip())
            normalized = dict(payload) if isinstance(payload, dict) else {}
            if not isinstance(payload, dict):
                error = f"string_action_json_{type(payload).__name__}"
        except (json.JSONDecodeError, ValueError):
            normalized = {}
            error = "string_action_not_json"
    else:
        try:
            normalized = dict(action)
        except Exception:
            normalized = {}
            error = f"unsupported_action_type:{type(action).__name__}"
    if error is not None:
        normalized["_normalize_error"] = error
        normalized["_raw_action_repr"] = repr(action)[:1000]
    normalized["x_norm"] = coerce_norm_scalar(normalized.get("x_norm"))
    normalized["y_norm"] = coerce_norm_scalar(normalized.get("y_norm"))
    normalized["region"] = normalized.get("region") or region_from_action(normalized)
    return normalized


def init_metric_accumulator() -> dict[str, Any]:
    return {
        "coord_thresholds": COORD_THRESHOLDS,
        "action_type_match": 0,
        "action_exact_match": 0,
        "region_match": 0,
        "terminate_match": 0,
        "pointer_count": 0,
        "pointer_action_type_match": 0,
        "pointer_region_match": 0,
        "pointer_exact_match": 0,
        "non_pointer_count": 0,
        "non_pointer_action_type_match": 0,
        "non_pointer_exact_match": 0,
        "per_action_type_count": {},
        "per_action_type_match": {},
        "per_action_type_exact_match": {},
        "coord_hit_counts": {threshold: 0 for threshold in COORD_THRESHOLDS},
        "action_exact_with_coord_counts": {threshold: 0 for threshold in COORD_THRESHOLDS},
        "pointer_exact_with_coord_counts": {threshold: 0 for threshold in COORD_THRESHOLDS},
        "x_errors": [],
        "y_errors": [],
        "l1_errors": [],
        "l2_errors": [],
        "count": 0,
    }


def normalize_metric_accumulator_for_resume(accumulator: dict[str, Any] | None) -> dict[str, Any]:
    normalized = init_metric_accumulator()
    if not isinstance(accumulator, dict):
        return normalized
    thresholds = tuple(float(value) for value in accumulator.get("coord_thresholds", COORD_THRESHOLDS))
    normalized["coord_thresholds"] = thresholds
    for key in (
        "action_type_match", "action_exact_match", "region_match", "terminate_match",
        "pointer_count", "pointer_action_type_match", "pointer_region_match", "pointer_exact_match",
        "non_pointer_count", "non_pointer_action_type_match", "non_pointer_exact_match", "count",
    ):
        normalized[key] = int(accumulator.get(key, 0) or 0)
    for key in ("x_errors", "y_errors", "l1_errors", "l2_errors"):
        values = accumulator.get(key, [])
        normalized[key] = [float(value) for value in values] if isinstance(values, list) else []
    for key in ("per_action_type_count", "per_action_type_match", "per_action_type_exact_match"):
        mapping = accumulator.get(key, {})
        normalized[key] = {str(name): int(value or 0) for name, value in mapping.items()} if isinstance(mapping, dict) else {}
    for key in ("coord_hit_counts", "action_exact_with_coord_counts", "pointer_exact_with_coord_counts"):
        source = accumulator.get(key, {})
        normalized[key] = {
            threshold: int(source.get(threshold, source.get(str(threshold), 0)) or 0)
            for threshold in thresholds
        }
    return normalized


def update_metric_accumulator(
    accumulator: dict[str, Any],
    *,
    pred_action: dict[str, Any],
    teacher_action_type: str,
    teacher_region: str,
    teacher_terminate: str,
    has_pointer_target: bool,
    x_norm: float,
    y_norm: float,
) -> None:
    accumulator["count"] += 1
    pred_type = pred_action.get("type")
    pred_region = pred_action.get("region")
    pred_terminate = pred_action.get("status") or "success"
    type_match = pred_type == teacher_action_type
    region_match = pred_region == teacher_region
    terminate_match = pred_terminate == teacher_terminate

    accumulator["per_action_type_count"][teacher_action_type] = accumulator["per_action_type_count"].get(teacher_action_type, 0) + 1
    if type_match:
        accumulator["action_type_match"] += 1
        accumulator["per_action_type_match"][teacher_action_type] = accumulator["per_action_type_match"].get(teacher_action_type, 0) + 1
    accumulator["region_match"] += int(region_match)
    accumulator["terminate_match"] += int(terminate_match)

    exact_match = type_match
    if teacher_action_type == "terminate":
        exact_match = exact_match and terminate_match
    elif has_pointer_target:
        exact_match = exact_match and region_match
    if exact_match:
        accumulator["action_exact_match"] += 1
        accumulator["per_action_type_exact_match"][teacher_action_type] = accumulator["per_action_type_exact_match"].get(teacher_action_type, 0) + 1

    group = "pointer" if has_pointer_target else "non_pointer"
    accumulator[f"{group}_count"] += 1
    accumulator[f"{group}_action_type_match"] += int(type_match)
    if has_pointer_target:
        accumulator["pointer_region_match"] += int(region_match)
    accumulator[f"{group}_exact_match"] += int(exact_match)

    if has_pointer_target:
        pred_x = coerce_norm_scalar(pred_action.get("x_norm")) or 0.0
        pred_y = coerce_norm_scalar(pred_action.get("y_norm")) or 0.0
        x_error = abs(pred_x - float(x_norm))
        y_error = abs(pred_y - float(y_norm))
        accumulator["x_errors"].append(x_error)
        accumulator["y_errors"].append(y_error)
        accumulator["l1_errors"].append(x_error + y_error)
        accumulator["l2_errors"].append((x_error**2 + y_error**2) ** 0.5)
        for threshold in accumulator["coord_thresholds"]:
            hit = x_error <= threshold and y_error <= threshold
            accumulator["coord_hit_counts"][threshold] += int(hit)
            accumulator["action_exact_with_coord_counts"][threshold] += int(exact_match and hit)
            accumulator["pointer_exact_with_coord_counts"][threshold] += int(exact_match and hit)
    elif exact_match:
        for threshold in accumulator["coord_thresholds"]:
            accumulator["action_exact_with_coord_counts"][threshold] += 1


def finalize_metric_accumulator(accumulator: dict[str, Any]) -> dict[str, Any]:
    count = int(accumulator["count"])
    pointer_count = int(accumulator["pointer_count"])
    non_pointer_count = int(accumulator["non_pointer_count"])

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    metrics: dict[str, Any] = {
        "action_type_accuracy": ratio(accumulator["action_type_match"], count),
        "action_exact_match_accuracy": ratio(accumulator["action_exact_match"], count),
        "region_accuracy": ratio(accumulator["region_match"], count),
        "terminate_accuracy": ratio(accumulator["terminate_match"], count),
        "pointer_count": pointer_count,
        "pointer_action_type_accuracy": ratio(accumulator["pointer_action_type_match"], pointer_count),
        "pointer_region_accuracy": ratio(accumulator["pointer_region_match"], pointer_count),
        "pointer_exact_match_accuracy": ratio(accumulator["pointer_exact_match"], pointer_count),
        "non_pointer_count": non_pointer_count,
        "non_pointer_action_type_accuracy": ratio(accumulator["non_pointer_action_type_match"], non_pointer_count),
        "non_pointer_exact_match_accuracy": ratio(accumulator["non_pointer_exact_match"], non_pointer_count),
    }
    for name in ("x", "y", "l1"):
        values = accumulator[f"{name}_errors"]
        metrics[f"{name}_mae"] = sum(values) / len(values) if values else None
    l2_values = accumulator["l2_errors"]
    metrics["l2_rmse_like"] = sum(l2_values) / len(l2_values) if l2_values else None
    for threshold in accumulator["coord_thresholds"]:
        suffix = f"{threshold:.2f}".replace(".", "p")
        metrics[f"coord_hit_accuracy@{suffix}"] = ratio(accumulator["coord_hit_counts"][threshold], pointer_count)
        metrics[f"action_exact_match_with_coord_accuracy@{suffix}"] = ratio(accumulator["action_exact_with_coord_counts"][threshold], count)
        metrics[f"pointer_exact_match_with_coord_accuracy@{suffix}"] = ratio(accumulator["pointer_exact_with_coord_counts"][threshold], pointer_count)
    metrics["per_action_type_accuracy"] = {
        name: accumulator["per_action_type_match"].get(name, 0) / total
        for name, total in sorted(accumulator["per_action_type_count"].items())
    }
    metrics["per_action_type_exact_match_accuracy"] = {
        name: accumulator["per_action_type_exact_match"].get(name, 0) / total
        for name, total in sorted(accumulator["per_action_type_count"].items())
    }
    return metrics


def summarize_latencies(latencies: list[float]) -> dict[str, Any]:
    if not latencies:
        return {}
    return {
        "avg_seconds": sum(latencies) / len(latencies),
        "median_seconds": statistics.median(latencies),
        "min_seconds": min(latencies),
        "max_seconds": max(latencies),
        "all_seconds": latencies,
    }


def build_step_analysis_row(
    *,
    base_row: dict[str, Any],
    normalized_action: dict[str, Any],
) -> dict[str, Any]:
    teacher_type = str(base_row["teacher_action_type"])
    teacher_region = str(base_row["teacher_region"])
    teacher_status = str(base_row["teacher_terminate_status"])
    pointer = bool(base_row["teacher_has_pointer_target"])
    pred_type = str(normalized_action.get("type", "") or "")
    pred_region = str(normalized_action.get("region", "") or "")
    pred_status = str(normalized_action.get("status", "success") or "success")
    type_match = pred_type == teacher_type
    region_match = pred_region == teacher_region
    terminate_match = pred_status == teacher_status
    pred_x = coerce_norm_scalar(normalized_action.get("x_norm"))
    pred_y = coerce_norm_scalar(normalized_action.get("y_norm"))
    x_error = y_error = l1_error = l2_error = None
    hits: dict[str, bool | None] = {}
    if pointer:
        pred_x = pred_x or 0.0
        pred_y = pred_y or 0.0
        x_error = abs(pred_x - float(base_row["teacher_x_norm"]))
        y_error = abs(pred_y - float(base_row["teacher_y_norm"]))
        l1_error = x_error + y_error
        l2_error = (x_error**2 + y_error**2) ** 0.5
    for threshold in COORD_THRESHOLDS:
        key = f"coord_hit@{threshold:.2f}".replace(".", "p")
        hits[key] = bool(x_error <= threshold and y_error <= threshold) if pointer else None
    exact = type_match and (terminate_match if teacher_type == "terminate" else region_match if pointer else True)
    exact_with_coord = {
        f"action_exact_match_with_coord@{threshold:.2f}".replace(".", "p"): bool(
            exact and (hits[f"coord_hit@{threshold:.2f}".replace('.', 'p')] if pointer else True)
        )
        for threshold in COORD_THRESHOLDS
    }
    return {
        "pred_action_type": pred_type,
        "pred_region": pred_region,
        "pred_terminate_status": pred_status,
        "pred_x_norm": pred_x,
        "pred_y_norm": pred_y,
        "action_type_match": type_match,
        "region_match": region_match,
        "terminate_match": terminate_match,
        "action_exact_match": exact,
        "x_error": x_error,
        "y_error": y_error,
        "l1_error": l1_error,
        "l2_error": l2_error,
        **hits,
        **exact_with_coord,
    }
