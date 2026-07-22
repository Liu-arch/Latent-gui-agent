from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch


def _atomic_torch_save(payload: Any, save_path: Path) -> None:
    """Write a torch payload without ever exposing a partial final file."""
    temp_path = save_path.with_name(save_path.name + ".tmp")
    torch.save(payload, temp_path)
    os.replace(temp_path, save_path)


def _atomic_write_text(path: Path, text: str) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def _checkpoint_values_equal(field: str, saved: Any, current: Any) -> bool:
    if field in {"steps", "dataset_root", "model"}:
        return os.path.normpath(str(saved)) == os.path.normpath(str(current))
    if isinstance(saved, bool) or isinstance(current, bool):
        return type(saved) is type(current) and saved == current
    if isinstance(saved, (int, float)) and isinstance(current, (int, float)):
        return math.isclose(float(saved), float(current), rel_tol=1e-12, abs_tol=1e-12)
    return saved == current


def validate_training_checkpoint_compatibility(
    *,
    payload: dict[str, Any],
    current_args: argparse.Namespace,
    compatibility_fields: tuple[str, ...],
    expected_extra_state: dict[str, Any] | None = None,
    legacy_arg_defaults: dict[str, Any] | None = None,
) -> None:
    saved_args = payload.get("args")
    if not isinstance(saved_args, dict):
        raise RuntimeError("Checkpoint has no saved argument configuration and cannot be resumed safely.")
    current_values = vars(current_args)
    legacy_defaults = legacy_arg_defaults or {}
    mismatches: list[str] = []
    for field in compatibility_fields:
        if field not in saved_args:
            # Older checkpoints cannot record arguments that did not exist yet.
            # Accept those fields only when the current run still uses the
            # historical default; non-default values remain a hard mismatch.
            if (
                field in legacy_defaults
                and field in current_values
                and _checkpoint_values_equal(
                    field,
                    legacy_defaults[field],
                    current_values[field],
                )
            ):
                continue
            mismatches.append(f"{field}: missing from checkpoint")
            continue
        if field not in current_values:
            mismatches.append(f"{field}: missing from current arguments")
            continue
        saved = saved_args[field]
        current = current_values[field]
        if not _checkpoint_values_equal(field, saved, current):
            mismatches.append(f"{field}: checkpoint={saved!r}, current={current!r}")

    saved_extra = payload.get("extra_state") or {}
    for field, expected in (expected_extra_state or {}).items():
        if field not in saved_extra:
            mismatches.append(f"extra_state.{field}: missing from checkpoint")
            continue
        saved = saved_extra[field]
        if not _checkpoint_values_equal(field, saved, expected):
            mismatches.append(f"extra_state.{field}: checkpoint={saved!r}, current={expected!r}")

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise RuntimeError(
            "Checkpoint configuration is incompatible with this run. "
            "Use the original settings or a new RUN_NAME/output directory:\n  - " + details
        )


def _extract_checkpoint_state_dict(agent_model: Any) -> dict[str, Any]:
    if hasattr(agent_model, "adapter_state_dict"):
        return agent_model.adapter_state_dict()
    return agent_model.state_dict()


def _extract_gradient_state_dict(agent_model: Any) -> dict[str, Any]:
    gradient_state_dict: dict[str, Any] = {}
    if not hasattr(agent_model, "named_parameters"):
        return gradient_state_dict
    for name, parameter in agent_model.named_parameters():
        if parameter.grad is None:
            continue
        gradient_state_dict[name] = parameter.grad.detach().cpu()
    return gradient_state_dict


def save_training_checkpoint(
    *,
    checkpoint_path: str | Path,
    agent_model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    extra_state: dict[str, Any] | None = None,
) -> Path:
    save_path = Path(checkpoint_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": _extract_checkpoint_state_dict(agent_model),
        "gradient_state_dict": _extract_gradient_state_dict(agent_model),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "extra_state": extra_state or {},
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    _atomic_torch_save(payload, save_path)
    meta_path = save_path.with_suffix(save_path.suffix + ".json")
    _atomic_write_text(
        meta_path,
        json.dumps(
            {
                "epoch": int(epoch),
                "global_step": int(global_step),
                "checkpoint_path": str(save_path),
                "extra_state": extra_state or {},
            },
            ensure_ascii=False,
            indent=2,
        ),
    )
    return save_path


def load_training_checkpoint(
    *,
    checkpoint_path: str | Path,
    agent_model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None = None,
    current_args: argparse.Namespace | None = None,
    compatibility_fields: tuple[str, ...] = (),
    expected_extra_state: dict[str, Any] | None = None,
    legacy_arg_defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was introduced.
        payload = torch.load(Path(checkpoint_path), map_location="cpu")
    if current_args is not None and compatibility_fields:
        validate_training_checkpoint_compatibility(
            payload=payload,
            current_args=current_args,
            compatibility_fields=compatibility_fields,
            expected_extra_state=expected_extra_state,
            legacy_arg_defaults=legacy_arg_defaults,
        )
    agent_model.load_state_dict(payload["model_state_dict"], strict=False)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    return payload


def restore_gradient_state(
    *,
    agent_model: Any,
    gradient_state_dict: dict[str, Any] | None,
) -> None:
    if not gradient_state_dict or not hasattr(agent_model, "named_parameters"):
        return
    parameter_map = dict(agent_model.named_parameters())
    for name, grad_tensor in gradient_state_dict.items():
        parameter = parameter_map.get(name)
        if parameter is None:
            continue
        parameter.grad = grad_tensor.to(device=parameter.device, dtype=parameter.dtype)


def align_optimizer_state_with_params(
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    After loading a checkpoint, make sure optimizer state tensors follow the
    current parameter device/dtype. This is especially important for bf16
    training resumes where exp_avg/exp_avg_sq may otherwise remain float32 and
    break AdamW's foreach path.
    """
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if param is None:
                continue
            if param.grad is not None and torch.is_floating_point(param.grad):
                if param.grad.device != param.device or param.grad.dtype != param.dtype:
                    param.grad = param.grad.to(device=param.device, dtype=param.dtype)
            state = optimizer.state.get(param)
            if not state:
                continue
            for key, value in list(state.items()):
                if not torch.is_tensor(value):
                    continue
                if key == "step":
                    # Keep step as its native dtype; just ensure tensor states are
                    # materialized on a valid device when needed.
                    if value.device.type != "cpu" and value.device != param.device:
                        state[key] = value.to(device=param.device)
                    continue
                if torch.is_floating_point(value):
                    state[key] = value.to(device=param.device, dtype=param.dtype)
                else:
                    state[key] = value.to(device=param.device)


def force_safe_adamw_runtime_flags(
    optimizer: torch.optim.Optimizer,
) -> None:
    """
    `optimizer.load_state_dict(...)` restores param_group options from the
    checkpoint and can silently re-enable foreach/fused AdamW paths even if the
    optimizer was freshly constructed with foreach=False.

    For our bf16 resume path we want the conservative single-tensor codepath to
    avoid dtype mismatches inside torch._foreach_* kernels.
    """
    optimizer.defaults["foreach"] = False
    optimizer.defaults["fused"] = False
    for group in optimizer.param_groups:
        group["foreach"] = False
        group["fused"] = False
