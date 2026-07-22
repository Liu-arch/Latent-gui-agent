from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    warmup_steps = max(0, min(total_steps - 1, int(total_steps * max(0.0, warmup_ratio))))

    def lr_lambda(current_step: int) -> float:
        if total_steps <= 1:
            return 1.0
        if warmup_steps > 0 and current_step < warmup_steps:
            return max(1e-8, float(current_step + 1) / float(warmup_steps))
        denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(current_step - warmup_steps) / float(denominator)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def infer_trajectory_key(step: dict[str, Any]) -> str:
    for field in ("episode_id", "trajectory_id"):
        value = str(step.get(field, "") or "").strip()
        if value:
            return f"{field}:{value}"
    sample_id = str(step.get("sample_id", "") or "").strip()
    if sample_id:
        normalized = sample_id
        for pattern in (
            r"([_\-])step[_\-]?\d+$",
            r"([_\-])s\d+$",
            r"([_\-])turn[_\-]?\d+$",
            r"([_\-])frame[_\-]?\d+$",
        ):
            updated = re.sub(pattern, "", normalized, flags=re.I)
            if updated != normalized:
                normalized = updated
                break
        return f"sample_group:{normalized}"
    before_screenshot = str(step.get("before_screenshot", "") or "").strip()
    if before_screenshot:
        prefix = before_screenshot.rsplit("/", 1)[0].rsplit("\\", 1)[0]
        if prefix:
            return f"before_dir:{prefix}"
    task = str(step.get("task", "") or "").strip()
    temporal_anchor = str(step.get("temporal_anchor", "") or "").strip()
    return f"task:{task}|anchor:{temporal_anchor}"


def resolve_dataset_image(dataset_root: Path, image_name: str) -> Path | None:
    candidates = (
        dataset_root / "ubuntu_images" / image_name,
        dataset_root / "win_mac_images" / image_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for image_root in (dataset_root / "ubuntu_images", dataset_root / "win_mac_images"):
        if image_root.exists():
            match = next(image_root.rglob(image_name), None)
            if match is not None:
                return match
    return None


def resolve_torch_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported --torch-dtype: {dtype_name}")
    return mapping[dtype_name]


def resolve_device_map(name: str | None) -> Any:
    if name is None or str(name).strip().lower() in {"", "none", "null"}:
        return None
    return name


def build_loss_artifact_paths(loss_out_dir: str, adapter_out: str) -> dict[str, Path]:
    root = Path(loss_out_dir)
    stem = Path(adapter_out).stem
    return {
        "root": root,
        "jsonl": root / f"{stem}.train_loss.jsonl",
        "csv": root / f"{stem}.train_loss.csv",
        "png": root / f"{stem}.train_loss.png",
    }


def write_loss_artifacts(
    *,
    artifact_paths: dict[str, Path],
    loss_rows: list[dict[str, Any]],
) -> None:
    root = artifact_paths["root"]
    root.mkdir(parents=True, exist_ok=True)
    artifact_paths["jsonl"].write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in loss_rows)
        + ("\n" if loss_rows else ""),
        encoding="utf-8",
    )

    header = sorted({key for row in loss_rows for key in row})
    csv_lines = [",".join(header)]
    for row in loss_rows:
        csv_lines.append(",".join(json.dumps(row.get(key, ""), ensure_ascii=False) for key in header))
    artifact_paths["csv"].write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    if plt is None or not loss_rows:
        return
    steps = [int(row.get("global_step", index + 1)) for index, row in enumerate(loss_rows)]
    figure, axis = plt.subplots(figsize=(10, 6))
    for key in ("loss", "lm_loss", "action_head_loss", "reasoning_alignment_loss", "future_frame_loss"):
        if any(key in row for row in loss_rows):
            axis.plot(steps, [float(row.get(key, 0.0)) for row in loss_rows], label=key)
    axis.set_title("Training Loss")
    axis.set_xlabel("optimizer step")
    axis.set_ylabel("loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(artifact_paths["png"], dpi=160)
    plt.close(figure)


# Compatibility names retained for the single-sample trainer.
_build_loss_artifact_paths = build_loss_artifact_paths
_write_loss_artifacts = write_loss_artifacts
