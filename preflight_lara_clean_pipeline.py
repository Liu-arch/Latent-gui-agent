from __future__ import annotations

import argparse
import inspect
import json
import py_compile
import shutil
from pathlib import Path
from typing import Any

import torch

from qwen3_gui_agent.distributed_training import DistributedRuntime, shard_ordered_items
from qwen3_gui_agent.rl.schema import read_jsonl
from train_lara_style_qwen3vl import build_trajectories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-fast preflight for the two-GPU clean LaRA pipeline.")
    parser.add_argument("--train-steps", required=True)
    parser.add_argument("--test-steps", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-samples", type=int, required=True)
    parser.add_argument("--test-samples", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--batch-size-per-rank", type=int, default=4)
    parser.add_argument("--min-free-gb", type=float, default=40.0)
    parser.add_argument("--skip-gpu-check", action="store_true")
    return parser.parse_args()


def _compile_runtime_files(code_dir: Path) -> None:
    files = (
        "train_lara_style_qwen3vl_active_batch_ddp.py",
        "train_lara_style_qwen3vl_active_batch.py",
        "train_lara_style_qwen3vl.py",
        "evaluate_lara_style_qwen3vl_batched.py",
        "prepare_lara_fixed_splits.py",
        "summarize_lara_clean_pipeline.py",
        "qwen3_gui_agent/lara_style_qwen3vl_agent.py",
        "qwen3_gui_agent/training_checkpoint.py",
        "qwen3_gui_agent/distributed_training.py",
    )
    for relative in files:
        path = code_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required pipeline source is missing: {path}")
        py_compile.compile(str(path), doraise=True)


def _validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Model config is missing: {model_dir / 'config.json'}")
    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    index_files = list(model_dir.glob("*.index.json"))
    if not weight_files and not index_files:
        raise FileNotFoundError(f"No model weight or shard-index files found under {model_dir}")


def _validate_cuda(world_size: int) -> dict[str, Any]:
    visible = int(torch.cuda.device_count())
    if visible < world_size:
        raise RuntimeError(f"Expected at least {world_size} visible GPUs, found {visible}.")
    if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
        raise RuntimeError("PyTorch distributed NCCL support is unavailable.")
    devices: list[dict[str, Any]] = []
    for index in range(world_size):
        major, minor = torch.cuda.get_device_capability(index)
        if major < 8:
            raise RuntimeError(
                f"GPU {index} ({torch.cuda.get_device_name(index)}) does not provide reliable bfloat16 support."
            )
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": f"{major}.{minor}",
                "total_memory_gb": round(torch.cuda.get_device_properties(index).total_memory / 2**30, 2),
            }
        )
    return {"visible_gpu_count": visible, "devices": devices}


def _validate_checkpoint_apis() -> None:
    from torch.utils.checkpoint import checkpoint
    from transformers.modeling_utils import PreTrainedModel

    if "use_reentrant" not in inspect.signature(checkpoint).parameters:
        raise RuntimeError("This PyTorch build lacks non-reentrant activation checkpointing support.")
    signature = inspect.signature(PreTrainedModel.gradient_checkpointing_enable)
    if "gradient_checkpointing_kwargs" not in signature.parameters:
        raise RuntimeError(
            "This Transformers build cannot request use_reentrant=False. Upgrade Transformers before training."
        )


def _load_valid_trajectories(path: Path, dataset_root: Path, expected_samples: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Fixed split is missing: {path}")
    trajectories = build_trajectories(
        steps=read_jsonl(path),
        dataset_root=dataset_root,
        max_samples=expected_samples,
        prep_log_every=max(1, expected_samples + 1),
    )
    prepared = sum(len(trajectory["rows"]) for trajectory in trajectories)
    if prepared != expected_samples:
        raise RuntimeError(
            f"Expected {expected_samples} usable samples in {path}, but trajectory preparation produced {prepared}."
        )
    missing_images: list[str] = []
    for trajectory in trajectories:
        for row in trajectory["rows"]:
            for key in ("image_path", "after_image_path"):
                image_path = Path(row[key])
                if not image_path.is_file():
                    missing_images.append(str(image_path))
                    if len(missing_images) >= 8:
                        break
            if len(missing_images) >= 8:
                break
        if len(missing_images) >= 8:
            break
    if missing_images:
        raise FileNotFoundError("Missing GUI images:\n  - " + "\n  - ".join(missing_images))
    return trajectories


def _validate_rank_shards(
    trajectories: list[dict[str, Any]],
    *,
    world_size: int,
    batch_size_per_rank: int,
) -> list[dict[str, int]]:
    shard_summaries: list[dict[str, int]] = []
    for rank in range(world_size):
        runtime = DistributedRuntime(
            enabled=True,
            rank=rank,
            local_rank=rank,
            world_size=world_size,
            device=torch.device("cpu"),
        )
        shard = shard_ordered_items(trajectories, runtime)
        rows = sum(len(trajectory["rows"]) for trajectory in shard)
        if not shard or rows <= 0:
            raise RuntimeError(
                f"Rank {rank} receives no trajectories; {world_size}-GPU DDP cannot proceed safely."
            )
        shard_summaries.append(
            {
                "rank": rank,
                "trajectory_count": len(shard),
                "row_count": rows,
                "max_active_trajectories": min(batch_size_per_rank, len(shard)),
            }
        )
    return shard_summaries


def main() -> None:
    args = parse_args()
    code_dir = Path(__file__).resolve().parent
    dataset_root = Path(args.dataset_root)
    model_dir = Path(args.model)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    _compile_runtime_files(code_dir)
    _validate_model_dir(model_dir)
    _validate_checkpoint_apis()
    cuda_summary = None if args.skip_gpu_check else _validate_cuda(max(1, int(args.world_size)))

    train_trajectories = _load_valid_trajectories(
        Path(args.train_steps), dataset_root, max(1, int(args.train_samples))
    )
    test_trajectories = _load_valid_trajectories(
        Path(args.test_steps), dataset_root, max(1, int(args.test_samples))
    )
    train_keys = {str(item["trajectory_key"]) for item in train_trajectories}
    test_keys = {str(item["trajectory_key"]) for item in test_trajectories}
    overlap = sorted(train_keys & test_keys)
    if overlap:
        raise RuntimeError(f"Train/test trajectory overlap detected: {overlap[:8]}")

    shard_summary = _validate_rank_shards(
        train_trajectories,
        world_size=max(1, int(args.world_size)),
        batch_size_per_rank=max(1, int(args.batch_size_per_rank)),
    )

    write_probe = output_root / ".preflight_write_probe"
    write_probe.write_text("ok", encoding="utf-8")
    write_probe.unlink()
    disk = shutil.disk_usage(output_root)
    free_gb = disk.free / 2**30
    if free_gb < float(args.min_free_gb):
        raise RuntimeError(
            f"Only {free_gb:.1f} GiB free under {output_root}; at least {args.min_free_gb:.1f} GiB is required."
        )

    print(
        json.dumps(
            {
                "stage": "lara_clean_preflight_passed",
                "torch_version": torch.__version__,
                "cuda": cuda_summary,
                "train_samples": sum(len(item["rows"]) for item in train_trajectories),
                "train_trajectories": len(train_trajectories),
                "test_samples": sum(len(item["rows"]) for item in test_trajectories),
                "test_trajectories": len(test_trajectories),
                "rank_shards": shard_summary,
                "free_disk_gb": round(free_gb, 2),
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
