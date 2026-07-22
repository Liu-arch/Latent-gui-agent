from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable

import torch

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
except Exception:  # pragma: no cover
    dist = None  # type: ignore[assignment]
    DDP = None  # type: ignore[assignment]


@dataclass
class DistributedRuntime:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedRuntime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return DistributedRuntime(
            enabled=False,
            rank=0,
            local_rank=0,
            world_size=1,
            device=device,
        )

    if dist is None or DDP is None:
        raise ImportError("torch.distributed is required for multi-GPU training.")
    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA.")
    if not dist.is_initialized():
        timeout_seconds = max(600, int(os.environ.get("DDP_TIMEOUT_SECONDS", "3600")))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
    torch.cuda.set_device(local_rank)
    return DistributedRuntime(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def cleanup_distributed(runtime: DistributedRuntime, *, skip_destroy: bool = False) -> None:
    if skip_destroy:
        # A CUDA device-side assert poisons the context. Calling the synchronous
        # NCCL destroy path then blocks healthy peers until the full DDP timeout.
        # Let torchrun observe this process exit and terminate its peers instead.
        return
    if runtime.enabled and dist is not None and dist.is_initialized():
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_disable_device_map_for_ddp(device_map: Any, runtime: DistributedRuntime) -> Any:
    if runtime.enabled:
        return None
    return device_map


def maybe_wrap_ddp(module: torch.nn.Module, runtime: DistributedRuntime) -> torch.nn.Module:
    if not runtime.enabled:
        return module
    assert DDP is not None
    module.to(runtime.device)
    return DDP(
        module,
        device_ids=[runtime.local_rank],
        output_device=runtime.local_rank,
        broadcast_buffers=False,
        # This training stack has data-dependent branches (history length,
        # temporal reuse, partially-filled active trajectory batches). Some
        # trainable parameters may be unused on a given rank for a given step.
        # Keeping this False causes intermittent cross-rank hangs/timeouts.
        find_unused_parameters=True,
    )


def _item_shard_weight(item: Any) -> int:
    if isinstance(item, dict):
        rows = item.get("rows")
        if isinstance(rows, list):
            return max(1, len(rows))
    return 1


def shard_ordered_items(items: list[Any], runtime: DistributedRuntime) -> list[Any]:
    if not runtime.enabled:
        return list(items)
    indexed_items = list(enumerate(items))
    sorted_indices = sorted(
        indexed_items,
        key=lambda pair: (-_item_shard_weight(pair[1]), pair[0]),
    )
    rank_loads = [0 for _ in range(runtime.world_size)]
    rank_assignments: list[list[tuple[int, Any]]] = [[] for _ in range(runtime.world_size)]
    for original_index, item in sorted_indices:
        target_rank = min(range(runtime.world_size), key=lambda rank: (rank_loads[rank], rank))
        rank_assignments[target_rank].append((original_index, item))
        rank_loads[target_rank] += _item_shard_weight(item)
    rank_assignments[runtime.rank].sort(key=lambda pair: pair[0])
    return [item for _, item in rank_assignments[runtime.rank]]


def epoch_ordered_trajectories(
    trajectories: list[dict[str, Any]],
    *,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    ordered = list(trajectories)
    if shuffle:
        random.Random(seed + epoch).shuffle(ordered)
    return ordered


def sum_across_ranks(value: float | int, runtime: DistributedRuntime) -> float:
    if not runtime.enabled:
        return float(value)
    assert dist is not None
    tensor = torch.tensor(float(value), device=runtime.device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def gather_objects(obj: Any, runtime: DistributedRuntime) -> list[Any]:
    if not runtime.enabled:
        return [obj]
    assert dist is not None
    gathered: list[Any] = [None for _ in range(runtime.world_size)]
    dist.all_gather_object(gathered, obj)
    return gathered


def weighted_merge_numeric_dicts(
    summaries: Iterable[dict[str, Any]],
    weights: Iterable[float | int],
) -> dict[str, Any]:
    summary_list = list(summaries)
    weight_list = [float(weight) for weight in weights]
    total_weight = sum(weight_list)
    if not summary_list or total_weight <= 0:
        return {}

    merged: dict[str, Any] = {}
    keys: set[str] = set()
    for summary in summary_list:
        keys.update(summary.keys())

    for key in sorted(keys):
        values = [summary.get(key) for summary in summary_list]
        exemplar = next((value for value in values if value is not None), None)
        if exemplar is None:
            continue
        if isinstance(exemplar, (int, float)):
            numerator = 0.0
            for value, weight in zip(values, weight_list):
                numerator += float(value or 0.0) * weight
            merged[key] = numerator / total_weight
            continue
        if isinstance(exemplar, list):
            max_len = max((len(value) if isinstance(value, list) else 0) for value in values)
            merged_list: list[float] = []
            for idx in range(max_len):
                numerator = 0.0
                for value, weight in zip(values, weight_list):
                    if isinstance(value, list) and idx < len(value):
                        numerator += float(value[idx]) * weight
                merged_list.append(numerator / total_weight)
            merged[key] = merged_list
    return merged
