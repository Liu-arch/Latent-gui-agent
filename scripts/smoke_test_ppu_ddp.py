from __future__ import annotations

import torch
import torch.distributed as dist

from qwen3_gui_agent.distributed_training import (
    cleanup_distributed,
    init_distributed,
)


def main() -> None:
    runtime = init_distributed()
    if not runtime.enabled:
        raise RuntimeError("Run this check with torchrun and at least two processes")

    try:
        value = torch.tensor(
            [float(runtime.rank + 1)],
            dtype=torch.float32,
            device=runtime.device,
        )
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = runtime.world_size * (runtime.world_size + 1) / 2
        if float(value.item()) != float(expected):
            raise RuntimeError(
                f"all_reduce mismatch: expected={expected}, actual={value.item()}"
            )
        dist.barrier()
        print(
            f"rank={runtime.rank} local_rank={runtime.local_rank} "
            f"device={torch.cuda.get_device_name(runtime.local_rank)} "
            f"all_reduce={value.item()} PASS",
            flush=True,
        )
    finally:
        cleanup_distributed(runtime)


if __name__ == "__main__":
    main()
