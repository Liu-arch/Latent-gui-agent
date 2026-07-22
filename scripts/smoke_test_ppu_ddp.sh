#!/usr/bin/env bash
set -euo pipefail

PPU_WORLD_SIZE="${PPU_WORLD_SIZE:-2}"
if [[ "${PPU_WORLD_SIZE}" -lt 2 ]]; then
  echo "PPU_WORLD_SIZE must be at least 2" >&2
  exit 2
fi

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${PPU_WORLD_SIZE}" \
  scripts/smoke_test_ppu_ddp.py

echo "[ppu-ddp-smoke] ${PPU_WORLD_SIZE}-device NCCL check passed"
